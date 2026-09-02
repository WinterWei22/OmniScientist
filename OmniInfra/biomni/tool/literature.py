import ipaddress
import json
import os
import re
import socket
import time
from datetime import date
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlencode, urljoin, urlsplit

import PyPDF2
import requests
from bs4 import BeautifulSoup

from biomni.config import default_config

DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_HTML_LIMIT_BYTES = 5 * 1024 * 1024
DEFAULT_DOWNLOAD_LIMIT_BYTES = 50 * 1024 * 1024
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
CROSSREF_WORKS_URL = "https://api.crossref.org/works"
BIORXIV_DETAILS_URL = "https://api.biorxiv.org/details/biorxiv"
PUBLIC_ENDPOINT_REGISTRY = {
    "chembl": {
        "official_hosts": {"www.ebi.ac.uk"},
        "api_path_prefixes": (
            "/chembl/api/data/activity",
            "/chembl/api/data/assay",
            "/chembl/api/data/atc_class",
            "/chembl/api/data/drug",
            "/chembl/api/data/drug_indication",
            "/chembl/api/data/mechanism",
            "/chembl/api/data/molecule",
            "/chembl/api/data/target",
        ),
    },
    "rcsb_files": {
        "official_hosts": {"files.rcsb.org"},
        "api_path_prefixes": ("/download",),
    },
}


def _validate_public_url(url: str) -> None:
    """Reject non-HTTP and non-public destinations before making a request."""
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("URL must use http or https and include a hostname")
    if parsed.username or parsed.password:
        raise ValueError("URL credentials are not allowed")

    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, parsed.port or 443)}
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve URL hostname: {parsed.hostname}") from exc

    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError(f"URL resolves to a non-public address: {address}")


def _fetch_public_url(
    url: str,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_HTML_LIMIT_BYTES,
    headers: dict | None = None,
    raise_for_status: bool = True,
) -> tuple[requests.Response, bytes, str]:
    """Fetch a public URL with redirect validation and a bounded response body."""
    request_headers = {"User-Agent": "Biomni/1.0 (+https://github.com/snap-stanford/biomni)"}
    if headers:
        request_headers.update(headers)

    current_url = url
    for _ in range(6):
        _validate_public_url(current_url)
        response = requests.get(
            current_url,
            headers=request_headers,
            timeout=timeout_seconds,
            allow_redirects=False,
            stream=True,
        )
        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("Location")
            if not location:
                raise requests.RequestException("Redirect response did not include a Location header")
            current_url = urljoin(current_url, location)
            response.close()
            continue

        if raise_for_status:
            response.raise_for_status()
        content_length = int(response.headers.get("Content-Length", 0))
        if content_length > max_bytes:
            response.close()
            raise ValueError(f"Response exceeds the {max_bytes}-byte limit")

        chunks = []
        downloaded_bytes = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            downloaded_bytes += len(chunk)
            if downloaded_bytes > max_bytes:
                response.close()
                raise ValueError(f"Response exceeds the {max_bytes}-byte limit")
            chunks.append(chunk)
        return response, b"".join(chunks), current_url

    raise requests.TooManyRedirects("Too many redirects")


def _registered_public_endpoint(url: str) -> str | None:
    """Return a registered source ID for an exact public API resource family."""
    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").lower()
    path = parsed.path.lower().rstrip("/")
    for source_id, descriptor in PUBLIC_ENDPOINT_REGISTRY.items():
        if hostname not in descriptor["official_hosts"]:
            continue
        if any(
            path in {prefix, f"{prefix}.json", f"{prefix}.xml", f"{prefix}.yaml"} or path.startswith(f"{prefix}/")
            for prefix in descriptor["api_path_prefixes"]
        ):
            return source_id
    return None


def _safe_download_name(url: str, content_disposition: str | None, index: int) -> str:
    """Return a filesystem-safe attachment name without overwriting paths."""
    candidate = ""
    if content_disposition:
        match = re.search(r"filename\*?=(?:UTF-8''|\")?([^\";]+)", content_disposition, flags=re.IGNORECASE)
        if match:
            candidate = unquote(match.group(1).strip())
    if not candidate:
        candidate = unquote(Path(urlsplit(url).path).name)
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(candidate).name).strip("._")
    return candidate or f"supplementary_{index}.bin"


def fetch_supplementary_info_from_doi(
    doi: str,
    output_dir: str = "supplementary_info",
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_download_bytes: int = DEFAULT_DOWNLOAD_LIMIT_BYTES,
) -> dict:
    """Fetch supplementary information for a DOI with bounded, safe downloads.

    Args:
        doi: The paper DOI.
        output_dir: Directory to save supplementary files.

    Returns:
        dict: A dictionary containing a research log and the downloaded file paths.

    """
    research_log = [f"Starting process for DOI: {doi}"]
    query_info = {"doi": doi, "output_dir": output_dir}

    if not isinstance(doi, str) or not doi.strip():
        return {"success": False, "error": "doi must be a non-empty string", "query_info": query_info}
    if timeout_seconds <= 0 or max_download_bytes <= 0:
        return {
            "success": False,
            "error": "timeout_seconds and max_download_bytes must be positive",
            "query_info": query_info,
        }

    # CrossRef API to resolve DOI to a publisher page
    crossref_url = f"https://doi.org/{doi}"
    try:
        _, publisher_content, publisher_url = _fetch_public_url(
            crossref_url,
            timeout_seconds=timeout_seconds,
            max_bytes=DEFAULT_HTML_LIMIT_BYTES,
        )
        research_log.append(f"Resolved DOI to publisher page: {publisher_url}")
        soup = BeautifulSoup(publisher_content, "html.parser")
    except (requests.RequestException, ValueError) as exc:
        research_log.append(f"Failed to resolve or access DOI {doi}: {exc}")
        return {
            "success": False,
            "error": str(exc),
            "query_info": query_info,
            "result": {"log": research_log, "files": []},
        }
    supplementary_links = []

    # Look for supplementary materials by keywords or links
    for link in soup.find_all("a", href=True):
        href = link.get("href")
        text = link.get_text().lower()
        if "supplementary" in text or "supplemental" in text or "appendix" in text:
            full_url = urljoin(publisher_url, href)
            supplementary_links.append(full_url)
            research_log.append(f"Found supplementary material link: {full_url}")

    if not supplementary_links:
        log_message = f"No supplementary materials found for DOI {doi}."
        research_log.append(log_message)
        return {
            "success": True,
            "query_info": query_info,
            "result": {"log": research_log, "files": [], "links_found": 0},
        }

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    research_log.append(f"Created output directory: {output_dir}")

    # Download supplementary materials
    downloaded_files = []
    for index, link in enumerate(dict.fromkeys(supplementary_links), start=1):
        try:
            file_response, file_content, final_url = _fetch_public_url(
                link,
                timeout_seconds=timeout_seconds,
                max_bytes=max_download_bytes,
            )
            file_name = _safe_download_name(
                final_url,
                file_response.headers.get("Content-Disposition"),
                index,
            )
            file_path = Path(output_dir) / file_name
            if file_path.exists():
                file_path = file_path.with_name(f"{file_path.stem}_{index}{file_path.suffix}")
            file_path.write_bytes(file_content)
            downloaded_files.append(str(file_path))
            research_log.append(f"Downloaded file: {file_path}")
        except (requests.RequestException, ValueError, OSError) as exc:
            research_log.append(f"Failed to download file from {link}: {exc}")

    if downloaded_files:
        research_log.append(f"Successfully downloaded {len(downloaded_files)} file(s).")
    else:
        research_log.append(f"No files could be downloaded for DOI {doi}.")

    return {
        "success": True,
        "query_info": query_info,
        "result": {
            "log": research_log,
            "files": downloaded_files,
            "links_found": len(supplementary_links),
        },
    }


def query_arxiv(query: str, max_papers: int = 10) -> dict:
    """Query arXiv for papers based on the provided search query.

    Parameters
    ----------
    - query (str): The search query string.
    - max_papers (int): The maximum number of papers to retrieve (default: 10).

    Returns
    -------
    - dict: Structured papers or a clear error.

    """
    try:
        import arxiv

        if not query.strip() or not 1 <= max_papers <= 100:
            return {"success": False, "error": "query must be non-empty and max_papers must be between 1 and 100"}
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_papers, sort_by=arxiv.SortCriterion.Relevance)
        papers = [
            {
                "title": paper.title,
                "summary": paper.summary,
                "entry_id": paper.entry_id,
                "published": paper.published.isoformat() if paper.published else None,
            }
            for paper in client.results(search)
        ]
        return {
            "success": True,
            "query_info": {"query": query, "max_papers": max_papers, "source": "arXiv"},
            "result": {"papers": papers, "count": len(papers)},
        }
    except ImportError:
        return {
            "success": False,
            "error": "The optional 'arxiv' package is not installed. Rebuild or update the Biomni environment.",
        }
    except Exception as e:
        return {"success": False, "error": f"Error querying arXiv: {e}"}


def query_biorxiv(
    query: str,
    max_papers: int = 10,
    start_date: str | None = None,
    end_date: str | None = None,
    category: str | None = None,
    max_records: int = 300,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """Query bioRxiv preprint metadata through the official bioRxiv API.

    A DOI is looked up directly. A keyword query scans metadata in the supplied
    date interval, or the most recent 30 days when dates are omitted, and ranks
    locally matched title, abstract, author, category, and DOI fields.

    Args:
        query: Keyword query or a bioRxiv DOI.
        max_papers: Maximum number of matching preprints to return.
        start_date: Inclusive interval start in YYYY-MM-DD format.
        end_date: Inclusive interval end in YYYY-MM-DD format.
        category: Optional bioRxiv subject category filter.
        max_records: Maximum metadata records to scan for a keyword query.
        timeout_seconds: HTTP timeout for each API request.

    Returns:
        A serializable dictionary containing query metadata and preprints.

    """
    normalized_query = query.strip() if isinstance(query, str) else ""
    normalized_category = category.strip() if isinstance(category, str) else category
    query_info = {
        "query": normalized_query,
        "source": "bioRxiv",
        "api_url": BIORXIV_DETAILS_URL,
        "start_date": start_date,
        "end_date": end_date,
        "category": normalized_category or None,
        "max_papers": max_papers,
        "max_records": max_records,
    }

    if not normalized_query:
        return {"success": False, "error": "query must be a non-empty string", "query_info": query_info}
    if not isinstance(max_papers, int) or not 1 <= max_papers <= 100:
        return {
            "success": False,
            "error": "max_papers must be an integer between 1 and 100",
            "query_info": query_info,
        }
    if not isinstance(max_records, int) or not 30 <= max_records <= 3000:
        return {
            "success": False,
            "error": "max_records must be an integer between 30 and 3000",
            "query_info": query_info,
        }
    if not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= 120:
        return {
            "success": False,
            "error": "timeout_seconds must be an integer between 1 and 120",
            "query_info": query_info,
        }
    if category is not None and not isinstance(category, str):
        return {"success": False, "error": "category must be a string or None", "query_info": query_info}
    if normalized_category is not None and not normalized_category:
        return {"success": False, "error": "category must not be blank", "query_info": query_info}

    doi_query = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", normalized_query, flags=re.IGNORECASE).rstrip("/")
    is_doi = bool(re.fullmatch(r"10\.1101/[A-Za-z0-9._()/:+-]+", doi_query, flags=re.IGNORECASE))

    if is_doi:
        interval_path = f"{doi_query}/na"
        query_info["lookup_type"] = "doi"
        query_info["doi"] = doi_query
    else:
        if (start_date is None) != (end_date is None):
            return {
                "success": False,
                "error": "start_date and end_date must either both be provided or both be omitted",
                "query_info": query_info,
            }
        if start_date is not None:
            try:
                parsed_start = date.fromisoformat(start_date)
                parsed_end = date.fromisoformat(end_date)
            except (TypeError, ValueError):
                return {
                    "success": False,
                    "error": "start_date and end_date must use YYYY-MM-DD format",
                    "query_info": query_info,
                }
            if parsed_start > parsed_end:
                return {
                    "success": False,
                    "error": "start_date must be earlier than or equal to end_date",
                    "query_info": query_info,
                }
            interval_path = f"{start_date}/{end_date}"
        else:
            interval_path = "30d"
        query_info["lookup_type"] = "keyword"
        query_info["interval"] = interval_path

    def fetch_page(cursor: int) -> tuple[list[dict], dict]:
        page_url = (
            f"{BIORXIV_DETAILS_URL}/{interval_path}/json"
            if is_doi
            else f"{BIORXIV_DETAILS_URL}/{interval_path}/{cursor}/json"
        )
        params = {"category": normalized_category} if normalized_category else None
        response = requests.get(
            page_url,
            params=params,
            headers={"User-Agent": "Biomni/1.0 (+https://github.com/snap-stanford/biomni)"},
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("bioRxiv API returned a non-object JSON response")
        messages = payload.get("messages") or []
        message = messages[0] if messages and isinstance(messages[0], dict) else {}
        if message.get("status") not in {None, "ok"}:
            raise ValueError(f"bioRxiv API error: {message.get('status')}")
        collection = payload.get("collection") or []
        if not isinstance(collection, list):
            raise ValueError("bioRxiv API returned an invalid collection")
        return collection, message

    try:
        records = []
        cursor = 0
        total_available = None
        while True:
            page, message = fetch_page(cursor)
            if total_available is None:
                try:
                    total_available = int(message.get("total"))
                except (TypeError, ValueError):
                    total_available = None
            if is_doi:
                records.extend(page)
                break

            remaining = max_records - len(records)
            records.extend(page[:remaining])
            cursor += len(page)
            if not page or len(records) >= max_records or (total_available is not None and cursor >= total_available):
                break
    except requests.Timeout:
        return {"success": False, "error": "bioRxiv API request timed out", "query_info": query_info}
    except requests.RequestException as exc:
        return {
            "success": False,
            "error": f"Could not query the bioRxiv API: {exc}",
            "query_info": query_info,
        }
    except ValueError as exc:
        return {"success": False, "error": str(exc), "query_info": query_info}

    query_terms = re.findall(r"[\w]+", normalized_query.casefold())

    def score_record(record: dict) -> int | None:
        if is_doi:
            return 1
        title = str(record.get("title") or "").casefold()
        abstract = str(record.get("abstract") or "").casefold()
        metadata = " ".join(str(record.get(field) or "").casefold() for field in ("authors", "category", "doi"))
        searchable = f"{title} {abstract} {metadata}"
        if not query_terms or not all(term in searchable for term in query_terms):
            return None
        phrase = normalized_query.casefold()
        return (
            (12 if phrase in title else 0)
            + (5 if phrase in abstract else 0)
            + sum(4 for term in query_terms if term in title)
            + sum(2 for term in query_terms if term in abstract)
            + sum(1 for term in query_terms if term in metadata)
        )

    matches_by_doi = {}
    for record in records:
        score = score_record(record)
        if score is None:
            continue
        doi = str(record.get("doi") or "")
        version = str(record.get("version") or "")
        paper = {
            "title": record.get("title"),
            "authors": record.get("authors"),
            "abstract": record.get("abstract"),
            "doi": doi or None,
            "date": record.get("date"),
            "version": version or None,
            "type": record.get("type"),
            "category": record.get("category"),
            "license": record.get("license"),
            "published_doi": None if record.get("published") in {None, "NA"} else record.get("published"),
            "url": f"https://www.biorxiv.org/content/{doi}v{version}" if doi and version else None,
            "jatsxml": record.get("jatsxml"),
            "relevance_score": score,
        }
        previous = matches_by_doi.get(doi)
        try:
            current_version = int(version)
            previous_version = int(previous["version"]) if previous else -1
        except (TypeError, ValueError):
            current_version = previous_version = 0
        if previous is None or current_version >= previous_version:
            matches_by_doi[doi] = paper

    papers = sorted(
        matches_by_doi.values(),
        key=lambda paper: (paper["relevance_score"], paper.get("date") or ""),
        reverse=True,
    )[:max_papers]
    scanned_count = len(records)
    query_info["scanned_records"] = scanned_count
    query_info["total_available"] = total_available
    query_info["scan_truncated"] = bool(not is_doi and total_available is not None and scanned_count < total_available)
    return {
        "success": True,
        "query_info": query_info,
        "result": {"papers": papers, "count": len(papers)},
    }


def _query_crossref_scholar_fallback(query: str, primary_error: str) -> dict | None:
    """Return the best Crossref work when anonymous OpenAlex is unavailable."""
    query_info = {
        "query": query,
        "source": "Crossref fallback",
        "api_url": CROSSREF_WORKS_URL,
        "authenticated": False,
        "api_key_source": "not_required",
        "primary_error": primary_error,
    }
    try:
        response = requests.get(
            CROSSREF_WORKS_URL,
            params={
                "query.bibliographic": query,
                "rows": 1,
                "select": "DOI,title,author,published-print,published-online,container-title,URL,is-referenced-by-count,abstract",
            },
            headers={"User-Agent": "Biomni/1.0 (+https://github.com/snap-stanford/biomni; mailto:biomni@stanford.edu)"},
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        items = payload.get("message", {}).get("items", []) if isinstance(payload, dict) else []
        if not isinstance(items, list):
            return None
        if not items:
            return {"success": True, "query_info": query_info, "result": None}
        work = items[0]
        if not isinstance(work, dict):
            return None
        title = work.get("title") or []
        venue = work.get("container-title") or []
        published = work.get("published-print") or work.get("published-online") or {}
        date_parts = published.get("date-parts") or [] if isinstance(published, dict) else []
        year = date_parts[0][0] if date_parts and date_parts[0] else None
        authors = []
        for author in work.get("author") or []:
            if not isinstance(author, dict):
                continue
            display_name = " ".join(
                part for part in (author.get("given", "").strip(), author.get("family", "").strip()) if part
            )
            if display_name:
                authors.append(display_name)
        doi = work.get("DOI")
        abstract = work.get("abstract")
        if isinstance(abstract, str):
            abstract = BeautifulSoup(abstract, "html.parser").get_text(" ", strip=True) or None
        return {
            "success": True,
            "query_info": query_info,
            "result": {
                "title": title[0] if isinstance(title, list) and title else None,
                "year": year,
                "venue": venue[0] if isinstance(venue, list) and venue else None,
                "abstract": abstract,
                "url": work.get("URL") or (f"https://doi.org/{doi}" if doi else None),
                "doi": f"https://doi.org/{doi}" if doi else None,
                "openalex_id": None,
                "authors": authors,
                "citation_count": work.get("is-referenced-by-count"),
            },
        }
    except (requests.RequestException, ValueError, TypeError):
        return None


def query_scholar(query: str) -> dict:
    """Query OpenAlex for the first work matching a scholarly search query.

    Parameters
    ----------
    - query (str): The search query string.

    Returns
    -------
    - dict: The first structured result or a clear error.

    """
    custom_api_key = os.getenv("OPENALEX_API_KEY", "").strip() or None
    api_key = custom_api_key or default_config.openalex_api_key
    api_key_source = "environment" if custom_api_key else "default_config" if api_key else "anonymous"
    query_info = {
        "query": query,
        "source": "OpenAlex",
        "api_url": OPENALEX_WORKS_URL,
        "authenticated": bool(api_key),
        "api_key_source": api_key_source,
    }
    if not isinstance(query, str) or not query.strip():
        return {"success": False, "error": "query must be a non-empty string", "query_info": query_info}

    params = {
        "search": query.strip(),
        "per-page": 1,
        "select": (
            "id,display_name,publication_year,authorships,primary_location,doi,cited_by_count,abstract_inverted_index"
        ),
    }
    if api_key:
        params["api_key"] = api_key

    try:
        response = requests.get(
            OPENALEX_WORKS_URL,
            params=params,
            headers={"User-Agent": "Biomni/1.0 (+https://github.com/snap-stanford/biomni)"},
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            try:
                error_payload = response.json()
                if isinstance(error_payload, dict):
                    error_detail = error_payload.get("message") or error_payload.get("error") or str(error_payload)
                else:
                    error_detail = str(error_payload)
            except ValueError:
                error_detail = response.text.strip()[:300]
            if api_key:
                error_detail = error_detail.replace(api_key, "[REDACTED]")
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                retry_advice = (
                    f" Retry after {retry_after} seconds." if retry_after else " Wait for the daily quota to reset."
                )
                primary_error = f"OpenAlex API rate limit exceeded (HTTP 429).{retry_advice} {error_detail}"
                if not api_key:
                    fallback = _query_crossref_scholar_fallback(query.strip(), primary_error)
                    if fallback is not None:
                        return fallback
                return {
                    "success": False,
                    "error": primary_error,
                    "query_info": query_info,
                    "suggestions": [
                        "Retry later or configure another free key with OPENALEX_API_KEY.",
                        "Use query_pubmed for biomedical literature or query_arxiv for preprints.",
                        "Use Crossref or Europe PMC directly when their coverage fits the query.",
                    ],
                }
            return {
                "success": False,
                "error": f"OpenAlex API returned HTTP {response.status_code}: {error_detail}",
                "query_info": query_info,
                "suggestions": [
                    "Check OpenAlex service status and retry later.",
                    "Use query_pubmed for biomedical literature or query_arxiv for preprints.",
                    "Use Crossref or Europe PMC directly when their coverage fits the query.",
                ],
            }

        payload = response.json()
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            return {
                "success": False,
                "error": "OpenAlex API returned an invalid response payload.",
                "query_info": query_info,
                "suggestions": [
                    "Retry later because the OpenAlex response format may be temporarily unavailable.",
                    "Use query_pubmed for biomedical literature or query_arxiv for preprints.",
                ],
            }
        if not results:
            return {"success": True, "query_info": query_info, "result": None}

        work = results[0]
        if not isinstance(work, dict):
            return {
                "success": False,
                "error": "OpenAlex API returned an invalid work record.",
                "query_info": query_info,
                "suggestions": [
                    "Retry with a different query.",
                    "Use query_pubmed for biomedical literature or query_arxiv for preprints.",
                ],
            }

        abstract = None
        abstract_index = work.get("abstract_inverted_index")
        if isinstance(abstract_index, dict):
            positioned_words = []
            for word, positions in abstract_index.items():
                if isinstance(word, str) and isinstance(positions, list):
                    positioned_words.extend(
                        (position, word) for position in positions if isinstance(position, int) and position >= 0
                    )
            if positioned_words:
                abstract = " ".join(word for _, word in sorted(positioned_words))

        primary_location = work.get("primary_location") or {}
        source = primary_location.get("source") or {} if isinstance(primary_location, dict) else {}
        authorships = work.get("authorships") or []
        authors = [
            authorship["author"]["display_name"]
            for authorship in authorships
            if isinstance(authorship, dict)
            and isinstance(authorship.get("author"), dict)
            and isinstance(authorship["author"].get("display_name"), str)
        ]
        url = primary_location.get("landing_page_url") if isinstance(primary_location, dict) else None
        return {
            "success": True,
            "query_info": query_info,
            "result": {
                "title": work.get("display_name"),
                "year": work.get("publication_year"),
                "venue": source.get("display_name") if isinstance(source, dict) else None,
                "abstract": abstract,
                "url": url or work.get("doi") or work.get("id"),
                "doi": work.get("doi"),
                "openalex_id": work.get("id"),
                "authors": authors,
                "citation_count": work.get("cited_by_count"),
            },
        }
    except requests.Timeout:
        return {
            "success": False,
            "error": f"OpenAlex API request timed out after {DEFAULT_TIMEOUT_SECONDS} seconds.",
            "query_info": query_info,
            "suggestions": [
                "Check network connectivity and retry later.",
                "Use query_pubmed for biomedical literature or query_arxiv for preprints.",
                "Use Crossref or Europe PMC directly when their coverage fits the query.",
            ],
        }
    except requests.ConnectionError as exc:
        error_detail = str(exc)
        if api_key:
            error_detail = error_detail.replace(api_key, "[REDACTED]")
        return {
            "success": False,
            "error": f"Could not connect to the OpenAlex API: {error_detail}",
            "query_info": query_info,
            "suggestions": [
                "Check DNS, firewall, proxy, and internet connectivity, then retry.",
                "Use query_pubmed for biomedical literature or query_arxiv for preprints.",
                "Use Crossref or Europe PMC directly when their coverage fits the query.",
            ],
        }
    except (requests.RequestException, ValueError) as exc:
        error_detail = str(exc)
        if api_key:
            error_detail = error_detail.replace(api_key, "[REDACTED]")
        return {
            "success": False,
            "error": f"Error querying OpenAlex: {error_detail}",
            "query_info": query_info,
            "suggestions": [
                "Retry later or verify the OpenAlex API response.",
                "Use query_pubmed for biomedical literature or query_arxiv for preprints.",
                "Use Crossref or Europe PMC directly when their coverage fits the query.",
            ],
        }


def query_pubmed(
    query: str,
    max_papers: int = 10,
    max_retries: int = 3,
    email: str | None = None,
) -> dict:
    """Query PubMed for papers based on the provided search query.

    Parameters
    ----------
    - query (str): The search query string.
    - max_papers (int): The maximum number of papers to retrieve (default: 10).
    - max_retries (int): Maximum number of retry attempts with modified queries (default: 3).

    Returns
    -------
    - dict: Structured papers or a clear error.

    """
    try:
        from pymed import PubMed

        if not query.strip() or not 1 <= max_papers <= 100 or not 0 <= max_retries <= 10:
            return {"success": False, "error": "Invalid query, max_papers, or max_retries"}

        contact_email = email or os.getenv("BIOMNI_PUBMED_EMAIL")
        pubmed_kwargs = {"tool": "Biomni"}
        if contact_email:
            pubmed_kwargs["email"] = contact_email
        pubmed = PubMed(**pubmed_kwargs)

        # Initial attempt
        papers = list(pubmed.query(query, max_results=max_papers))

        # Retry with modified queries if no results
        retries = 0
        while not papers and retries < max_retries:
            retries += 1
            # Simplify query with each retry by removing the last word
            simplified_query = " ".join(query.split()[:-retries]) if len(query.split()) > retries else query
            time.sleep(1)  # Add delay between requests
            papers = list(pubmed.query(simplified_query, max_results=max_papers))

        results = [
            {
                "title": paper.title,
                "abstract": paper.abstract,
                "journal": paper.journal,
                "doi": getattr(paper, "doi", None),
                "pubmed_id": getattr(paper, "pubmed_id", None),
            }
            for paper in papers
        ]
        return {
            "success": True,
            "query_info": {"query": query, "max_papers": max_papers, "source": "PubMed"},
            "result": {"papers": results, "count": len(results)},
        }
    except ImportError:
        return {"success": False, "error": "The optional 'pymed' package is not installed."}
    except Exception as e:
        return {"success": False, "error": f"Error querying PubMed: {e}"}


def search_web(
    query: str,
    num_results: int = 3,
    language: str = "en",
    provider: str = "bing",
    timeout_seconds: int = 15,
) -> dict:
    """Search the web through Bing or Baidu and return structured results.

    Args:
        query: The search query.
        num_results: Number of results to return (1-20).
        language: Language code for search results.
        provider: Search provider, either ``bing`` or ``baidu``.
        timeout_seconds: HTTP request timeout.

    Returns:
        A structured result containing titles, direct URLs, and snippets.

    """
    query_info = {
        "query": query,
        "num_results": num_results,
        "language": language,
        "provider": provider,
    }
    if not isinstance(query, str) or not query.strip():
        return {
            "success": False,
            "provider": provider,
            "query": query,
            "results": [],
            "error": "query must be a non-empty string",
            "query_info": query_info,
        }
    if not 1 <= num_results <= 20:
        return {
            "success": False,
            "provider": provider,
            "query": query,
            "results": [],
            "error": "num_results must be between 1 and 20",
            "query_info": query_info,
        }
    if provider not in {"bing", "baidu"}:
        return {
            "success": False,
            "provider": provider,
            "query": query,
            "results": [],
            "error": "provider must be 'bing' or 'baidu'",
            "query_info": query_info,
        }

    if provider == "bing":
        search_url = "https://www.bing.com/search?" + urlencode({"q": query, "count": num_results, "setlang": language})
    else:
        search_url = "https://www.baidu.com/s?" + urlencode({"wd": query, "rn": num_results})

    try:
        last_requests = getattr(search_web, "_last_request_times", {})
        elapsed = time.monotonic() - last_requests.get(provider, 0.0)
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        last_requests[provider] = time.monotonic()
        search_web._last_request_times = last_requests

        _, content, _ = _fetch_public_url(
            search_url,
            timeout_seconds=timeout_seconds,
            max_bytes=DEFAULT_HTML_LIMIT_BYTES,
            headers={"Accept-Language": language},
        )
        soup = BeautifulSoup(content, "html.parser")
        results = []
        selectors = "li.b_algo" if provider == "bing" else "div.result, div.c-container"
        for item in soup.select(selectors):
            link = item.select_one("h2 a") if provider == "bing" else item.select_one("h3 a")
            if not link or not link.get("href"):
                continue
            snippet = (
                item.select_one("p") if provider == "bing" else item.select_one(".c-abstract, .content-right_8Zs40")
            )
            results.append(
                {
                    "title": link.get_text(" ", strip=True),
                    "url": urljoin(search_url, link["href"]),
                    "description": snippet.get_text(" ", strip=True) if snippet else "",
                }
            )
            if len(results) >= num_results:
                break
        if not results:
            return {
                "success": False,
                "provider": provider,
                "query": query,
                "results": [],
                "error": f"{provider} returned no parsable results; its HTML layout may have changed",
                "query_info": query_info,
            }
        return {
            "success": True,
            "provider": provider,
            "query": query,
            "results": results,
            "error": None,
            "query_info": query_info,
            "result": {"results": results, "count": len(results)},
        }
    except (requests.RequestException, ValueError) as exc:
        return {
            "success": False,
            "provider": provider,
            "query": query,
            "results": [],
            "error": f"{provider} search failed: {exc}",
            "query_info": query_info,
        }


def search_google(query: str, num_results: int = 3, language: str = "en") -> dict:
    """Deprecated compatibility alias that now searches with Bing."""
    result = search_web(query, num_results=num_results, language=language, provider="bing")
    result["deprecated_alias"] = "search_google"
    return result


def advanced_web_search_claude(
    query: str,
    max_searches: int = 1,
    max_retries: int = 3,
) -> tuple[str, list[dict[str, str]], list]:
    """
    Initiate an advanced web search by launching a specialized agent to collect relevant information and citations through multiple rounds of web searches for a given query.
    Craft the query carefully for the search agent to find the most relevant information.

    Parameters
    ----------
    query : str
        The search phrase you want Claude to look up.
    max_searches : int, optional
        Upper-bound on searches Claude may issue inside this request.
    max_retries : int, optional
        Maximum number of retry attempts with exponential backoff.

    Returns
    -------
    full_text : str
        A formatted string containing the full text response from Claude and the citations.
    """
    import random

    import anthropic

    try:
        from biomni.config import default_config

        model = default_config.llm
        api_key = default_config.api_key
        if not api_key:
            api_key = os.getenv("ANTHROPIC_API_KEY")
    except ImportError:
        model = "claude-4-sonnet-latest"
        api_key = os.getenv("ANTHROPIC_API_KEY")

    if "claude" not in model:
        raise ValueError("Model must be a Claude model.")

    if not api_key:
        raise ValueError("Set your api_key explicitly.")

    client = anthropic.Anthropic(api_key=api_key)
    tool_def = {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": max_searches,
    }

    delay = random.randint(1, 10)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": query}],
                tools=[tool_def],
            )

            paragraphs, citations = [], []
            response.content = response.content
            formatted_response = ""
            for blk in response.content:
                if blk.type == "text":
                    paragraphs.append(blk.text)
                    formatted_response += blk.text

                    if blk.citations:
                        for cite in blk.citations:
                            citations.append({"url": cite.url, "title": cite.title, "cited_text": cite.cited_text})
                            formatted_response += f"(Citation: {cite.title} - {cite.url})"
            return formatted_response

        except Exception as e:
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
                continue
            print(f"Error performing web search after {max_retries} attempts: {str(e)}")
            return f"Error performing web search after {max_retries} attempts: {str(e)}"


def extract_url_content(
    url: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_content_bytes: int = DEFAULT_HTML_LIMIT_BYTES,
    max_characters: int = 100_000,
) -> dict:
    """Extract bounded text content from a public webpage.

    Args:
        url: Webpage URL to extract content from

    Returns:
        Text content of the webpage

    """
    query_info = {"url": url, "max_content_bytes": max_content_bytes, "max_characters": max_characters}
    if max_content_bytes <= 0 or max_characters <= 0:
        return {"success": False, "error": "Content limits must be positive", "query_info": query_info}

    try:
        response, body, final_url = _fetch_public_url(
            url,
            timeout_seconds=timeout_seconds,
            max_bytes=max_content_bytes,
        )
        content_type = response.headers.get("Content-Type", "").lower()
        encoding = response.encoding or "utf-8"
        decoded = body.decode(encoding, errors="replace")
        if "text/plain" in content_type or "application/json" in content_type:
            extracted = decoded.strip()
        else:
            soup = BeautifulSoup(decoded, "html.parser")
            content = soup.find("main") or soup.find("article") or soup.body
            if content is None:
                return {"success": False, "error": "Page did not contain extractable content", "query_info": query_info}
            for element in content(["script", "style", "nav", "header", "footer", "aside", "iframe"]):
                element.decompose()
            paragraphs = content.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
            extracted = "\n\n".join(text for paragraph in paragraphs if (text := paragraph.get_text(" ", strip=True)))

        if not extracted:
            return {"success": False, "error": "Page content was empty", "query_info": query_info}
        truncated = len(extracted) > max_characters
        return {
            "success": True,
            "query_info": {**query_info, "final_url": final_url},
            "result": {"content": extracted[:max_characters], "truncated": truncated, "content_type": content_type},
        }
    except (requests.RequestException, ValueError, UnicodeError) as exc:
        return {"success": False, "error": f"Error extracting URL content: {exc}", "query_info": query_info}


def query_public_endpoint(
    url: str,
    params: dict | None = None,
    documentation_url: str | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_content_bytes: int = 1_048_576,
) -> dict:
    """Perform one bounded read-only request to a registered or documented public endpoint."""
    query_info = {
        "url": url,
        "params": params or {},
        "documentation_url": documentation_url,
        "method": "GET",
    }
    if not isinstance(url, str) or not url.strip():
        return {
            "success": False,
            "failure_kind": "invalid_arguments",
            "error": "url must be non-empty",
            "query_info": query_info,
        }
    if params is not None and not isinstance(params, dict):
        return {
            "success": False,
            "failure_kind": "invalid_arguments",
            "error": "params must be a dict or None",
            "query_info": query_info,
        }
    if not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= 120:
        return {
            "success": False,
            "failure_kind": "invalid_arguments",
            "error": "timeout_seconds must be between 1 and 120",
            "query_info": query_info,
        }
    if not isinstance(max_content_bytes, int) or not 1 <= max_content_bytes <= DEFAULT_HTML_LIMIT_BYTES:
        return {
            "success": False,
            "failure_kind": "invalid_arguments",
            "error": f"max_content_bytes must be between 1 and {DEFAULT_HTML_LIMIT_BYTES}",
            "query_info": query_info,
        }

    try:
        _validate_public_url(url)
    except ValueError as exc:
        return {
            "success": False,
            "failure_kind": "invalid_arguments",
            "error": f"Invalid public endpoint URL: {exc}",
            "query_info": query_info,
        }

    try:
        source_id = _registered_public_endpoint(url)
        provenance = "registered_endpoint" if source_id else None

        if source_id is None:
            if not isinstance(documentation_url, str) or not documentation_url.strip():
                return {
                    "success": False,
                    "failure_kind": "endpoint_provenance_unverified",
                    "error": "An unregistered endpoint requires an official documentation_url",
                    "query_info": query_info,
                }
            endpoint_host = (urlsplit(url).hostname or "").lower()
            documentation_host = (urlsplit(documentation_url).hostname or "").lower()
            if endpoint_host != documentation_host:
                return {
                    "success": False,
                    "failure_kind": "endpoint_provenance_unverified",
                    "error": "The documentation URL must use the same official host as the endpoint",
                    "query_info": query_info,
                }
            try:
                _validate_public_url(documentation_url)
            except ValueError as exc:
                return {
                    "success": False,
                    "failure_kind": "endpoint_provenance_unverified",
                    "error": f"Invalid official documentation URL: {exc}",
                    "query_info": query_info,
                }
            doc_response, doc_body, final_documentation_url = _fetch_public_url(
                documentation_url,
                timeout_seconds=timeout_seconds,
                max_bytes=min(max_content_bytes, 524_288),
            )
            if (urlsplit(final_documentation_url).hostname or "").lower() != endpoint_host:
                return {
                    "success": False,
                    "failure_kind": "endpoint_provenance_unverified",
                    "error": "Official documentation redirected to a different host",
                    "query_info": {**query_info, "final_documentation_url": final_documentation_url},
                }
            documentation_text = doc_body.decode(doc_response.encoding or "utf-8", errors="replace")
            endpoint_path = unquote(urlsplit(url).path).rstrip("/")
            if not endpoint_path or endpoint_path not in unquote(documentation_text):
                return {
                    "success": False,
                    "failure_kind": "endpoint_provenance_unverified",
                    "error": "The official documentation did not contain the requested endpoint path",
                    "query_info": {**query_info, "final_documentation_url": final_documentation_url},
                }
            provenance = "documented_endpoint"

        request_url = url
        if params:
            separator = "&" if "?" in request_url else "?"
            request_url = f"{request_url}{separator}{urlencode(params, doseq=True)}"
        response, body, final_url = _fetch_public_url(
            request_url,
            timeout_seconds=timeout_seconds,
            max_bytes=max_content_bytes,
            headers={"Accept": "application/json, text/plain;q=0.9, */*;q=0.1"},
            raise_for_status=False,
        )
        content_type = response.headers.get("Content-Type", "").lower()
        if (urlsplit(final_url).hostname or "").lower() != (urlsplit(url).hostname or "").lower():
            return {
                "success": False,
                "failure_kind": "endpoint_provenance_unverified",
                "error": "The endpoint redirected to a different host",
                "query_info": {**query_info, "requested_url": request_url, "final_url": final_url},
            }
        decoded = body.decode(response.encoding or "utf-8", errors="replace")
        try:
            data = json.loads(decoded) if "json" in content_type or decoded.lstrip().startswith(("{", "[")) else decoded
        except json.JSONDecodeError:
            data = decoded

        success = 200 <= response.status_code < 300
        if success:
            failure_kind = None
        elif response.status_code == 404:
            failure_kind = "not_found"
        elif response.status_code == 429:
            failure_kind = "rate_limited"
        elif response.status_code in {401, 403}:
            failure_kind = "authentication_required"
        elif response.status_code >= 500:
            failure_kind = "transient_network"
        else:
            failure_kind = "upstream_error"
        result = {
            "success": success,
            "failure_kind": failure_kind,
            "query_info": {
                **query_info,
                "requested_url": request_url,
                "final_url": final_url,
                "source_id": source_id,
                "endpoint_provenance": provenance,
            },
            "result": {
                "status_code": response.status_code,
                "content_type": content_type,
                "data": data,
            },
        }
        if not success:
            result["error"] = f"Public endpoint returned HTTP {response.status_code}"
        return result
    except requests.Timeout:
        return {
            "success": False,
            "failure_kind": "timeout",
            "error": "Public endpoint request timed out",
            "query_info": query_info,
        }
    except (requests.RequestException, ValueError, UnicodeError) as exc:
        return {
            "success": False,
            "failure_kind": "transient_network",
            "error": f"Public endpoint request failed: {exc}",
            "query_info": query_info,
        }


def extract_pdf_content(
    url: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_download_bytes: int = DEFAULT_DOWNLOAD_LIMIT_BYTES,
    max_pages: int = 100,
    max_characters: int = 200_000,
) -> dict:
    """Extract bounded text from a PDF hosted at a public URL.

    Args:
        url: URL of the PDF file to extract text from

    Returns:
        The extracted text content from the PDF

    """
    try:
        query_info = {"url": url, "max_download_bytes": max_download_bytes, "max_pages": max_pages}
        if max_download_bytes <= 0 or max_pages <= 0 or max_characters <= 0:
            return {"success": False, "error": "PDF limits must be positive", "query_info": query_info}

        if not url.lower().endswith(".pdf"):
            response, page_content, page_url = _fetch_public_url(
                url,
                timeout_seconds=timeout_seconds,
                max_bytes=DEFAULT_HTML_LIMIT_BYTES,
            )
            page_text = page_content.decode(response.encoding or "utf-8", errors="replace")
            soup = BeautifulSoup(page_text, "html.parser")
            pdf_link = next(
                (link.get("href") for link in soup.find_all("a", href=True) if ".pdf" in link.get("href", "").lower()),
                None,
            )
            if not pdf_link:
                return {"success": False, "error": f"No PDF file found at {url}", "query_info": query_info}
            url = urljoin(page_url, pdf_link)

        response, pdf_content, final_url = _fetch_public_url(
            url,
            timeout_seconds=timeout_seconds,
            max_bytes=max_download_bytes,
        )

        # Check if we actually got a PDF file (by checking content type or magic bytes)
        content_type = response.headers.get("Content-Type", "").lower()
        if "application/pdf" not in content_type and not pdf_content.startswith(b"%PDF"):
            return {
                "success": False,
                "error": f"The URL did not return a valid PDF file. Content type: {content_type}",
                "query_info": query_info,
            }

        pdf_file = BytesIO(pdf_content)

        # Try with PyPDF2 first
        try:
            text = ""
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            pages_processed = min(len(pdf_reader.pages), max_pages)
            for page_num in range(pages_processed):
                page = pdf_reader.pages[page_num]
                text += (page.extract_text() or "") + "\n\n"
        except Exception as e:
            return {"success": False, "error": f"Error extracting text from PDF: {e}", "query_info": query_info}

        # Clean up the text
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return {
                "success": False,
                "error": "The PDF did not contain extractable text; it may require OCR",
                "query_info": query_info,
            }

        truncated = len(text) > max_characters
        return {
            "success": True,
            "query_info": {**query_info, "final_url": final_url},
            "result": {
                "content": text[:max_characters],
                "truncated": truncated,
                "pages_processed": pages_processed,
                "total_pages": len(pdf_reader.pages),
            },
        }

    except (requests.RequestException, ValueError) as e:
        return {"success": False, "error": f"Error downloading PDF: {e}", "query_info": {"url": url}}
    except Exception as e:
        return {"success": False, "error": f"Error extracting text from PDF: {e}", "query_info": {"url": url}}
