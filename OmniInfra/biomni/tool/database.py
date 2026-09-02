import csv
import json
import os
import pickle
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests
from Bio.Blast import NCBIWWW, NCBIXML
from Bio.Seq import Seq
from langchain_core.messages import HumanMessage, SystemMessage

from biomni.llm import get_llm
from biomni.utils import parse_hpo_obo


def query_bindingdb(gene_symbol, data_lake_path, max_results=20):
    """Read target-linked BindingDB records without failing on malformed TSV rows."""
    path = os.path.join(data_lake_path, "BindingDB_All_202409.tsv")
    query = str(gene_symbol or "").strip().casefold()
    if not query:
        return {"success": False, "error": "gene_symbol must be non-empty"}
    if not os.path.isfile(path):
        return {"success": False, "error": f"BindingDB file does not exist: {path}"}
    if not isinstance(max_results, int) or max_results < 1:
        return {"success": False, "error": "max_results must be a positive integer"}

    records, malformed_rows, rows_read = [], 0, 0
    try:
        with open(path, encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t", quotechar='"')
            header = next(reader, [])
            normalized = [str(item).strip().casefold() for item in header]
            target_cols = [i for i, name in enumerate(normalized) if any(token in name for token in ("target name", "uniprot", "protein name"))]
            structure_cols = {label: next((i for i, name in enumerate(normalized) if label in name), None) for label in ("smiles", "inchikey")}
            if not target_cols:
                return {"success": False, "error": "BindingDB target columns were not found", "parse_stats": {"rows_read": 0, "malformed_rows": 0}}
            for row in reader:
                rows_read += 1
                if len(row) != len(header):
                    malformed_rows += 1
                    continue
                target_text = " ".join(row[i] for i in target_cols).casefold()
                if query not in target_text:
                    continue
                records.append({"target": row[target_cols[0]], "smiles": row[structure_cols["smiles"]] if structure_cols["smiles"] is not None else None, "inchikey": row[structure_cols["inchikey"]] if structure_cols["inchikey"] is not None else None})
                if len(records) >= max_results:
                    break
    except (OSError, csv.Error) as exc:
        return {"success": False, "error": f"BindingDB parse failed: {exc}", "parse_stats": {"rows_read": rows_read, "malformed_rows": malformed_rows}}
    return {"success": True, "query_info": {"gene_symbol": gene_symbol, "source": "BindingDB", "path": path}, "records": records, "parse_stats": {"rows_read": rows_read, "malformed_rows": malformed_rows}, "evidence_status": "sufficient" if records else "insufficient"}


def _normalize_uniprot_endpoint(endpoint: str) -> str:
    """Normalize known natural-language UniProt field aliases in a URL."""
    return endpoint.replace("cc_post_translational_modification", "cc_ptm")


# Backward-compatible import path for agent-generated code. PubMed is a
# literature tool, but older trajectories sometimes guessed this module.
def query_pubmed(query: str, max_papers: int = 10, max_retries: int = 3) -> dict:
    """Compatibility alias for :func:`biomni.tool.literature.query_pubmed`."""
    from biomni.tool.literature import query_pubmed as literature_query_pubmed

    return literature_query_pubmed(query, max_papers=max_papers, max_retries=max_retries)


# Function to map HPO terms to names
def get_hpo_names(hpo_terms: list[str], data_lake_path: str) -> list[str]:
    """Retrieve the names of given HPO terms.

    Args:
        hpo_terms (List[str]): A list of HPO terms (e.g., ['HP:0001250']).

    Returns:
        List[str]: A list of corresponding HPO term names.

    """
    hp_dict = parse_hpo_obo(data_lake_path + "/hp.obo")

    hpo_names = []
    for term in hpo_terms:
        name = hp_dict.get(term, f"Unknown term: {term}")
        hpo_names.append(name)
    return hpo_names


def _query_llm_for_api(prompt, schema, system_template):
    """Helper function to query LLMs for generating API calls based on natural language prompts.

    Uses the central tool LLM configuration (Qwen3.8 by default) via the
    unified get_llm interface.

    Parameters
    ----------
    prompt (str): Natural language query to process
    schema (dict): API schema to include in the system prompt
    system_template (str): Template string for the system prompt (should have {schema} placeholder)

    Returns
    -------
    dict: Dictionary with 'success', 'data' (if successful), 'error' (if failed), and optional 'raw_response'

    """
    # Use global config for model and api_key
    try:
        from biomni.config import default_config

        model = default_config.tool_llm
        source = default_config.tool_source
        api_key = default_config.api_key
    except ImportError:
        model = "Qwen3.8"
        source = "Qwen"
        api_key = None

    try:
        # Format the system prompt with schema if provided
        if schema is not None:
            schema_json = json.dumps(schema, indent=2)
            # Replace only the schema placeholder.  ``str.format`` treats
            # endpoint examples such as ``targets/{targetId}`` as additional
            # fields and can raise misleading KeyError parsing failures.
            system_prompt = system_template.replace("{schema}", schema_json)
        else:
            system_prompt = system_template

        # Get LLM instance using the unified interface with config
        try:
            from biomni.config import default_config

            llm = get_llm(
                model=model,
                temperature=0.0,
                api_key=api_key,
                source=source,
                config=default_config,
            )
        except ImportError:
            llm = get_llm(model=model, temperature=0.0, api_key=api_key or "EMPTY", source=source)

        # Compose messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]

        # Query the LLM
        response = llm.invoke(messages)
        llm_text = response.content.strip()

        # Find JSON boundaries (in case LLM adds explanations)
        json_start = llm_text.find("{")
        json_end = llm_text.rfind("}") + 1

        if json_start >= 0 and json_end > json_start:
            json_text = llm_text[json_start:json_end]
            result = json.loads(json_text)
        else:
            # If no JSON found, try the whole response
            result = json.loads(llm_text)

        return {"success": True, "data": result, "raw_response": llm_text}

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        return {
            "success": False,
            "error": f"Failed to parse LLM response: {str(e)}",
            "raw_response": llm_text if "llm_text" in locals() else "No content found",
        }
    except Exception as e:
        return {"success": False, "error": f"Error querying LLM: {str(e)}"}


def _query_rest_api(
    endpoint,
    method="GET",
    params=None,
    headers=None,
    json_data=None,
    description=None,
    timeout_seconds: int = 30,
    max_retries: int = 0,
):
    """General helper function to query REST APIs with consistent error handling.

    Parameters
    ----------
    endpoint (str): Full URL endpoint to query
    method (str): HTTP method ("GET" or "POST")
    params (dict, optional): Query parameters to include in the URL
    headers (dict, optional): HTTP headers for the request
    json_data (dict, optional): JSON data for POST requests
    description (str, optional): Description of this query for error messages
    timeout_seconds (int): Maximum seconds to wait for the external API

    Returns
    -------
    dict: Dictionary containing the result or error information

    """
    # Set default headers if not provided
    if headers is None:
        headers = {"Accept": "application/json"}

    # Set default description if not provided
    if description is None:
        description = f"{method} request to {endpoint}"

    url_error = None

    try:
        # Retry transient transport failures and server errors only. Client
        # errors (4xx) are deterministic and should be reported immediately.
        response = None
        attempts = max(0, int(max_retries)) + 1
        for attempt in range(attempts):
            try:
                if method.upper() == "GET":
                    response = requests.get(endpoint, params=params, headers=headers, timeout=timeout_seconds)
                elif method.upper() == "POST":
                    response = requests.post(endpoint, params=params, headers=headers, json=json_data, timeout=timeout_seconds)
                else:
                    return {"error": f"Unsupported HTTP method: {method}"}
                if response.status_code < 500 or attempt == attempts - 1:
                    break
            except requests.RequestException:
                if attempt == attempts - 1:
                    raise
            time.sleep(min(2 ** attempt, 4))

        url_error = str(response.text)
        response.raise_for_status()

        # Try to parse JSON response
        try:
            result = response.json()
        except ValueError:
            # Return raw text if not JSON
            result = {"raw_text": response.text}

        return {
            "success": True,
            "query_info": {
                "endpoint": endpoint,
                "method": method,
                "description": description,
            },
            "result": result,
        }

    except Exception as e:
        error_msg = str(e)
        response_text = ""

        # Try to get more detailed error info from response
        if hasattr(e, "response") and e.response:
            try:
                error_json = e.response.json()
                if "messages" in error_json:
                    error_msg = "; ".join(error_json["messages"])
                elif "message" in error_json:
                    error_msg = error_json["message"]
                elif "error" in error_json:
                    error_msg = error_json["error"]
                elif "detail" in error_json:
                    error_msg = error_json["detail"]
            except Exception:
                response_text = e.response.text

        return {
            "success": False,
            "error": f"API error: {error_msg}",
            "query_info": {
                "endpoint": endpoint,
                "method": method,
                "description": description,
            },
            "response_url_error": url_error,
            "response_text": response_text,
        }


def _query_ncbi_database(
    database: str,
    search_term: str,
    result_formatter=None,
    max_results: int = 3,
) -> dict[str, Any]:
    """Core function to query NCBI databases using Claude for query interpretation and NCBI eutils.

    Parameters
    ----------
    database (str): NCBI database to query (e.g., "clinvar", "gds", "geoprofiles")
    result_formatter (callable): Function to format results from the database
    api_key (str): Anthropic API key. If None, will look for ANTHROPIC_API_KEY environment variable
    model (str): Anthropic model to use
    max_results (int): Maximum number of results to return
    verbose (bool): Whether to return verbose results

    Returns
    -------
    dict: Dictionary containing both the structured query and the results

    """
    # Query NCBI API using the structured search term
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    esearch_params = {
        "db": database,
        "term": search_term,
        "retmode": "json",
        "retmax": 100,
        "usehistory": "y",  # Use history server to store results
    }

    # Get IDs of matching entries
    search_response = _query_rest_api(
        endpoint=esearch_url,
        method="GET",
        params=esearch_params,
        description="NCBI ESearch API query",
    )

    if not search_response["success"]:
        return search_response

    search_data = search_response["result"]

    # If we have results, fetch the details
    if "esearchresult" in search_data and int(search_data["esearchresult"]["count"]) > 0:
        # Extract WebEnv and query_key from the search results
        webenv = search_data["esearchresult"].get("webenv", "")
        query_key = search_data["esearchresult"].get("querykey", "")

        # Use WebEnv and query_key if available
        if webenv and query_key:
            # Get details using eSummary
            esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            esummary_params = {
                "db": database,
                "query_key": query_key,
                "WebEnv": webenv,
                "retmode": "json",
                "retmax": max_results,
            }

            details_response = _query_rest_api(
                endpoint=esummary_url,
                method="GET",
                params=esummary_params,
                description="NCBI ESummary API query",
            )

            if not details_response["success"]:
                return details_response

            results = details_response["result"]

        else:
            # Fall back to direct ID fetch
            id_list = search_data["esearchresult"]["idlist"][:max_results]

            # Get details for each ID
            esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            esummary_params = {
                "db": database,
                "id": ",".join(id_list),
                "retmode": "json",
            }

            details_response = _query_rest_api(
                endpoint=esummary_url,
                method="GET",
                params=esummary_params,
                description="NCBI ESummary API query",
            )

            if not details_response["success"]:
                return details_response

            results = details_response["result"]

        # Format results using the provided formatter
        formatted_results = result_formatter(results) if result_formatter else results

        # Return the combined information
        return {
            "database": database,
            "query_interpretation": search_term,
            "total_results": int(search_data["esearchresult"]["count"]),
            "formatted_results": formatted_results,
        }
    else:
        return {
            "database": database,
            "query_interpretation": search_term,
            "total_results": 0,
            "formatted_results": [],
        }


def _format_query_results(result, options=None):
    """A general-purpose formatter for query function results to reduce output size.

    Parameters
    ----------
    result (dict): The original API response dictionary
    options (dict, optional): Formatting options including:
        - max_items (int): Maximum number of items to include in lists (default: 5)
        - max_depth (int): Maximum depth to traverse in nested dictionaries (default: 2)
        - include_keys (list): Only include these top-level keys (overrides exclude_keys)
        - exclude_keys (list): Exclude these keys from the output
        - summarize_lists (bool): Whether to summarize long lists (default: True)
        - truncate_strings (int): Maximum length for string values (default: 100)

    Returns
    -------
    dict: A condensed version of the input results

    """

    def _format_value(value, depth, options):
        """Recursively format a value based on its type and formatting options.

        Parameters
        ----------
        value: The value to format
        depth (int): Current recursion depth
        options (dict): Formatting options

        Returns
        -------
        Formatted value

        """
        # Base case: reached max depth
        if depth >= options["max_depth"] and (isinstance(value, dict | list)):
            if isinstance(value, dict):
                return {
                    "_summary": f"Nested dictionary with {len(value)} keys",
                    "_keys": list(value.keys())[: options["max_items"]],
                }
            else:  # list
                return _summarize_list(value, options)

        # Process based on type
        if isinstance(value, dict):
            return _format_dict(value, depth, options)
        elif isinstance(value, list):
            return _format_list(value, depth, options)
        elif isinstance(value, str) and len(value) > options["truncate_strings"]:
            return value[: options["truncate_strings"]] + "... (truncated)"
        else:
            return value

    def _format_dict(d, depth, options):
        """Format a dictionary according to options."""
        result = {}

        # Filter keys based on include/exclude options
        keys_to_process = d.keys()
        if depth == 0 and options["include_keys"]:  # Only apply at top level
            keys_to_process = [k for k in keys_to_process if k in options["include_keys"]]
        elif depth == 0 and options["exclude_keys"]:  # Only apply at top level
            keys_to_process = [k for k in keys_to_process if k not in options["exclude_keys"]]

        # Process each key
        for key in keys_to_process:
            result[key] = _format_value(d[key], depth + 1, options)

        return result

    def _format_list(lst, depth, options):
        """Format a list according to options."""
        if options["summarize_lists"] and len(lst) > options["max_items"]:
            return _summarize_list(lst, options)

        result = []
        for i, item in enumerate(lst):
            if i >= options["max_items"]:
                remaining = len(lst) - options["max_items"]
                result.append(f"... {remaining} more items (omitted)")
                break
            result.append(_format_value(item, depth + 1, options))

        return result

    def _summarize_list(lst, options):
        """Create a summary for a list."""
        if not lst:
            return []

        # Sample a few items
        sample = lst[: min(3, len(lst))]
        sample_formatted = [_format_value(item, options["max_depth"], options) for item in sample]

        # For homogeneous lists, provide type info
        if len(lst) > 0:
            item_type = type(lst[0]).__name__
            homogeneous = all(isinstance(item, type(lst[0])) for item in lst)
            type_info = f"all {item_type}" if homogeneous else "mixed types"
        else:
            type_info = "empty"

        return {
            "_summary": f"List with {len(lst)} items ({type_info})",
            "_sample": sample_formatted,
        }

    if options is None:
        options = {}

    # Default options
    default_options = {
        "max_items": 5,
        "max_depth": 20,
        "include_keys": None,
        "exclude_keys": ["raw_response", "debug_info", "request_details"],
        "summarize_lists": True,
        "truncate_strings": 100,
    }

    # Merge provided options with defaults
    for key, value in default_options.items():
        if key not in options:
            options[key] = value

    # Filter and format the result
    formatted = _format_value(result, 0, options)
    return formatted


def query_uniprot(
    prompt=None,
    endpoint=None,
    max_results=5,
):
    """Query the UniProt REST API using either natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, optional): Natural language query about proteins (e.g., "Find information about human insulin")
    endpoint (str, optional): Full or partial UniProt API endpoint URL to query directly
                            (e.g., "https://rest.uniprot.org/uniprotkb/P01308")
    max_results (int): Maximum number of results to return

    Returns
    -------
    dict: Dictionary containing the query information and the UniProt API results

    Examples
    --------
    - Natural language: query_uniprot(prompt="Find information about human insulin protein")
    - Direct endpoint: query_uniprot(endpoint="https://rest.uniprot.org/uniprotkb/P01308")

    """
    # Base URL for UniProt API
    base_url = "https://rest.uniprot.org"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"success": False, "error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load UniProt schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "uniprot.pkl")
        with open(schema_path, "rb") as f:
            uniprot_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a protein biology expert specialized in using the UniProt REST API.

        Based on the user's natural language request, determine the appropriate UniProt REST API endpoint and parameters.

        UNIPROT REST API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including base URL, dataset, endpoint type, and parameters)
        2. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - Base URL is "https://rest.uniprot.org"
        - Search in reviewed (Swiss-Prot) entries first before using non-reviewed (TrEMBL) entries
        - Assume organism is human unless otherwise specified. Human taxonomy ID is 9606
        - Use gene_exact: for exact gene name searches
        - Use specific query fields like accession:, gene:, organism_id: in search queries
        - Use quotes for terms with spaces: organism_name:"Homo sapiens"
        - If using the optional fields= parameter, use UniProt field names only;
          for post-translational modifications the valid field is cc_ptm (not
          cc_post_translational_modification).

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=uniprot_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the full URL from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Use provided endpoint directly
        if endpoint.startswith("/"):
            endpoint = f"{base_url}{endpoint}"
        elif not endpoint.startswith("http"):
            endpoint = f"{base_url}/{endpoint.lstrip('/')}"
        description = "Direct query to provided endpoint"

    # Normalize a common natural-language field name hallucinated by the
    # model.  UniProt's REST API uses the compact field identifier ``cc_ptm``.
    endpoint = _normalize_uniprot_endpoint(endpoint)

    # Use the common REST API helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    return api_result


def query_alphafold(
    uniprot_id,
    endpoint="prediction",
    residue_range=None,
    download=False,
    output_dir=None,
    file_format="pdb",
    model_version=None,
    model_number=1,
):
    """Query the AlphaFold Database API for protein structure predictions.

    Parameters
    ----------
    uniprot_id (str): UniProt accession ID (e.g., "P04637")
    endpoint (str, optional): Specific AlphaFold API endpoint to query:
                            "prediction", "summary", or "annotations"
    residue_range (str, optional): Specific residue range in format "start-end" (e.g., "1-100")
    download (bool): Whether to download structure files
    output_dir (str, optional): Directory to save downloaded files (default: current directory)
    file_format (str): Format of the structure file to download - "pdb" or "cif"
    model_version (str, optional): Explicit AlphaFold model version override (for example, "v6").
                                  By default, use the versioned URL returned by the API.
    model_number (int): Model number (1-5, with 1 being the highest confidence model)

    Returns
    -------
    dict: Dictionary containing both the query information and the AlphaFold results

    Examples
    --------
    - Basic query: query_alphafold(uniprot_id="P04637")
    - Download structure: query_alphafold(uniprot_id="P04637", download=True, output_dir="./structures")
    - Get annotations: query_alphafold(uniprot_id="P04637", endpoint="annotations")

    """
    # Base URL for AlphaFold API
    base_url = "https://alphafold.ebi.ac.uk/api"

    # Ensure we have a UniProt ID
    if not uniprot_id:
        return {"success": False, "error": "UniProt ID is required"}

    if not re.fullmatch(r"[A-Za-z0-9]+", uniprot_id):
        return {"success": False, "error": "UniProt ID must contain only letters and digits"}

    # Validate endpoint
    valid_endpoints = ["prediction", "summary", "annotations"]
    if endpoint not in valid_endpoints:
        return {"success": False, "error": f"Invalid endpoint. Must be one of: {', '.join(valid_endpoints)}"}

    file_format = file_format.lower()
    if file_format not in {"pdb", "cif"}:
        return {"success": False, "error": "file_format must be either 'pdb' or 'cif'"}

    if model_version is not None and not re.fullmatch(r"v\d+", model_version):
        return {"success": False, "error": "model_version must look like 'v6' or be None"}

    if not 1 <= model_number <= 5:
        return {"success": False, "error": "model_number must be between 1 and 5"}

    # Construct the API URL based on endpoint
    if endpoint == "prediction":
        url = f"{base_url}/prediction/{uniprot_id}"
    elif endpoint == "summary":
        url = f"{base_url}/uniprot/summary/{uniprot_id}.json"
    elif endpoint == "annotations":
        if residue_range:
            url = f"{base_url}/annotations/{uniprot_id}/{residue_range}"
        else:
            url = f"{base_url}/annotations/{uniprot_id}"

    try:
        # Make the API request
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Parse the response as JSON
        result = response.json()

        # Handle download request if specified
        download_info = None
        if download:
            # Ensure output directory exists
            if not output_dir:
                output_dir = "."
            os.makedirs(output_dir, exist_ok=True)

            metadata = result[0] if isinstance(result, list) and result else result
            url_key = "pdbUrl" if file_format == "pdb" else "cifUrl"
            download_url = metadata.get(url_key) if isinstance(metadata, dict) else None
            if not download_url:
                return {
                    "success": False,
                    "error": f"AlphaFold response did not contain {url_key}",
                    "query_info": {"uniprot_id": uniprot_id, "endpoint": endpoint, "url": url},
                }

            if model_version is not None:
                download_url = re.sub(r"model_v\d+", f"model_{model_version}", download_url)

            filename = os.path.basename(urlsplit(download_url).path)
            file_path = os.path.join(output_dir, filename)

            # Download the file
            download_response = requests.get(download_url, timeout=60, stream=True)
            if download_response.status_code == 200:
                max_download_bytes = 100 * 1024 * 1024
                content_length = int(download_response.headers.get("Content-Length", 0))
                if content_length > max_download_bytes:
                    return {"success": False, "error": "AlphaFold structure exceeds the 100 MiB download limit"}
                downloaded_bytes = 0
                with open(file_path, "wb") as f:
                    for chunk in download_response.iter_content(chunk_size=64 * 1024):
                        downloaded_bytes += len(chunk)
                        if downloaded_bytes > max_download_bytes:
                            f.close()
                            os.unlink(file_path)
                            return {
                                "success": False,
                                "error": "AlphaFold structure exceeds the 100 MiB download limit",
                            }
                        f.write(chunk)
                download_info = {
                    "success": True,
                    "file_path": file_path,
                    "url": download_url,
                }
            else:
                download_info = {
                    "success": False,
                    "error": f"Failed to download file (status code: {download_response.status_code})",
                    "url": download_url,
                }

        # Return the query information and results
        response_data = {
            "success": True,
            "query_info": {
                "uniprot_id": uniprot_id,
                "endpoint": endpoint,
                "residue_range": residue_range,
                "url": url,
            },
            "result": result,
        }

        if download_info:
            response_data["download"] = download_info
            if not download_info["success"]:
                response_data["success"] = False
                response_data["error"] = download_info["error"]

        return response_data

    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        response_text = ""

        # Try to get more detailed error info from response
        if hasattr(e, "response") and e.response:
            try:
                error_json = e.response.json()
                if "message" in error_json:
                    error_msg = error_json["message"]
            except Exception:
                response_text = e.response.text

        return {
            "success": False,
            "error": f"AlphaFold API error: {error_msg}",
            "query_info": {
                "uniprot_id": uniprot_id,
                "endpoint": endpoint,
                "residue_range": residue_range,
                "url": url,
            },
            "response_text": response_text,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error: {str(e)}",
            "query_info": {
                "uniprot_id": uniprot_id,
                "endpoint": endpoint,
                "residue_range": residue_range,
            },
        }


def query_interpro(
    prompt=None,
    endpoint=None,
    max_results=3,
):
    """Query the InterPro REST API using natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, required): Natural language query about protein domains or families
    endpoint (str, optional): Direct endpoint path or full URL (e.g., "/entry/interpro/IPR023411"
                             or "https://www.ebi.ac.uk/interpro/api/entry/interpro/IPR023411")
    max_results (int): Maximum number of results to return per page

    Returns
    -------
    dict: Dictionary containing both the query information and the InterPro API results

    Examples
    --------
    - Natural language: query_interpro("Find information about kinase domains in InterPro")
    - Direct endpoint: query_interpro(endpoint="/entry/interpro/IPR023411")

    """
    # Base URL for InterPro API
    base_url = "https://www.ebi.ac.uk/interpro/api"

    # Default parameters
    format = "json"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load InterPro schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "interpro.pkl")
        with open(schema_path, "rb") as f:
            interpro_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a protein domain expert specialized in using the InterPro REST API.

        Based on the user's natural language request, determine the appropriate InterPro REST API endpoint.

        INTERPRO REST API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including the base URL "https://www.ebi.ac.uk/interpro/api")
        2. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - Path components for data types: entry, protein, structure, set, taxonomy, proteome
        - Common sources: interpro, pfam, cdd, uniprot, pdb
        - Protein subtypes can be "reviewed" or "unreviewed"
        - For specific entries, use lowercase accessions (e.g., "ipr000001" instead of "IPR000001")
        - Endpoints can be hierarchical like "/entry/interpro/protein/uniprot/P04637"

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=interpro_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Extract the full URL from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Process provided endpoint
        # If it's just a path, add the base URL
        if endpoint.startswith("/"):
            endpoint = f"{base_url}{endpoint}"
        elif not endpoint.startswith("http"):
            endpoint = f"{base_url}/{endpoint.lstrip('/')}"

        description = "Direct query to provided endpoint"

    # Add pagination parameters
    params = {"page": 1, "page_size": max_results}

    # Add format parameter if not json
    if format and format != "json":
        params["format"] = format

    # Make the API request
    api_result = _query_rest_api(endpoint=endpoint, method="GET", params=params, description=description)

    return api_result


def query_pdb(
    prompt=None,
    query=None,
    max_results=3,
    gene_symbol=None,
    organism=None,
    experimental_method=None,
    released_before=None,
):
    """Query the RCSB PDB database using natural language or a direct structured query.

    Parameters
    ----------
    prompt (str, optional): Natural language query about protein structures
    query (dict, optional): Direct structured query or complete RCSB Search API request
    max_results (int): Maximum number of results to return
    gene_symbol (str, optional): Gene symbol or protein name for a deterministic full-text search
    organism (str, optional): Scientific organism name, for example ``Homo sapiens``
    experimental_method (str, optional): Experimental method, for example ``SOLUTION NMR``
    released_before (str, optional): Inclusive initial-release cutoff in YYYY-MM-DD format

    Returns
    -------
    dict: Dictionary containing the structured query, search results, and identifiers

    Examples
    --------
    - Natural language: query_pdb("Find structures of human insulin")
    - Direct query: query_pdb(query={"query": {"type": "terminal", "service": "full_text",
                           "parameters": {"value": "insulin"}}, "return_type": "entry"})

    """
    # Default parameters
    return_type = "entry"

    high_level_values = (gene_symbol, organism, experimental_method, released_before)
    if prompt is None and query is None and not any(high_level_values):
        return {
            "success": False,
            "error": "Provide prompt, query, or at least one structured filter such as gene_symbol",
        }

    if max_results < 1:
        return {"success": False, "error": "max_results must be at least 1"}

    if released_before and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(released_before)):
        return {"success": False, "error": "released_before must use YYYY-MM-DD format"}

    if any(high_level_values):
        nodes = []
        if gene_symbol:
            nodes.append(
                {
                    "type": "terminal",
                    "service": "full_text",
                    "parameters": {"value": str(gene_symbol).strip()},
                }
            )
        attribute_filters = (
            ("rcsb_entity_source_organism.ncbi_scientific_name", "exact_match", organism),
            ("exptl.method", "exact_match", experimental_method),
            ("rcsb_accession_info.initial_release_date", "less_or_equal", released_before),
        )
        for attribute, operator, value in attribute_filters:
            if value:
                nodes.append(
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": attribute,
                            "operator": operator,
                            "value": str(value).strip(),
                        },
                    }
                )
        query_json = {
            "query": nodes[0]
            if len(nodes) == 1
            else {"type": "group", "logical_operator": "and", "nodes": nodes},
            "return_type": return_type,
        }

    # Generate search query from natural language if prompt is provided and query is not
    elif prompt and not query:
        # Load schema from pickle file
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "pdb.pkl")

        with open(schema_path, "rb") as f:
            schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a structural biology expert that creates precise RCSB PDB Search API queries based on natural language requests.

        SEARCH API SCHEMA:
        {schema}

        IMPORTANT GUIDELINES:
        1. Choose the appropriate search_service based on the query:
           - Use "text" for attribute-specific searches (REQUIRES attribute, operator, and value)
           - Use "full_text" for general keyword searches across multiple fields
           - Use appropriate specialized services for sequence, structure, motif searches

        2. For "text" searches, you MUST specify:
           - attribute: The specific field to search (use common_attributes from schema)
           - operator: The comparison method (exact_match, contains_words, less_or_equal, etc.)
           - value: The search term or value

        3. For "full_text" searches, only specify:
           - value: The search term(s)

        4. For combined searches, use "group" nodes with logical_operator ("and" or "or")

        5. Always specify the appropriate return_type based on what the user is looking for

        Generate a well-formed Search API query JSON object. Return ONLY the JSON with no additional explanation.
        """

        # Query Claude to generate the search query
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return {
                "error": llm_result["error"],
                "llm_response": llm_result.get("raw_response", "No response"),
            }

        # Get the query from Claude's response
        query_json = llm_result["data"]
    else:
        # Use provided query directly
        if query and "query" not in query and query.get("type") in {"terminal", "group"}:
            query_json = {"query": query, "return_type": return_type}
        else:
            query_json = query

    # Ensure return_type is set
    if "return_type" not in query_json:
        query_json["return_type"] = return_type

    # Add request options for pagination
    if "request_options" not in query_json:
        query_json["request_options"] = {}

    if "paginate" not in query_json["request_options"]:
        query_json["request_options"]["paginate"] = {"start": 0, "rows": max_results}

    # Use query_rest_api to execute the search
    search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
    api_result = _query_rest_api(
        endpoint=search_url,
        method="POST",
        json_data=query_json,
        description="PDB Search API query",
    )

    if api_result.get("success") and isinstance(api_result.get("result"), dict):
        result = api_result["result"]
        if result.get("raw_text", None) == "":
            api_result["result"] = {"total_count": 0, "result_set": [], "identifiers": []}
        elif isinstance(result.get("result_set"), list):
            result["identifiers"] = [item.get("identifier") for item in result["result_set"] if item.get("identifier")]
    if api_result.get("query_info") is not None:
        api_result["query_info"]["request"] = query_json

    return api_result


def query_pdb_target_ligand_structures(
    uniprot_accession: str,
    max_resolution: float | None = None,
    experimental_method: str = "X-RAY DIFFRACTION",
    ligand_names: list[str] | None = None,
    ligand_inchikeys: dict | None = None,
    require_ligand: bool = True,
    max_results: int = 50,
    timeout_seconds: int = 30,
) -> dict:
    """Retrieve compact, UniProt-verified PDB structures and their nonpolymer ligands.

    This deterministic helper avoids requiring an agent to construct RCSB schema
    attributes or guess polymer/nonpolymer entity identifiers. Requested ligand
    matches are limited to the filtered PDB entries and are not claims of global
    absence from the PDB.
    """
    query_info = {
        "uniprot_accession": uniprot_accession,
        "max_resolution": max_resolution,
        "experimental_method": experimental_method,
        "ligand_names": ligand_names or [],
        "ligand_inchikeys": ligand_inchikeys or {},
        "require_ligand": require_ligand,
        "max_results": max_results,
        "source": "RCSB PDB Search and Data APIs",
        "query_date": datetime.now().astimezone().date().isoformat(),
    }

    def error(message: str, *, code: str = "invalid_arguments") -> dict:
        return {"success": False, "error": str(message), "error_code": code, "query_info": query_info}

    if not isinstance(uniprot_accession, str) or not re.fullmatch(
        r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})",
        uniprot_accession.strip().upper(),
    ):
        return error("uniprot_accession must be a valid UniProt accession")
    accession = uniprot_accession.strip().upper()
    query_info["uniprot_accession"] = accession
    if max_resolution is not None and (not isinstance(max_resolution, (int, float)) or max_resolution <= 0):
        return error("max_resolution must be a positive number or null")
    if not isinstance(experimental_method, str) or not experimental_method.strip():
        return error("experimental_method must be a non-empty string")
    if ligand_names is not None and (
        not isinstance(ligand_names, list)
        or not all(isinstance(name, str) and name.strip() for name in ligand_names)
    ):
        return error("ligand_names must be null or a list of non-empty strings")
    if ligand_inchikeys is not None and (
        not isinstance(ligand_inchikeys, dict)
        or not all(isinstance(name, str) and name.strip() and isinstance(key, str) and key.strip() for name, key in ligand_inchikeys.items())
    ):
        return error("ligand_inchikeys must be null or a mapping of ligand names to InChIKeys")
    if not isinstance(require_ligand, bool):
        return error("require_ligand must be a boolean")
    if not isinstance(max_results, int) or not 1 <= max_results <= 200:
        return error("max_results must be an integer from 1 through 200")
    if not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= 120:
        return error("timeout_seconds must be an integer from 1 through 120")

    search_request = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": (
                    "rcsb_polymer_entity_container_identifiers."
                    "reference_sequence_identifiers.database_accession"
                ),
                "operator": "exact_match",
                "value": accession,
            },
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": max_results}},
    }
    try:
        search_response = requests.post(
            "https://search.rcsb.org/rcsbsearch/v2/query",
            json=search_request,
            timeout=timeout_seconds,
        )
        if search_response.status_code == 204:
            search_payload = {"total_count": 0, "result_set": []}
        else:
            search_response.raise_for_status()
            search_payload = search_response.json()
    except requests.Timeout:
        return error("RCSB PDB search timed out", code="timeout")
    except (requests.RequestException, ValueError) as exc:
        return error(f"RCSB PDB search failed: {exc}", code="upstream_error")

    result_set = search_payload.get("result_set", []) if isinstance(search_payload, dict) else []
    entry_ids = [
        str(item.get("identifier", "")).upper()
        for item in result_set
        if isinstance(item, dict) and item.get("identifier")
    ]
    partial_errors = []
    chemcomp_cache = {}
    response_cache = {}
    structures = []

    def fetch_json(url: str, record_id: str, record_type: str):
        if url in response_cache:
            return response_cache[url]
        try:
            response = requests.get(url, timeout=timeout_seconds)
            response.raise_for_status()
            response_cache[url] = response.json()
            return response_cache[url]
        except requests.Timeout:
            partial_errors.append({"record_type": record_type, "identifier": record_id, "error_code": "timeout"})
        except (requests.RequestException, ValueError) as exc:
            partial_errors.append(
                {"record_type": record_type, "identifier": record_id, "error_code": "upstream_error", "message": str(exc)[:300]}
            )
        response_cache[url] = None
        return None

    def prefetch(records):
        unique_records = list(dict.fromkeys(records))
        if not unique_records:
            return
        with ThreadPoolExecutor(max_workers=min(8, len(unique_records))) as executor:
            list(executor.map(lambda item: fetch_json(*item), unique_records))

    entry_records = [
        (f"https://data.rcsb.org/rest/v1/core/entry/{entry_id}", entry_id, "entry")
        for entry_id in entry_ids
    ]
    prefetch(entry_records)
    entity_records = []
    for entry_id in entry_ids:
        entry = response_cache.get(f"https://data.rcsb.org/rest/v1/core/entry/{entry_id}") or {}
        methods = {
            str(item.get("method", "")).upper() for item in entry.get("exptl") or [] if isinstance(item, dict)
        }
        if experimental_method.strip().upper() not in methods:
            continue
        resolution_values = (entry.get("rcsb_entry_info") or {}).get("resolution_combined") or []
        numeric_resolutions = [float(value) for value in resolution_values if isinstance(value, (int, float))]
        resolution = min(numeric_resolutions) if numeric_resolutions else None
        if max_resolution is not None and (resolution is None or resolution > float(max_resolution)):
            continue
        container = entry.get("rcsb_entry_container_identifiers") or {}
        entity_records.extend(
            (
                f"https://data.rcsb.org/rest/v1/core/polymer_entity/{entry_id}/{entity_id}",
                f"{entry_id}_{entity_id}",
                "polymer_entity",
            )
            for entity_id in container.get("polymer_entity_ids") or []
        )
        entity_records.extend(
            (
                f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{entry_id}/{entity_id}",
                f"{entry_id}_{entity_id}",
                "nonpolymer_entity",
            )
            for entity_id in container.get("non_polymer_entity_ids") or []
        )
    prefetch(entity_records)
    component_ids = set()
    for url, _identifier, record_type in entity_records:
        if record_type != "nonpolymer_entity":
            continue
        nonpolymer = response_cache.get(url) or {}
        identifiers = nonpolymer.get("rcsb_nonpolymer_entity_container_identifiers") or {}
        comp_id = str(identifiers.get("nonpolymer_comp_id") or identifiers.get("chem_ref_def_id") or "").upper()
        if comp_id:
            component_ids.add(comp_id)
    prefetch(
        [
            (f"https://data.rcsb.org/rest/v1/core/chemcomp/{comp_id}", comp_id, "chem_comp")
            for comp_id in sorted(component_ids)
        ]
    )

    for entry_id in entry_ids:
        entry = fetch_json(f"https://data.rcsb.org/rest/v1/core/entry/{entry_id}", entry_id, "entry")
        if not isinstance(entry, dict):
            continue
        container = entry.get("rcsb_entry_container_identifiers") or {}
        verified_entities = []
        organisms = set()
        for entity_id in container.get("polymer_entity_ids") or []:
            identifier = f"{entry_id}_{entity_id}"
            polymer = fetch_json(
                f"https://data.rcsb.org/rest/v1/core/polymer_entity/{entry_id}/{entity_id}",
                identifier,
                "polymer_entity",
            )
            if not isinstance(polymer, dict):
                continue
            polymer_ids = polymer.get("rcsb_polymer_entity_container_identifiers") or {}
            references = polymer_ids.get("reference_sequence_identifiers") or []
            accessions = {
                str(reference.get("database_accession", "")).upper()
                for reference in references
                if isinstance(reference, dict) and str(reference.get("database_name", "")).lower() == "uniprot"
            }
            accessions.update(str(value).upper() for value in polymer_ids.get("uniprot_ids") or [])
            if accession not in accessions:
                continue
            verified_entities.append(
                {
                    "entity_id": str(entity_id),
                    "auth_asym_ids": polymer_ids.get("auth_asym_ids") or [],
                    "uniprot_accessions": sorted(accessions),
                }
            )
            for source in polymer.get("rcsb_entity_source_organism") or []:
                if isinstance(source, dict) and source.get("ncbi_scientific_name"):
                    organisms.add(str(source["ncbi_scientific_name"]))
        if not verified_entities:
            partial_errors.append(
                {"record_type": "entry", "identifier": entry_id, "error_code": "uniprot_verification_failed"}
            )
            continue

        methods = sorted(
            {str(item.get("method", "")).upper() for item in entry.get("exptl") or [] if isinstance(item, dict)}
        )
        if experimental_method and experimental_method.strip().upper() not in methods:
            continue
        resolution_values = (entry.get("rcsb_entry_info") or {}).get("resolution_combined") or []
        numeric_resolutions = [float(value) for value in resolution_values if isinstance(value, (int, float))]
        resolution = min(numeric_resolutions) if numeric_resolutions else None
        if max_resolution is not None and (resolution is None or resolution > float(max_resolution)):
            continue

        ligands = []
        for entity_id in container.get("non_polymer_entity_ids") or []:
            identifier = f"{entry_id}_{entity_id}"
            nonpolymer = fetch_json(
                f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{entry_id}/{entity_id}",
                identifier,
                "nonpolymer_entity",
            )
            if not isinstance(nonpolymer, dict):
                continue
            entity_ids = nonpolymer.get("rcsb_nonpolymer_entity_container_identifiers") or {}
            comp_id = str(entity_ids.get("nonpolymer_comp_id") or entity_ids.get("chem_ref_def_id") or "").upper()
            if not comp_id:
                continue
            if comp_id not in chemcomp_cache:
                chemcomp_cache[comp_id] = fetch_json(
                    f"https://data.rcsb.org/rest/v1/core/chemcomp/{comp_id}", comp_id, "chem_comp"
                )
            chemical = chemcomp_cache.get(comp_id) or {}
            descriptor = chemical.get("rcsb_chem_comp_descriptor") or {}
            component = chemical.get("chem_comp") or {}
            component_info = chemical.get("rcsb_chem_comp_info") or {}
            annotations = nonpolymer.get("rcsb_nonpolymer_entity_annotation") or []
            ligands.append(
                {
                    "entity_id": str(entity_id),
                    "chem_comp_id": comp_id,
                    "name": component.get("name")
                    or (nonpolymer.get("pdbx_entity_nonpoly") or {}).get("name"),
                    "inchi_key": descriptor.get("InChIKey"),
                    "heavy_atom_count": component_info.get("atom_count_heavy"),
                    "is_subject_of_investigation": any(
                        str(annotation.get("type", "")).upper() == "SUBJECT_OF_INVESTIGATION"
                        for annotation in annotations
                        if isinstance(annotation, dict)
                    ),
                }
            )
        subject_ligands = [ligand for ligand in ligands if ligand["is_subject_of_investigation"]]
        if subject_ligands:
            ligands = subject_ligands
            ligand_selection = "rcsb_subject_of_investigation"
        else:
            ligands = [
                ligand
                for ligand in ligands
                if isinstance(ligand.get("heavy_atom_count"), int) and ligand["heavy_atom_count"] >= 6
            ]
            ligand_selection = "fallback_heavy_atom_count_at_least_6"
        for ligand in ligands:
            ligand.pop("heavy_atom_count", None)
        if require_ligand and not ligands:
            continue
        accession_info = entry.get("rcsb_accession_info") or {}
        database_status = entry.get("pdbx_database_status") or {}
        pdb_format_compatible = str(database_status.get("pdb_format_compatible") or "").upper() or None
        preferred_structure_format = "mmcif" if pdb_format_compatible == "N" else "pdb"
        download_urls = {
            "pdb": f"https://files.rcsb.org/download/{entry_id}.pdb",
            "mmcif": f"https://files.rcsb.org/download/{entry_id}.cif",
        }
        structures.append(
            {
                "pdb_id": entry_id,
                "uniprot_accession": accession,
                "verified_polymer_entities": verified_entities,
                "organisms": sorted(organisms),
                "experimental_methods": methods,
                "resolution_angstroms": resolution,
                "title": (entry.get("struct") or {}).get("title"),
                "initial_release_date": accession_info.get("initial_release_date"),
                "ligands": ligands,
                "ligand_selection": ligand_selection,
                "pdb_format_compatible": pdb_format_compatible,
                "preferred_structure_format": preferred_structure_format,
                "download_url": download_urls[preferred_structure_format],
                "download_urls": download_urls,
            }
        )

    structures.sort(
        key=lambda item: (
            item["resolution_angstroms"] is None,
            item["resolution_angstroms"] if item["resolution_angstroms"] is not None else float("inf"),
            item["pdb_id"],
        )
    )

    def normalized(value: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", str(value).upper())

    all_ligands = [(structure["pdb_id"], ligand) for structure in structures for ligand in structure["ligands"]]
    requested_matches = []
    requested = list(ligand_names or [])
    for name in (ligand_inchikeys or {}):
        if name not in requested:
            requested.append(name)
    for name in requested:
        wanted_name = normalized(name)
        wanted_key = str((ligand_inchikeys or {}).get(name, "")).strip().upper()
        matches = []
        for pdb_id, ligand in all_ligands:
            ligand_name = normalized(ligand.get("name") or "")
            comp_id = normalized(ligand.get("chem_comp_id") or "")
            actual_key = str(ligand.get("inchi_key") or "").upper()
            match_basis = None
            if wanted_key and actual_key == wanted_key:
                match_basis = "inchi_key"
            elif wanted_name and (wanted_name == comp_id or wanted_name == ligand_name):
                match_basis = "name_or_chem_comp_id"
            if match_basis:
                matches.append(
                    {"pdb_id": pdb_id, "chem_comp_id": ligand["chem_comp_id"], "match_basis": match_basis}
                )
        requested_matches.append(
            {
                "requested_ligand": name,
                "status": (
                    "matched"
                    if matches
                    else (
                        "no_inchikey_match_in_filtered_entries"
                        if wanted_key
                        else "no_exact_name_or_chem_comp_id_match"
                    )
                ),
                "identity_method": "inchi_key" if wanted_key else "exact_name_or_chem_comp_id",
                "matches": matches,
            }
        )

    return {
        "success": True,
        "query_info": query_info,
        "result": {
            "total_uniprot_entries": int(search_payload.get("total_count", len(entry_ids)) or 0),
            "entries_considered": len(entry_ids),
            "entries_returned": len(structures),
            "structures": structures,
            "requested_ligand_matches": requested_matches,
            "partial_errors": partial_errors,
            "evidence_scope": (
                "Matches and non-matches apply only to RCSB entries returned for the exact UniProt accession "
                "and the stated method/resolution filters on the query date. Name-only checks are exact RCSB "
                "component-name or CCD-ID comparisons; provide InChIKeys for identity-grade comparisons."
            ),
        },
    }


def _download_rcsb_structure(pdb_id, output_dir, preferred_formats, timeout=60):
    """Download one RCSB coordinate file with bounded, atomic format fallback."""
    max_download_bytes = 100 * 1024 * 1024
    attempts = []
    os.makedirs(output_dir, exist_ok=True)
    for structure_format in preferred_formats:
        suffix = "pdb" if structure_format == "pdb" else "cif"
        url = f"https://files.rcsb.org/download/{pdb_id}.{suffix}"
        response = None
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            status_code = int(response.status_code)
            attempt = {"format": structure_format, "url": url, "http_status": status_code}
            attempts.append(attempt)
            if status_code != 200:
                continue
            content_length = int(response.headers.get("Content-Length", 0))
            if content_length > max_download_bytes:
                attempt["error"] = "structure file exceeds the 100 MiB download limit"
                continue

            destination = os.path.abspath(os.path.join(output_dir, f"{pdb_id}.{suffix}"))
            downloaded_bytes = 0
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{pdb_id}.",
                    suffix=".part",
                    dir=output_dir,
                    delete=False,
                ) as handle:
                    temporary_path = handle.name
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        downloaded_bytes += len(chunk)
                        if downloaded_bytes > max_download_bytes:
                            raise ValueError("structure file exceeds the 100 MiB download limit")
                        handle.write(chunk)
                os.replace(temporary_path, destination)
            except Exception:
                if temporary_path and os.path.exists(temporary_path):
                    os.unlink(temporary_path)
                raise
            return {
                "success": True,
                "structure_file_path": destination,
                "structure_format": structure_format,
                "download_url": url,
                "downloaded_bytes": downloaded_bytes,
                "download_attempts": attempts,
            }
        except (OSError, ValueError, requests.RequestException) as exc:
            if attempts and attempts[-1].get("url") == url:
                attempts[-1]["error"] = str(exc)[:300]
            else:
                attempts.append({"format": structure_format, "url": url, "error": str(exc)[:300]})
        finally:
            if response is not None:
                response.close()
    return {
        "success": False,
        "download_error": "RCSB returned no downloadable PDB or mmCIF coordinate file",
        "download_attempts": attempts,
    }


def query_pdb_identifiers(identifiers, return_type="entry", download=False, attributes=None, output_dir=None):
    """Retrieve detailed data and/or download files for PDB identifiers.

    Parameters
    ----------
    identifiers (list): List of PDB identifiers (from query_pdb)
    return_type (str): Type of results: "entry", "assembly", "polymer_entity", etc.
    download (bool): Whether to download PDB or mmCIF coordinate files
    attributes (list, optional): List of specific attributes to retrieve
    output_dir (str, optional): Directory for downloaded coordinate files

    Returns
    -------
    dict: Dictionary containing the detailed data and file paths if downloaded

    Example:
    - Search and then get details:
        results = query_pdb("Find structures of human insulin")
        details = get_pdb_details(results["identifiers"], download=True)

    """
    supported_return_types = {
        "entry",
        "assembly",
        "polymer_entity",
        "nonpolymer_entity",
        "polymer_instance",
        "mol_definition",
    }
    query_info = {
        "identifiers": identifiers,
        "return_type": return_type,
        "download": download,
        "attributes": attributes,
        "output_dir": output_dir,
        "source": "RCSB PDB Data API",
    }
    if not isinstance(identifiers, list) or not identifiers or not all(
        isinstance(identifier, str) and identifier.strip() for identifier in identifiers
    ):
        return {"success": False, "error": "identifiers must be a non-empty list of strings", "query_info": query_info}
    if return_type not in supported_return_types:
        return {
            "success": False,
            "error": f"Unsupported return_type {return_type!r}; expected one of {sorted(supported_return_types)}",
            "query_info": query_info,
        }
    if not isinstance(download, bool):
        return {"success": False, "error": "download must be a boolean", "query_info": query_info}
    if attributes is not None and (
        not isinstance(attributes, list) or not all(isinstance(attribute, str) and attribute for attribute in attributes)
    ):
        return {"success": False, "error": "attributes must be null or a list of non-empty strings", "query_info": query_info}
    if output_dir is not None and (not isinstance(output_dir, str) or not output_dir.strip()):
        return {"success": False, "error": "output_dir must be null or a non-empty string", "query_info": query_info}

    resolved_output_dir = os.path.abspath(
        os.path.expanduser(
            output_dir
            or os.environ.get("BIOMNI_TASK_OUTPUT_DIR")
            or os.path.join(os.path.dirname(__file__), "data", "pdb")
        )
    )
    if download:
        query_info["output_dir"] = resolved_output_dir

    try:
        # Fetch detailed data using Data API
        detailed_results = []
        pdb_format_compatibility = {}
        for raw_identifier in identifiers:
            identifier = raw_identifier.strip()
            try:
                # Determine the appropriate endpoint based on return_type and identifier format
                if return_type == "entry":
                    data_url = f"https://data.rcsb.org/rest/v1/core/entry/{identifier}"
                elif return_type == "polymer_entity":
                    entry_id, entity_id = identifier.split("_")
                    data_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{entry_id}/{entity_id}"
                elif return_type == "nonpolymer_entity":
                    entry_id, entity_id = identifier.split("_")
                    data_url = f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{entry_id}/{entity_id}"
                elif return_type == "polymer_instance":
                    entry_id, asym_id = identifier.split(".")
                    data_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity_instance/{entry_id}/{asym_id}"
                elif return_type == "assembly":
                    entry_id, assembly_id = identifier.split("-")
                    data_url = f"https://data.rcsb.org/rest/v1/core/assembly/{entry_id}/{assembly_id}"
                elif return_type == "mol_definition":
                    data_url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{identifier}"

                # Fetch data
                data_response = requests.get(data_url, timeout=30)
                data_response.raise_for_status()
                entity_data = data_response.json()
                if return_type == "entry":
                    database_status = entity_data.get("pdbx_database_status") or {}
                    pdb_format_compatibility[identifier.upper()] = str(
                        database_status.get("pdb_format_compatible") or ""
                    ).upper()

                # Filter attributes if specified
                if attributes:
                    filtered_data = {}
                    for attr in attributes:
                        parts = attr.split(".")
                        current = entity_data
                        try:
                            for part in parts[:-1]:
                                current = current[part]
                            filtered_data[attr] = current[parts[-1]]
                        except (KeyError, TypeError):
                            filtered_data[attr] = None
                    entity_data = filtered_data

                detailed_results.append({"identifier": identifier, "data": entity_data})
            except Exception as e:
                detailed_results.append({"identifier": identifier, "error": str(e)})

        # Download structure files if requested
        if download:
            for identifier in identifiers:
                if "_" in identifier or "." in identifier or "-" in identifier:
                    # For non-entry identifiers, extract the PDB ID
                    if "_" in identifier:
                        pdb_id = identifier.split("_")[0]
                    elif "." in identifier:
                        pdb_id = identifier.split(".")[0]
                    elif "-" in identifier:
                        pdb_id = identifier.split("-")[0]
                else:
                    pdb_id = identifier

                try:
                    if not re.fullmatch(r"[A-Za-z0-9]{4}", pdb_id):
                        raise ValueError(f"Invalid PDB identifier for download: {pdb_id}")

                    compatibility = pdb_format_compatibility.get(pdb_id.upper())
                    preferred_formats = ("mmcif", "pdb") if compatibility == "N" else ("pdb", "mmcif")
                    download_result = _download_rcsb_structure(
                        pdb_id.upper(),
                        resolved_output_dir,
                        preferred_formats,
                    )
                    for result in detailed_results:
                        if result["identifier"] == identifier or result["identifier"].startswith(pdb_id):
                            result.update({key: value for key, value in download_result.items() if key != "success"})
                            if download_result.get("structure_format") == "pdb":
                                result["pdb_file_path"] = download_result["structure_file_path"]
                except Exception as e:
                    for result in detailed_results:
                        if result["identifier"] == identifier or result["identifier"].startswith(pdb_id):
                            result["download_error"] = str(e)

        success_count = sum("data" in result for result in detailed_results)
        error_count = len(detailed_results) - success_count
        download_success_count = sum("structure_file_path" in result for result in detailed_results)
        download_error_count = sum("download_error" in result for result in detailed_results)
        summary = {
            "requested_count": len(identifiers),
            "success_count": success_count,
            "error_count": error_count,
        }
        if download:
            summary.update(
                {
                    "download_success_count": download_success_count,
                    "download_error_count": download_error_count,
                }
            )
        downloads = [
            {
                key: result.get(key)
                for key in (
                    "identifier",
                    "structure_file_path",
                    "structure_format",
                    "download_url",
                    "downloaded_bytes",
                    "download_attempts",
                    "download_error",
                )
                if result.get(key) is not None
            }
            for result in detailed_results
        ]
        response = {
            "success": success_count > 0 and (not download or download_success_count > 0),
            "query_info": query_info,
            "summary": summary,
        }
        if download:
            response["downloads"] = downloads
        response["detailed_results"] = detailed_results
        if success_count == 0:
            response["error"] = "RCSB PDB returned no successful identifier results"
        elif download and download_success_count == 0:
            response["error"] = "RCSB PDB metadata succeeded but no coordinate file could be downloaded"
        return response

    except Exception as e:
        return {"success": False, "error": f"Error retrieving PDB details: {str(e)}", "query_info": query_info}


def query_kegg(prompt=None, endpoint=None, verbose=True):
    """Take a natural language prompt and convert it to a structured KEGG API query.

    Parameters
    ----------
    prompt (str): Natural language query about KEGG data (e.g., "Find human pathways related to glycolysis")
    endpoint (str, optional): Direct KEGG API endpoint to query
    verbose (bool): Whether to print verbose output

    Returns
    -------
    dict: Dictionary containing both the structured query and the KEGG results

    """
    base_url = "https://rest.kegg.jp"

    if not prompt and not endpoint:
        return {"success": False, "error": "Either a prompt or an endpoint must be provided"}

    if prompt:
        # Load schema from pickle file
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "kegg.pkl")
        with open(schema_path, "rb") as f:
            kegg_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a bioinformatics expert that helps convert natural language queries into KEGG API requests.

        Based on the user's natural language request, you will generate a structured query for the KEGG API.

        The KEGG API has the following general form:
        https://rest.kegg.jp/<operation>/<argument>[/<argument2>[/<argument3> ...]]

        Where <operation> can be one of: info, list, find, get, conv, link, ddi

        Here is the schema of available operations, databases, and other details:
        {schema}

        Output only a JSON object with the following fields:
        1. "full_url": The complete URL to query (including the base URL "https://rest.kegg.jp")
        2. "description": A brief description of what the query is doing

        IMPORTANT: Your response must ONLY contain a JSON object with the required fields.

        EXAMPLES OF CORRECT OUTPUTS:
        - For "Find information about glycolysis pathway": {{"full_url": "https://rest.kegg.jp/info/pathway/hsa00010", "description": "Finding information about the glycolysis pathway"}}
        - For "Get information about the human BRCA1 gene": {{"full_url": "https://rest.kegg.jp/get/hsa:672", "description": "Retrieving information about BRCA1 gene in human"}}
        - For "List all human pathways": {{"full_url": "https://rest.kegg.jp/list/pathway/hsa", "description": "Listing all human-specific pathways"}}
        - For "Convert NCBI gene ID 672 to KEGG ID": {{"full_url": "https://rest.kegg.jp/conv/genes/ncbi-geneid:672", "description": "Converting NCBI Gene ID 672 to KEGG gene identifier"}}
        """

        # Query LLM to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=kegg_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

            # Extract the query info from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info["full_url"]
        description = query_info["description"]

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }

    if endpoint:
        if endpoint.startswith("/"):
            endpoint = f"{base_url}{endpoint}"
        elif not endpoint.startswith("http"):
            endpoint = f"{base_url}/{endpoint.lstrip('/')}"
        description = "Direct query to KEGG API"

    # Execute the KEGG API request using the helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        api_result["result"] = _format_query_results(api_result["result"])

    return api_result


def query_stringdb(
    prompt=None,
    endpoint=None,
    download_image=False,
    output_dir=None,
    verbose=True,
    identifiers=None,
    species=9606,
    required_score=400,
):
    """Query the STRING protein interaction database using natural language or direct endpoint.

    Parameters
    ----------
    prompt (str, optional): Natural language query about protein interactions
    endpoint (str, optional): Full URL to query directly (overrides prompt)
    download_image (bool): Whether to download image results (for image endpoints)
    output_dir (str, optional): Directory to save downloaded files (default: current directory)
    identifiers (str or list, optional): Protein or gene identifiers for a structured network query
    species (int): NCBI taxonomy identifier used with ``identifiers``
    required_score (int): Minimum STRING score from 0 to 1000 used with ``identifiers``

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_stringdb("Show protein interactions for BRCA1 and BRCA2 in humans")
    - Direct endpoint: query_stringdb(endpoint="https://string-db.org/api/json/network?identifiers=BRCA1,BRCA2&species=9606")

    """
    # Base URL for STRING API
    base_url = "https://version-12-0.string-db.org/api"

    if not isinstance(required_score, int) or not 0 <= required_score <= 1000:
        return {"success": False, "error": "required_score must be an integer from 0 to 1000"}

    if identifiers is not None:
        if isinstance(identifiers, str):
            identifier_list = [value.strip() for value in re.split(r"[\r\n,]+", identifiers) if value.strip()]
        elif isinstance(identifiers, (list, tuple)):
            identifier_list = [str(value).strip() for value in identifiers if str(value).strip()]
        else:
            return {"success": False, "error": "identifiers must be a string or list of strings"}
        if not identifier_list:
            return {"success": False, "error": "identifiers must contain at least one value"}

        endpoint = f"{base_url}/json/network"
        params = {
            "identifiers": "\r".join(identifier_list),
            "species": species,
            "required_score": required_score,
            "add_nodes": 0,
            "caller_identity": "biomni",
        }
        api_result = _query_rest_api(
            endpoint=endpoint,
            method="GET",
            params=params,
            description="Structured STRING protein interaction network query",
        )
        if not api_result.get("success"):
            return api_result
        rows = api_result.get("result")
        if not isinstance(rows, list):
            return {
                "success": False,
                "error": "STRING returned an unexpected non-list response",
                "query_info": api_result.get("query_info"),
            }
        interactions = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            normalized = dict(row)
            for score_name in (
                "score",
                "nscore",
                "fscore",
                "pscore",
                "ascore",
                "escore",
                "dscore",
                "tscore",
            ):
                if score_name in normalized:
                    try:
                        normalized[score_name] = float(normalized[score_name])
                    except (TypeError, ValueError):
                        pass
            interactions.append(normalized)
        interactions.sort(
            key=lambda item: (
                -float(item.get("score", 0.0)),
                str(item.get("preferredName_A", "")),
                str(item.get("preferredName_B", "")),
            )
        )
        api_result["query_info"].update(
            {
                "identifiers": identifier_list,
                "species": species,
                "required_score": required_score,
                "candidate_scope": "requested identifiers only; no additional network nodes",
            }
        )
        api_result["result"] = {
            "interaction_count": len(interactions),
            "interactions": interactions,
        }
        return api_result

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"success": False, "error": "Provide identifiers, prompt, or endpoint"}

    # If using prompt, parse with Claude
    if prompt:
        # Load STRING schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "stringdb.pkl")
        with open(schema_path, "rb") as f:
            stringdb_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a protein interaction expert specialized in using the STRING database API.

        Based on the user's natural language request, determine the appropriate STRING API endpoint and parameters.

        STRING API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including all parameters)
        2. "description": A brief description of what the query is doing
        3. "output_format": The format of the output (json, tsv, image, svg)

        SPECIAL NOTES:
        - Common species IDs: 9606 (human), 10090 (mouse), 7227 (fruit fly), 4932 (yeast)
        - For protein identifiers, use either gene names (e.g., "BRCA1") or UniProt IDs (e.g., "P38398")
        - The "required_score" parameter accepts values from 0 to 1000 (higher means more stringent)
        - Add "caller_identity=bioagentos_api" as a parameter

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=stringdb_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the full URL from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")
        output_format = query_info.get("output_format", "json")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Use direct endpoint
        if endpoint.startswith("/"):
            endpoint = f"{base_url}{endpoint}"
        elif not endpoint.startswith("http"):
            endpoint = f"{base_url}/{endpoint.lstrip('/')}"
        description = "Direct query to STRING API"
        output_format = "json"

        # Try to determine output format from URL
        if "image" in endpoint or "svg" in endpoint:
            output_format = "image"

    # Check if we're dealing with an image request
    is_image = output_format in ["image", "highres_image", "svg"]

    if is_image:
        if download_image:
            # For images, we need to handle the download manually
            try:
                response = requests.get(endpoint, stream=True, timeout=60)
                response.raise_for_status()

                max_image_bytes = 20 * 1024 * 1024
                if int(response.headers.get("Content-Length", 0)) > max_image_bytes:
                    return {"success": False, "error": "STRING image exceeds the 20 MiB download limit"}

                # Create output directory if needed
                if not output_dir:
                    output_dir = "."
                os.makedirs(output_dir, exist_ok=True)

                # Generate filename based on endpoint
                endpoint_parts = endpoint.split("/")
                endpoint_label = re.sub(r"[^A-Za-z0-9._-]", "_", endpoint_parts[-2])
                filename = f"string_{endpoint_label}_{int(time.time())}.{output_format}"
                file_path = os.path.join(output_dir, filename)

                # Save the image
                downloaded_bytes = 0
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            downloaded_bytes += len(chunk)
                            if downloaded_bytes > max_image_bytes:
                                f.close()
                                os.unlink(file_path)
                                return {"success": False, "error": "STRING image exceeds the 20 MiB download limit"}
                            f.write(chunk)

                return {
                    "success": True,
                    "query_info": {
                        "endpoint": endpoint,
                        "description": description,
                        "output_format": output_format,
                    },
                    "result": {
                        "image_saved": True,
                        "file_path": file_path,
                        "content_type": response.headers.get("Content-Type"),
                    },
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Error downloading image: {str(e)}",
                    "query_info": {"endpoint": endpoint, "description": description},
                }
        else:
            # Just report that an image is available but not downloaded
            return {
                "success": True,
                "query_info": {
                    "endpoint": endpoint,
                    "description": description,
                    "output_format": output_format,
                },
                "result": {
                    "image_available": True,
                    "download_url": endpoint,
                    "note": "Set download_image=True to save the image",
                },
            }

    # For non-image requests, use the REST API helper
    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        api_result["result"] = _format_query_results(api_result["result"])

    return api_result


def query_iucn(
    prompt=None,
    endpoint=None,
    token="",
    verbose=True,
):
    """Query the IUCN Red List API using natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, required): Natural language query about species conservation status
    endpoint (str, optional): API endpoint name (e.g., "species/id/12392") or full URL
    token (str): IUCN API token - required for all queries
    verbose (bool): Whether to print verbose output

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_iucn("Get conservation status of white rhinoceros", token="your-token")
    - Direct endpoint: query_iucn(endpoint="species/id/12392", token="your-token")

    """
    # Base URL for IUCN API
    base_url = "https://apiv3.iucnredlist.org/api/v3"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # Ensure we have a token
    if not token:
        return {"error": "IUCN API token is required. Get one at https://apiv3.iucnredlist.org/api/v3/token"}

    # If using prompt, parse with Claude
    if prompt:
        # Load IUCN schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "iucn.pkl")
        with open(schema_path, "rb") as f:
            iucn_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a conservation biology expert specialized in using the IUCN Red List API.

        Based on the user's natural language request, determine the appropriate IUCN API endpoint.

        IUCN API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including the base URL "https://apiv3.iucnredlist.org/api/v3" and any path parameters)
        2. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - The token parameter will be added automatically, do not include it in your URL
        - For taxonomic queries, prefer using scientific names over common names
        - For region-specific queries, use region identifiers from the schema
        - For species queries, try to use the species ID if known, otherwise use scientific name

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=iucn_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the full URL from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Process provided endpoint
        if not endpoint.startswith("http"):
            endpoint = f"{base_url}{endpoint}" if endpoint.startswith("/") else f"{base_url}/{endpoint}"
        description = "Direct query to IUCN API"

    # Add token as query parameter
    params = {"token": token}

    # Execute the IUCN API request using the helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", params=params, description=description)

    # For security, remove token from the results
    if "query_info" in api_result and "endpoint" in api_result["query_info"]:
        api_result["query_info"]["endpoint"] = api_result["query_info"]["endpoint"].replace(token, "TOKEN_HIDDEN")

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])

    return api_result


def query_paleobiology(
    prompt=None,
    endpoint=None,
    verbose=True,
):
    """Query the Paleobiology Database (PBDB) API using natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, required): Natural language query about fossil records
    endpoint (str, optional): API endpoint name or full URL
    verbose (bool): Whether to print verbose output

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_paleobiology("Find fossil records of Tyrannosaurus rex")
    - Direct endpoint: query_paleobiology(endpoint="data1.2/taxa/list.json?name=Tyrannosaurus")

    """
    # Base URL for PBDB API
    base_url = "https://paleobiodb.org/data1.2"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load PBDB schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "pbdb.pkl")
        with open(schema_path, "rb") as f:
            pbdb_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a paleobiology expert specialized in using the Paleobiology Database (PBDB) API.

        Based on the user's natural language request, determine the appropriate PBDB API endpoint and parameters.

        PBDB API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including the base URL "https://paleobiodb.org/data1.2" and format extension)
        2. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - For taxonomic queries, be specific about taxonomic ranks and names
        - For geographic queries, use standard country/continent names or coordinate bounding boxes
        - For time interval queries, use standard geological time names (e.g., "Cretaceous", "Maastrichtian")
        - Use appropriate format extension (.json, .txt, .csv, .tsv) based on the query
        - If appropriate, use "vocab=pbdb" (default) or "vocab=com" (compact) parameter in the URL
        - For detailed occurrence data, include "show=paleoloc,phylo" in the parameters

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=pbdb_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the full URL from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Process provided endpoint
        if not endpoint.startswith("http"):
            # Add base URL if it's just a path
            endpoint = f"{base_url}/{endpoint}" if not endpoint.startswith("/") else f"{base_url}{endpoint}"

        description = "Direct query to PBDB API"

    # Check if we're dealing with an image request
    is_image = endpoint.endswith(".png")

    if is_image:
        # For image queries, we need special handling
        try:
            response = requests.get(endpoint, timeout=30)
            response.raise_for_status()

            # Return image metadata without the binary data
            return {
                "success": True,
                "query_info": {
                    "endpoint": endpoint,
                    "description": description,
                    "format": "png",
                },
                "result": {
                    "content_type": response.headers.get("Content-Type"),
                    "size_bytes": len(response.content),
                    "note": "Binary image data not included in response",
                },
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error retrieving image: {str(e)}",
                "query_info": {"endpoint": endpoint, "description": description},
            }

    # For non-image requests, use the REST API helper
    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])

    return api_result


def query_jaspar(
    prompt=None,
    endpoint=None,
    verbose=True,
):
    """Query the JASPAR REST API using natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, required): Natural language query about transcription factor binding profiles
    endpoint (str, optional): API endpoint path (e.g., "/matrix/MA0002.2/") or full URL
    verbose (bool): Whether to print verbose output

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_jaspar("Find all transcription factor matrices for human")
    - Direct endpoint: query_jaspar(endpoint="/matrix/MA0002.2/")

    """
    # Base URL for JASPAR API
    base_url = "https://jaspar.elixir.no/api/v1"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load JASPAR schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "jaspar.pkl")
        with open(schema_path, "rb") as f:
            jaspar_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a transcription factor binding site expert specialized in using the JASPAR REST API.

        Based on the user's natural language request, determine the appropriate JASPAR REST API endpoint and parameters.

        JASPAR REST API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including the base URL "https://jaspar.elixir.no/api/v1" and any parameters)
        2. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - Common taxonomic groups include: vertebrates, plants, fungi, insects, nematodes, urochordates
        - Common collections include: CORE, UNVALIDATED, PENDING, etc.
        - Matrix IDs follow the format MA####.# (e.g., MA0002.2)
        - For inferring matrices from sequences, provide the protein sequence directly in the path

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=jaspar_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the full URL from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Process provided endpoint
        if not endpoint.startswith("http"):
            # Clean up endpoint format
            if not endpoint.startswith("/"):
                endpoint = "/" + endpoint

            # Ensure endpoint ends with /
            if not endpoint.endswith("/"):
                endpoint = endpoint + "/"

            # Add base URL
            endpoint = f"{base_url}{endpoint}"

        description = "Direct query to JASPAR API"

    # Execute the JASPAR API request using the helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])

    return api_result


def query_worms(
    prompt=None,
    endpoint=None,
    verbose=True,
):
    """Query the World Register of Marine Species (WoRMS) REST API using natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, required): Natural language query about marine species
    endpoint (str, optional): Full URL or endpoint specification
    verbose (bool): Whether to print verbose output

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_worms("Find information about the blue whale")
    - Direct endpoint: query_worms(endpoint="https://www.marinespecies.org/rest/AphiaRecordByName/Balaenoptera%20musculus")

    """
    # Base URL for WoRMS API
    base_url = "https://www.marinespecies.org/rest"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load WoRMS schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "worms.pkl")
        with open(schema_path, "rb") as f:
            worms_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a marine biology expert specialized in using the World Register of Marine Species (WoRMS) API.

        Based on the user's natural language request, determine the appropriate WoRMS API endpoint and parameters.

        WORMS API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including the base URL "https://www.marinespecies.org/rest" and any path/query parameters)
        2. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - For taxonomic searches, be precise with scientific names and use proper capitalization
        - For fuzzy matching, include "fuzzy=true" in the URL query parameters
        - When searching by name, prefer "AphiaRecordByName" for exact matches and "AphiaRecordsByName" for broader results
        - AphiaID is the main identifier in WoRMS (e.g., Blue Whale is 137087)
        - For multiple IDs or names, use the appropriate POST endpoint

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=worms_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the full URL and details from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Process provided endpoint
        if not endpoint.startswith("http"):
            # Add base URL if it's just a path
            endpoint = f"{base_url}/{endpoint}" if not endpoint.startswith("/") else f"{base_url}{endpoint}"

        description = "Direct query to WoRMS API"

    # Execute the WoRMS API request using the helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])

    return api_result


def query_cbioportal(
    prompt=None,
    endpoint=None,
    verbose=True,
):
    """Query the cBioPortal REST API using natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, required): Natural language query about cancer genomics data
    endpoint (str, optional): API endpoint path (e.g., "/studies/brca_tcga/patients") or full URL
    verbose (bool): Whether to print verbose output

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_cbioportal("Find mutations in BRCA1 for breast cancer")
    - Direct endpoint: query_cbioportal(endpoint="/studies/brca_tcga/molecular-profiles")

    """
    # Base URL for cBioPortal API
    base_url = "https://www.cbioportal.org/api"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load cBioPortal schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "cbioportal.pkl")
        with open(schema_path, "rb") as f:
            cbioportal_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a cancer genomics expert specialized in using the cBioPortal REST API.

        Based on the user's natural language request, determine the appropriate cBioPortal REST API endpoint and parameters.

        CBIOPORTAL REST API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including the base URL "https://www.cbioportal.org/api" and any parameters)
        2. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - For gene queries, use either Hugo symbol (e.g., "BRCA1") or Entrez ID (e.g., 672)
        - For pagination, include parameters "pageNumber" and "pageSize" if needed
        - For mutation data queries, always include appropriate sample identifiers
        - Common studies include: "brca_tcga" (breast cancer), "gbm_tcga" (glioblastoma), "luad_tcga" (lung adenocarcinoma)
        - For molecular profiles, common IDs follow pattern: "[study]_[data_type]" (e.g., "brca_tcga_mutations")
        - Consider including "projection=DETAILED" for more comprehensive results when appropriate

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=cbioportal_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the full URL from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Process provided endpoint
        if not endpoint.startswith("http"):
            # Clean up endpoint format
            if not endpoint.startswith("/"):
                endpoint = "/" + endpoint

            # Add base URL
            endpoint = f"{base_url}{endpoint}"

        description = "Direct query to cBioPortal API"

    # Execute the cBioPortal API request using the helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])

    return api_result


def query_clinvar(
    prompt=None,
    search_term=None,
    max_results=3,
):
    """Take a natural language prompt and convert it to a structured ClinVar query.

    Parameters
    ----------
    prompt (str): Natural language query about genetic variants (e.g., "Find pathogenic BRCA1 variants")
    search_term (str): Direct search term in ClinVar syntax
    max_results (int): Maximum number of results to return

    Returns
    -------
    dict: Dictionary containing both the structured query and the ClinVar results

    """
    if not prompt and not search_term:
        return {"error": "Either a prompt or an endpoint must be provided"}

    if prompt:
        # Load ClinVar schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "clinvar.pkl")
        with open(schema_path, "rb") as f:
            clinvar_schema = pickle.load(f)

        # ClinVar system prompt template
        system_prompt_template = """
        You are a genetics research assistant that helps convert natural language queries into structured ClinVar search queries.

        Based on the user's natural language request, you will generate a structured search for the ClinVar database.

        Output only a JSON object with the following fields:
        1. "search_term": The exact search query to use with the ClinVar API

        IMPORTANT: Your response must ONLY contain a JSON object with the search term field.

        Your "search_term" MUST strictly follow these ClinVar search syntax rules/tags:

        {schema}

        For combining terms: Use AND, OR, NOT (must be capitalized)
        For complex logic: Use parentheses
        For terms with multiple words: use double quotes escaped with a backslash or underscore (e.g. breast_cancer[dis] or \"breast cancer\"[dis])
        Example: "BRCA1[gene] AND (pathogenic[clinsig] OR likely_pathogenic[clinsig])"


        EXAMPLES OF CORRECT QUERIES:
        - For "pathogenic BRCA1 variants": "BRCA1[gene] AND clinsig_pathogenic[prop]"
        - For "Specific RS": "rs6025[rsid]"
        - For "Combined search with multiple criteria": "BRCA1[gene] AND origin_germline[prop]"
        - For "Find variants in a specific genomic region": "17[chr] AND 43000000:44000000[chrpos37]"
        - If query asks for pathogenicity of a variant, it's asking for all possible germline classifications of the variant, so just [gene] AND [variant] is needed
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=clinvar_schema,
            system_template=system_prompt_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the full URL from Claude's response
        query_info = llm_result["data"]
        search_term = query_info.get("search_term", "")

        if not search_term:
            return {
                "error": "Failed to generate a valid search term from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }

    return _query_ncbi_database(
        database="clinvar",
        search_term=search_term,
        max_results=max_results,
    )


def query_geo(
    prompt=None,
    search_term=None,
    max_results=3,
):
    """Query the NCBI Gene Expression Omnibus (GEO) using natural language or a direct search term.

    Parameters
    ----------
    prompt (str, required): Natural language query about RNA-seq, microarray, or other expression data
    search_term (str, optional): Direct search term in GEO syntax
    max_results (int): Maximum number of results to return

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_geo("Find RNA-seq datasets for breast cancer")
    - Direct search: query_geo(search_term="RNA-seq AND breast cancer AND gse[ETYP]")

    """
    if not prompt and not search_term:
        return {"error": "Either a prompt or a search term must be provided"}

    database = "gds"  # Default database

    if prompt:
        # Load GEO schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "geo.pkl")
        with open(schema_path, "rb") as f:
            geo_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a bioinformatics research assistant that helps convert natural language queries into structured GEO (Gene Expression Omnibus) search queries.

        Based on the user's natural language request, you will generate a structured search for the GEO database.

        Output only a JSON object with the following fields:
        1. "search_term": The exact search query to use with the GEO API
        2. "database": The specific GEO database to search (either "gds" for GEO DataSets or "geoprofiles" for GEO Profiles)

        IMPORTANT: Your response must ONLY contain a JSON object with the required fields.

        Your "search_term" MUST strictly follow these GEO search syntax rules/tags:

        {schema}

        For combining terms: Use AND, OR, NOT (must be capitalized)
        For complex logic: Use parentheses
        For terms with multiple words: use double quotes or underscore (e.g. "breast cancer"[Title])
        Date ranges use colon format: 2015/01:2020/12[PDAT]

        Choose the appropriate database based on the user's query:
        - gds: GEO DataSets (contains Series, Datasets, Platforms, Samples metadata)
        - geoprofiles: GEO Profiles (contains gene expression data)

        If database isn't clearly specified, default to "gds" as it contains most common experiment metadata.

        EXAMPLES OF CORRECT OUTPUTS:
        - For "RNA-seq data in breast cancer": {"search_term": "RNA-seq AND breast cancer AND gse[ETYP]", "database": "gds"}
        - For "Mouse microarray data from 2020": {"search_term": "Mus musculus[ORGN] AND 2020[PDAT] AND microarray AND gse[ETYP]", "database": "gds"}
        - For "Expression profiles of TP53 in lung cancer": {"search_term": "TP53[Gene Symbol] AND lung cancer", "database": "geoprofiles"}
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=geo_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the search term and database from Claude's response
        query_info = llm_result["data"]
        search_term = query_info.get("search_term", "")
        database = query_info.get("database", "gds")

        if not search_term:
            return {
                "error": "Failed to generate a valid search term from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }

    # Execute the GEO query using the helper function
    result = _query_ncbi_database(
        database=database,
        search_term=search_term,
        max_results=max_results,
    )

    return result


def query_dbsnp(
    prompt=None,
    search_term=None,
    max_results=3,
):
    """Query the NCBI dbSNP database using natural language or a direct search term.

    Parameters
    ----------
    prompt (str, required): Natural language query about genetic variants/SNPs
    search_term (str, optional): Direct search term in dbSNP syntax
    max_results (int): Maximum number of results to return

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_dbsnp("Find pathogenic variants in BRCA1")
    - Direct search: query_dbsnp(search_term="BRCA1[Gene Name] AND pathogenic[Clinical Significance]")

    """
    if not prompt and not search_term:
        return {"error": "Either a prompt or a search term must be provided"}

    if prompt:
        # Load dbSNP schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "dbsnp.pkl")
        with open(schema_path, "rb") as f:
            dbsnp_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a genetics research assistant that helps convert natural language queries into structured dbSNP search queries.

        Based on the user's natural language request, you will generate a structured search for the dbSNP database.

        Output only a JSON object with the following fields:
        1. "search_term": The exact search query to use with the dbSNP API

        IMPORTANT: Your response must ONLY contain a JSON object with the search term field.

        Your "search_term" MUST strictly follow these dbSNP search syntax rules/tags:

        {schema}

        For combining terms: Use AND, OR, NOT (must be capitalized)
        For complex logic: Use parentheses
        For terms with multiple words: use double quotes (e.g. "breast cancer"[Disease Name])

        EXAMPLES OF CORRECT QUERIES:
        - For "pathogenic variants in BRCA1": "BRCA1[Gene Name] AND pathogenic[Clinical Significance]"
        - For "specific SNP rs6025": "rs6025[rs]"
        - For "SNPs in a genomic region": "17[Chromosome] AND 41196312:41277500[Base Position]"
        - For "common SNPs in EGFR": "EGFR[Gene Name] AND common[COMMON]"
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=dbsnp_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the search term from Claude's response
        query_info = llm_result["data"]
        search_term = query_info.get("search_term", "")

        if not search_term:
            return {
                "error": "Failed to generate a valid search term from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }

    # Execute the dbSNP query using the helper function
    result = _query_ncbi_database(
        database="snp",
        search_term=search_term,
        max_results=max_results,
    )

    return result


def query_ucsc(
    prompt=None,
    endpoint=None,
    verbose=True,
):
    """Query the UCSC Genome Browser API using natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, required): Natural language query about genomic data
    endpoint (str, optional): Full URL or endpoint specification with parameters
    verbose (bool): Whether to return detailed results

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_ucsc("Get DNA sequence of chromosome M positions 1-100 in human genome")
    - Direct endpoint: query_ucsc(endpoint="https://api.genome.ucsc.edu/getData/sequence?genome=hg38&chrom=chrM&start=1&end=100")

    """
    # Base URL for UCSC API
    base_url = "https://api.genome.ucsc.edu"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load UCSC schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "ucsc.pkl")
        with open(schema_path, "rb") as f:
            ucsc_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a genomics expert specialized in using the UCSC Genome Browser API.

        Based on the user's natural language request, determine the appropriate UCSC Genome Browser API endpoint and parameters.

        UCSC GENOME BROWSER API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including the base URL "https://api.genome.ucsc.edu" and all parameters)
        2. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - For chromosome names, always include the "chr" prefix (e.g., "chr1", "chrX", "chrM")
        - Genomic positions are 0-based (first base is position 0)
        - For "start" and "end" parameters, both must be provided together
        - The "maxItemsOutput" parameter can be used to limit the amount of data returned
        - Common genomes include: "hg38" (human), "mm39" (mouse), "danRer11" (zebrafish)
        - For sequence data, use "getData/sequence" endpoint
        - For chromosome listings, use "list/chromosomes" endpoint
        - For available genomes, use "list/ucscGenomes" endpoint

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=ucsc_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the full URL from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }

    else:
        # Process provided endpoint
        if not endpoint.startswith("http"):
            # Add base URL if it's just a path
            endpoint = f"{base_url}/{endpoint}"

        description = "Direct query to UCSC Genome Browser API"

    # Execute the UCSC API request using the helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    # Format the results if successful
    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])

    return api_result


def query_ensembl(
    prompt=None,
    endpoint=None,
    verbose=True,
):
    """Query the Ensembl REST API using natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, required): Natural language query about genomic data
    endpoint (str, optional): Direct API endpoint to query (e.g., "lookup/symbol/human/BRCA2") or full URL
    verbose (bool): Whether to return detailed results

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_ensembl("Get information about the human BRCA2 gene")
    - Direct endpoint: query_ensembl(endpoint="lookup/symbol/homo_sapiens/BRCA2")

    """
    # Base URL for Ensembl API
    base_url = "https://rest.ensembl.org"

    # Ensure we have either a prompt or an endpoint
    if not prompt and not endpoint:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load Ensembl schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "ensembl.pkl")
        with open(schema_path, "rb") as f:
            ensembl_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a genomics and bioinformatics expert specialized in using the Ensembl REST API.

        Based on the user's natural language request, determine the appropriate Ensembl REST API endpoint and parameters.

        ENSEMBL REST API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "endpoint": The API endpoint to query (e.g., "lookup/symbol/homo_sapiens/BRCA2")
        2. "params": An object containing query parameters specific to the endpoint
        3. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - Chromosome region queries have a maximum length of 4900000 bp inclusive, so bp of start and end should be 4900000 bp apart. If the user's query exceeds this limit, Ensembl will return an error.
        - For symbol lookups, the format is "lookup/symbol/[species]/[symbol]"
        - To find the coordinates of a band on a chromosome, use /info/assembly/homo_sapiens/[chromosome] with parameters "band":1
        - To find the overlapping genes of a genomic region, use /overlap/region/homo_sapiens/[chromosome]:[start]-[end]
        - For sequence queries, specify the sequence type in parameters (genomic, cdna, cds, protein)
        - For converting rsID to hg38 genomic coordinates, use the "GET id/variation/[species]/[rsid]" endpoint
        - Many endpoints support "content-type" parameter for format specification (application/json, text/xml)

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=ensembl_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the endpoint and parameters from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("endpoint", "")
        params = query_info.get("params", {})
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }

        # ``:region`` and bracketed placeholders are schema examples, not
        # valid Ensembl paths.  Return a repairable error before making a
        # guaranteed 400 request; the agent can then ask for coordinates or
        # use a local dataset for genome-wide counting.
        if ":region" in endpoint or any(token in endpoint for token in ("[start]", "[end]", "{region}")):
            return {
                "success": False,
                "error": (
                    "Ensembl region endpoint requires concrete coordinates, "
                    "for example overlap/region/homo_sapiens/1:100-200. "
                    "Use a local dataset or chromosome-by-chromosome queries for genome-wide counts."
                ),
                "endpoint": endpoint,
            }
    else:
        # Process provided endpoint
        if endpoint.startswith("http"):
            # If a full URL is provided, extract the endpoint part
            if endpoint.startswith(base_url):
                endpoint = endpoint[len(base_url) :].lstrip("/")

        params = {}
        description = "Direct query to Ensembl API"

    # Remove leading slash if present
    if endpoint.startswith("/"):
        endpoint = endpoint[1:]

    # Prepare headers for JSON response
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    # Construct the URL
    url = f"{base_url}/{endpoint}"

    # Execute the Ensembl API request using the helper function
    api_result = _query_rest_api(
        endpoint=url,
        method="GET",
        params=params,
        headers=headers,
        description=description,
    )

    # Format the results if successful
    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])

    return api_result


def query_opentarget(
    prompt=None,
    query=None,
    variables=None,
    verbose=False,
    disease_name=None,
    disease_id=None,
    max_results=10,
):
    """Query the OpenTargets Platform API using natural language or a direct GraphQL query.

    Parameters
    ----------
    prompt (str, optional): Natural language query about drug targets, diseases, and mechanisms
    query (str, optional): Direct GraphQL query string
    variables (dict, optional): Variables for the GraphQL query
    verbose (bool): Whether to return detailed results
    disease_name (str, optional): Disease name to resolve before retrieving associated targets
    disease_id (str, optional): Open Targets disease identifier, bypassing name resolution
    max_results (int): Maximum number of associated targets to return

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_opentarget("Find drug targets for Alzheimer's disease")
    - Direct query: query_opentarget(query="query diseaseAssociations($diseaseId: String!) {...}",
                                     variables={"diseaseId": "EFO_0000249"})

    """
    # Constants and initialization
    OPENTARGETS_URL = "https://api.platform.opentargets.org/api/v4/graphql"

    if not isinstance(max_results, int) or max_results < 1:
        return {"success": False, "error": "max_results must be a positive integer"}

    if disease_name or disease_id:
        resolved_id = str(disease_id).strip() if disease_id else None
        resolved_name = None
        if not resolved_id:
            search_query = """
            query SearchDisease($queryString: String!) {
              search(queryString: $queryString, entityNames: ["disease"]) {
                total
                hits { id name entity }
              }
            }
            """
            search_result = _query_rest_api(
                endpoint=OPENTARGETS_URL,
                method="POST",
                json_data={"query": search_query, "variables": {"queryString": str(disease_name).strip()}},
                headers={"Content-Type": "application/json"},
                description="Resolve an Open Targets disease name",
            )
            if not search_result.get("success"):
                return search_result
            search_payload = search_result.get("result", {})
            if search_payload.get("errors"):
                return {
                    "success": False,
                    "error": "OpenTargets disease search returned GraphQL errors",
                    "graphql_errors": search_payload["errors"],
                }
            search_data = search_payload.get("data") or {}
            hits = (search_data.get("search") or {}).get("hits") or []
            disease_hits = [hit for hit in hits if hit.get("entity") == "disease" and hit.get("id")]
            if not disease_hits:
                return {"success": False, "error": f"No Open Targets disease found for {disease_name!r}"}
            requested_name = str(disease_name).strip().casefold()
            selected = next(
                (hit for hit in disease_hits if str(hit.get("name", "")).strip().casefold() == requested_name),
                disease_hits[0],
            )
            resolved_id = selected["id"]
            resolved_name = selected.get("name")

        association_query = """
        query DiseaseAssociations($efoId: String!, $size: Int!) {
          disease(efoId: $efoId) {
            id
            name
            associatedTargets(page: {index: 0, size: $size}) {
              count
              rows {
                score
                target { id approvedSymbol approvedName }
              }
            }
          }
        }
        """
        association_result = _query_rest_api(
            endpoint=OPENTARGETS_URL,
            method="POST",
            json_data={"query": association_query, "variables": {"efoId": resolved_id, "size": max_results}},
            headers={"Content-Type": "application/json"},
            description="Retrieve Open Targets disease-associated targets",
        )
        if not association_result.get("success"):
            return association_result
        payload = association_result.get("result", {})
        if payload.get("errors"):
            return {
                "success": False,
                "error": "OpenTargets association query returned GraphQL errors",
                "graphql_errors": payload["errors"],
            }
        disease = (payload.get("data") or {}).get("disease")
        if not disease:
            return {
                "success": False,
                "error": f"Open Targets returned no disease data for {resolved_id}",
                "query_info": {"disease_name": disease_name, "disease_id": resolved_id},
            }
        associated = disease.get("associatedTargets") or {}
        rows = associated.get("rows") or []
        targets = []
        for rank, row in enumerate(rows[:max_results], start=1):
            target = row.get("target") or {}
            targets.append(
                {
                    "rank": rank,
                    "score": row.get("score"),
                    "target_id": target.get("id"),
                    "gene_symbol": target.get("approvedSymbol"),
                    "target_name": target.get("approvedName"),
                }
            )
        return {
            "success": True,
            "query_info": {
                "disease_name": disease_name or resolved_name or disease.get("name"),
                "disease_id": resolved_id,
                "max_results": max_results,
            },
            "result": {
                "disease": {"id": disease.get("id"), "name": disease.get("name")},
                "total_count": associated.get("count", len(targets)),
                "associated_targets": targets,
            },
        }

    # Ensure we have either a prompt or a query
    if prompt is None and query is None:
        return {"success": False, "error": "Provide disease_name, disease_id, prompt, or GraphQL query"}

    # If using prompt, parse with Claude
    if prompt:
        # Load OpenTargets schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "opentarget.pkl")
        with open(schema_path, "rb") as f:
            opentarget_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are an expert in translating natural language requests into GraphQL queries for the OpenTargets Platform API.

        Here is a schema of the main types and queries available in the OpenTargets Platform API:
        {schema}

        Translate the user's natural language request into a valid GraphQL query for this API.
        Return only a JSON object with two fields:
        1. "query": The complete GraphQL query string
        2. "variables": A JSON object containing the variables needed for the query

        SPECIAL NOTES:
        - Disease IDs typically use EFO ontology (e.g., "EFO_0000249" for Alzheimer's disease)
        - Target IDs typically use Ensembl IDs (e.g., "ENSG00000197386" for ENSG00000197386)
        - The API can provide information about drug-target associations, disease-target associations, etc.
        - Always limit results to a reasonable number using "first" parameter (e.g., first: 10)
        - Always escape special characters, including quotes, in the query string (eg. \\" instead of ")

        Return ONLY the JSON object with no additional text or explanations.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=opentarget_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the query and variables from Claude's response
        query_info = llm_result["data"]
        query = query_info.get("query", "")
        if variables is None:  # Only use Claude's variables if none provided
            variables = query_info.get("variables", {})

        if not query:
            return {
                "error": "Failed to generate a valid GraphQL query from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }

    # Execute the GraphQL query
    api_result = _query_rest_api(
        endpoint=OPENTARGETS_URL,
        method="POST",
        json_data={"query": query, "variables": variables or {}},
        headers={"Content-Type": "application/json"},
        description="OpenTargets Platform GraphQL query",
    )

    if api_result.get("success") and isinstance(api_result.get("result"), dict):
        graphql_errors = api_result["result"].get("errors")
        if graphql_errors:
            return {
                "success": False,
                "error": "OpenTargets GraphQL returned errors",
                "graphql_errors": graphql_errors,
                "query_info": api_result.get("query_info"),
            }
        business_data = api_result["result"].get("data")
        if business_data is None or (
            isinstance(business_data, dict) and "disease" in business_data and business_data["disease"] is None
        ):
            return {
                "success": False,
                "error": "OpenTargets returned no business data",
                "query_info": api_result.get("query_info"),
            }

    # Format the results if not verbose and successful
    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        api_result["result"] = _format_query_results(api_result["result"])

    return api_result


def query_pharos_target(
    gene_symbol: str,
    disease_name: str | None = None,
    max_results: int = 10,
) -> dict:
    """Retrieve target, disease, and ligand evidence for one HGNC symbol from Pharos."""
    from biomni.tool._a1_evidence_tools import query_pharos_target_impl

    return query_pharos_target_impl(gene_symbol, disease_name, max_results)


def query_opentargets_genetic_evidence(
    gene_symbol: str,
    disease_name: str | None = None,
    max_results: int = 10,
) -> dict:
    """Retrieve Open Targets genetic evidence for one HGNC symbol."""
    from biomni.tool._a1_evidence_tools import query_opentargets_genetic_evidence_impl

    return query_opentargets_genetic_evidence_impl(gene_symbol, disease_name, max_results)


def query_gtex_expression(
    gene_symbol: str,
    data_lake_path: str,
    top_n: int = 10,
) -> dict:
    """Read the normal-tissue expression profile for one HGNC symbol from a GTEx snapshot."""
    from biomni.tool._a1_evidence_tools import query_gtex_expression_impl

    return query_gtex_expression_impl(gene_symbol, data_lake_path, top_n)


def query_depmap_gene_dependency(
    gene_symbol: str,
    data_lake_path: str,
    lineage: str | None = None,
    include_expression: bool = False,
    top_n: int = 20,
) -> dict:
    """Read cancer-model dependency evidence for one HGNC symbol from a DepMap snapshot."""
    from biomni.tool._a1_evidence_tools import query_depmap_gene_dependency_impl

    return query_depmap_gene_dependency_impl(gene_symbol, data_lake_path, lineage, include_expression, top_n)


# Monarch Initiative integration
def query_monarch(
    prompt=None,
    endpoint=None,
    max_results=2,
    verbose=False,
):
    """Query the Monarch Initiative API using natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, optional): Natural language query about genes, diseases, phenotypes, etc.
    endpoint (str, optional): Direct Monarch API endpoint or full URL
    max_results (int): Maximum number of results to return (if supported by endpoint)
    verbose (bool): Whether to return detailed results

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_monarch("Find phenotypes associated with BRCA1")
    - Direct endpoint: query_monarch(endpoint="https://api.monarchinitiative.org/v3/api/search?q=marfan&category=biolink:Disease&limit=10")
    - Direct endpoint: query_monarch(endpoint="https://api.monarchinitiative.org/v3/api/entity/MONDO:0007947")
    """
    base_url = "https://api.monarchinitiative.org/v3/api"

    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, use Claude to generate the endpoint
    if prompt:
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "monarch.pkl")
        if os.path.exists(schema_path):
            with open(schema_path, "rb") as f:
                monarch_schema = pickle.load(f)
        else:
            monarch_schema = None

        system_template = """
        You are an expert in translating natural language requests into REST API calls for the Monarch Initiative Platform API.

        Here is the API schema with available endpoints and parameters:
        {schema}

        Translate the user's natural language request into a valid REST API call for this API.
        Return only a JSON object with three fields:
        1. "endpoint": The specific endpoint name from the schema
        2. "url": The complete URL with path parameters filled in
        3. "params": A JSON object containing query parameters needed for the request

        SPECIAL NOTES:
        - Disease IDs typically use MONDO ontology (e.g., "MONDO:0007947" for Marfan syndrome)
        - Gene IDs typically use HGNC (e.g., "HGNC:3603" for FBN1) or other standard identifiers
        - Phenotype IDs use Human Phenotype Ontology (e.g., "HP:0002616" for aortic root dilatation)
        - Association categories use biolink model terms (e.g., "biolink:DiseaseToPhenotypicFeatureAssociation")
        - For example: to find phenotypes associated with BRCA1, use the following endpoint: /entity/HGNC:1100/biolink:GeneToPhenotypicFeatureAssociation
        - For search queries, use the 'q' parameter with relevant keywords
        - When looking for associations, use the association_table endpoint with entity ID and category
        - For similarity searches, use semsim endpoints with comma-separated term lists
        - Entity categories include: biolink:Disease, biolink:Gene, biolink:PhenotypicFeature, etc.
        - Format parameter defaults to 'json' but can be 'tsv' for tabular data
        - Use autocomplete endpoint for entity name suggestions before exact searches

        COMMON PATTERNS:
        - Search for entities: Use 'search' endpoint with 'q' and 'category' parameters
        - Get entity details: Use 'get_entity' endpoint with specific ID
        - Find associations: Use 'association_table' endpoint with ID and association category
        - Compare phenotypes: Use 'semsim_compare' with lists of phenotype IDs
        - Find similar diseases: Use 'semsim_search' with phenotype profile

        Return ONLY the JSON object with no additional text or explanations.
        """

        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=monarch_schema,
            system_template=system_template,
        )
        if not llm_result["success"]:
            return llm_result
        query_info = llm_result["data"]
        endpoint = query_info.get("url", "")  # Changed from "full_url" to "url"
        description = f"Monarch API query: {query_info.get('endpoint', 'unknown endpoint')}"
        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Use provided endpoint directly
        if endpoint is not None:
            if endpoint.startswith("/"):
                endpoint = f"{base_url}{endpoint}"
            elif not endpoint.startswith("http"):
                endpoint = f"{base_url}/{endpoint.lstrip('/')}"
        description = "Direct query to Monarch API"

    # Add max_results as a query parameter if not already present
    if "?" in endpoint:
        if "rows=" not in endpoint and "limit=" not in endpoint:
            endpoint += f"&limit={max_results}"
    else:
        endpoint += f"?limit={max_results}"

    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])

    return api_result


# OpenFDA integration
def query_openfda(
    prompt=None,
    endpoint=None,
    max_results=100,
    verbose=True,
    search_params=None,
    sort_params=None,
    count_params=None,
    skip_results=0,
):
    """Query the OpenFDA API using natural language or direct parameters.

    Parameters
    ----------
    prompt (str, optional): Natural language query about drugs, adverse events, recalls, etc.
    endpoint (str, optional): Direct OpenFDA API endpoint or full URL
    max_results (int): Maximum number of results to return (if supported by endpoint)
    verbose (bool): Whether to return detailed results
    search_params (dict, optional): Search parameters in format {"field": "term"} or {"field": ["term1", "term2"]}
    sort_params (dict, optional): Sort parameters in format {"field": "asc|desc"}
    count_params (str, optional): Field to count unique values for
    skip_results (int): Number of results to skip for pagination (max 25000)

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_openfda("Find adverse events for Lipitor")
    - Direct endpoint: query_openfda(endpoint="https://api.fda.gov/drug/event.json?search=patient.drug.medicinalproduct:lipitor")
    - Search params: query_openfda(search_params={"patient.drug.medicinalproduct": "lipitor"}, endpoint="/drug/event.json")
    - Count reactions: query_openfda(count_params="patient.reaction.reactionmeddrapt.exact", endpoint="/drug/event.json")
    """
    base_url = "https://api.fda.gov"

    if not 1 <= max_results <= 1000:
        return {"success": False, "error": "max_results must be between 1 and 1000"}
    if not 0 <= skip_results <= 25000:
        return {"success": False, "error": "skip_results must be between 0 and 25000"}
    if sort_params is not None and not isinstance(sort_params, dict):
        return {"success": False, "error": "sort_params must be a dictionary"}
    if search_params is not None and not isinstance(search_params, dict):
        return {"success": False, "error": "search_params must be a dictionary"}

    if prompt is None and endpoint is None and search_params is None and count_params is None:
        return {
            "success": False,
            "error": "Either a prompt, endpoint, search_params, or count_params must be provided",
        }

    # If using prompt, use LLM to generate the endpoint
    if prompt:
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "openfda.pkl")
        if os.path.exists(schema_path):
            with open(schema_path, "rb") as f:
                openfda_schema = pickle.load(f)
        else:
            openfda_schema = None

        system_template = """
        You are a biomedical informatics expert specialized in using the OpenFDA API.

        Based on the user's natural language request, determine the appropriate OpenFDA API endpoint and parameters.

        OPENFDA API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including the base URL "https://api.fda.gov" and any parameters)
        2. "description": A brief description of what the query is doing

        QUERY PARAMETERS:
        - search: Use field:term syntax (e.g., "patient.drug.medicinalproduct:lipitor")
        - sort: Use field:asc or field:desc (e.g., "receivedate:desc")
        - count: Use field.exact for exact phrase counting (e.g., "patient.reaction.reactionmeddrapt.exact")
        - limit: Maximum results (max 1000)
        - skip: Skip results for pagination (max 25000)

        SEARCH SYNTAX:
        - Basic: search=field:term
        - AND: search=field1:term1+AND+field2:term2
        - OR: search=field1:term1+field2:term2
        - Exact: search=field:"exact phrase"

        Return ONLY the JSON object with no additional text.
        """

        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=openfda_schema,
            system_template=system_template,
        )
        if not llm_result["success"]:
            return llm_result
        query_info = llm_result["data"]
        endpoint = query_info.get("full_url", "")
        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Build endpoint from parameters
        if endpoint is None:
            endpoint = "/drug/event.json"

        # Ensure endpoint has proper format
        if endpoint.startswith("/"):
            endpoint = f"{base_url}{endpoint}"
        elif not endpoint.startswith("http"):
            endpoint = f"{base_url}/{endpoint.lstrip('/')}"

    # Merge public parameters with query values already present in a direct
    # endpoint. This ensures none of the advertised options are silently lost.
    split_endpoint = urlsplit(endpoint)
    query_params = dict(parse_qsl(split_endpoint.query, keep_blank_values=True))

    if search_params:
        expressions = []
        for field, value in search_params.items():
            values = value if isinstance(value, list) else [value]
            field_expressions = [f"{field}:{item}" for item in values]
            expressions.append(" OR ".join(field_expressions))
        query_params["search"] = " AND ".join(f"({item})" for item in expressions)

    if sort_params:
        invalid_directions = [value for value in sort_params.values() if value not in {"asc", "desc"}]
        if invalid_directions:
            return {"success": False, "error": "sort directions must be 'asc' or 'desc'"}
        query_params["sort"] = ",".join(f"{field}:{direction}" for field, direction in sort_params.items())

    if count_params:
        query_params["count"] = count_params
    query_params.setdefault("limit", str(max_results))
    if skip_results:
        query_params["skip"] = str(skip_results)

    endpoint = urlunsplit(
        (split_endpoint.scheme, split_endpoint.netloc, split_endpoint.path, urlencode(query_params), "")
    )

    # Make the API request using the REST API helper
    description = "OpenFDA API query"
    if prompt:
        description = f"OpenFDA API query for: {prompt}"

    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    # Format results based on verbose setting
    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        api_result["result"] = _format_query_results(api_result["result"])

    return api_result


def query_gwas_catalog(
    prompt=None,
    endpoint=None,
    max_results=3,
):
    """Query the GWAS Catalog API using natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, required): Natural language query about GWAS data
    endpoint (str, optional): Full API endpoint to query (e.g., "https://www.ebi.ac.uk/gwas/rest/api/studies?diseaseTraitId=EFO_0001360")
    max_results (int): Maximum number of results to return

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_gwas_catalog("Find GWAS studies related to Type 2 diabetes")
    - Direct endpoint: query_gwas_catalog(endpoint="studies", params={"diseaseTraitId": "EFO_0001360"})

    """
    # Base URL for GWAS Catalog API
    base_url = "https://www.ebi.ac.uk/gwas/rest/api"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load GWAS Catalog schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "gwas_catalog.pkl")
        with open(schema_path, "rb") as f:
            gwas_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a genomics expert specialized in using the GWAS Catalog API.

        Based on the user's natural language request, determine the appropriate GWAS Catalog API endpoint and parameters.

        GWAS CATALOG API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "endpoint": The API endpoint to query (e.g., "studies", "associations")
        2. "params": An object containing query parameters specific to the endpoint
        3. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - For disease/trait searches, consider using the "EFO" identifiers when possible
        - Common endpoints include: "studies", "associations", "singleNucleotidePolymorphisms", "efoTraits"
        - For pagination, use "size" and "page" parameters
        - For filtering by p-value, use "pvalueMax" parameter
        - GWAS Catalog uses a HAL-based REST API

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=gwas_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the endpoint and parameters from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("endpoint", "")
        params = query_info.get("params", {})
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        if endpoint is None:
            endpoint = ""  # Use root endpoint
        params = {"size": max_results}
        description = f"Direct query to {endpoint}"

    # Remove leading slash if present
    if endpoint.startswith("/"):
        endpoint = endpoint[1:]

    # Construct the URL
    url = f"{base_url}/{endpoint}"

    # Execute the GWAS Catalog API request using the helper function
    api_result = _query_rest_api(endpoint=url, method="GET", params=params, description=description)

    return api_result


def query_gnomad(
    prompt=None,
    gene_symbol=None,
    verbose=True,
):
    """Query gnomAD for variants in a gene using natural language or direct gene symbol.

    Parameters
    ----------
    prompt (str, required): Natural language query about genetic variants
    gene_symbol (str, optional): Gene symbol (e.g., "BRCA1")
    verbose (bool): Whether to print verbose output

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Direct gene: query_gnomad(gene_symbol="BRCA1")
    - Natural language: query_gnomad(prompt="Find variants in the TP53 gene")

    """
    # Base URL for gnomAD API
    base_url = "https://gnomad.broadinstitute.org/api"

    # Ensure we have either a prompt or a gene_symbol
    if prompt is None and gene_symbol is None:
        return {"error": "Either a prompt or a gene_symbol must be provided"}

    # If using prompt, parse with Claude
    if prompt and not gene_symbol:
        # Load gnomAD schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "gnomad.pkl")
        with open(schema_path, "rb") as f:
            gnomad_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a genomics expert specialized in using the gnomAD GraphQL API.

        Based on the user's natural language request, extract the gene symbol and relevant parameters and create the gnomAD GraphQL query.

        GnomAD GraphQL API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "query": The complete GraphQL query string

        SPECIAL NOTES:
        - The gene_symbol should be the official gene symbol (e.g., "BRCA1" not "breast cancer gene 1")
        - If no reference genome is specified, default to GRCh38
        - If no dataset is specified, default to gnomad_r4
        - Return only a single gene symbol, even if multiple are mentioned
        - Always escape special characters, including quotes, in the query string (eg. \" instead of ")



        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=gnomad_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the gene symbol from Claude's response
        query_info = llm_result["data"]
        query_str = query_info.get("query", "")

        if not query_str:
            return {
                "error": "Failed to extract a valid query from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        description = f"Query gnomAD for variants in {gene_symbol}"
        # replace BRCA1 with gene_symbol
        query_str = gnomad_schema.replace("BRCA1", gene_symbol)

    api_result = _query_rest_api(
        endpoint=base_url,
        method="POST",
        json_data={"query": query_str},
        headers={"Content-Type": "application/json"},
        description=description,
    )

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])

    return api_result


def blast_sequence(sequence: str, database: str, program: str) -> dict[str, str | float] | str:
    """Identifies a DNA sequence using NCBI BLAST with improved error handling, timeout management, and debugging.

    Args:
        sequence (str): The sequence to identify. If DNA, use database: core_nt, program: blastn;
                        if protein, use database: nr, program: blastp
        database (str): The BLAST database to search against
        program (str): The BLAST program to use

    Returns:
        dict: A dictionary containing the title, e-value, identity percentage, and coverage percentage of the best alignment

    """
    max_attempts = 1  # One initial attempt plus one retry
    attempts = 0
    max_runtime = 600  # 10 minutes in seconds

    while attempts < max_attempts:
        try:
            attempts += 1
            query_sequence = Seq(sequence)

            # Start timer
            start_time = time.time()

            # Submit BLAST job
            print(f"Submitting BLAST job (attempt {attempts}/{max_attempts})...")
            result_handle = NCBIWWW.qblast(
                program,
                database,
                query_sequence,
                expect=100,
                word_size=7,
                megablast=True,
            )

            # Parse results with timeout check
            blast_records = NCBIXML.parse(result_handle)
            blast_record = None

            # Try to get the first record with timeout check
            while time.time() - start_time < max_runtime:
                try:
                    # Set a short timeout for next operation
                    blast_record = next(blast_records)  # Get first record
                    break  # Successfully got the record
                except StopIteration:
                    # No more records
                    return "No BLAST results found"
                except Exception:
                    # Check if we've exceeded the time limit
                    if time.time() - start_time >= max_runtime:
                        if attempts < max_attempts:
                            print("BLAST job timeout exceeded. Resubmitting...")
                            break  # Break to retry
                        else:
                            return "BLAST search failed after maximum attempts due to timeout"
                    # Brief pause before trying again
                    time.sleep(1)

            # Check if we timed out during record retrieval
            if blast_record is None:
                if attempts < max_attempts:
                    continue  # Retry
                else:
                    return "BLAST search failed after maximum attempts due to timeout"

            # Debug information
            print(f"Number of alignments found: {len(blast_record.alignments)}")

            if blast_record.alignments:
                for alignment in blast_record.alignments:
                    print("\nAlignment:")
                    print(f"hit_id: {alignment.hit_id}")
                    print(f"hit_def: {alignment.hit_def}")
                    print(f"accession: {alignment.accession}")
                    for hsp in alignment.hsps:
                        print(f"E-value: {hsp.expect}")
                        print(f"Score: {hsp.score}")
                        print(f"Identities: {hsp.identities}/{hsp.align_length}")

                        return {
                            "hit_id": alignment.hit_id,
                            "hit_def": alignment.hit_def,
                            "accession": alignment.accession,
                            "e_value": hsp.expect,
                            "identity": (hsp.identities / float(hsp.align_length)) * 100,
                            "coverage": len(hsp.query) / len(sequence) * 100,
                        }
            else:
                return "No alignments found - sequence might be too short or low complexity"

        except Exception as e:
            if attempts < max_attempts:
                print(f"Error during BLAST search: {str(e)}. Retrying...")
                time.sleep(2)  # Wait briefly before retrying
            else:
                return f"Error during BLAST search after maximum attempts: {str(e)}"

    return "BLAST search failed after maximum attempts"


def query_reactome(
    prompt=None,
    endpoint=None,
    download=False,
    output_dir=None,
    verbose=True,
):
    """Query the Reactome database using natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, optional): Natural language query about biological pathways
    endpoint (str, optional): Direct API endpoint or full URL
    download (bool): Whether to download pathway diagrams
    output_dir (str, optional): Directory to save downloaded files
    verbose (bool): Whether to return detailed results

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_reactome("Find pathways related to DNA repair")
    - Direct endpoint: query_reactome(endpoint="data/pathways/R-HSA-73894")

    """
    # Base URLs for Reactome APIs
    content_base_url = "https://reactome.org/ContentService"
    analysis_base_url = "https://reactome.org/AnalysisService"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"success": False, "error": "Either a prompt or an endpoint must be provided"}

    # Create output directory if downloading and directory doesn't exist
    if download and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # If using prompt, parse with Claude
    if prompt:
        # Load Reactome schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "reactome.pkl")
        with open(schema_path, "rb") as f:
            reactome_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a bioinformatics expert specialized in using the Reactome API.

        Based on the user's natural language request, determine the appropriate Reactome API endpoint and parameters.

        REACTOME API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "endpoint": The API endpoint to query (e.g., "data/pathways/PATHWAY_ID", "data/query/GENE_SYMBOL")
        2. "base": Which base URL to use ("content" for ContentService or "analysis" for AnalysisService)
        3. "params": An object containing query parameters specific to the endpoint
        4. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - Reactome has two primary APIs: ContentService (for retrieving specific pathway data) and AnalysisService (for analyzing gene lists)
        - For pathway queries, use "data/pathways/PATHWAY_ID" with the pathway stable identifier (e.g., R-HSA-73894)
        - For gene queries, use "data/query/GENE" with official gene symbol (e.g., "BRCA1")
        - For pathway diagrams, include "download: true" in your response if the query is for pathway visualization
        - Common human pathway IDs start with "R-HSA-"

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=reactome_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the endpoint and parameters from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("endpoint", "")
        base = query_info.get("base", "content")  # Default to ContentService
        params = query_info.get("params", {})
        description = query_info.get("description", "")
        should_download = query_info.get("download", download)  # Override download if specified

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Process provided endpoint
        if endpoint.startswith("http"):
            # Full URL already provided
            if "ContentService" in endpoint:
                base = "content"
            elif "AnalysisService" in endpoint:
                base = "analysis"
            else:
                base = "content"  # Default
        else:
            # Just endpoint provided, assume ContentService by default
            base = "content"

        params = {}
        description = f"Direct query to Reactome {base} API: {endpoint}"
        should_download = download

    # Select base URL based on API type
    base_url = content_base_url if base == "content" else analysis_base_url

    # Remove leading slash if present
    if endpoint.startswith("/"):
        endpoint = endpoint[1:]

    # --- ✅ FIX: Handle old 'data/query/GENE' endpoints to avoid 404 ---
    if endpoint.startswith("http"):
        url = endpoint
    else:
        if endpoint.startswith("data/query/"):
            query_text = endpoint.replace("data/query/", "").strip()
            url = f"{content_base_url}/search/query"
            params = {"query": query_text, "species": "Homo sapiens"}
            description = f"Redirected Reactome search for '{query_text}'"
        else:
            url = f"{base_url}/{endpoint}"
    # --- ✅ END FIX ---

    # Execute the Reactome API request using the helper function
    api_result = _query_rest_api(endpoint=url, method="GET", params=params, description=description)

    # Handle downloading pathway diagrams if requested
    if should_download and api_result.get("success") and "result" in api_result:
        result = api_result["result"]
        pathway_id = None

        # Try to extract pathway ID from result
        if isinstance(result, dict):
            pathway_id = result.get("stId") or result.get("dbId")

        # If we have a pathway ID and output directory, download diagram
        if pathway_id and output_dir:
            safe_pathway_id = re.sub(r"[^A-Za-z0-9._-]", "_", str(pathway_id))
            diagram_url = f"{content_base_url}/data/pathway/{safe_pathway_id}/diagram"
            try:
                diagram_response = requests.get(diagram_url, timeout=60, stream=True)
                diagram_response.raise_for_status()

                max_diagram_bytes = 20 * 1024 * 1024
                if int(diagram_response.headers.get("Content-Length", 0)) > max_diagram_bytes:
                    raise ValueError("Reactome diagram exceeds the 20 MiB download limit")

                # Save diagram file
                diagram_path = os.path.join(output_dir, f"{safe_pathway_id}_diagram.png")
                downloaded_bytes = 0
                with open(diagram_path, "wb") as f:
                    for chunk in diagram_response.iter_content(chunk_size=64 * 1024):
                        downloaded_bytes += len(chunk)
                        if downloaded_bytes > max_diagram_bytes:
                            f.close()
                            os.unlink(diagram_path)
                            raise ValueError("Reactome diagram exceeds the 20 MiB download limit")
                        f.write(chunk)

                api_result["diagram_path"] = diagram_path
            except Exception as e:
                api_result["diagram_error"] = f"Failed to download diagram: {str(e)}"

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        api_result["result"] = _format_query_results(api_result["result"])

    return api_result


def query_regulomedb(
    prompt=None,
    endpoint=None,
    verbose=False,
):
    """Query the RegulomeDB database using natural language or direct variant/coordinate specification.

    Parameters
    ----------
    prompt (str, required): Natural language query about regulatory elements
    endpoint (str, optional): The full endpoint to query (e.g., "https://regulomedb.org/regulome-search/?regions=chr11:5246919-5246919&genome=GRCh38")
    verbose (bool): Whether to return detailed results

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_regulomedb("Find regulatory elements for rs35675666")
    - Direct variant: query_regulomedb(variant="rs35675666")
    - Coordinates: query_regulomedb(coordinates="chr11:5246919-5246919")

    """
    # Base URL for RegulomeDB API

    # Ensure we have either a prompt, variant, or coordinates
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt, variant ID, or genomic coordinates must be provided"}

    # If using prompt, parse with Claude
    if prompt and not endpoint:
        # Create system prompt template
        system_template = """
        You are a genomics expert specialized in using the RegulomeDB API.

        Based on the user's natural language request, extract the variant ID or genomic coordinates they want to query.

        Your response should be a JSON object with ONLY ONE of the following fields:
        1. "endpoint": The API endpoint to query (e.g., "https://regulomedb.org/regulome-search/?regions=chr11:5246919-5246919&genome=GRCh38")


        SPECIAL NOTES:
        - RegulomeDB only works with human genome data
        - Variant IDs should be rsIDs from dbSNP when possible. The endpoint should be in the format https://regulomedb.org/regulome-search/?regions=rsID&genome=GRCh38
        - Thumbnails for chip and chromatin should be in the format https://regulomedb.org/regulome-search?regions=chr11:5246919-5246919&genome=GRCh38/thumbnail=chip
        - Coordinates should be in GRCh37/hg19 format
        - For single base queries, use the same position for start and end (e.g., "chr11:5246919-5246919")
        - Chromosome should be specified with "chr" prefix (e.g., "chr11" not just "11")

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=None,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the variant or coordinates from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("endpoint", "")

        if not endpoint:
            return {
                "error": "Failed to extract a valid variant ID or coordinates from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        pass

    # Construct the request URL
    endpoint = endpoint

    # Execute the RegulomeDB API request using the helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", headers={"Accept": "application/json"})

    # Format the results if not verbose and successful
    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        api_result["result"] = _format_query_results(api_result["result"])

    return api_result


def query_pride(
    prompt=None,
    endpoint=None,
    max_results=3,
):
    """Query the PRIDE (PRoteomics IDEntifications) database using natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, required): Natural language query about proteomics data
    endpoint (str, optional): The full endpoint to query (e.g., "https://www.ebi.ac.uk/pride/ws/archive/v2/projects?keyword=breast%20cancer")
    max_results (int): Maximum number of results to return

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_pride("Find proteomics data related to breast cancer")
    - Direct endpoint: query_pride(endpoint="projects", params={"keyword": "breast cancer"})

    """
    # Base URL for PRIDE API
    base_url = "https://www.ebi.ac.uk/pride/ws/archive/v2"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load PRIDE schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "pride.pkl")
        with open(schema_path, "rb") as f:
            pride_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a proteomics expert specialized in using the PRIDE API.

        Based on the user's natural language request, determine the appropriate PRIDE API endpoint and parameters.

        PRIDE API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "endpoint": The full url endpoint to query
        2. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - PRIDE is a repository for proteomics data stored at EBI
        - Common endpoints include: "projects", "assays", "files", "proteins", "peptideevidences"
        - For searching projects, you can use parameters like "keyword", "species", "tissue", "disease"
        - For pagination, use "page" and "pageSize" parameters
        - Most results include PagingObject and FieldsObject structures

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=pride_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the endpoint and parameters from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("endpoint", "")
        params = query_info.get("params", {})
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Process provided endpoint
        params = {"pageSize": max_results, "page": 0}
        description = f"Direct query to PRIDE {endpoint}"

    # Remove leading slash if present
    if endpoint.startswith("/"):
        endpoint = f"{base_url}{endpoint}"
    elif not endpoint.startswith("http"):
        endpoint = f"{base_url}/{endpoint.lstrip('/')}"
    description = "Direct query to provided endpoint"

    # Execute the PRIDE API request using the helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", params=params, description=description)

    return api_result


def query_gtopdb(
    prompt=None,
    endpoint=None,
    verbose=True,
):
    """Query the Guide to PHARMACOLOGY database (GtoPdb) using natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, required): Natural language query about drug targets, ligands, and interactions
    endpoint (str, optional): Full API endpoint to query (e.g., "https://www.guidetopharmacology.org/services/targets?type=GPCR&name=beta-2")
    verbose (bool): Whether to return detailed results

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_gtopdb("Find ligands that target the beta-2 adrenergic receptor")
    - Direct endpoint: query_gtopdb(endpoint="targets", params={"type": "GPCR", "name": "beta-2"})

    """
    # Base URL for GtoPdb API
    base_url = "https://www.guidetopharmacology.org/services"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load GtoPdb schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "gtopdb.pkl")
        with open(schema_path, "rb") as f:
            gtopdb_schema = pickle.load(f)

        # Create system prompt template
        system_template = r"""
        You are a pharmacology expert specialized in using the Guide to PHARMACOLOGY API.

        Based on the user's natural language request, determine the appropriate GtoPdb API endpoint and parameters.

        GTOPDB API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "endpoint": The full API endpoint to query
        2. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - Main endpoints include: "targets", "ligands", "interactions", "diseases", "refs"
        - Target types include: "GPCR", "NHR", "LGIC", "VGIC", "OtherIC", "Enzyme", "CatalyticReceptor", "Transporter", "OtherProtein"
        - Ligand types include: "Synthetic organic", "Metabolite", "Natural product", "Endogenous peptide", "Peptide", "Antibody", "Inorganic", "Approved", "Withdrawn", "Labelled", "INN"
        - Interaction types include: "Activator", "Agonist", "Allosteric modulator", "Antagonist", "Antibody", "Channel blocker", "Gating inhibitor", "Inhibitor", "Subunit-specific"
        - For specific target/ligand details, use formats like "targets/\{targetId\}" or "ligands/\{ligandId\}"
        - For subresources, use formats like "targets/\{targetId\}/interactions" or "ligands/\{ligandId\}/structure"

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=gtopdb_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the endpoint and parameters from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("endpoint", "")
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Process provided endpoint
        description = f"Direct query to GtoPdb {endpoint}"

    # Remove leading slash if present
    if endpoint.startswith("/"):
        endpoint = f"{base_url}{endpoint}"
    elif not endpoint.startswith("http"):
        endpoint = f"{base_url}/{endpoint.lstrip('/')}"
    description = "Direct query to provided endpoint"

    # Execute the GtoPdb API request using the helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    # Format the results if not verbose and successful
    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        api_result["result"] = _format_query_results(api_result["result"])

    return api_result


def region_to_ccre_screen(coord_chrom: str, coord_start: int, coord_end: int, assembly: str = "GRCh38") -> str:
    """Given starting and ending coordinates, this function retrieves information of intersecting cCREs.

    Args:
        assembly (str): Assembly of the genome, formatted like 'GRCh38'. Default is 'GRCh38'.
        coord_chrom (str): Chromosome of the gene, formatted like 'chr12'.
        coord_start (int): Starting chromosome coordinate.
        coord_end (int): Ending chromosome coordinate.

    Returns:
        str: A detailed string explaining the steps and the intersecting cCRE data or any error encountered.

    """
    steps = []
    try:
        steps.append(
            f"Starting cCRE data retrieval for coordinates: {coord_chrom}:{coord_start}-{coord_end} (Assembly: {assembly})."
        )

        # Build the URL and request payload
        url = "https://screen-beta-api.wenglab.org/dataws/cre_table"
        data = {
            "assembly": assembly,
            "coord_chrom": coord_chrom,
            "coord_start": coord_start,
            "coord_end": coord_end,
        }

        steps.append("Sending POST request to API with the following data:")
        steps.append(str(data))

        # Make the request
        response = requests.post(url, json=data, timeout=30)

        # Check if the response is successful
        if not response.ok:
            raise Exception(f"Request failed with status code {response.status_code}. Response: {response.text}")

        steps.append("Request executed successfully. Parsing the response...")

        # Parse the JSON response
        response_json = response.json()
        if "errors" in response_json:
            raise Exception(f"API error: {response_json['errors']}")

        # Function to reduce and filter response data
        def reduce_tokens(res_json):
            # Remove unnecessary fields and round floats
            res = sorted(res_json["cres"], key=lambda x: x["dnase_zscore"], reverse=True)
            filtered_res = []

            for item in res:
                new_item = {
                    "chrom": item["chrom"],
                    "start": item["start"],
                    "len": item["len"],
                    "pct": item["pct"],
                    "ctcf_zscore": round(item["ctcf_zscore"], 2),
                    "dnase_zscore": round(item["dnase_zscore"], 2),
                    "enhancer_zscore": round(item["enhancer_zscore"], 2),
                    "promoter_zscore": round(item["promoter_zscore"], 2),
                    "accession": item["info"]["accession"],
                    "isproximal": item["info"]["isproximal"],
                    "concordance": item["info"]["concordant"],
                    "ctcfmax": round(item["info"]["ctcfmax"], 2),
                    "k4me3max": round(item["info"]["k4me3max"], 2),
                    "k27acmax": round(item["info"]["k27acmax"], 2),
                }
                filtered_res.append(new_item)
            return filtered_res

        # Process the response data
        filtered_data = reduce_tokens(response_json)

        if not filtered_data:
            steps.append(f"No intersecting cCREs found for coordinates: {coord_chrom}:{coord_start}-{coord_end}.")
            return "\n".join(steps + ["No cCRE data available for this genomic region."])

        # Format the result into a readable string
        ccre_data_string = f"Intersecting cCREs for {coord_chrom}:{coord_start}-{coord_end} (Assembly: {assembly}):\n"
        for i, ccre in enumerate(filtered_data, 1):
            ccre_data_string += (
                f"cCRE {i}:\n"
                f"  Chromosome: {ccre['chrom']}\n"
                f"  Start: {ccre['start']}\n"
                f"  Length: {ccre['len']}\n"
                f"  PCT: {ccre['pct']}\n"
                f"  CTCF Z-score: {ccre['ctcf_zscore']}\n"
                f"  DNase Z-score: {ccre['dnase_zscore']}\n"
                f"  Enhancer Z-score: {ccre['enhancer_zscore']}\n"
                f"  Promoter Z-score: {ccre['promoter_zscore']}\n"
                f"  Accession: {ccre['accession']}\n"
                f"  Is Proximal: {ccre['isproximal']}\n"
                f"  Concordance: {ccre['concordance']}\n"
                f"  CTCFmax: {ccre['ctcfmax']}\n"
                f"  K4me3max: {ccre['k4me3max']}\n"
                f"  K27acmax: {ccre['k27acmax']}\n\n"
            )

        steps.append(f"cCRE data successfully retrieved and formatted for {coord_chrom}:{coord_start}-{coord_end}.")
        return "\n".join(steps + [ccre_data_string])

    except Exception as e:
        steps.append(f"Exception encountered: {str(e)}")
        return "\n".join(steps + [f"Error: {str(e)}"])


def get_genes_near_ccre(accession: str, assembly: str, chromosome: str, k: int = 10) -> str:
    """Given a cCRE (Candidate cis-Regulatory Element), this function returns a string containing the
    steps it performs and the k nearest genes sorted by distance.

    Parameters
    ----------
    - accession (str): ENCODE Accession ID of query cCRE, e.g., EH38E1516980.
    - assembly (str): Assembly of the gene, e.g., 'GRCh38'.
    - chromosome (str): Chromosome of the gene, e.g., 'chr12'.
    - k (int): Number of nearby genes to return, sorted by distance. Default is 10.

    Returns
    -------
    - str: Steps performed and the result.

    """
    steps_log = (
        f"Starting process with accession: {accession}, assembly: {assembly}, chromosome: {chromosome}, k: {k}\n"
    )

    url = "https://screen-beta-api.wenglab.org/dataws/re_detail/nearbyGenomic"
    data = {"accession": accession, "assembly": assembly, "coord_chrom": chromosome}

    steps_log += "Sending POST request to API with given data.\n"
    response = requests.post(url, json=data, timeout=30)

    if not response.ok:
        steps_log += f"API request failed with response: {response.text}\n"
        return steps_log

    response_json = response.json()

    if "errors" in response_json:
        steps_log += f"API returned errors: {response_json['errors']}\n"
        return steps_log

    nearby_genes = response_json.get(accession, {}).get("nearby_genes", [])
    if not nearby_genes:
        steps_log += "No nearby genes found for the given accession.\n"
        return steps_log

    steps_log += "Successfully retrieved nearby genes. Sorting them by distance.\n"
    sorted_genes = sorted(nearby_genes, key=lambda x: x["distance"])[:k]

    steps_log += f"Returning the top {k} nearest genes.\n"
    steps_log += "Result:\n"

    for gene in sorted_genes:
        gene_name = gene.get("name", "Unknown")
        distance = gene.get("distance", "N/A")
        ensembl_id = gene.get("ensemblid_ver", "N/A")
        start = gene.get("start", "N/A")
        stop = gene.get("stop", "N/A")
        chrom = gene.get("chrom", "N/A")
        steps_log += f"Gene: {gene_name}, Distance: {distance}, Ensembl ID: {ensembl_id}, Chromosome: {chrom}, Start: {start}, Stop: {stop}\n"

    return steps_log


def query_remap(
    prompt=None,
    endpoint=None,
    verbose=True,
):
    """Query the ReMap database for regulatory elements and transcription factor binding sites.

    Parameters
    ----------
    prompt (str, required): Natural language query about transcription factors and binding sites
    endpoint (str, optional): Full API endpoint to query (e.g., "https://remap.univ-amu.fr/api/v1/catalogue/tf?tf=CTCF")
    verbose (bool): Whether to return detailed results

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_remap("Find CTCF binding sites in chromosome 1")
    - Direct endpoint: query_remap(endpoint="catalogue/tf", params={"tf": "CTCF"})

    """
    # Base URL for ReMap API
    base_url = "https://remap.univ-amu.fr/api/v1"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load ReMap schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "remap.pkl")
        with open(schema_path, "rb") as f:
            remap_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a genomics expert specialized in using the ReMap database API.

        Based on the user's natural language request, determine the appropriate ReMap API endpoint and parameters.

        REMAP API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "endpoint": The full url endpoint to query
        2. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - ReMap is a database of regulatory regions and transcription factor binding sites based on ChIP-seq experiments
        - Common endpoints include: "catalogue/tf" (transcription factors), "catalogue/biotype" (biotypes), "browse/peaks" (binding sites)
        - For searching binding sites, you can filter by transcription factor (tf), cell line, biotype, chromosome, etc.
        - Genomic coordinates should be specified with "chr", "start", and "end" parameters
        - For limiting results, use "limit" parameter (default is 100)

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=remap_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the endpoint and parameters from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("endpoint", "")
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Process provided endpoint
        description = f"Direct query to ReMap {endpoint}"

    # Remove leading slash if present
    if endpoint.startswith("/"):
        endpoint = f"{base_url}{endpoint}"
    elif not endpoint.startswith("http"):
        endpoint = f"{base_url}/{endpoint.lstrip('/')}"
    description = "Direct query to provided endpoint"

    # Execute the ReMap API request using the helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    # Format the results if not verbose and successful
    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        api_result["result"] = _format_query_results(api_result["result"])

    return api_result


def query_mpd(
    prompt=None,
    endpoint=None,
    verbose=True,
):
    """Query the Mouse Phenome Database (MPD) for mouse strain phenotype data.

    Parameters
    ----------
    prompt (str, required): Natural language query about mouse phenotypes, strains, or measurements
    endpoint (str, optional): Full API endpoint to query (e.g., "https://phenomedoc.jax.org/MPD_API/strains")
    verbose (bool): Whether to return detailed results

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_mpd("Find phenotype data for C57BL/6J mice related to blood glucose")
    - Direct endpoint: query_mpd(endpoint="strains/C57BL/6J/measures")

    """
    # Base URL for MPD API
    base_url = "https://phenome.jax.org"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load MPD schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "mpd.pkl")
        with open(schema_path, "rb") as f:
            mpd_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a mouse genetics expert specialized in using the Mouse Phenome Database (MPD) API.

        Based on the user's natural language request, determine the appropriate MPD API endpoint and parameters.

        MPD API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "endpoint": The full url endpoint to query (e.g. https://phenome.jax.org/api/strains)
        2. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - The MPD contains phenotype data for diverse strains of laboratory mice
        - Common endpoints include: "strains" (mouse strains), "measures" (phenotypic measurements), "genes" (gene info)
        - Use the url to construct the endpoint, not the endpoint name
        - Common mouse strains include: "C57BL/6J", "DBA/2J", "BALB/cJ", "A/J", "129S1/SvImJ"
        - Common phenotypic domains include: "behavior", "blood_chemistry", "body_weight", "cardiovascular", "growth", "metabolism"

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=mpd_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the endpoint and parameters from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("endpoint", "")
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Process provided endpoint
        description = f"Direct query to MPD {endpoint}"

    # Remove leading slash if present
    if endpoint.startswith("/"):
        endpoint = f"{base_url}{endpoint}"
    elif not endpoint.startswith("http"):
        endpoint = f"{base_url}/{endpoint.lstrip('/')}"
    description = "Direct query to provided endpoint"

    # Execute the MPD API request using the helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    # Format the results if not verbose and successful
    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        api_result["result"] = _format_query_results(api_result["result"])

    return api_result


def query_emdb(
    prompt=None,
    endpoint=None,
    verbose=True,
):
    """Query the Electron Microscopy Data Bank (EMDB) for 3D macromolecular structures.

    Parameters
    ----------
    prompt (str, required): Natural language query about EM structures and associated data
    endpoint (str, optional): Full API endpoint to query (e.g., "https://www.ebi.ac.uk/emdb/api/search")
    verbose (bool): Whether to return detailed results

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_emdb("Find cryo-EM structures of ribosomes at resolution better than 3Å")
    - Direct endpoint: query_emdb(endpoint="entry/EMD-10000")

    """
    # Base URL for EMDB API
    base_url = "https://www.ebi.ac.uk/emdb/api"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load EMDB schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "emdb.pkl")
        with open(schema_path, "rb") as f:
            emdb_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a structural biology expert specialized in using the Electron Microscopy Data Bank (EMDB) API.

        Based on the user's natural language request, determine the appropriate EMDB API endpoint and parameters.

        EMDB API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "endpoint": The API endpoint to query (e.g., "search", "entry/EMD-XXXXX")
        2. "params": An object containing query parameters specific to the endpoint
        3. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - EMDB contains 3D macromolecular structures determined by electron microscopy
        - Common endpoints include: "search" (search for entries), "entry/EMD-XXXXX" (specific entry details)
        - For searching, you can filter by resolution, specimen, authors, release date, etc.
        - Resolution filters should be specified with "resolution_low" and "resolution_high" parameters
        - For specific entry retrieval, use the format "entry/EMD-XXXXX" where XXXXX is the EMDB ID
        - Common specimen types include: "ribosome", "virus", "membrane protein", "filament"

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=emdb_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the endpoint and parameters from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("endpoint", "")
        params = query_info.get("params", {})
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Process provided endpoint
        params = {}
        description = f"Direct query to EMDB {endpoint}"

    # Remove leading slash if present
    if endpoint.startswith("/"):
        endpoint = f"{base_url}{endpoint}"
    elif not endpoint.startswith("http"):
        endpoint = f"{base_url}/{endpoint.lstrip('/')}"
    description = "Direct query to provided endpoint"

    # Execute the EMDB API request using the helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", params=params, description=description)

    # Format the results if not verbose and successful
    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        api_result["result"] = _format_query_results(api_result["result"])

    return api_result


def query_synapse(
    prompt: str | None = None,
    query_term: str | list[str] | None = None,
    return_fields: list[str] | None = None,
    max_results: int = 20,
    query_type: str = "dataset",
    verbose: bool = True,
):
    """Query Synapse REST API for biomedical datasets and files.

    Synapse is a platform for sharing and analyzing biomedical data, particularly
    genomics and clinical research datasets. Supports optional authentication via
    SYNAPSE_AUTH_TOKEN environment variable for access to private datasets.

    Parameters
    ----------
    prompt : str, optional
        Natural language query about biomedical data (e.g., "Find drug screening datasets")
    query_term : str or list of str, optional
        Specific search terms for Synapse search. When multiple terms are provided
        as a list, they are combined with AND logic (more terms = more restrictive). Start with 1-2 most relevant search terms.
    return_fields : list of str, optional
        Fields to return in results. Default: ["name", "node_type", "description"]
    max_results : int, default 20
        Maximum number of results to return. Default 20 is optimal for most searches.
        Use up to 50 if extensive results are desired for comprehensive analysis.
    query_type : str, default "dataset"
        Type of entity to search for ("dataset", "file", "folder")
    verbose : bool, default True
        Whether to return full API response or formatted results

    Returns
    -------
    dict
        Dictionary containing query information and Synapse API results

    Notes
    -----
    Authentication is optional but recommended for access to private datasets.
    Set SYNAPSE_AUTH_TOKEN environment variable with your Synapse personal access token
    to enable authenticated requests.

    Examples
    --------
    # Natural language
    query_synapse(prompt="Find drug screening datasets")

    # Direct search (AND logic - finds datasets with both "cancer" AND "genomics")
    query_synapse(query_term=["cancer", "genomics"], max_results=10)

    # Extensive search
    query_synapse(query_term="alzheimer", max_results=50)

    """
    base_url = "https://repo-prod.prod.sagebase.org"

    # Default return fields
    if return_fields is None:
        return_fields = ["name", "node_type", "description"]

    # Check for optional authentication
    headers = {"Content-Type": "application/json"}
    synapse_token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    if synapse_token:
        headers["Authorization"] = f"Bearer {synapse_token}"

    # If natural language prompt provided, convert to search terms
    if prompt and not query_term:
        system_template = (
            "You extract search terms from natural language queries for biomedical data search.\n"
            "Return ONLY a JSON object with this structure, where query_term combines search terms using AND for each entry:\n"
            '{"query_term": ["term1", "term2"], "query_type": "dataset", "max_results": 20}.\n'
            "query_type should be 'dataset' for datasets, 'file' for data files, or 'folder' for collections.\n"
            "max_results should be 20 for typical searches, or up to 50 if extensive/comprehensive results are desired.\n"
            "Use 1-2 most relevant search terms (these are combined with AND; more terms = more restrictive). Only include main term (disease, gene, etc.) of the search query and do not include any other terms/adjectives/modifiers. Do not include explanations.\n"
            "Try to remove hyphens and other special characters from the search terms. For example, use RNAseq instead of RNA-seq."
        )

        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=None,
            system_template=system_template,
        )

        if llm_result.get("success"):
            mapping = llm_result["data"] or {}
            query_term = mapping.get("query_term", [])
            query_type = mapping.get("query_type", query_type)
            max_results = mapping.get("max_results", max_results)

    # Build search request
    search_url = f"{base_url}/repo/v1/search"

    # Ensure query_term is a list
    if isinstance(query_term, str):
        query_term = [query_term]
    elif query_term is None:
        query_term = [""]

    # Build search payload
    search_payload = {
        "queryTerm": query_term,
        "returnFields": return_fields,
        "start": 0,
        "size": max_results,
        "booleanQuery": [{"key": "node_type", "value": query_type}],
    }

    description = f"Synapse search for terms: {query_term} (query type: {query_type})"

    # Execute search
    api_result = _query_rest_api(
        endpoint=search_url,
        method="POST",
        json_data=search_payload,
        headers=headers,
        description=description,
    )

    # Augment results with access control information
    if api_result.get("success") and "result" in api_result:
        result_data = api_result["result"]
        if isinstance(result_data, dict) and "hits" in result_data:
            for hit in result_data["hits"]:
                if "id" in hit:
                    # Check access requirements for this entity
                    access_url = f"{base_url}/repo/v1/entity/{hit['id']}/accessRequirement"
                    access_result = _query_rest_api(
                        endpoint=access_url,
                        method="GET",
                        headers=headers,
                        description=f"Check access requirements for {hit['id']}",
                    )

                    # Add access_restricted property based on access requirements
                    if access_result.get("success") and "result" in access_result:
                        access_data = access_result["result"]
                        total_requirements = access_data.get("totalNumberOfResults", 0)
                        hit["access_restricted"] = total_requirements > 0
                    else:
                        # If we can't check access, assume it might be restricted
                        hit["access_restricted"] = True

    # Format results if not verbose and successful
    if not verbose and api_result.get("success") and "result" in api_result:
        api_result["result"] = _format_query_results(api_result["result"])

    return api_result


def query_pubchem(
    prompt=None,
    endpoint=None,
    max_results=5,
    verbose=True,
):
    """Query the PubChem PUG-REST API using natural language or a direct endpoint.
    Parameters
    ----------
    prompt (str, optional): Natural language query about chemical compounds
    endpoint (str, optional): Direct PubChem API endpoint to query
    max_results (int): Maximum number of results to return
    verbose (bool): Whether to return detailed results
    Returns
    -------
    dict: Dictionary containing the query results or error information
    Examples
    --------
    - Natural language: query_pubchem("Find molecular weight of aspirin")
    - Direct endpoint: query_pubchem(endpoint="compound/cid/2244/property/MolecularWeight/txt")
    """
    # Base URL for PubChem API
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"success": False, "error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load PubChem schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "pubchem.pkl")
        with open(schema_path, "rb") as f:
            pubchem_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a chemistry expert specialized in using the PubChem PUG-REST API.
        Based on the user's natural language request, determine the appropriate PubChem API endpoint and parameters.
        PUBCHEM API SCHEMA:
        {schema}
        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including base URL and parameters)
        2. "description": A brief description of what the query is doing
        SPECIAL NOTES:
        - Base URL is "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        - Common operations: property, synonyms, record, xrefs
        - For properties, use CSV format for multiple properties, TXT for single property
        - For images, use PNG format with optional image_size parameter
        - Rate limit: maximum 5 requests per second
        - Use compound/name/ for chemical names, compound/cid/ for PubChem IDs
        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=pubchem_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the full URL from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Use provided endpoint directly
        if endpoint is not None:
            if endpoint.startswith("/"):
                endpoint = f"{base_url}{endpoint}"
            elif not endpoint.startswith("http"):
                endpoint = f"{base_url}/{endpoint.lstrip('/')}"
        description = "Direct query to provided endpoint"

    # Rate limiting: allow user to configure or disable; only sleep if last request was too recent
    if not hasattr(query_pubchem, "_last_request_time"):
        query_pubchem._last_request_time = 0
    min_interval = 1.0 / 5  # 5 requests per second by default
    now = time.time()
    elapsed = now - query_pubchem._last_request_time
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    query_pubchem._last_request_time = time.time()

    # Use the common REST API helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        api_result["result"] = _format_query_results(api_result["result"])

    return api_result


def query_chembl(
    prompt=None,
    endpoint=None,
    chembl_id=None,
    smiles=None,
    molecule_name=None,
    max_results=20,
    verbose=True,
):
    """Query the ChEMBL REST API using natural language, direct endpoint, or specific identifiers.
    Parameters
    ----------
    prompt (str, optional): Natural language query about bioactivity data
    endpoint (str, optional): Direct ChEMBL API endpoint to query
    chembl_id (str, optional): Specific ChEMBL ID to query (e.g., 'CHEMBL25')
    smiles (str, optional): SMILES string for similarity/substructure search
    molecule_name (str, optional): Molecule name for lookup
    max_results (int): Maximum number of results to return
    verbose (bool): Whether to return detailed results
    Returns
    -------
    dict: Dictionary containing the query results or error information
    Examples
    --------
    - Natural language: query_chembl("Find approved drugs with kinase activity")
    - Direct endpoint: query_chembl(endpoint="molecule?max_phase=4")
    - ChEMBL ID: query_chembl(chembl_id="CHEMBL25")
    - SMILES similarity: query_chembl(smiles="CC(=O)OC1=CC=CC=C1C(=O)O")
    - Molecule name: query_chembl(molecule_name="aspirin")
    """
    # Base URL for ChEMBL API
    base_url = "https://www.ebi.ac.uk/chembl/api/data"

    prompt_chembl_id = None
    if prompt:
        match = re.search(r"\bCHEMBL\d+\b", prompt, flags=re.IGNORECASE)
        if match:
            prompt_chembl_id = match.group(0).upper()

    # Handle specific identifier parameters first (most reliable)
    if chembl_id:
        endpoint = f"{base_url}/molecule/{chembl_id}.json"
        description = f"Direct lookup for ChEMBL ID: {chembl_id} (most reliable method)"
    elif smiles:
        endpoint = f"{base_url}/similarity/{smiles}/80.json"  # Default similarity cutoff
        description = f"Similarity search for SMILES: {smiles} with 80% cutoff"
    elif molecule_name:
        endpoint = f"{base_url}/molecule/search.json?{urlencode({'q': molecule_name, 'limit': max_results})}"
        description = f"Search for molecule with name containing: {molecule_name}"
    elif prompt:
        # Try LLM-based parsing with fallback
        try:
            # Load ChEMBL schema
            schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "chembl.pkl")
            with open(schema_path, "rb") as f:
                chembl_schema = pickle.load(f)

            # Create system prompt template
            system_template = """
            You are a bioactivity data expert specialized in using the ChEMBL REST API.
            Based on the user's natural language request, determine the appropriate ChEMBL API endpoint and parameters.
            CHEMBL API SCHEMA:
            {schema}
            Your response should be a JSON object with the following fields:
            1. "full_url": The complete URL to query (including base URL and parameters)
            2. "description": A brief description of what the query is doing
            SPECIAL NOTES:
            - Base URL is "https://www.ebi.ac.uk/chembl/api/data"

            # IMPORTANT ENDPOINTS:
            - Molecule search: /molecule/search.json?q={search_term} (full-text search)
            - Molecule by ID: /molecule/{chembl_id}.json (direct lookup)
            - Image: /image/{chembl_id}.svg or /molecule/{chembl_id}.svg
            - Substructure: /substructure/{smiles}.json (valid SMILES required)
            - Similarity: /similarity/{smiles}/{cutoff}.json (cutoff 70-90 typical)

            # BIOACTIVITY DATA:
            - Activities: /activity.json?molecule_chembl_id={chembl_id}&limit=20
            - Assays: /assay.json?molecule_chembl_id={chembl_id}&limit=20
            - Use only= parameter to reduce fields: &only=target_chembl_id,standard_type,standard_value

            # DRUG METADATA:
            - Drug info: /drug.json?molecule_chembl_id={chembl_id} (use parent ID)
            - Indications: /drug_indication.json?molecule_chembl_id={chembl_id}
            - Mechanisms: /mechanism.json?molecule_chembl_id={chembl_id}
            - ATC: /atc_class.json?molecule_chembl_id={chembl_id}

            # COMMON FILTERS:
            - max_phase=4 (approved drugs)
            - assay_type=B (binding), F (functional), A (ADMET)
            - standard_type=IC50, Ki, EC50
            - pchembl_value__gte=5 (activity threshold)

            # FORMAT NOTES:
            - Add .json for JSON output (default is XML)
            - Use /search.json for full-text search (not ?search=)
            - Use parent ChEMBL IDs for drug endpoints
            - Use raw SMILES (don't double-encode)
            Return ONLY the JSON object with no additional text.
            """

            # Query LLM to generate the API call
            llm_result = _query_llm_for_api(
                prompt=prompt,
                schema=chembl_schema,
                system_template=system_template,
            )

            if llm_result["success"]:
                # Get the full URL from LLM's response
                query_info = llm_result["data"]
                endpoint = query_info.get("full_url", "")
                description = query_info.get("description", "")

                if endpoint:
                    if prompt_chembl_id and prompt_chembl_id.lower() not in endpoint.lower():
                        endpoint = f"{base_url}/molecule/{prompt_chembl_id}.json"
                        description = f"Direct lookup for ChEMBL ID: {prompt_chembl_id}"
                else:
                    raise Exception("No endpoint generated from LLM")
            else:
                raise Exception(f"LLM failed: {llm_result.get('error', 'Unknown error')}")

        except Exception:
            # Fall back to generic endpoint mapping for common query types
            prompt_lower = prompt.lower()

            # Extract potential molecule names or keywords from the prompt
            words = prompt.split()
            potential_molecule = None

            # Look for common molecule indicators - skip common words and look for longer, more specific terms
            common_words = {
                "find",
                "search",
                "get",
                "show",
                "list",
                "target",
                "targets",
                "binding",
                "for",
                "the",
                "a",
                "an",
                "and",
                "or",
                "with",
                "using",
                "via",
                "through",
                "from",
                "in",
                "on",
                "at",
                "to",
                "of",
                "by",
            }

            for word in words:
                word_lower = word.lower()
                # Skip common words and look for longer, more specific terms that could be molecule names
                if (
                    len(word) > 4
                    and word.isalpha()
                    and word_lower not in common_words
                    and not word_lower.startswith("che")  # Skip words starting with common prefixes
                    and not word_lower.endswith("ing")
                ):  # Skip gerunds
                    potential_molecule = word
                    break

            if prompt_chembl_id:
                endpoint = f"{base_url}/molecule/{prompt_chembl_id}.json"
                description = f"Direct lookup for ChEMBL ID: {prompt_chembl_id}"
            elif "binding" in prompt_lower and "target" in prompt_lower:
                # Try to find binding targets - use molecule if found, otherwise generic
                if potential_molecule:
                    endpoint = f"{base_url}/molecule/search.json?q={potential_molecule}&limit={max_results}"
                    description = f"Search for {potential_molecule} binding targets in ChEMBL database"
                else:
                    endpoint = f"{base_url}/activity.json?standard_type=IC50&limit={max_results}"
                    description = "Search for binding activities with IC50 values"
            elif "molecule" in prompt_lower or "compound" in prompt_lower or "drug" in prompt_lower:
                # Molecule search
                if potential_molecule:
                    endpoint = f"{base_url}/molecule/search.json?q={potential_molecule}&limit={max_results}"
                    description = f"Search for molecule {potential_molecule} in ChEMBL database"
                else:
                    endpoint = f"{base_url}/molecule/search.json?q=molecule&limit={max_results}"
                    description = "Search for molecules in ChEMBL database"
            elif "activity" in prompt_lower or "bioactivity" in prompt_lower:
                # Bioactivity search
                endpoint = f"{base_url}/activity.json?limit={max_results}"
                description = "Search for bioactivity data in ChEMBL database"
            elif "assay" in prompt_lower:
                # Assay search
                endpoint = f"{base_url}/assay.json?limit={max_results}"
                description = "Search for assay data in ChEMBL database"
            elif "target" in prompt_lower:
                # Target search
                endpoint = f"{base_url}/target.json?limit={max_results}"
                description = "Search for target data in ChEMBL database"
            elif "image" in prompt_lower:
                # Image search
                if potential_molecule:
                    endpoint = f"{base_url}/molecule/search.json?q={potential_molecule}&limit={max_results}"
                    description = f"Search for {potential_molecule} images in ChEMBL database"
                else:
                    endpoint = f"{base_url}/molecule/search.json?q=molecule&limit={max_results}"
                    description = "Search for molecule images in ChEMBL database"
            else:
                # Generic search - use first meaningful word or fallback
                if potential_molecule:
                    endpoint = f"{base_url}/molecule/search.json?q={potential_molecule}&limit={max_results}"
                    description = f"Generic search for {potential_molecule} in ChEMBL database"
                else:
                    endpoint = f"{base_url}/molecule/search.json?q=molecule&limit={max_results}"
                    description = f"Generic search in ChEMBL database for: {prompt[:50]}..."
    elif endpoint:
        # Use provided endpoint directly
        if endpoint.startswith("/"):
            endpoint = f"{base_url}{endpoint}"
        elif not endpoint.startswith("http"):
            endpoint = f"{base_url}/{endpoint.lstrip('/')}"
        description = "Direct query to provided endpoint"
    else:
        # No valid parameters provided
        return {
            "success": False,
            "error": "No query parameters provided. Use prompt, endpoint, chembl_id, smiles, or molecule_name.",
        }

    # Do not execute a syntactically valid but semantically empty search.  A
    # generic token ("known", "molecule", etc.) is not evidence for a target.
    if prompt and not (chembl_id or smiles or molecule_name):
        generic_terms = {
            "known", "molecule", "compound", "drug", "list", "find", "search",
            "inhibitor", "inhibitors", "agonist", "antagonist",
        }
        parsed_query = dict(parse_qsl(urlsplit(endpoint).query)).get("q", "").strip().casefold()
        if parsed_query in generic_terms:
            return {
                "success": False,
                "failure_kind": "irrelevant_result",
                "error": "Generated ChEMBL query lost the target-specific terms; retry with a target/activity endpoint.",
                "query_info": {"endpoint": endpoint, "prompt": prompt},
            }

    # Add pagination if not already specified
    if "?" in endpoint:
        if "limit=" not in endpoint:
            endpoint += f"&limit={max_results}"
    else:
        endpoint += f"?limit={max_results}"

    # Use the common REST API helper function
    api_result = _query_rest_api(
        endpoint=endpoint,
        method="GET",
        description=description,
        timeout_seconds=15,
        max_retries=2,
    )

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        api_result["result"] = _format_query_results(api_result["result"])

    return api_result


def query_unichem(
    prompt=None,
    endpoint=None,
    method=None,
    data=None,
    verbose=True,
):
    """Query the UniChem 2.0 REST API using natural language or a direct endpoint.
    Parameters
    ----------
    prompt (str, optional): Natural language query about chemical cross-references
    endpoint (str, optional): Direct UniChem API endpoint to query
    verbose (bool): Whether to return detailed results
    Returns
    -------
    dict: Dictionary containing the query results or error information
    Examples
    --------
    - Natural language: query_unichem("Find cross-references for aspirin")
    - Direct endpoint: query_unichem(endpoint="/compounds")
    - Compound search: query_unichem(endpoint="/compounds", data={"type": "inchikey", "compound": "LMXNVOREDXZICN-WDSOQIARSA-N"})
    - Connectivity search: query_unichem(endpoint="/connectivity", data={"type": "inchi", "compound": "InChI=1S/C7H8N4O2/c1-10-5-4(8-3-9-5)6(12)11(2)7(10)13/h3H,1-2H3,(H,8,9)", "searchComponents": True})
    - Get sources: query_unichem(endpoint="/sources")
    """
    # Base URL for UniChem API (corrected from beta to production)
    base_url = "https://www.ebi.ac.uk/unichem/api/v1"

    def _normalize_body(body):
        """Accept the common ``sourceID=CHEMBL...`` shorthand."""
        if not isinstance(body, dict):
            return body
        normalized = dict(body)
        if normalized.get("type") == "sourceID" and "compound" not in normalized:
            value = normalized.get("sourceID")
            if isinstance(value, str) and not value.isdigit():
                normalized["compound"] = value
                normalized["sourceID"] = 1  # ChEMBL in UniChem
        return normalized

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load UniChem schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "unichem.pkl")
        with open(schema_path, "rb") as f:
            unichem_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a chemical cross-reference expert specialized in using the UniChem 2.0 REST API.
        Based on the user's natural language request, determine the appropriate UniChem API endpoint and parameters.
        UNICHEM API SCHEMA:
        {schema}
        Your response should be a JSON object with the following fields:
        1. "endpoint": The API endpoint to use (e.g., "/compounds", "/sources", "/connectivity")
        2. "method": HTTP method ("GET" or "POST")
        3. "data": POST data if method is POST (null for GET requests)
        4. "description": A brief description of what the query is doing
        SPECIAL NOTES:
        - Base URL is "https://www.ebi.ac.uk/unichem/api/v1"
        - Compound searches use POST method to /compounds endpoint
        - Connectivity searches use POST method to /connectivity endpoint
        - Source information uses GET method to /sources endpoint
        - Valid identifier types: uci, inchi, inchikey, sourceID
        - For sourceID searches, use type=sourceID, sourceID=1 (ChEMBL) and compound=<source ID>
        - For connectivity searches, can include searchComponents boolean parameter
        - Common source IDs: 1=ChEMBL, 2=DrugBank, 5=PubChem, 7=ChEBI
        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=unichem_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the API call details from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("endpoint", "")
        method = query_info.get("method", "GET").upper()
        data = _normalize_body(query_info.get("data", None))
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
        if method == "POST" and endpoint.rstrip("/").endswith(("/compounds", "/connectivity")) and not isinstance(data, dict):
            return {
                "success": False,
                "error": "UniChem POST requests require a JSON object in data; generated data was empty or invalid",
                "query_info": {"endpoint": endpoint, "method": method},
            }

        # Construct full URL
        if endpoint.startswith("/"):
            full_url = f"{base_url}{endpoint}"
        else:
            full_url = f"{base_url}/{endpoint.lstrip('/')}"

    else:
        # Use provided endpoint directly
        if endpoint is None:
            return {"error": "Endpoint cannot be None when prompt is not provided"}

        if endpoint.startswith("/"):
            full_url = f"{base_url}{endpoint}"
        elif not endpoint.startswith("http"):
            full_url = f"{base_url}/{endpoint.lstrip('/')}"
        else:
            full_url = endpoint
        endpoint_path = full_url.split("?", 1)[0].rstrip("/")
        method = (method or ("POST" if endpoint_path.endswith(("/compounds", "/connectivity")) else "GET")).upper()
        if method not in {"GET", "POST"}:
            return {"success": False, "error": f"Unsupported UniChem HTTP method: {method}"}
        if method == "POST" and not isinstance(data, dict):
            return {
                "success": False,
                "error": "UniChem POST requests require data as a JSON object",
                "query_info": {"endpoint": full_url, "method": method},
            }
        data = _normalize_body(data)
        description = "Direct query to provided endpoint"

    # Use the common REST API helper function
    api_result = _query_rest_api(endpoint=full_url, method=method, json_data=data, description=description)

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        api_result["result"] = _format_query_results(api_result["result"])

    return api_result


def query_clinicaltrials(
    prompt=None,
    endpoint=None,
    max_results=10,
    verbose=True,
):
    """Query the ClinicalTrials.gov API v2 using natural language or a direct endpoint.
    Parameters
    ----------
    prompt (str, optional): Natural language query about clinical trials
    endpoint (str, optional): Direct ClinicalTrials.gov API endpoint to query
    max_results (int): Maximum number of results to return
    verbose (bool): Whether to return detailed results
    Returns
    -------
    dict: Dictionary containing the query results or error information
    Examples
    --------
    - Natural language: query_clinicaltrials("Find recruiting cancer trials")
    - Direct endpoint: query_clinicaltrials(endpoint="/studies?query.cond=cancer&filter.overallStatus=RECRUITING")
    """
    # Base URL for ClinicalTrials.gov API
    base_url = "https://clinicaltrials.gov/api/v2"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"success": False, "error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load ClinicalTrials.gov schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "clinicaltrials.pkl")
        with open(schema_path, "rb") as f:
            clinicaltrials_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a clinical research expert specialized in using the ClinicalTrials.gov API v2.
        Based on the user's natural language request, determine the appropriate ClinicalTrials.gov API endpoint and parameters.
        CLINICALTRIALS.GOV API SCHEMA:
        {schema}
        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including base URL and parameters)
        2. "description": A brief description of what the query is doing
        SPECIAL NOTES:
        - Base URL is "https://clinicaltrials.gov/api/v2"
        - Main endpoint is /studies for searching clinical trials
        - Use query.cond for conditions/diseases, query.intr for interventions
        - Use filter.overallStatus for study status (RECRUITING, COMPLETED, etc.)
        - Use filter.phase for study phases (PHASE1, PHASE2, PHASE3, PHASE4)
        - Use filter.studyType for study types (INTERVENTIONAL, OBSERVATIONAL)
        - Use pageSize parameter to limit results (max 1000)
        - For specific studies, use /studies/{{nctId}}

        CORRECT PHASE FILTERING:
        - Use filter.phase=PHASE1, PHASE2, PHASE3, PHASE4 (comma-separated for multiple phases)
        - Do NOT use filter.phase=PHASE3 (single value with equals)
        - Example: filter.phase=PHASE1,PHASE2 for early phase trials
        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=clinicaltrials_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the full URL from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Use provided endpoint directly
        if endpoint is not None:
            if endpoint.startswith("/"):
                endpoint = f"{base_url}{endpoint}"
            elif not endpoint.startswith("http"):
                endpoint = f"{base_url}/{endpoint.lstrip('/')}"
        description = "Direct query to provided endpoint"

    # Add pageSize if not already specified and not a specific study lookup
    if "/studies/" not in endpoint and "pageSize=" not in endpoint:
        separator = "&" if "?" in endpoint else "?"
        endpoint += f"{separator}pageSize={max_results}"

    # Use the common REST API helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    # Handle API parameter errors with fallback for ClinicalTrials.gov
    if not api_result.get("success", False) and "400" in str(api_result.get("error", "")):
        # Try simplified query without problematic filters
        if "filter.phase" in endpoint:
            simplified_endpoint = endpoint.replace("&filter.phase=PHASE3", "").replace("filter.phase=PHASE3&", "")
            if simplified_endpoint != endpoint:
                api_result = _query_rest_api(
                    endpoint=simplified_endpoint, method="GET", description=f"{description} (simplified)"
                )
                if api_result.get("success", False):
                    api_result["note"] = "Query simplified due to API parameter restrictions"

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        api_result["result"] = _format_query_results(api_result["result"])

    return api_result


def query_dailymed(
    prompt=None,
    endpoint=None,
    format="json",
    verbose=True,
):
    """Query the DailyMed RESTful API using natural language or a direct endpoint.
    Parameters
    ----------
    prompt (str, optional): Natural language query about drug labeling information
    endpoint (str, optional): Direct DailyMed API endpoint to query
    format (str): Response format ('json' or 'xml')
    verbose (bool): Whether to return detailed results
    Returns
    -------
    dict: Dictionary containing the query results or error information
    Examples
    --------
    - Natural language: query_dailymed("Find all drug names")
    - Direct endpoint: query_dailymed(endpoint="/drugnames.json")
    - Get specific SPL: query_dailymed(endpoint="/spls/12345678-1234-1234-1234-123456789012.json")
    - Get SPL history: query_dailymed(endpoint="/spls/12345678-1234-1234-1234-123456789012/history.json")
    """
    # Base URL for DailyMed API
    base_url = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"success": False, "error": "Either a prompt or an endpoint must be provided"}

    # Validate format
    if format not in ["json", "xml"]:
        return {"success": False, "error": "format must be either 'json' or 'xml'"}

    # If using prompt, parse with Claude
    if prompt:
        # Load DailyMed schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "dailymed.pkl")
        with open(schema_path, "rb") as f:
            dailymed_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a pharmaceutical labeling expert specialized in using the DailyMed RESTful API.
        Based on the user's natural language request, determine the appropriate DailyMed API endpoint and parameters.
        DAILYMED API SCHEMA:
        {schema}
        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including base URL and format extension)
        2. "description": A brief description of what the query is doing
        SPECIAL NOTES:
        - Base URL is "https://dailymed.nlm.nih.gov/dailymed/services/v2"
        - Available resources: applicationnumbers, drugclasses, drugnames, ndcs, rxcuis, spls, uniis
        - For specific SPL documents, use /spls/{{SETID}} format
        - For SPL-related data, use /spls/{{SETID}}/history, /spls/{{SETID}}/media, /spls/{{SETID}}/ndcs, /spls/{{SETID}}/packaging
        - Always append format extension (.json or .xml)
        - API only supports GET method
        - HTTPS is required (HTTP disabled since 2016)
        - Each resource may have optional query parameters to filter or control output
        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=dailymed_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the full URL from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Use provided endpoint directly
        if endpoint is not None:
            if endpoint.startswith("/"):
                endpoint = f"{base_url}{endpoint}"
            elif not endpoint.startswith("http"):
                endpoint = f"{base_url}/{endpoint.lstrip('/')}"
        description = "Direct query to provided endpoint"

        # Add format extension if not present
        if not endpoint.endswith(f".{format}") and not endpoint.endswith(".json") and not endpoint.endswith(".xml"):
            endpoint += f".{format}"

    # Use the common REST API helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        api_result["result"] = _format_query_results(api_result["result"])

    return api_result


def query_quickgo(
    prompt=None,
    endpoint=None,
    max_results=25,
    verbose=True,
):
    """Query the QuickGO API using natural language or a direct endpoint.
    Parameters
    ----------
    prompt (str, optional): Natural language query about Gene Ontology terms, annotations, or gene products
    endpoint (str, optional): Direct QuickGO API endpoint to query
    max_results (int): Maximum number of results to return (max 100)
    verbose (bool): Whether to return detailed results
    Returns
    -------
    dict: Dictionary containing the query results or error information
    Examples
    --------
    - Natural language: query_quickgo("Find GO terms related to apoptosis")
    - Direct endpoint: query_quickgo(endpoint="/ontology/go/search?query=apoptosis&limit=10")
    - Get specific term: query_quickgo(endpoint="/ontology/go/terms/GO:0006915")
    """
    # Base URL for QuickGO API (corrected from documentation)
    base_url = "https://www.ebi.ac.uk/QuickGO/services"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # Validate max_results
    if max_results > 100:
        import warnings

        warnings.warn(
            f"max_results ({max_results}) exceeds QuickGO API limit (100). Setting max_results to 100.", stacklevel=2
        )
        max_results = 100

    # If using prompt, parse with Claude
    if prompt:
        # Load QuickGO schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "quickgo.pkl")
        with open(schema_path, "rb") as f:
            quickgo_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a Gene Ontology expert specialized in using the QuickGO REST API.
        Based on the user's natural language request, determine the appropriate QuickGO API endpoint and parameters.
        QUICKGO API SCHEMA:
        {schema}
        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including base URL and parameters)
        2. "description": A brief description of what the query is doing
        SPECIAL NOTES:
        - Base URL is "https://www.ebi.ac.uk/QuickGO/services"
        - Main services: /ontology (GO/ECO terms), /annotation (GO annotations), /geneproduct (gene products)
        - For GO term search, use /ontology/go/search with query parameter
        - For specific GO terms, use /ontology/go/terms/{{go_id}}
        - For GO term relationships, use /ontology/go/terms/{{go_id}}/children, /descendants, /ancestors
        - For annotations, use /annotation/search with various filters
        - For gene products, use /geneproduct/search
        - Use limit parameter to control results (max 100)
        - Common organisms: 9606 (human), 10090 (mouse), 7227 (fly)
        - GO aspects: biological_process, molecular_function, cellular_component
        - Evidence codes: IEA, IDA, IPI, IMP, IGI, etc.
        - Qualifiers: enables, involved_in, is_active_in, part_of, etc.
        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=quickgo_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the full URL from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Use provided endpoint directly
        if endpoint is not None:
            if endpoint.startswith("/"):
                endpoint = f"{base_url}{endpoint}"
            elif not endpoint.startswith("http"):
                endpoint = f"{base_url}/{endpoint.lstrip('/')}"
        description = "Direct query to provided endpoint"

    # Add limit parameter if not already specified
    if "limit=" not in endpoint and "/terms/" not in endpoint:
        separator = "&" if "?" in endpoint else "?"
        endpoint += f"{separator}limit={max_results}"

    # Use the common REST API helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        api_result["result"] = _format_query_results(api_result["result"])

    return api_result


def query_encode(
    prompt=None,
    endpoint=None,
    max_results=25,
    verbose=True,
):
    """Query the ENCODE Portal API to help users locate functional genomics data.
    This function is designed to help users find and explore ENCODE data including:
    - Experiments (ChIP-seq, RNA-seq, ATAC-seq, DNase-seq, WGBS, etc.)
    - Files (BAM, BED, bigWig, fastq, etc.)
    - Biosamples (cell lines, tissues, primary cells)
    - Datasets and replicates
    Parameters
    ----------
    prompt (str, required): Natural language query about functional genomics data you want to find
    endpoint (str, optional): Direct ENCODE Portal API endpoint to query
    max_results (int): Maximum number of results to return (use "all" for all results)
    verbose (bool): Whether to return detailed results
    Returns
    -------
    dict: Dictionary containing the query results with data location information
    Examples
    --------
    - Find experiments: query_encode("Find ChIP-seq experiments for CTCF in human K562 cells")
    - Find files: query_encode("Find BAM files from ATAC-seq experiments in mouse brain")
    - Find biosamples: query_encode("Find human primary T cells from blood")
    - Find datasets: query_encode("Find RNA-seq datasets from human liver tissue")
    - Direct endpoint: query_encode(endpoint="/search/?type=Experiment&assay_title=ChIP-seq&format=json")
    """
    # Base URL for ENCODE Portal API
    base_url = "https://www.encodeproject.org"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load ENCODE schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "encode.pkl")
        with open(schema_path, "rb") as f:
            encode_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a functional genomics expert specialized in helping users locate data in the ENCODE Portal.
        Your goal is to help users find the specific functional genomics data they need. Based on the user's request,
        determine the most appropriate ENCODE Portal API endpoint and parameters to locate their data.
        ENCODE PORTAL API SCHEMA:
        {schema}
        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including base URL and parameters)
        2. "description": A clear description of what data the query will help locate
        3. "data_type": The type of data being searched (Experiment, File, Biosample, etc.)
        4. "search_strategy": Brief explanation of the search approach used
        CRITICAL RULES FOR SIMPLE, EFFECTIVE QUERIES:
        1. KEEP QUERIES SIMPLE - use only 1-3 parameters maximum for better results
        2. Start with basic searches and let users refine based on results
        3. Use searchTerm for text-based searches (most reliable for complex terms)
        4. Avoid complex nested property paths when possible
        5. For organism filtering, use simple organism names: "Homo sapiens", "Mus musculus"
        SIMPLE QUERY PATTERNS (PREFERRED):
        - Basic experiment search: /search/?type=Experiment&assay_title=ChIP-seq&format=json
        - Text-based search: /search/?searchTerm=CTCF&format=json
        - File type search: /search/?type=File&file_format=bam&format=json
        - Biosample search: /search/?type=Biosample&format=json
        - Dataset search: /search/?type=Dataset&format=json
        COMMON ASSAY TYPES (choose ONE per query):
        - ChIP-seq, RNA-seq, ATAC-seq, DNase-seq, WGBS, Hi-C, CAGE, ChIA-PET
        COMMON FILE FORMATS:
        - bam, fastq, bigWig, bigBed, bed, narrowPeak, broadPeak
        SIMPLE EXAMPLES:
        - Find ChIP-seq experiments: /search/?type=Experiment&assay_title=ChIP-seq&format=json
        - Find CTCF data: /search/?searchTerm=CTCF&format=json
        - Find BAM files: /search/?type=File&file_format=bam&format=json
        - Find human experiments: /search/?type=Experiment&searchTerm=human&format=json
        - Find mouse brain data: /search/?type=Experiment&searchTerm=mouse%20brain&format=json
        IMPORTANT: Return ONLY a valid JSON object with no additional text, code comments, or explanations.
        The response must be parseable JSON starting with {{ and ending with }}.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=encode_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the full URL from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Use provided endpoint directly
        if endpoint is not None:
            if endpoint.startswith("/"):
                endpoint = f"{base_url}{endpoint}"
            elif not endpoint.startswith("http"):
                endpoint = f"{base_url}/{endpoint.lstrip('/')}"
        description = "Direct query to provided endpoint"

    # Ensure format=json is included for API access
    if "format=json" not in endpoint and "/search/" in endpoint:
        separator = "&" if "?" in endpoint else "?"
        endpoint += f"{separator}format=json"

    # Add limit parameter if not already specified and it's a search endpoint
    if "/search/" in endpoint and "limit=" not in endpoint:
        separator = "&" if "?" in endpoint else "?"
        limit_value = "all" if max_results == "all" or max_results > 100 else max_results
        endpoint += f"{separator}limit={limit_value}"

    # Use the common REST API helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    # Add data location information to the result
    if api_result.get("success", False):
        # Extract data_type and search_strategy from the query_info if available
        data_type = query_info.get("data_type", "Unknown") if "query_info" in locals() else "Unknown"
        search_strategy = (
            query_info.get("search_strategy", "Direct query") if "query_info" in locals() else "Direct query"
        )

        api_result["data_type"] = data_type
        api_result["search_strategy"] = search_strategy
        api_result["data_location_info"] = {
            "description": description,
            "data_type": data_type,
            "search_strategy": search_strategy,
            "endpoint_used": endpoint,
        }

    # Handle API parameter errors with fallback for ENCODE
    if not api_result.get("success", False) and "404" in str(api_result.get("error", "")):
        # Try simplified query with basic search
        if prompt and "transcription factor" in prompt.lower():
            simplified_endpoint = f"{base_url}/search/?type=Experiment&assay_title=ChIP-seq&searchTerm=transcription%20factor&format=json&limit={max_results}"
            api_result = _query_rest_api(
                endpoint=simplified_endpoint, method="GET", description=f"{description} (simplified)"
            )
            if api_result.get("success", False):
                api_result["note"] = "Query simplified due to API endpoint restrictions"

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        api_result["result"] = _format_query_results(api_result["result"])

    return api_result


def query_zinc15_compounds(
    query: str,
    search_type: str = "smiles",
    max_results: int = 20,
    timeout_seconds: int = 30,
    task_id: str = "",
    poll_interval_seconds: float = 1.0,
) -> dict:
    """Query ZINC, falling back to Cartblanche22 when ZINC15 requires browser verification."""
    query_info = {
        "input": query,
        "source": "ZINC15 public API with Cartblanche22 fallback",
        "parameters": {
            "search_type": search_type,
            "max_results": max_results,
            "task_id": task_id,
            "poll_interval_seconds": poll_interval_seconds,
        },
    }

    def compounds_from_payload(payload: Any) -> list[dict]:
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            rows = payload.get("results", payload.get("substances", payload.get("data", [])))
        else:
            rows = []
        if not isinstance(rows, list):
            return []
        compounds = []
        allowed = ("zinc_id", "smiles", "name", "inchi", "inchikey", "formula", "sub_id", "tranche")
        for row in rows[:max_results]:
            if isinstance(row, dict):
                compounds.append({key: row[key] for key in allowed if row.get(key) is not None})
        return compounds

    try:
        if (not isinstance(query, str) or not query.strip()) and not task_id.strip():
            return {"success": False, "error": "query must be a non-empty string.", "query_info": query_info}
        if search_type not in {"smiles", "zinc_id", "similarity"}:
            return {
                "success": False,
                "error": "search_type must be smiles, zinc_id, or similarity.",
                "query_info": query_info,
            }
        if not 1 <= max_results <= 1000:
            return {"success": False, "error": "max_results must be between 1 and 1000.", "query_info": query_info}
        if timeout_seconds < 1:
            return {"success": False, "error": "timeout_seconds must be positive.", "query_info": query_info}
        if poll_interval_seconds <= 0:
            return {"success": False, "error": "poll_interval_seconds must be positive.", "query_info": query_info}

        primary_error = ""
        if not task_id.strip():
            try:
                response = requests.get(
                    "https://zinc15.docking.org/substances.json",
                    params={"q": query.strip(), "count": max_results},
                    timeout=timeout_seconds,
                )
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").lower()
                if "json" not in content_type:
                    raise ValueError("ZINC15 returned browser-verification HTML instead of JSON")
                payload = response.json()
                compounds = compounds_from_payload(payload)
                return {
                    "success": True,
                    "result": {
                        "compounds": compounds,
                        "count": len(compounds),
                        "completed": True,
                        "http_status": response.status_code,
                    },
                    "query_info": query_info,
                }
            except (requests.RequestException, ValueError, TypeError) as exc:
                primary_error = str(exc)

        cartblanche_root = "https://cartblanche22.docking.org"
        current_task_id = task_id.strip()
        if not current_task_id:
            if search_type in {"smiles", "similarity"}:
                endpoint = f"{cartblanche_root}/smiles.json"
                distance = "4" if search_type == "similarity" else "0"
                files = {
                    "smiles": (None, query.strip()),
                    "dist": (None, distance),
                    "adist": (None, distance),
                    "database": (None, "zinc22,zinc20"),
                }
            else:
                endpoint = f"{cartblanche_root}/substances.json"
                files = {"zinc_ids": (None, query.strip()), "smiles_only": (None, "true")}
            submitted = requests.post(endpoint, files=files, timeout=timeout_seconds)
            submitted.raise_for_status()
            submission = submitted.json()
            current_task_id = str(submission.get("task", "")).strip() if isinstance(submission, dict) else ""
            if not current_task_id:
                raise ValueError(f"Cartblanche22 submission did not return a task ID: {submission}")

        result_url = f"{cartblanche_root}/search/saveResult/{current_task_id}.json"
        query_info["fallback"] = {
            "source": "Cartblanche22/ZINC22",
            "task_id": current_task_id,
            "result_url": result_url,
            "primary_error": primary_error or None,
        }
        deadline = time.monotonic() + timeout_seconds
        last_status = "PENDING"
        while time.monotonic() < deadline:
            result_response = requests.get(result_url, timeout=timeout_seconds)
            result_response.raise_for_status()
            payload = result_response.json()
            if isinstance(payload, dict) and str(payload.get("status", "")).upper() in {"PENDING", "STARTED"}:
                last_status = str(payload.get("status", "PENDING")).upper()
                time.sleep(min(poll_interval_seconds, max(0.0, deadline - time.monotonic())))
                continue
            compounds = compounds_from_payload(payload)
            return {
                "success": True,
                "result": {
                    "compounds": compounds,
                    "count": len(compounds),
                    "completed": True,
                    "status": "COMPLETE",
                    "task_id": current_task_id,
                    "result_url": result_url,
                },
                "query_info": query_info,
            }
        return {
            "success": True,
            "result": {
                "compounds": [],
                "count": 0,
                "completed": False,
                "status": last_status,
                "task_id": current_task_id,
                "result_url": result_url,
                "message": "The public Cartblanche22 job is still running; call this tool again with task_id to continue polling.",
            },
            "query_info": query_info,
        }
    except requests.RequestException as exc:
        return {"success": False, "error": f"ZINC/Cartblanche22 network/API request failed: {exc}", "query_info": query_info}
    except (ValueError, TypeError) as exc:
        return {"success": False, "error": f"ZINC15 returned an invalid response: {exc}", "query_info": query_info}
    except Exception as exc:
        return {"success": False, "error": f"ZINC15 query failed: {exc}", "query_info": query_info}
