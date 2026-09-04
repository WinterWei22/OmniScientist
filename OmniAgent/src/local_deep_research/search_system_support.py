import re
import ast
import json
import uuid
import asyncio
from typing import Dict, List, Optional, Union, Any

from pydantic import BaseModel, Field

# utils/reference_utils.py
import urllib.parse as u

_URL_TAIL_PUNCT = r'\)\]\}\>,.;\'"!?$'

import json, re, html
from typing import Any, Dict, List, Tuple


class SourcesReference(BaseModel):
    title: str = Field(description="title")
    subtitle: str = Field("", description="subtitle")
    link: str = Field("", description="link")

def _to_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        x = x[0] if x else ""
    try:
        return str(x).strip()
    except Exception:
        return ""


PRIMARY_FIELDS = (
    "answer", "content", "summary", "text",
    "result", "output", "response"      
)

def _to_text(x: Any) -> str:
    """ Convert various types to a clean text string."""
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):

        for t in x:
            txt = _to_text(t)
            if txt:
                return txt
        return ""
    if isinstance(x, dict):

        for k in ("href", "url", "link", "value"):
            if k in x:
                return _to_text(x[k])
        return json.dumps(x, ensure_ascii=False)
    return str(x).strip()

def _clean_url(u: str) -> str:
    """ Clean and validate a URL string."""
    if not u:
        return ""
    while u and u[-1] in _URL_TAIL_PUNCT:
        u = u[:-1]
    return u if u.startswith(("http://", "https://")) else ""


async def parse_single(raw: Dict[str, Any], query: str | None = None) -> Dict[str, Any]:
    """
    Parse a single tool result into a structured format.
    """
    out: Dict[str, Any] = {
        "tool_name": _to_text(raw.get("tool_name", "unknown"))[:100],
        "query": query,
        "primary": "",
        "description_by_urls": [],   # [{title,url,meta}]
        "urls": [],                  
        "error": None,
    }

    try:
        txt: str = _to_text(raw.get("content", ""))
        parsed: Any
        try:
            parsed = json.loads(txt) if txt.lstrip()[:1] in "{[" else txt
        except Exception:
            parsed = txt

        if isinstance(parsed, dict):
            for k in ("answer", "content", "summary", "text"):
                v = _to_text(parsed.get(k))
                if v:
                    out["primary"] = v[:5000]
                    break
            else:
                out["primary"] = json.dumps(parsed, ensure_ascii=False)[:5000]
        else:
            out["primary"] = _to_text(parsed)[:5000]

        temp_sources: List[Dict[str, str]] = []
        if isinstance(parsed, dict):
            for key in ("links", "citations", "sources"):
                for item in parsed.get(key, []):
                    if not isinstance(item, dict):
                        continue
                    url = _clean_url(_to_text(item.get("url")))
                    if not url:
                        continue
                    temp_sources.append({
                        "title": _to_text(item.get("title") or url),
                        "url": url,
                        "meta": _to_text(item.get("date") or item.get("author")),
                    })

        for title, url in re.findall(r'\[([^\]]+?)\]\((https?://[^\)]+)\)', txt):
            url = _clean_url(url)
            if url:
                temp_sources.append({"title": _to_text(title) or url, "url": url, "meta": ""})

        for m in re.finditer(r'https?://\S+', txt):
            url = _clean_url(m.group(0))
            if not url:
                continue
            preceding = txt[max(0, m.start() - 200): m.start()]
            mt = re.search(r'"title"\s*:\s*"([^"]+)"', preceding, re.IGNORECASE)
            title = _to_text(mt.group(1)) if mt else ""
            if not title:
                snippet = preceding.rsplit('\n', 1)[-1]
                snippet = snippet.strip().strip('":')
                title = snippet if snippet else url
            temp_sources.append({"title": title, "url": url, "meta": ""})

        seen: set[Tuple[str, str]] = set()
        for s in temp_sources:
            title = _to_text(s["title"])[:400]
            url   = _clean_url(s["url"])
            if not url:
                continue
            key = (title, url)
            if key in seen:
                continue
            seen.add(key)
            out["description_by_urls"].append({"title": title, "url": url, "meta": _to_text(s["meta"])})
        out["urls"] = [s["url"] for s in out["description_by_urls"]]

    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"

    return out


async def compress_single_llm(
    fast_model, parsed: Dict[str, Any], query: str
    ) -> Dict[str, Any]:
        """

        retrun:
        {
            "extracted_facts": [ ... ],
            "references": [                 # ← only one source of references
                {
                    "url": "https://...",
                    "description": "",
                    "apa_citation": "Author, A. A. (YYYY). Title. Journal, volume(issue), pages."
                },
                ...
            ]
        }
        """
        if parsed.get("error") or not parsed.get("primary", "").strip():
            return {"extracted_facts": [], "references": []}

        tool_name   = parsed["tool_name"][:1000]
        primary_txt = parsed["primary"][:10000]
        urls_info   = parsed.get("description_by_urls", [])[:5000]   

        prompt =  f"""
        Your task is to **extract information only** from this tool-calling result.
        Do **NOT** analyse, judge, interpret or invent – just copy what is there.
        Extract only information that clearly stated in raw data and directly relevant to Tool query，skip the information that contradict or error source data or lack direct evidence.

        ### Context
        - Tool Query: {query}
        - Tool name: {tool_name}

        ### Raw Data
        ```text
        {primary_txt}
        {json.dumps(urls_info, ensure_ascii=False)}
        
        ### Extraction Rules
        1. Facts or important detials – quote verbatim from the raw data.

        2. references – for every distinct HTTP/HTTPS link produce an object:
        -- "url" : the link itself

        -- "description" : short phrase copied verbatim from title / meta / primary

        -- "apa_citation" : if the URL points to a scholarly paper (DOI / PubMed / arXiv / journal),recover an APA-style citation from nearby useful text(but not invent);otherwise output an empty string.

        3. Never invent information; leave fields empty rather than guessing.

        4. No interpretation, no main-query answer.

        5. For isolated facts (no clear context), add Tool query to indicate their origin

        6. If nothing fits a section, return an empty JSON array [].


        Output JSON (and nothing else)
        {{
        "extracted_facts": [ "Fact 1 …", "Fact 2 …" ],
        "references": [
            {{ "url": "...", "description": "...", "apa_citation": "..." }}
        ]
        }}

        """
        compress_results = await fast_model.ainvoke(prompt)
        data = safe_json_from_text(compress_results.content) or {}

        data.setdefault("extracted_facts", [])
        data.setdefault("references", [])
        if not isinstance(data["extracted_facts"], list):
            data["extracted_facts"] = []
        if not isinstance(data["references"], list):
            data["references"] = []


        for ref in data["references"]:
            ref.setdefault("description", "")
            ref.setdefault("apa_citation", "")
        return data

async def compress_all_llm(model, parsed_list: List[Dict[str,Any]], limit=5, query=""):
    sem = asyncio.Semaphore(limit)
    async def wrap(p):
        async with sem: return await compress_single_llm(model, p, query)
    return await asyncio.gather(*(wrap(p) for p in parsed_list))




def pick_url(results: list[dict]) -> tuple[str, str] | None:
    """
    """
    for r in results:
        # additional_urls / additional_info
        for key in ("additional_urls", "additional_info", "urls"):
            if key in r and r[key]:
                val = r[key]
                if isinstance(val, str) and val.startswith(("http://", "https://")):
                    return val, detect_content_type(val)
                if isinstance(val, list):
                    for v in val:
                        if isinstance(v, str) and v.startswith(("http://", "https://")):
                            return v, detect_content_type(v)
        for key in ("link", "url"):
            if (
                key in r
                and isinstance(r[key], str)
                and r[key].startswith(("http://", "https://"))
            ):
                return r[key], detect_content_type(r[key])

        if isinstance(r.get("content"), str):
            try:
                obj = json.loads(r["content"])
                for sub in obj.get("results", []):
                    u = sub.get("url")
                    if isinstance(u, str) and u.startswith(("http://", "https://")):
                        return u, detect_content_type(u)
            except Exception:
                pass

    return None

def safe_json_from_text(text: str) -> Optional[Dict]:
    """
    Extract and parse the first valid JSON object from LLM output text.

    Args:
        text: Raw text response from LLM that may contain JSON

    Returns:
        Parsed JSON dict or None if no valid JSON found
    """
    if not text:
        return None

    # Remove code block markers ```json ... ``` or ``` ... ```
    cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", text, flags=re.I)

    # Find and try to parse each potential JSON object starting with {
    for match in re.finditer(r"{", cleaned):
        try:
            candidate = cleaned[match.start() :]
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def extract_and_convert_list(text: str) -> Optional[List]:
    """
    Extract and convert a list from text using regex and ast.literal_eval.

    Args:
        text: Text containing a Python list representation

    Returns:
        Parsed list or None if extraction/conversion fails
    """
    if not text:
        return None

    # Regex pattern to match list structures, including nested ones
    pattern = r"\[(?:[^\[\]]*|\[(?:[^\[\]]*|\[[^\[\]]*\])*\])*\]"
    match = re.search(pattern, text)

    if match:
        list_str = match.group()
        try:
            python_list = ast.literal_eval(list_str)
            return python_list if isinstance(python_list, list) else None
        except (SyntaxError, ValueError):
            return None
    return None


def extract_json_from_response(response_content: str) -> Optional[Dict]:
    """
    Extract JSON from LLM response content with multiple fallback strategies.

    Args:
        response_content: Raw response content from LLM

    Returns:
        Extracted JSON dict or None
    """
    if not response_content:
        return None

    # Strategy 1: Try direct JSON parsing
    try:
        return json.loads(response_content.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2: Use safe_json_from_text for structured extraction
    result = safe_json_from_text(response_content)
    if result:
        return result

    # Strategy 3: Look for JSON between specific markers
    markers = [
        (r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE),
        (r"```\s*(.*?)\s*```", re.DOTALL),
        (r"\{.*\}", re.DOTALL),
    ]

    for pattern, flags in markers:
        matches = re.findall(pattern, response_content, flags)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

    return None


# ============================================================================
# Text Processing Utilities
# ============================================================================


def clean_text_format(text: Union[str, Any]) -> str:
    """
    Clean and format text, handling newline characters and formatting issues.

    Args:
        text: Text to clean (any type, will be converted to string)

    Returns:
        Cleaned text string
    """
    if not text:
        return ""

    # Convert to string and handle escaped characters
    text = str(text).replace("\\n", "\n").replace("\\t", " ")

    # Remove excessive empty lines while preserving structure
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line:  # Only keep non-empty lines
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def detect_content_type(url: str) -> str:
    """
    Detect the type of content from a URL based on file extension.

    Args:
        url: URL to analyze

    Returns:
        Content type string: 'image', 'video', 'pdf', or 'iframe'
    """
    if not url:
        return "iframe"

    url_lower = url.lower().strip()

    # Common image extensions
    image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp"]
    if any(url_lower.endswith(ext) for ext in image_extensions):
        return "image"

    # Common video extensions
    video_extensions = [".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm", ".mkv"]
    if any(url_lower.endswith(ext) for ext in video_extensions):
        return "video"

    # PDF files
    if url_lower.endswith(".pdf"):
        return "pdf"

    # Default to iframe for web pages
    return "iframe"


# ============================================================================
# Debug and Utility Functions
# ============================================================================
def highlight_print(obj: Any, name: Optional[str] = None) -> None:
    """
    Print an object with highlighted formatting for debugging.

    Args:
        obj: Object to print
        name: Optional name/label for the object
    """
    print("\n" + "#" * 80)
    if name:
        print(f"{name}: {obj}")
    else:
        print(f"{obj}")
    print("#" * 80 + "\n")


def format_progress_message(
    message: str, progress_percent: Optional[int] = None
) -> str:
    """
    Format a progress message with optional percentage.

    Args:
        message: Progress message
        progress_percent: Optional progress percentage (0-100)

    Returns:
        Formatted progress message
    """
    progress_text = f" ({progress_percent}%)" if progress_percent else ""
    return f"🔄 PROGRESS: {message}{progress_text}"






