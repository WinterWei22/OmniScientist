import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class BasicEvidence:
    title: Optional[str]
    text: str
    url: Optional[str]
    doi: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class BasicToolResult:
    tool_name: str
    sub_question: str
    success: bool
    round_index: int
    raw_payload: Any
    raw_text: str
    evidences: List[BasicEvidence]

    def get(self, key: str, default: Any = None) -> Any:
        if key == "success":
            return self.success
        return getattr(self, key, default)


@dataclass
class miniEvidenceEntry:
    evidence_id: str
    round_index: int
    tool: str
    brief: str
    quote: str
    fetched_at: str
    hash_value: str
    relevance: float
    credibility: float
    sub_question: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[str] = None
    doi: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    reference_chain_parent: Optional[str] = None
    raw_content: Optional[dict] = None
    reference_key: Optional[str] = None
    citation_label: Optional[str] = None
    journal: Optional[str] = None
    full_abstract: Optional[str] = None
    key_sentences: List[str] = field(default_factory=list)
    citation_reason: Optional[str] = None
    citation_context: Optional[str] = None
    detailed_findings: Optional[str] = None
    llm_summary: Optional[str] = None


class ToolResultParser:
    MAX_ITEMS = 40

    def __init__(
        self, tool_name: str, content: Any, success: bool, sub_question: str, round_index: int
    ) -> None:
        self.tool_name = tool_name
        self.content = content
        self.sub_question = sub_question
        self.round_index = round_index
        self.success = success

    def parse_basic(self) -> BasicToolResult:
        payload = self._load_json(self.content)

        if (
            self.tool_name == "paper_search"
        ):  
            payload = payload["papers"] if isinstance(payload, dict) and "papers" in payload else payload

        evidences = self._extract_basic_evidences(payload)
        raw_text = self._normalise_to_text(payload)

        return BasicToolResult(
            tool_name=self.tool_name,
            sub_question=self.sub_question,
            round_index=self.round_index,
            raw_payload=payload,
            raw_text=raw_text,
            evidences=evidences,
            success=self.success,
        )

    def _load_json(self, content: Any) -> Any:
        if isinstance(content, str):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return content
        return content

    def _normalise_to_text(self, payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        try:
            return json.dumps(payload, ensure_ascii=False)
        except TypeError:
            return str(payload)

    def _extract_basic_evidences(self, payload: Any) -> List[BasicEvidence]:
        evidences: List[BasicEvidence] = []
        if isinstance(payload, dict):
            evidences.extend(self._extract_from_dict(payload))
        elif isinstance(payload, list):
            for item in payload[: self.MAX_ITEMS]:
                if isinstance(item, dict):
                    evidences.extend(self._extract_from_dict(item))
                else:
                    evidences.append(
                        BasicEvidence(
                            title=None, text=str(item), url=None, doi=None, metadata={}
                        )
                    )
        else:
            evidences.append(
                BasicEvidence(
                    title=None, text=str(payload), url=None, doi=None, metadata={}
                )
            )
        return evidences[: self.MAX_ITEMS]

    def _extract_from_dict(self, data: Dict[str, Any]) -> List[BasicEvidence]:
        evidences: List[BasicEvidence] = []
        text_fields = [
            data.get("summary"),
            data.get("content"),
            data.get("text"),
            data.get("raw_content"),
            data.get("description"),
            data.get("full_abstract"),
        ]
        text = next(
            (field for field in text_fields if isinstance(field, str) and field.strip()),
            None,
        )
        if text:
            evidences.append(
                BasicEvidence(
                    title=data.get("title"),
                    text=text,
                    url=data.get("url"),
                    doi=data.get("doi", None),
                    metadata={
                        k: v
                        for k, v in data.items()
                        if k
                        not in {
                            "summary",
                            "content",
                            "text",
                            "raw_content",
                            "description",
                            "full_abstract",
                        }
                    },
                )
            )
        else:
            try:
                evidences.append(
                    BasicEvidence(
                        title=data.get("title"),
                        text=json.dumps(data, ensure_ascii=False),
                        url=data.get("url"),
                        doi=data.get("doi", None),
                        metadata={},
                    )
                )
            except TypeError:
                evidences.append(
                    BasicEvidence(
                        title=data.get("title"),
                        text=str(data),
                        url=data.get("url"),
                        doi=data.get("doi", None),
                        metadata={},
                    )
                )
        return evidences

    def _build_evidence(
        self,
        brief: str,
        quote: str,
        url: Optional[str] = None,
        title: Optional[str] = None,
        authors: Optional[str] = None,
        year: Optional[str] = None,
        doi: Optional[str] = None,
        raw: Optional[Dict] = None,
    ) -> miniEvidenceEntry:
        hash_val = str(hash(quote + (url or "")))
        return miniEvidenceEntry(
            evidence_id=f"{self.round_index}-{hash_val[-6:]}",
            round_index=self.round_index,
            tool=self.tool_name,
            brief=brief,
            quote=quote,
            fetched_at=datetime.now().isoformat(),
            hash_value=hash_val,
            relevance=0.0,
            credibility=0.0,
            sub_question=self.sub_question,
            url=url,
            title=title,
            authors=authors,
            year=year,
            doi=doi,
            raw_content=raw,
        )

    def _parse_tavily_search(self, payload: Dict[str, Any]):
        summaries: List[str] = []
        sources: List[str] = []
        entries: List[miniEvidenceEntry] = []
        for item in payload.get("results", [])[: self.MAX_ITEMS]:
            title = item.get("title") or "Untitled"
            summary = item.get("summary") or item.get("content", "")
            url = item.get("url")
            raw = item.get("raw_content") or item
            published = item.get("published_date")
            year = None
            if isinstance(published, str) and published:
                try:
                    year = str(
                        datetime.fromisoformat(published.replace("Z", "+00:00")).year
                    )
                except ValueError:
                    year = published[:4]
            summaries.append(f"- {title}: {summary}")
            if url:
                sources.append(url)
            entries.append(
                self._build_evidence(
                    summary,
                    raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)[
                        :6000
                    ],
                    url=url,
                    title=title,
                    authors=None,
                    year=year,
                    doi=None,
                    raw=item if isinstance(item, dict) else None,
                )
            )
        return summaries, sources, entries

    def _parse_tavily_deep_extract(self, payload: Dict[str, Any]):
        summaries: List[str] = []
        sources: List[str] = []
        entries: List[miniEvidenceEntry] = []
        page_content = payload.get("content") or payload.get("raw_content") or ""
        title = payload.get("title") or payload.get("metadata", {}).get("title")
        url = payload.get("url") or payload.get("metadata", {}).get("url")
        summary = (
            payload.get("summary")
            or payload.get("metadata", {}).get("summary")
            or page_content[:500]
        )
        authors = None
        metadata = payload.get("metadata") or {}
        if isinstance(metadata, dict):
            authors = metadata.get("authors") or metadata.get("author")
        year = metadata.get("year") if isinstance(metadata, dict) else None
        doi = metadata.get("doi") if isinstance(metadata, dict) else None
        summaries.append(f"- {title or self.tool_name}: {summary}")
        if url:
            sources.append(url)
        entries.append(
            self._build_evidence(
                summary,
                page_content,
                url=url,
                title=title,
                authors=", ".join(authors) if isinstance(authors, list) else authors,
                year=str(year) if year else None,
                doi=doi,
                raw=payload if isinstance(payload, dict) else None,
            )
        )
        references = payload.get("references") or payload.get("metadata", {}).get(
            "references"
        )
        if isinstance(references, list):
            for ref in references[: self.MAX_ITEMS]:
                ref_title = ref.get("title") if isinstance(ref, dict) else str(ref)
                ref_url = ref.get("url") if isinstance(ref, dict) else None
                summaries.append(f"- Reference: {ref_title}")
                if ref_url:
                    sources.append(ref_url)
        return summaries, sources, entries

    def _parse_dict_generic(self, payload: Dict[str, Any]):
        serialised = json.dumps(payload, ensure_ascii=False)[:5000]
        summary = f"- {self.tool_name}: {serialised}"
        entry = self._build_evidence(summary, serialised, raw=payload)
        return [summary], [], [entry]

    def _parse_list(self, payload: List[Any]):
        summaries: List[str] = []
        entries: List[miniEvidenceEntry] = []
        for idx, item in enumerate(payload[: self.MAX_ITEMS], 1):
            text = str(item)[:5000]
            summary = f"- Item {idx}: {text}"
            summaries.append(summary)
            entries.append(
                self._build_evidence(
                    summary,
                    text,
                    title=f"Item {idx}",
                    raw=item if isinstance(item, dict) else None,
                )
            )
        return summaries, [], entries

    def _parse_text(self, text: str):
        summary = f"- {self.tool_name}: {text[:5000]}"
        return [summary], [], [self._build_evidence(summary, text)]


def parse_tool_result(tool_result: dict[str, Any], sub_question: str, round_index: int):
    try:
        return ToolResultParser(
            tool_name=tool_result.get("tool") or tool_result.get("tool_name"),
            content=tool_result.get("content"),
            sub_question=sub_question,
            round_index=round_index,
            success=tool_result.get("success"),
        ).parse_basic()
    except Exception as e:
        logging.error(f"Error parsing tool result: {e}")
        return None
