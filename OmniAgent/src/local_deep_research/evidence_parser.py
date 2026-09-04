from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .tool_parsers import BasicToolResult, miniEvidenceEntry

logger = logging.getLogger(__name__)


@dataclass
class BibliographyEntry:
    key: str
    title: str
    authors: str | None = None
    journal: str | None = None
    year: str | None = None
    doi: str | None = None
    url: str | None = None
    fetched_at: str | None = None
    first_seen_round: int | None = None


class BibliographyRegistry:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: Dict[str, BibliographyEntry] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for key, value in data.items():
                self._entries[key] = BibliographyEntry(**value)
        except Exception as e:
            logger.warning(f"Failed to load bibliography from {self.path}: {e}")

    def _persist(self) -> None:
        serialized = {key: entry.__dict__ for key, entry in self._entries.items()}
        self.path.write_text(
            json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def add_or_update(self, entry: BibliographyEntry) -> None:
        existing = self._entries.get(entry.key)
        if existing:
            if entry.first_seen_round is not None:
                if (
                    existing.first_seen_round is None
                    or entry.first_seen_round < existing.first_seen_round
                ):
                    existing.first_seen_round = entry.first_seen_round
            for field_name in [
                "title",
                "authors",
                "journal",
                "year",
                "doi",
                "url",
                "fetched_at",
            ]:
                value = getattr(entry, field_name)
                if value:
                    setattr(existing, field_name, value)
        else:
            self._entries[entry.key] = entry
        self._persist()

    def get(self, key: str) -> Optional[BibliographyEntry]:
        return self._entries.get(key)

    def all_entries(self) -> List[BibliographyEntry]:
        return list(self._entries.values())

    def merge_entries(self, old_key: str, new_key: str) -> None:
        if old_key not in self._entries or new_key not in self._entries:
            return

        old_entry = self._entries[old_key]
        new_entry = self._entries[new_key]

        for field in ["authors", "journal", "year", "doi", "url"]:
            old_val = getattr(old_entry, field)
            new_val = getattr(new_entry, field)
            if old_val and not new_val:
                setattr(new_entry, field, old_val)

        del self._entries[old_key]
        self._persist()


@dataclass
class ParsedToolResult:
    basic: BasicToolResult
    evidence_entries: List[miniEvidenceEntry]
    summary: str


class ToolParseAgent:
    def __init__(self, llm: ChatOpenAI, max_concurrent: int = 10) -> None:
        self.llm = llm
        self.max_concurrent = max_concurrent
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You transform raw MCP tool outputs into reliable evidence entries with rich citation support."
                    " STRICT POLICY: You must stay completely faithful to the MCP output — do NOT fabricate, infer, or guess any factual content."
                    " If the MCP output only contains the query and all other fields (results, metadata, text, etc.) are empty, meaningless, or null,"
                    " then output an empty evidence structure with no evidence items (i.e., {{\"summary\": \"\", \"items\": []}})."
                    " Preserve all critical details actually present in MCP output (claims, statistics, citations, experimental data) without alteration."
                    " If data quality is weak or uncertain, note it explicitly — but never invent missing data."
                    " Output valid JSON matching schema: {{\"summary\": string, \"items\": [ {{"
                    " \"title\": string?, \"brief\": string, \"quote\": string, \"url\": string?,"
                    " \"doi\": string?, \"year\": string?, \"authors\": string?, \"relevance\": number?,"
                    " \"tags\": [string]?, \"full_abstract\": string?, \"key_sentences\": [string]?,"
                    " \"citation_reason\": string?, \"detailed_findings\": string? }} ] }}."
                    " NEW FIELDS EXPLANATION:"
                    " - full_abstract: If MCP source provides abstract/summary, extract it in full (up to 3000 chars)."
                    " - key_sentences: Extract 3–5 most important sentences that contain core findings, claims, or data, directly from MCP text."
                    " - citation_reason: 1–2 sentences explaining WHY this source should be cited (what it demonstrates), based only on MCP content."
                    " - detailed_findings: Expanded factual description including methods, sample sizes, key results, effect sizes, p-values, mechanisms, or clinical implications (up to 2000 chars), only if explicitly present."
                    " - When the title is incomplete, DO NOT complete it yourself; just use '...' to mark incompleteness."
                    " - DO NOT invent or fill any of these factual terms: title, url, doi, year, authors, full_abstract, key_sentences."
                    " For relevance scoring (0.0–1.0), assess only based on MCP-provided evidence:"
                    " - 0.9–1.0: Primary research with direct experimental/clinical data answering the core question."
                    " - 0.7–0.8: Strong mechanistic insights, comprehensive reviews, or key supporting studies."
                    " - 0.5–0.6: Relevant background or partial evidence."
                    " - 0.3–0.4: Tangential connections."
                    " - below 0.3: Weak or low-quality source."
                    " Evaluate credibility only when the source type is explicitly available in MCP output."
                    " Only extract items with verifiable content; skip generic, empty, or duplicate entries."
                    " Keep all quotes and key_sentences verbatim (no paraphrasing)."
                    " If URL or DOI is missing, leave it null — never fabricate."
                ),
                (
                    "human",
                    "Sub-problem objective: {objective}\n"
                    "Tool: {tool_name}\nSub-question: {sub_question}\n"
                    "Raw text (may be truncated):\n{raw_text}\n"
                    "Extracted snippets (JSON):\n{snippets}\n"
                    "Respond with JSON only.",
                ),
            ]
        )

    async def parse_results(
        self, query: str, basics: List[BasicToolResult]
    ) -> List[ParsedToolResult]:
        logger.info(
            f"Starting parallel parsing of {len(basics)} tool results (max concurrent: {self.max_concurrent})..."
        )

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def _parse_with_limit(idx, basic):
            async with semaphore:
                return await self._parse_single(query, basic, idx)

        tasks = [_parse_with_limit(idx, basic) for idx, basic in enumerate(basics)]
        parsed = await asyncio.gather(*tasks, return_exceptions=True)

        results: List[ParsedToolResult] = []
        for idx, result in enumerate(parsed):
            if isinstance(result, ParsedToolResult):
                results.append(result)
                logger.debug(f"  ✓ Parsed result {idx+1}/{len(basics)}")
            else:
                logger.warning(f"Failed to parse tool result {idx}: {result}")

        logger.info(f"Parsing complete: {len(results)}/{len(basics)} successful")
        return results

    async def _parse_single(
        self, query: str, basic: BasicToolResult, idx: int = 0
    ) -> ParsedToolResult:
        import time

        start = time.time()

        if basic.tool_name == "read_url":
            raw_text = basic.raw_text[:50000]
            snippet_text_limit = 30000
        elif basic.tool_name == "doi2text":
            raw_text = basic.raw_text[:50000]
            snippet_text_limit = 30000
        else:
            raw_text = basic.raw_text[:30000]
            snippet_text_limit = 20000

        snippets = [
            {
                "title": evidence.title,
                "text": evidence.text[:snippet_text_limit],
                "url": evidence.url,
                "doi": evidence.doi,
                "metadata": evidence.metadata,
            }
            for evidence in basic.evidences
        ]
        if basic.tool_name == "read_url":  # fixed typo readl_url
            tool_specific_hint = (
                "\n\nCRITICAL: This is FULL CONTENT from deep extraction (up to 50k chars). "
                "You have access to complete article text. MUST extract:"
                "\n- full_abstract: Complete abstract/summary (2000-3000 chars if available)"
                "\n- detailed_findings: Experimental design, methods, sample sizes, statistical results, "
                "mechanisms, dosages, effect sizes, p-values (aim for 1500-2000 chars)"
                "\n- key_sentences: 4-5 most data-rich sentences containing specific numbers, percentages, "
                "p-values, or quantitative findings"
                "\nDo NOT just extract brief summaries - use the full content to provide comprehensive details."
            )
        elif basic.tool_name == "tavily_search":
            tool_specific_hint = (
                """The result from tavily_search contains two parts: (1) raw contents retrieved from multiple sources, and (2) an LLM-generated 'answer' summarizing those sources.
                Be aware that the LLM-generated answer may include hallucinations, while the raw content is generally reliable.
                Always verify the 'answer' against the raw content and make your own judgment about relevance.
                When building the 'brief', rely only on the corresponding raw content, not on the generated answer."""
            )
        elif basic.tool_name == "paper_search":
            tool_specific_hint = (
                "\nThe raw_text may contain multiple output papers' information."
                " One doi record in raw_context means one individual reasarch paper."
                " Organize each of them according to the output format and output them as an item."
                " Output the item even its information is imcomplete."
            )
        else:
            tool_specific_hint = ""

        response = await (self.prompt | self.llm).ainvoke(
            {
                "objective": query + tool_specific_hint,
                "tool_name": basic.tool_name,
                "sub_question": basic.sub_question,
                "raw_text": raw_text,
                "snippets": json.dumps(snippets, ensure_ascii=False),
            }
        )
        evidence_entries: List[miniEvidenceEntry] = []
        summary = ""
        try:
            data = json.loads(response.content)
            summary = str(data.get("summary", "")).strip()
            items = data.get("items", [])
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    entry = self._build_entry(basic, item)
                    evidence_entries.append(entry)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        if not evidence_entries:
            # Fallback: use first snippet if parsing failed
            for evidence in basic.evidences[:1]:
                evidence_entries.append(self._fallback_entry(basic, evidence))
            if not summary:
                summary = f"{basic.tool_name} returned data but structured parsing failed; manual review required."

        if not summary:
            if evidence_entries:
                summary = "\n".join(entry.brief for entry in evidence_entries[:2])[:500]
            else:
                summary = "No evidence extracted."

        duration = time.time() - start
        logger.debug(f"  Parse #{idx+1} completed in {duration:.1f}s - {basic.tool_name}")

        return ParsedToolResult(
            basic=basic, evidence_entries=evidence_entries, summary=summary
        )

    def _build_entry(
        self, basic: BasicToolResult, item: Dict[str, Any]
    ) -> miniEvidenceEntry:
        brief = str(
            item.get("brief")
            or item.get("summary")
            or item.get("text")
            or basic.raw_text[:500]
        ).strip()
        quote = str(item.get("quote") or item.get("text") or basic.raw_text)[:6000]
        url = item.get("url") or None
        doi = item.get("doi") or None
        authors = item.get("authors") or None
        year = item.get("year")
        if year is not None:
            year = str(year)
        title = item.get("title") or url or basic.tool_name
        relevance = item.get("relevance")
        credibility = item.get("credibility")
        journal = item.get("journal") or None
        try:
            relevance_value = float(relevance)
        except (TypeError, ValueError):
            relevance_value = 0.0
        try:
            credibility_value = float(credibility)
        except (TypeError, ValueError):
            credibility_value = 0.0
        tags = item.get("tags")
        if isinstance(tags, list):
            tag_list = [str(tag) for tag in tags]
        elif tags:
            tag_list = [str(tags)]
        else:
            tag_list = []

        full_abstract = item.get("full_abstract")
        if full_abstract:
            full_abstract = str(full_abstract)[:3000]

        key_sentences = item.get("key_sentences")
        if isinstance(key_sentences, list):
            key_sentences_list = [str(s) for s in key_sentences if s][:5]
        else:
            key_sentences_list = []

        citation_reason = item.get("citation_reason")
        if citation_reason:
            citation_reason = str(citation_reason)[:500]

        detailed_findings = item.get("detailed_findings")
        if detailed_findings:
            detailed_findings = str(detailed_findings)[:2000]

        hash_value = str(hash((quote, url, title)))
        return miniEvidenceEntry(
            evidence_id=f"{basic.round_index}-{hash_value[-6:]}",
            # sab_id=basic.sab_id,
            round_index=basic.round_index,
            tool=basic.tool_name,
            sub_question=basic.sub_question,
            url=url,
            title=title,
            authors=authors,
            year=year,
            doi=doi,
            brief=brief,
            quote=quote,
            fetched_at="",
            hash_value=hash_value,
            relevance=relevance_value,
            credibility=credibility_value,
            journal=journal,
            tags=tag_list,
            reference_chain_parent=None,
            raw_content=item,
            reference_key=None,
            citation_label=None,
            full_abstract=full_abstract,
            key_sentences=key_sentences_list,
            citation_reason=citation_reason,
            detailed_findings=detailed_findings,
        )

    def _fallback_entry(self, basic: BasicToolResult, evidence) -> miniEvidenceEntry:
        brief = evidence.text[:500]
        hash_value = str(hash((brief, evidence.url)))
        return miniEvidenceEntry(
            evidence_id=f"{basic.round_index}-{hash_value[-6:]}",
            # sab_id=basic.sab_id,
            round_index=basic.round_index,
            tool=basic.tool_name,
            sub_question=basic.sub_question,
            url=evidence.url,
            title=evidence.title or basic.tool_name,
            authors=None,
            year=None,
            doi=None,
            brief=brief,
            quote=evidence.text[:6000],
            fetched_at="",
            hash_value=hash_value,
            relevance=0.0,
            credibility=0.0,
            tags=[],
            reference_chain_parent=None,
            raw_content={"fallback": True},
            reference_key=None,
            citation_label=None,
            full_abstract=None,
            key_sentences=[],
            citation_reason=None,
            detailed_findings=None,
        )

    async def summarise(self, question: str, e_entry: miniEvidenceEntry) -> str:
        evidence_text = e_entry.quote[:30000]
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You synthesise raw tool outputs into concise research notes."
                    " Keep the style aligned with the user's instructions."
                    " Cite sources if URLs are available.",
                ),
                (
                    "human",
                    "Objective: {objective}\n\nEvidence:\n{evidence}",
                ),
            ]
        )
        if not evidence_text:
            return ""
        response = await (self.prompt | self.llm).ainvoke(
            {
                "objective": question,
                "evidence": evidence_text,
            }
        )
        return response.content.strip()

