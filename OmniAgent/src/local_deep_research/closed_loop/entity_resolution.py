from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .scientific_state import CanonicalEntityRecord


class EntityResolutionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ProteinEntityResolution:
    record: CanonicalEntityRecord
    corrections: dict[str, str]


class ProteinEntityResolver:
    """Resolve a planner-proposed protein name against UniProtKB."""

    _IDENTIFIER_KEYS = frozenset(
        {
            "alternative_names",
            "aliases",
            "gene_name",
            "gene_symbol",
            "uniprot_id",
            "uniprot_accession",
            "accession",
            "cd_antigen",
            "cd_antigen_name",
        }
    )

    def __init__(
        self,
        *,
        base_url: str = "https://rest.uniprot.org",
        timeout_seconds: float = 15.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def resolve(
        self,
        query_name: str,
        *,
        organism: str = "",
        entity_context: dict[str, Any] | None = None,
    ) -> ProteinEntityResolution:
        query = str(query_name or "").strip()
        if not query:
            raise EntityResolutionError("protein entity name is empty")
        payload, resolved_query = await asyncio.to_thread(
            self._fetch_first_nonempty,
            query,
            self._clean_optional(organism),
        )
        record = self._select_record(
            query,
            payload,
            resolved_query=resolved_query,
            organism=self._clean_optional(organism),
        )
        return ProteinEntityResolution(
            record=record,
            corrections=self._derive_corrections(record, entity_context or {}),
        )

    def reuse(
        self,
        record: CanonicalEntityRecord,
        entity_context: dict[str, Any],
    ) -> ProteinEntityResolution:
        return ProteinEntityResolution(
            record=record,
            corrections=self._derive_corrections(record, entity_context),
        )

    def _fetch_first_nonempty(
        self,
        query_name: str,
        organism: str,
    ) -> tuple[dict[str, Any], str]:
        errors: list[str] = []
        for query in self._query_variants(query_name):
            try:
                payload = self._fetch(query, organism)
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                continue
            if isinstance(payload.get("results"), list) and payload["results"]:
                return payload, query
        if errors:
            raise EntityResolutionError(
                "UniProtKB lookup failed: " + "; ".join(errors[-2:])
            )
        raise EntityResolutionError(
            f"UniProtKB returned no protein record for {query_name!r}"
        )

    def _fetch(self, query_name: str, organism: str) -> dict[str, Any]:
        query = query_name
        if organism:
            if organism.isdigit():
                query += f" AND organism_id:{organism}"
            else:
                escaped = organism.replace('"', "")
                query += f' AND organism_name:"{escaped}"'
        url = f"{self.base_url}/uniprotkb/search?" + urlencode(
            {"query": query, "format": "json", "size": 25}
        )
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "OmniAgent-Agent/0.1 entity-resolution",
            },
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise EntityResolutionError("UniProtKB response is not a JSON object")
        return payload

    @classmethod
    def _select_record(
        cls,
        query_name: str,
        payload: dict[str, Any],
        *,
        resolved_query: str = "",
        organism: str = "",
    ) -> CanonicalEntityRecord:
        entries = [
            item for item in payload.get("results", []) if isinstance(item, dict)
        ]
        candidates: list[tuple[int, CanonicalEntityRecord]] = []
        for entry in entries:
            record = cls._record_from_entry(query_name, entry)
            if record is None:
                continue
            candidates.append(
                (
                    cls._score_record(
                        record,
                        query_name=query_name,
                        resolved_query=resolved_query,
                        organism=organism,
                        reviewed="reviewed"
                        in str(entry.get("entryType", "")).casefold(),
                    ),
                    record,
                )
            )
        if not candidates:
            raise EntityResolutionError(
                f"UniProtKB returned no usable protein record for {query_name!r}"
            )
        candidates.sort(
            key=lambda item: (
                item[0],
                item[1].tax_id == 9606,
                item[1].uniprot_accession,
            ),
            reverse=True,
        )
        best_score, best = candidates[0]
        if best_score < 80:
            raise EntityResolutionError(
                f"UniProtKB records do not contain an exact alias for {query_name!r}"
            )
        competing_genes = {
            cls.normalize_identifier(record.gene_symbol)
            for score, record in candidates
            if score >= best_score - 3 and record.gene_symbol
        }
        if len(competing_genes) > 1:
            raise EntityResolutionError(
                f"UniProtKB lookup for {query_name!r} is ambiguous across genes"
            )
        return best

    @classmethod
    def _record_from_entry(
        cls,
        query_name: str,
        entry: dict[str, Any],
    ) -> CanonicalEntityRecord | None:
        accession = str(entry.get("primaryAccession", "")).strip()
        if not accession:
            return None
        protein_description = entry.get("proteinDescription", {})
        genes = entry.get("genes", [])
        protein_names = cls._nested_values(protein_description)
        gene_names = cls._nested_values(genes)
        preferred_name = cls._preferred_name(protein_description)
        gene_symbol = gene_names[0] if gene_names else ""
        organism_value = entry.get("organism", {})
        organism_data = organism_value if isinstance(organism_value, dict) else {}
        aliases = tuple(
            dict.fromkeys(
                item
                for item in (
                    query_name,
                    gene_symbol,
                    *gene_names,
                    *protein_names,
                    str(entry.get("uniProtkbId", "")).strip(),
                )
                if item
            )
        )
        tax_id_value = organism_data.get("taxonId")
        tax_id = (
            int(tax_id_value)
            if isinstance(tax_id_value, int | str) and str(tax_id_value).isdigit()
            else None
        )
        return CanonicalEntityRecord(
            entity_id=f"uniprot:{accession}",
            entity_type="protein",
            query_name=query_name,
            preferred_name=preferred_name or gene_symbol or query_name,
            gene_symbol=gene_symbol,
            aliases=aliases[:40],
            uniprot_accession=accession,
            organism=str(organism_data.get("scientificName", "")).strip(),
            tax_id=tax_id,
            source="UniProtKB",
            source_url=f"https://rest.uniprot.org/uniprotkb/{accession}",
        )

    @classmethod
    def _score_record(
        cls,
        record: CanonicalEntityRecord,
        *,
        query_name: str,
        resolved_query: str,
        organism: str,
        reviewed: bool,
    ) -> int:
        query = cls.normalize_identifier(query_name)
        resolved = cls.normalize_identifier(resolved_query)
        aliases = {cls.normalize_identifier(item) for item in record.aliases}
        score = 0
        if query and query in aliases:
            score += 100
        elif resolved and resolved in aliases:
            score += 100
        elif any(query and query in item for item in aliases):
            score += 25
        if query == cls.normalize_identifier(record.gene_symbol):
            score += 30
        if reviewed:
            score += 5
        if organism and cls.normalize_identifier(organism) == cls.normalize_identifier(
            record.organism
        ):
            score += 15
        elif not organism and record.tax_id == 9606:
            # A human entry is only the canonical reference when the task did not
            # constrain species; it must not become an execution species filter.
            score += 2
        if (
            "receptor" in record.preferred_name.casefold()
            and "receptor" not in query_name.casefold()
        ):
            score -= 40
        return score

    @classmethod
    def _derive_corrections(
        cls,
        record: CanonicalEntityRecord,
        entity_context: dict[str, Any],
    ) -> dict[str, str]:
        accepted = {
            cls.normalize_identifier(item)
            for item in (
                record.query_name,
                record.preferred_name,
                record.gene_symbol,
                record.uniprot_accession,
                *record.aliases,
            )
            if item
        }
        corrections: dict[str, str] = {}
        for key, raw in entity_context.items():
            if str(key).strip().casefold() not in cls._IDENTIFIER_KEYS:
                continue
            for proposed in cls._flatten_identifiers(raw):
                normalized = cls.normalize_identifier(proposed)
                if not normalized or normalized in accepted:
                    continue
                replacement = cls._replacement_for(proposed, key, record)
                if replacement and cls.normalize_identifier(replacement) != normalized:
                    corrections[proposed] = replacement
        return corrections

    @classmethod
    def canonical_context(
        cls,
        original: dict[str, Any],
        resolution: ProteinEntityResolution,
    ) -> dict[str, Any]:
        record = resolution.record
        context = dict(original)
        context.update(
            {
                "protein_name": record.query_name,
                "canonical_preferred_name": record.preferred_name,
                "gene_symbol": record.gene_symbol,
                "alternative_names": list(record.aliases),
                "uniprot_accession": record.uniprot_accession,
                "canonical_entity_id": record.entity_id,
                "entity_resolution_source": record.source,
                "reference_organism": record.organism,
            }
        )
        return context

    @classmethod
    def apply_corrections(cls, value: Any, corrections: dict[str, str]) -> Any:
        if isinstance(value, str):
            text = value
            for rejected, replacement in sorted(
                corrections.items(), key=lambda item: len(item[0]), reverse=True
            ):
                text = re.sub(
                    rf"(?<![A-Za-z0-9]){re.escape(rejected)}(?![A-Za-z0-9])",
                    replacement,
                    text,
                    flags=re.IGNORECASE,
                )
            return text
        if isinstance(value, list):
            return [cls.apply_corrections(item, corrections) for item in value]
        if isinstance(value, tuple):
            return tuple(cls.apply_corrections(item, corrections) for item in value)
        if isinstance(value, dict):
            return {
                key: cls.apply_corrections(item, corrections)
                for key, item in value.items()
            }
        return value

    @classmethod
    def find_existing(
        cls,
        records: dict[str, CanonicalEntityRecord],
        query_name: str,
    ) -> CanonicalEntityRecord | None:
        query = cls.normalize_identifier(query_name)
        for record in records.values():
            identifiers = (
                record.query_name,
                record.preferred_name,
                record.gene_symbol,
                record.uniprot_accession,
                *record.aliases,
            )
            if query and query in {
                cls.normalize_identifier(item) for item in identifiers
            }:
                return record
        return None

    @staticmethod
    def normalize_identifier(value: Any) -> str:
        return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())

    @staticmethod
    def _query_variants(value: str) -> tuple[str, ...]:
        original = " ".join(value.split())
        compact = re.sub(r"[\s_-]+", "", original)
        return tuple(dict.fromkeys(item for item in (original, compact) if item))

    @staticmethod
    def _clean_optional(value: Any) -> str:
        text = str(value or "").strip()
        return "" if text.casefold() in {"none", "null", "unknown", "n/a"} else text

    @staticmethod
    def _nested_values(value: Any) -> list[str]:
        found: list[str] = []

        def visit(item: Any) -> None:
            if isinstance(item, dict):
                raw = item.get("value")
                if isinstance(raw, str) and raw.strip():
                    found.append(raw.strip())
                for child in item.values():
                    visit(child)
            elif isinstance(item, list):
                for child in item:
                    visit(child)

        visit(value)
        return list(dict.fromkeys(found))

    @staticmethod
    def _preferred_name(value: Any) -> str:
        if not isinstance(value, dict):
            return ""
        recommended = value.get("recommendedName", {})
        if isinstance(recommended, dict):
            full_name = recommended.get("fullName", {})
            if isinstance(full_name, dict):
                name = str(full_name.get("value", "")).strip()
                if name:
                    return name
        names = ProteinEntityResolver._nested_values(value)
        return names[0] if names else ""

    @staticmethod
    def _flatten_identifiers(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list | tuple | set):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    @staticmethod
    def _replacement_for(
        proposed: str,
        key: Any,
        record: CanonicalEntityRecord,
    ) -> str:
        key_name = str(key).casefold()
        if re.fullmatch(r"CD\s*[-_]?\s*\d+", proposed, flags=re.IGNORECASE):
            return next(
                (
                    alias
                    for alias in record.aliases
                    if re.fullmatch(r"CD\s*[-_]?\s*\d+", alias, flags=re.IGNORECASE)
                ),
                record.gene_symbol,
            )
        if "accession" in key_name or key_name == "uniprot_id":
            return record.uniprot_accession
        if "gene" in key_name or re.fullmatch(r"[A-Za-z][A-Za-z0-9-]{1,15}", proposed):
            return record.gene_symbol
        return record.query_name
