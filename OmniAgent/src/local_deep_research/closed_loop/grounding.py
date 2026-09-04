from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from typing import Any

from .result_payload import material_result_values
from .scientific_state import ClaimRecord, ClaimStatus


_ENVELOPE_KEYS = frozenset(
    {
        "data",
        "evidence",
        "output",
        "payload",
        "response",
        "result",
        "results",
    }
)
_PATH_TOKEN = re.compile(r"([^\.\[\]]+)|\[(\d+)\]")


@dataclass(frozen=True, slots=True)
class GroundingMatch:
    claim_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _EvidenceFact:
    path: str
    value: str | bool | int | float
    evidence_id: str


class GroundingIndex:
    """Typed final-output grounding over verifier-admitted claim/evidence links."""

    def __init__(
        self,
        claims: dict[str, ClaimRecord],
        evidence: dict[str, Any],
    ) -> None:
        self._claims: list[ClaimRecord] = []
        self._facts_by_evidence: dict[str, tuple[_EvidenceFact, ...]] = {}
        for claim in claims.values():
            status = getattr(claim.status, "value", claim.status)
            if status in {
                ClaimStatus.RETRACTED.value,
                ClaimStatus.CONTRADICTED.value,
            }:
                continue
            evidence_ids = tuple(str(item) for item in claim.evidence_ids)
            if not evidence_ids or not set(evidence_ids).issubset(evidence):
                continue
            self._claims.append(claim)
            for evidence_id in evidence_ids:
                if evidence_id in self._facts_by_evidence:
                    continue
                record = evidence[evidence_id]
                payload = (
                    record.get("payload")
                    if isinstance(record, dict)
                    else getattr(record, "payload", None)
                )
                self._facts_by_evidence[evidence_id] = tuple(
                    self._evidence_facts(evidence_id, payload)
                )

    def match(
        self,
        path: str,
        value: str | bool | int | float,
    ) -> GroundingMatch | None:
        claim_ids: list[str] = []
        evidence_ids: list[str] = []
        for claim in self._claims:
            linked_ids = tuple(str(item) for item in claim.evidence_ids)
            matched_ids = [
                evidence_id
                for evidence_id in linked_ids
                if any(
                    self._paths_compatible(path, fact.path)
                    and self._values_equal(value, fact.value)
                    for fact in self._facts_by_evidence.get(evidence_id, ())
                )
            ]
            if not matched_ids and not self._claim_supports(claim.statement, path, value):
                continue
            claim_ids.append(claim.claim_id)
            evidence_ids.extend(matched_ids or linked_ids)
        if not claim_ids:
            return None
        return GroundingMatch(
            claim_ids=tuple(dict.fromkeys(claim_ids)),
            evidence_ids=tuple(dict.fromkeys(evidence_ids)),
        )

    def output_matches(
        self,
        payload: Any,
    ) -> tuple[list[tuple[str, GroundingMatch]], list[str]]:
        values = material_result_values(payload, limit=None)
        if not values:
            return [], ["Final output has no material result fields to ground."]
        matches: list[tuple[str, GroundingMatch]] = []
        blockers: list[str] = []
        for path, value in values:
            match = self.match(path, value)
            if match is None:
                blockers.append(
                    "Final output field is not represented by an "
                    f"evidence-linked claim: {path}"
                )
            else:
                matches.append((path, match))
        return matches, blockers

    @classmethod
    def _evidence_facts(
        cls,
        evidence_id: str,
        payload: Any,
    ) -> list[_EvidenceFact]:
        facts = [
            _EvidenceFact(path=path, value=value, evidence_id=evidence_id)
            for path, value in material_result_values(payload, limit=None)
        ]
        if not isinstance(payload, dict):
            return facts
        projection = payload.get("projection")
        if not isinstance(projection, list):
            return facts
        for item in projection:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path", "")).strip()
            value = item.get("value")
            if path and isinstance(value, str | bool | int | float):
                facts.append(_EvidenceFact(path=path, value=value, evidence_id=evidence_id))
        return facts

    @staticmethod
    def _claim_supports(
        statement: str,
        path: str,
        value: str | bool | int | float,
    ) -> bool:
        text = statement.strip()
        rendered = GroundingIndex._render_value(value)
        if not text or not rendered:
            return False
        if text.casefold() == rendered.casefold():
            return True
        expected = f"Final output {path}: {rendered}"
        if text.casefold() == expected.casefold():
            return True
        return False

    @staticmethod
    def _paths_compatible(left: str, right: str) -> bool:
        left_tokens = GroundingIndex._path_tokens(left)
        right_tokens = GroundingIndex._path_tokens(right)
        if not left_tokens or not right_tokens:
            return False
        return (
            left_tokens == right_tokens
            or len(left_tokens) < len(right_tokens)
            and right_tokens[-len(left_tokens) :] == left_tokens
            or len(right_tokens) < len(left_tokens)
            and left_tokens[-len(right_tokens) :] == right_tokens
        )

    @staticmethod
    def _path_tokens(path: str) -> tuple[str, ...]:
        tokens = tuple(
            key.casefold() if key else f"[{index}]"
            for key, index in _PATH_TOKEN.findall(path)
        )
        offset = 0
        while offset < len(tokens) and tokens[offset] in _ENVELOPE_KEYS:
            offset += 1
            while offset < len(tokens) and tokens[offset].startswith("["):
                offset += 1
        return tokens[offset:]

    @staticmethod
    def _render_value(value: str | bool | int | float) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    @staticmethod
    def _values_equal(
        left: str | bool | int | float,
        right: str | bool | int | float,
    ) -> bool:
        if isinstance(left, bool) or isinstance(right, bool):
            return isinstance(left, bool) and isinstance(right, bool) and left is right
        if isinstance(left, int | float) and isinstance(right, int | float):
            try:
                return Decimal(str(left)) == Decimal(str(right))
            except InvalidOperation:
                return False
        if isinstance(left, str) and isinstance(right, str):
            return left.strip() == right.strip()
        return False
