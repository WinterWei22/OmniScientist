from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class AnswerSemanticContract:
    property_name: str
    context: str
    answer_type: str
    accepted_method_classes: tuple[str, ...]
    prohibited_sole_sources: tuple[str, ...]
    requires_explicit_semantic_evidence: bool = True
    version: str = "omniagent.answer_validation.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_answer_semantic_contract(goal: str) -> AnswerSemanticContract | None:
    """Infer only deterministic property contracts from the public task prompt."""
    text = " ".join(goal.casefold().replace("-", " ").split())
    if re.search(r"\bnet charge\b", text):
        context = (
            "protein_bound"
            if "protein bound" in text or "bound inhibitor" in text
            else "molecular_state"
        )
        return AnswerSemanticContract(
            property_name="molecular_net_charge",
            context=context,
            answer_type="signed_integer",
            accepted_method_classes=(
                ("context_aware_computation",)
                if context == "protein_bound"
                else ("context_aware_computation", "context_aware_reference")
            ),
            prohibited_sole_sources=(
                "pdbx_formal_charge",
                "chem_comp.formal_charge",
            ),
        )
    return None


def assess_result_semantics(
    contract: AnswerSemanticContract | None,
    result: Any,
) -> dict[str, Any] | None:
    if contract is None:
        return None

    assertions = list(_semantic_assertions(getattr(result, "output", None)))
    assertions.extend(_semantic_assertions(getattr(result, "observations", None)))
    accepted: list[dict[str, Any]] = []
    rejected_reasons: list[str] = []
    for assertion in assertions:
        assessment, reason = _assess_assertion(contract, assertion)
        if assessment is not None:
            accepted.append(assessment)
        elif reason:
            rejected_reasons.append(reason)

    return {
        "contract": contract.to_dict(),
        "passed": bool(accepted),
        "values": list(dict.fromkeys(item["value"] for item in accepted)),
        "accepted_assertions": accepted,
        "assertion_count": len(assertions),
        "reasons": list(dict.fromkeys(rejected_reasons))[:8],
    }


def validate_final_answer_semantics(
    contract: AnswerSemanticContract | None,
    payload: Any,
    claims: Iterable[Any],
    evidence: dict[str, Any],
    supporting_evidence_ids: Iterable[str] = (),
) -> dict[str, Any]:
    if contract is None:
        return {
            "required": False,
            "passed": True,
            "blockers": [],
            "evidence_ids": [],
        }
    if not isinstance(payload, dict):
        return _failed("Semantic answer validation requires a JSON object output.")

    answer = payload.get("answer")
    answer_field = "answer"
    if answer is None:
        aliases = {
            _normal(contract.property_name),
            "netcharge",
            "molecularnetcharge",
        }
        for key, value in payload.items():
            if _normal(key) in aliases:
                answer = value
                answer_field = str(key)
                break
    parsed_answer = _parse_answer(answer, contract.answer_type)
    if parsed_answer is None:
        return _failed(
            f"Final answer does not satisfy semantic answer type {contract.answer_type}."
        )

    linked_ids = {
        evidence_id
        for claim in claims
        for evidence_id in getattr(claim, "evidence_ids", ())
    }
    linked_ids.update(
        str(evidence_id).strip()
        for evidence_id in supporting_evidence_ids
        if str(evidence_id).strip()
    )
    matching: list[str] = []
    for evidence_id in linked_ids:
        record = evidence.get(evidence_id)
        if record is None:
            continue
        provenance = getattr(record, "provenance", {})
        assessment = (
            provenance.get("semantic_validation")
            if isinstance(provenance, dict)
            else None
        )
        derived_support = _derived_evidence_supports(
            contract,
            payload,
            parsed_answer,
            evidence_id,
            record,
        )
        values = assessment.get("values", []) if isinstance(assessment, dict) else []
        if (
            isinstance(assessment, dict)
            and assessment.get("passed") is True
            and parsed_answer in values
        ) or derived_support:
            matching.append(evidence_id)

    if not matching:
        return _failed(
            "The final answer lacks verifier-admitted evidence for the requested "
            f"{contract.property_name} in context {contract.context}; metadata fields "
            "with a similar name are not semantically sufficient.",
            linked_ids=sorted(linked_ids),
        )
    return {
        "required": True,
        "passed": True,
        "blockers": [],
        "evidence_ids": matching,
        "property_name": contract.property_name,
        "context": contract.context,
        "answer_field": answer_field,
        "answer_value": parsed_answer,
    }


def synthesize_grounded_final_output(
    contract: AnswerSemanticContract | None,
    payload: Any,
    goal: str,
    supporting_evidence_ids: Iterable[str],
    evidence: dict[str, Any],
    analysis: Any = None,
) -> dict[str, Any] | None:
    """Complete a final payload only from a directly verifiable raw record.

    This is deliberately narrow. It does not infer chemistry from prose or
    turn an arbitrary metadata field into evidence. A deposited PDB charge is
    not sufficient for a protein-bound condition-sensitive charge, so that case
    must be produced by an explicit condition-aware computation.
    """
    if contract is None or not isinstance(payload, dict):
        return None
    if (
        _normal(contract.property_name) != "molecularnetcharge"
        or _normal(contract.context) != "proteinbound"
    ):
        return None
    # Do not turn a deposited CCD descriptor into a condition-aware result.
    # The domain workflow must provide explicit protonation and atom-level
    # charge evidence instead.
    return None
    answer = payload.get("answer")
    if answer is None:
        for key, value in payload.items():
            if _normal(key) in {"molecularnetcharge", "netcharge"}:
                answer = value
                break
    parsed_answer = _parse_answer(answer, contract.answer_type)
    if parsed_answer is None:
        return None

    analysis_text = " ".join(
        [
            str(getattr(analysis, "summary", "") or ""),
            *(
                str(item)
                for item in getattr(analysis, "observations", [])
                if item is not None
            ),
        ]
    ).casefold()
    pdb_match = re.search(r"\b[0-9][a-z0-9]{3}\b", goal.casefold())
    target_pdb = pdb_match.group(0) if pdb_match else ""
    target_id = _first_text_value(
        payload,
        ("ligand_id", "component_id", "entity_id"),
    )
    candidates: list[tuple[int, str, int, dict[str, Any]]] = []
    for evidence_id in dict.fromkeys(str(item) for item in supporting_evidence_ids):
        record = evidence.get(evidence_id)
        raw_payload = getattr(record, "payload", None)
        raw_text = str(raw_payload).casefold()
        if target_pdb and target_pdb not in raw_text and target_pdb not in analysis_text:
            continue
        for item in _structured_charge_records(raw_payload):
            if item["value"] != parsed_answer:
                continue
            component_id = item["component_id"]
            score = 0
            if target_id and component_id.casefold() == target_id.casefold():
                score += 100
            if component_id and component_id.casefold() in analysis_text:
                score += 15
            if len(item["name"]) >= 20:
                score += 10
            if "c" in item["formula"].casefold() and "n" in item["formula"].casefold():
                score += 8
            if component_id.casefold() in {
                "gdp",
                "mg",
                "gol",
                "edo",
                "hoh",
                "wat",
            }:
                score -= 100
            candidates.append((score, evidence_id, item["value"], item))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    best = candidates[0]
    if best[0] <= 0 or (len(candidates) > 1 and best[0] == candidates[1][0]):
        return None
    _, evidence_id, value, item = best
    result = dict(payload)
    result.setdefault("ligand_id", item["component_id"])
    pdb_id = _first_text_value(payload, ("pdb_id", "pdbid"))
    if pdb_id:
        result.setdefault("pdb_id", pdb_id)
    result.setdefault("evidence_id", evidence_id)
    result["semantic_evidence"] = {
        "property": contract.property_name,
        "value": value,
        "context": contract.context,
        "method": (
            "Resolved the drug-like non-polymer component in the referenced PDB "
            "entry and used its deposited Chemical Component Dictionary charge."
        ),
        "method_class": "context_aware_reference",
        "entity_id": item["component_id"],
        "source_fields": [
            "entry.rcsb_id",
            "chem_comp.id",
            "chem_comp.name",
            "chem_comp.formula",
            "chem_comp.pdbx_formal_charge",
        ],
        "evidence_id_ref": evidence_id,
    }
    return result


def _derived_evidence_supports(
    contract: AnswerSemanticContract,
    payload: Any,
    parsed_answer: int | float | str,
    evidence_id: str,
    record: Any,
) -> bool:
    for assertion in _semantic_assertions(payload):
        assessment, _ = _assess_assertion(contract, assertion)
        if assessment is None or assessment["value"] != parsed_answer:
            continue
        if str(assertion.get("evidence_id_ref", "")).strip() not in {"", evidence_id}:
            continue
        target_id = _first_text_value(
            assertion,
            ("entity_id", "ligand_id", "component_id"),
        ).casefold()
        records = _structured_charge_records(getattr(record, "payload", None))
        if any(
            item["value"] == parsed_answer
            and (not target_id or item["component_id"].casefold() == target_id)
            for item in records
        ):
            return True
    return False


def _structured_charge_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            charge = None
            for key in ("pdbx_formal_charge", "formal_charge"):
                if key in node:
                    charge = _parse_answer(node.get(key), "signed_integer")
                    if charge is not None:
                        break
            if charge is not None:
                component_id = _first_text_value(
                    node,
                    ("id", "component_id", "comp_id", "rcsb_id"),
                )
                name = _first_text_value(node, ("name", "chemical_name", "description"))
                formula = _first_text_value(node, ("formula", "chemical_formula"))
                key = (component_id.casefold(), charge, name.casefold())
                if key not in seen:
                    seen.add(key)
                    records.append(
                        {
                            "component_id": component_id,
                            "name": name,
                            "formula": formula,
                            "value": charge,
                        }
                    )
            for child in node.values():
                walk(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

    walk(value)
    return records


def _first_text_value(value: Any, keys: tuple[str, ...]) -> str:
    if not isinstance(value, dict):
        return ""
    for key in keys:
        candidate = value.get(key)
        if candidate not in (None, "") and not isinstance(candidate, (dict, list, tuple)):
            return str(candidate).strip()
    return ""


def _semantic_assertions(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if "property" in value and "value" in value:
            yield value
        for child in value.values():
            yield from _semantic_assertions(child)
    elif isinstance(value, list | tuple):
        for child in value:
            yield from _semantic_assertions(child)


def _assess_assertion(
    contract: AnswerSemanticContract,
    assertion: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    property_name = _normal(assertion.get("property"))
    if property_name not in {
        _normal(contract.property_name),
        "netcharge",
        "molecularnetcharge",
    }:
        return None, "semantic assertion targets a different property"

    value = _parse_answer(assertion.get("value"), contract.answer_type)
    if value is None:
        return None, "semantic assertion has an invalid value"
    if _normal(assertion.get("context")) != _normal(contract.context):
        return None, "semantic assertion does not match the requested context"

    method = str(assertion.get("method", "")).strip()
    method_class = _normal(assertion.get("method_class"))
    accepted_classes = {_normal(item) for item in contract.accepted_method_classes}
    if not method or method_class not in accepted_classes:
        return None, "semantic assertion lacks an accepted context-aware method"

    sources = assertion.get("source_fields", assertion.get("sources", []))
    if isinstance(sources, str):
        sources = [sources]
    if not isinstance(sources, list) or not sources:
        return None, "semantic assertion does not identify its source fields"
    normalized_sources = {_normal(item) for item in sources if str(item).strip()}
    prohibited = {_normal(item) for item in contract.prohibited_sole_sources}
    if normalized_sources and all(
        any(source == item or source.endswith(item) for item in prohibited)
        for source in normalized_sources
    ):
        return None, "a context-free metadata charge is the assertion's sole source"

    return (
        {
            "property": contract.property_name,
            "context": contract.context,
            "value": value,
            "method": method[:500],
            "method_class": assertion.get("method_class"),
            "source_fields": [str(item) for item in sources[:12]],
        },
        "",
    )


def _parse_answer(value: Any, answer_type: str) -> int | float | str | None:
    if answer_type == "signed_integer":
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str) and re.fullmatch(r"[+-]\d+", value.strip()):
            return int(value)
        return None
    return str(value).strip() or None


def _normal(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _failed(message: str, *, linked_ids: list[str] | None = None) -> dict[str, Any]:
    return {
        "required": True,
        "passed": False,
        "blockers": [message],
        "evidence_ids": [],
        "linked_evidence_ids": linked_ids or [],
    }
