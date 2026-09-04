from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from .scientific_state import ClaimStatus

if TYPE_CHECKING:
    from .contracts import ResearchState


_TERM_PATTERN = re.compile(r"[A-Za-z0-9_./:+-]+|[\u3400-\u9fff]")


def estimate_tokens(value: Any) -> int:
    """Conservative tokenizer-free estimate for mixed English/Chinese JSON."""
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    )
    ascii_count = sum(ord(char) < 128 for char in text)
    non_ascii_count = len(text) - ascii_count
    return max(1, math.ceil(ascii_count / 4) + non_ascii_count)


class WorkingMemoryProjector:
    """Project durable scientific state into a bounded, task-relevant model view."""

    def __init__(self, token_budget: int = 2400) -> None:
        if token_budget < 256:
            raise ValueError("working-memory token budget must be at least 256")
        self.token_budget = token_budget

    def project(
        self,
        state: ResearchState,
        *,
        purpose: str,
        focus: str = "",
    ) -> dict[str, Any]:
        scientific = state.scientific_state
        focus_terms = self._terms(" ".join((state.goal, focus)))
        output_config = state.task_manifest.get("output_config", {})
        parameters = state.task_manifest.get("task_parameters", {})
        required_feedback = state.last_critique

        view: dict[str, Any] = {
            "run": {
                "run_id": state.run_id,
                "task_id": scientific.task_id,
                "purpose": purpose,
                "iteration": len(state.iterations),
                "phase": state.phase.value,
            },
            "objective": {
                "goal": self._compact_text(state.goal, 1800),
                "focus": self._compact_text(focus, 1000),
                "constraints": [self._compact_text(item, 500) for item in state.constraints[:12]],
            },
            "task_contract": {
                "output_config": (
                    {
                        key: output_config[key]
                        for key in ("file_path", "format", "required")
                        if key in output_config
                    }
                    if isinstance(output_config, dict)
                    else {}
                ),
                "benchmark": (
                    parameters.get("benchmark", "")
                    if isinstance(parameters, dict)
                    else ""
                ),
            },
            "progress": {
                "state_version": scientific.state_version,
                "best_score": state.best_score,
                "stalled_iterations": state.stalled_iterations,
                "a1_call_count": state.a1_call_count,
            },
            "canonical_entities": [
                {
                    "entity_id": item.entity_id,
                    "entity_type": item.entity_type,
                    "query_name": item.query_name,
                    "preferred_name": item.preferred_name,
                    "gene_symbol": item.gene_symbol,
                    "aliases": list(item.aliases[:12]),
                    "uniprot_accession": item.uniprot_accession,
                    "reference_organism": item.organism,
                    "source": item.source,
                    "source_url": item.source_url,
                }
                for item in scientific.canonical_entities.values()
            ],
            "entity_corrections": dict(scientific.entity_corrections),
            "required_feedback": (
                {
                    "feedback_id": required_feedback.feedback_id,
                    "decision": required_feedback.decision.value,
                    "required_changes": required_feedback.required_changes[:8],
                    "next_experiment": self._compact_text(
                        required_feedback.next_experiment, 800
                    ),
                    "evidence_gaps": required_feedback.evidence_gaps[:8],
                }
                if required_feedback
                else None
            ),
            "recent_changes": {
                "attempts": [],
                "unresolved_questions": [
                    self._compact_text(item, 500)
                    for item in scientific.unresolved_questions[-8:]
                ],
                "conflicts": [
                    self._compact_text(item, 500) for item in scientific.conflicts[-8:]
                ],
                "failed_directions": [
                    self._compact_text(item, 500)
                    for item in scientific.failed_directions[-6:]
                ],
            },
            "claims": [],
            "evidence": [],
            "hypotheses": [],
            "prior_evaluations": [],
            "artifact_refs": [],
            "actions": [],
        }

        omitted = {
            "attempts": 0,
            "claims": 0,
            "evidence": 0,
            "hypotheses": 0,
            "prior_evaluations": 0,
            "artifact_refs": 0,
            "actions": 0,
        }
        reserve = 96

        claims = list(scientific.claims.values())
        ordered_claims = sorted(
            enumerate(claims),
            key=lambda pair: (
                pair[1].status is ClaimStatus.VERIFIED,
                self._relevance(pair[1].statement, focus_terms),
                pair[0],
            ),
            reverse=True,
        )
        selected_evidence_ids: set[str] = set()
        evidence_reserve = max(reserve, self.token_budget // 3)
        for _, item in ordered_claims:
            value = {
                "claim_id": item.claim_id,
                "statement": self._compact_text(item.statement, 1000),
                "status": item.status.value,
                "evidence_ids": list(item.evidence_ids),
            }
            if self._append_if_fits(
                view["claims"], value, view, evidence_reserve
            ):
                selected_evidence_ids.update(item.evidence_ids)
            else:
                omitted["claims"] += 1

        evidence = list(scientific.evidence.values())
        ordered_evidence = sorted(
            enumerate(evidence),
            key=lambda pair: (
                pair[1].evidence_id in selected_evidence_ids,
                self._relevance(pair[1].summary, focus_terms),
                pair[0],
            ),
            reverse=True,
        )
        selected_artifacts: list[str] = []
        for _, item in ordered_evidence:
            value = {
                "evidence_id": item.evidence_id,
                "type": item.evidence_type,
                "summary": self._compact_text(item.summary, 1200),
                "backend": item.source_backend,
                "capability_id": item.source_capability_id,
                "artifact_refs": list(item.artifact_refs),
            }
            if self._append_if_fits(view["evidence"], value, view, reserve):
                selected_artifacts.extend(item.artifact_refs)
            else:
                omitted["evidence"] += 1

        hypotheses = list(scientific.hypotheses.values())
        ordered_hypotheses = sorted(
            enumerate(hypotheses),
            key=lambda pair: (
                self._relevance(pair[1].statement, focus_terms),
                pair[0],
            ),
            reverse=True,
        )
        for _, item in ordered_hypotheses:
            value = {
                "hypothesis_id": item.hypothesis_id,
                "statement": self._compact_text(item.statement, 900),
                "status": item.status.value,
                "uncertainty": self._compact_text(item.uncertainty, 600),
                "next_test": self._compact_text(item.next_test, 600),
                "version": item.version,
            }
            if not self._append_if_fits(view["hypotheses"], value, view, reserve):
                omitted["hypotheses"] += 1

        attempts = list(scientific.attempts.values())
        omitted["attempts"] += max(0, len(attempts) - 4)
        for item in reversed(attempts[-4:]):
            value = asdict(item)
            value["status"] = item.status.value
            if not self._append_if_fits(
                view["recent_changes"]["attempts"], value, view, reserve
            ):
                omitted["attempts"] += 1

        for record in reversed(state.iterations):
            value = {
                "iteration": record.iteration,
                "evaluation_id": record.evaluation.evaluation_id,
                "score": record.evaluation.score,
                "metrics": record.evaluation.metrics,
                "failed_criteria": record.evaluation.failed_criteria[:8],
                "errors": record.evaluation.errors[:5],
                "feedback_id": record.critique.feedback_id,
                "required_changes": record.critique.required_changes[:8],
            }
            if not self._append_if_fits(
                view["prior_evaluations"], value, view, reserve
            ):
                omitted["prior_evaluations"] += 1

        artifact_candidates = list(
            dict.fromkeys(selected_artifacts + scientific.artifact_refs)
        )
        for item in reversed(artifact_candidates):
            if not self._append_if_fits(
                view["artifact_refs"], self._compact_text(item, 900), view, reserve
            ):
                omitted["artifact_refs"] += 1

        ledger = getattr(state, "action_ledger", None)
        action_records = list(ledger.records.values()) if hasattr(ledger, "records") else []
        for item in reversed(action_records[-8:]):
            value = {
                "action_id": item.action_id,
                "status": item.status,
                "backend": item.backend,
                "capability_id": item.capability_id,
                "request_id": item.request_id,
                "external_task_id": item.external_task_id,
                "last_error": self._compact_text(item.last_error, 500),
            }
            if not self._append_if_fits(view["actions"], value, view, reserve):
                omitted["actions"] += 1

        view["memory_metadata"] = {
            "token_budget": self.token_budget,
            "estimated_tokens": estimate_tokens(view),
            "omitted": omitted,
        }
        self._trim_to_budget(view)
        view["memory_metadata"]["estimated_tokens"] = estimate_tokens(view)
        return view

    def _append_if_fits(
        self,
        target: list[Any],
        value: Any,
        view: dict[str, Any],
        reserve: int,
    ) -> bool:
        target.append(value)
        if estimate_tokens(view) + reserve <= self.token_budget:
            return True
        target.pop()
        return False

    def _trim_to_budget(self, view: dict[str, Any]) -> None:
        removable = (
            view["actions"],
            view["artifact_refs"],
            view["hypotheses"],
            view["prior_evaluations"],
            view["claims"],
            view["evidence"],
            view["recent_changes"]["attempts"],
            view["objective"]["constraints"],
        )
        while estimate_tokens(view) > self.token_budget:
            target = next((items for items in removable if items), None)
            if target is None:
                goal = view["objective"]["goal"]
                if len(goal) <= 160:
                    break
                view["objective"]["goal"] = self._compact_text(
                    goal, max(160, len(goal) - 240)
                )
                continue
            target.pop()
        if estimate_tokens(view) > self.token_budget:
            view["required_feedback"] = None
            view["recent_changes"] = {"attempts": []}
            view["objective"]["constraints"] = []
            view["objective"]["focus"] = ""
            view["objective"]["goal"] = self._compact_text(
                view["objective"]["goal"], 120
            )
        if estimate_tokens(view) > self.token_budget:
            for key in (
                "actions",
                "artifact_refs",
                "hypotheses",
                "prior_evaluations",
                "claims",
                "evidence",
                "recent_changes",
            ):
                view.pop(key, None)

    @staticmethod
    def _compact_text(value: Any, limit: int) -> str:
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + " ...[truncated]"

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {item.casefold() for item in _TERM_PATTERN.findall(text) if item.strip()}

    def _relevance(self, text: str, focus_terms: set[str]) -> int:
        if not focus_terms:
            return 0
        return len(self._terms(text).intersection(focus_terms))
