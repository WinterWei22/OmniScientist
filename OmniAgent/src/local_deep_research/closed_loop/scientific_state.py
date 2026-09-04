from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ClaimStatus(str, Enum):
    PROPOSED = "proposed"
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    RETRACTED = "retracted"


class HypothesisStatus(str, Enum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    SUPPORTED = "supported"
    WEAKENED = "weakened"
    CONTRADICTED = "contradicted"
    REJECTED = "rejected"


class AttemptStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class CanonicalEntityRecord:
    entity_id: str
    entity_type: str
    query_name: str
    preferred_name: str
    gene_symbol: str = ""
    aliases: tuple[str, ...] = ()
    uniprot_accession: str = ""
    organism: str = ""
    tax_id: int | None = None
    source: str = ""
    source_url: str = ""


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    evidence_id: str
    evidence_type: str
    summary: str
    source_attempt_id: str
    source_backend: str
    source_capability_id: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    artifact_refs: tuple[str, ...] = ()
    verifier_id: str = ""


@dataclass(frozen=True, slots=True)
class ClaimRecord:
    claim_id: str
    statement: str
    evidence_ids: tuple[str, ...]
    status: ClaimStatus
    source_attempt_id: str
    verifier_id: str


@dataclass(frozen=True, slots=True)
class HypothesisRecord:
    hypothesis_id: str
    statement: str
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    supporting_evidence_ids: tuple[str, ...] = ()
    contradicting_evidence_ids: tuple[str, ...] = ()
    uncertainty: str = ""
    next_test: str = ""
    version: int = 1


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    attempt_id: str
    iteration: int
    step_id: str
    objective: str
    backend: str
    capability_id: str
    status: AttemptStatus
    result_status: str
    verifier_id: str
    reason: str = ""
    evidence_ids: tuple[str, ...] = ()
    artifact_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class VerifiedStateTransition:
    transition_id: str
    expected_state_version: int
    accepted: bool
    reason: str
    verifier_id: str
    attempt: AttemptRecord
    evidence: tuple[EvidenceRecord, ...] = ()
    claims: tuple[ClaimRecord, ...] = ()
    hypotheses: tuple[HypothesisRecord, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()


@dataclass(slots=True)
class ScientificState:
    task_id: str
    goal: str
    state_version: int = 0
    canonical_entities: dict[str, CanonicalEntityRecord] = field(default_factory=dict)
    entity_corrections: dict[str, str] = field(default_factory=dict)
    evidence: dict[str, EvidenceRecord] = field(default_factory=dict)
    claims: dict[str, ClaimRecord] = field(default_factory=dict)
    hypotheses: dict[str, HypothesisRecord] = field(default_factory=dict)
    attempts: dict[str, AttemptRecord] = field(default_factory=dict)
    unresolved_questions: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    failed_directions: list[str] = field(default_factory=list)
    artifact_refs: list[str] = field(default_factory=list)
    transition_ids: list[str] = field(default_factory=list)

    def policy_view(self) -> dict[str, Any]:
        verified_claims = [
            item
            for item in self.claims.values()
            if item.status is ClaimStatus.VERIFIED
        ]
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "state_version": self.state_version,
            "canonical_entities": [
                asdict(item) for item in self.canonical_entities.values()
            ],
            "entity_corrections": dict(self.entity_corrections),
            "verified_claims": [asdict(item) for item in verified_claims[-20:]],
            "hypotheses": [asdict(item) for item in list(self.hypotheses.values())[-12:]],
            "recent_evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "evidence_type": item.evidence_type,
                    "summary": item.summary,
                    "source_backend": item.source_backend,
                    "source_capability_id": item.source_capability_id,
                    "artifact_refs": list(item.artifact_refs),
                }
                for item in list(self.evidence.values())[-12:]
            ],
            "unresolved_questions": self.unresolved_questions[-12:],
            "conflicts": self.conflicts[-12:],
            "failed_directions": self.failed_directions[-8:],
            "artifact_refs": self.artifact_refs[-20:],
            "recent_attempts": [
                asdict(item) for item in list(self.attempts.values())[-8:]
            ],
        }


class ScientificStateReducer:
    """The only boundary that admits verified execution results into state."""

    def apply(
        self,
        state: ScientificState,
        transition: VerifiedStateTransition,
    ) -> None:
        if transition.expected_state_version != state.state_version:
            raise ValueError(
                "stale scientific transition: "
                f"expected {transition.expected_state_version}, current {state.state_version}"
            )
        if transition.transition_id in state.transition_ids:
            raise ValueError(f"duplicate transition ID: {transition.transition_id}")
        if transition.attempt.attempt_id in state.attempts:
            raise ValueError(f"duplicate attempt ID: {transition.attempt.attempt_id}")
        if transition.attempt.verifier_id != transition.verifier_id:
            raise ValueError("attempt verifier does not match transition verifier")
        if transition.accepted and transition.attempt.status is not AttemptStatus.SUCCEEDED:
            raise ValueError("accepted transition requires a succeeded attempt")
        if not transition.accepted and transition.attempt.status is AttemptStatus.SUCCEEDED:
            raise ValueError("rejected transition cannot contain a succeeded attempt")
        if transition.accepted and not transition.evidence:
            raise ValueError("accepted transition requires admitted evidence")
        if not transition.accepted and any(
            (transition.evidence, transition.claims, transition.hypotheses)
        ):
            raise ValueError("rejected transition cannot update scientific content")

        incoming_evidence = {item.evidence_id: item for item in transition.evidence}
        if len(incoming_evidence) != len(transition.evidence):
            raise ValueError("transition contains duplicate evidence IDs")
        if set(incoming_evidence).intersection(state.evidence):
            raise ValueError("transition reuses an existing evidence ID")
        available_evidence = set(state.evidence).union(incoming_evidence)
        incoming_claims = {item.claim_id: item for item in transition.claims}
        if len(incoming_claims) != len(transition.claims):
            raise ValueError("transition contains duplicate claim IDs")
        if set(incoming_claims).intersection(state.claims):
            raise ValueError("transition reuses an existing claim ID")
        incoming_hypotheses = {
            item.hypothesis_id: item for item in transition.hypotheses
        }
        if len(incoming_hypotheses) != len(transition.hypotheses):
            raise ValueError("transition patches one hypothesis more than once")

        for item in transition.evidence:
            if item.source_attempt_id != transition.attempt.attempt_id:
                raise ValueError("evidence is not bound to the transition attempt")
            if item.verifier_id != transition.verifier_id:
                raise ValueError("evidence verifier does not match transition verifier")
        for item in transition.claims:
            if not item.statement.strip() or not item.evidence_ids:
                raise ValueError("claims require a statement and supporting evidence")
            if not set(item.evidence_ids).issubset(available_evidence):
                raise ValueError("claim cites evidence outside scientific state")
            if item.source_attempt_id != transition.attempt.attempt_id:
                raise ValueError("claim is not bound to the transition attempt")
            if item.verifier_id != transition.verifier_id:
                raise ValueError("claim verifier does not match transition verifier")
        for item in transition.hypotheses:
            cited = set(item.supporting_evidence_ids).union(
                item.contradicting_evidence_ids
            )
            if not cited.issubset(available_evidence):
                raise ValueError("hypothesis cites evidence outside scientific state")
            current = state.hypotheses.get(item.hypothesis_id)
            expected_version = 1 if current is None else current.version + 1
            if item.version != expected_version:
                raise ValueError(
                    f"invalid hypothesis version for {item.hypothesis_id}: "
                    f"expected {expected_version}, got {item.version}"
                )

        state.attempts[transition.attempt.attempt_id] = transition.attempt
        if transition.accepted:
            state.evidence.update(incoming_evidence)
            state.claims.update(incoming_claims)
            state.hypotheses.update(incoming_hypotheses)
            state.unresolved_questions.extend(
                item
                for item in transition.unresolved_questions
                if item and item not in state.unresolved_questions
            )
            state.conflicts.extend(
                item for item in transition.conflicts if item and item not in state.conflicts
            )
            state.artifact_refs.extend(
                item
                for evidence in transition.evidence
                for item in evidence.artifact_refs
                if item not in state.artifact_refs
            )
        elif transition.reason and transition.reason not in state.failed_directions:
            state.failed_directions.append(transition.reason)

        state.transition_ids.append(transition.transition_id)
        state.state_version += 1
