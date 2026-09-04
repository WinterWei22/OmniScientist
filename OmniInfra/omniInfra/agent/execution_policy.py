"""Behavioral admission and evidence guards for A1-generated execution.

These checks constrain normal model behavior at the existing interpreter boundary.
They are not an adversarial security sandbox; OS-level network isolation remains a
separate deployment concern.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Literal

NetworkPolicy = Literal["controlled", "offline", "legacy_open"]
NETWORK_POLICIES = {"controlled", "offline", "legacy_open"}

_PYTHON_NETWORK_ROOTS = {
    "aiohttp",
    "boto3",
    "botocore",
    "ftplib",
    "http",
    "httpx",
    "imaplib",
    "nntplib",
    "paramiko",
    "poplib",
    "requests",
    "smtplib",
    "socket",
    "urllib",
    "websocket",
    "websockets",
    "xmlrpc",
}
_PYTHON_URL_READERS = {
    "read_csv",
    "read_excel",
    "read_html",
    "read_json",
    "read_parquet",
    "read_pickle",
    "read_table",
}
_SHELL_NETWORK_COMMAND = re.compile(
    r"(?:^|[;&|]\s*|\b(?:sudo|env)\s+)(?:curl|wget|nc|ncat|netcat|ssh|scp|sftp|ftp|telnet)\b",
    re.IGNORECASE | re.MULTILINE,
)
_SHELL_REMOTE_GIT = re.compile(r"\bgit\s+(?:clone|fetch|pull|push|remote)\b", re.IGNORECASE)
_SHELL_PACKAGE_NETWORK = re.compile(
    r"\b(?:pip|pip3|conda|mamba|apt|apt-get|yum|dnf|npm|yarn|pnpm)\s+(?:install|update|upgrade|add)\b",
    re.IGNORECASE,
)
_SHELL_INTERPRETER_URL = re.compile(
    r"\b(?:python(?:3(?:\.\d+)?)?|Rscript|ruby|perl|node)\b.*https?://",
    re.IGNORECASE | re.DOTALL,
)
_UNBOUNDED_FILESYSTEM_SEARCH = re.compile(
    r"\bfind\s+(?:/|/data(?:/|\b)|/bigdat2(?:/|\b))",
    re.IGNORECASE,
)
_R_NETWORK = re.compile(
    r"\b(?:download\.file|url|socketConnection|readLines)\s*\(|"
    r"\b(?:httr|httr2|curl|RCurl)::|\blibrary\s*\(\s*(?:httr|httr2|curl|RCurl)\s*\)|"
    r"\b(?:fromJSON|read\.csv|read\.table)\s*\(\s*['\"]https?://",
    re.IGNORECASE,
)
_URL = re.compile(r"https?://", re.IGNORECASE)
ToolAction = tuple[str, str | None, bool]


@dataclass(frozen=True)
class AdmissionDecision:
    allowed: bool
    policy: NetworkPolicy
    reason_code: str | None = None
    message: str | None = None
    detections: tuple[str, ...] = ()
    audit_warning: bool = False

    def observation(
        self,
        *,
        alternatives: list[str] | None = None,
        remaining_policy_budget: int | None = None,
    ) -> str:
        payload = {
            "success": False,
            "failure_kind": "policy_blocked",
            "reason": self.reason_code,
            "message": self.message,
            "detections": list(self.detections),
            "allowed_alternative": alternatives or [],
            "remaining_policy_budget": remaining_policy_budget,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


class ExecutionAdmissionController:
    """Detect direct networking before generated code reaches an interpreter."""

    def __init__(self, policy: NetworkPolicy = "controlled") -> None:
        if policy not in NETWORK_POLICIES:
            raise ValueError(f"network_policy must be one of: {', '.join(sorted(NETWORK_POLICIES))}")
        self.policy = policy

    def inspect(
        self,
        code: str,
        *,
        language: Literal["python", "bash", "r"],
        offline_network_tools: set[str] | None = None,
    ) -> AdmissionDecision:
        detections = self._detect(code, language=language)
        if self.policy == "offline" and offline_network_tools:
            referenced = sorted(name for name in offline_network_tools if re.search(rf"\b{re.escape(name)}\b", code))
            detections.extend(f"registered_network_tool:{name}" for name in referenced)

        unique = tuple(dict.fromkeys(detections))
        if not unique:
            return AdmissionDecision(allowed=True, policy=self.policy)
        if "unbounded_filesystem_search" in unique:
            return AdmissionDecision(
                allowed=False,
                policy=self.policy,
                reason_code="unbounded_filesystem_search",
                message="Unbounded recursive search is blocked; use a configured executable path or PATH lookup.",
                detections=unique,
            )
        if self.policy == "legacy_open":
            return AdmissionDecision(
                allowed=True,
                policy=self.policy,
                reason_code="legacy_open_raw_network",
                message="Raw networking was permitted by the explicit legacy_open compatibility mode.",
                detections=unique,
                audit_warning=True,
            )
        return AdmissionDecision(
            allowed=False,
            policy=self.policy,
            reason_code="offline_network_denied" if self.policy == "offline" else "raw_network_denied",
            message=(
                "External networking is disabled in offline mode."
                if self.policy == "offline"
                else "Use a listed OmniInfra tool or query_public_endpoint instead of direct networking."
            ),
            detections=unique,
        )

    def _detect(self, code: str, *, language: str) -> list[str]:
        if language == "bash":
            return self._detect_shell(code)
        if language == "r":
            return ["r_network_api"] if _R_NETWORK.search(code) else []
        return self._detect_python(code)

    @staticmethod
    def _call_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = ExecutionAdmissionController._call_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    @staticmethod
    def _literal_text(node: ast.AST) -> str | None:
        try:
            value = ast.literal_eval(node)
        except (ValueError, TypeError, SyntaxError):
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return " ".join(str(item) for item in value)
        return None

    def _detect_python(self, code: str) -> list[str]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        detections: list[str] = []
        importlib_aliases = {"importlib"}
        builtins_aliases = {"builtins"}
        dynamic_import_names = {"__import__"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "importlib":
                        importlib_aliases.add(alias.asname or alias.name)
                    elif alias.name == "builtins":
                        builtins_aliases.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
                for alias in node.names:
                    if alias.name == "import_module":
                        dynamic_import_names.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module == "builtins":
                for alias in node.names:
                    if alias.name == "__import__":
                        dynamic_import_names.add(alias.asname or alias.name)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root in _PYTHON_NETWORK_ROOTS:
                        detections.append(f"python_import:{root}")
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".", 1)[0]
                if root in _PYTHON_NETWORK_ROOTS:
                    detections.append(f"python_import:{root}")
            elif isinstance(node, ast.Call):
                call_name = self._call_name(node.func)
                is_dynamic_import = (
                    call_name in dynamic_import_names
                    or any(call_name == f"{alias}.import_module" for alias in importlib_aliases)
                    or any(call_name == f"{alias}.__import__" for alias in builtins_aliases)
                )
                if is_dynamic_import:
                    module_name = self._literal_text(node.args[0]) if node.args else None
                    root = (module_name or "").split(".", 1)[0]
                    if root in _PYTHON_NETWORK_ROOTS:
                        detections.append(f"python_dynamic_import:{root}")
                        continue
                root = call_name.split(".", 1)[0]
                if root in _PYTHON_NETWORK_ROOTS:
                    detections.append(f"python_call:{call_name}")
                    continue
                if call_name.rsplit(".", 1)[-1] in _PYTHON_URL_READERS:
                    values = [self._literal_text(arg) for arg in node.args]
                    values.extend(self._literal_text(keyword.value) for keyword in node.keywords)
                    if any(value and _URL.search(value) for value in values):
                        detections.append(f"python_url_reader:{call_name}")
                if call_name in {"os.system", "os.popen"} or call_name.startswith("subprocess."):
                    command = self._literal_text(node.args[0]) if node.args else None
                    if command:
                        shell_detections = self._detect_shell(command)
                        if any(item != "unbounded_filesystem_search" for item in shell_detections):
                            detections.append(f"python_process_network:{call_name}")
                        if "unbounded_filesystem_search" in shell_detections:
                            detections.append("unbounded_filesystem_search")
        return detections

    @staticmethod
    def _detect_shell(code: str) -> list[str]:
        detections = []
        if _UNBOUNDED_FILESYSTEM_SEARCH.search(code):
            detections.append("unbounded_filesystem_search")
        if _SHELL_NETWORK_COMMAND.search(code):
            detections.append("shell_network_command")
        if _SHELL_REMOTE_GIT.search(code):
            detections.append("shell_remote_git")
        if _SHELL_PACKAGE_NETWORK.search(code):
            detections.append("shell_network_package_manager")
        if _SHELL_INTERPRETER_URL.search(code):
            detections.append("shell_interpreter_network")
        return detections


@dataclass
class A1RunControl:
    max_execute_rounds: int = 16
    max_tool_calls: int = 24
    max_policy_rejections: int = 2
    max_tool_recoveries: int = 1
    max_endpoint_discoveries: int = 2
    max_fallback_requests: int = 4
    max_consecutive_no_evidence: int = 3
    max_response_format_failures: int = 4
    max_generated_code_failures: int = 6
    execute_rounds: int = 0
    tool_calls: int = 0
    policy_rejections: int = 0
    tool_recoveries: int = 0
    endpoint_discoveries: int = 0
    fallback_requests: int = 0
    consecutive_no_evidence: int = 0
    solution_rewrites: int = 0
    response_format_failures: int = 0
    generated_code_failures: int = 0
    termination_reason: str | None = None
    # Keep a small retry budget per action fingerprint.  A single retry is
    # useful for transient failures and output/format repair, while the third
    # identical attempt is almost certainly a loop.
    action_attempts: dict[str, int] = field(default_factory=dict)
    tool_action_attempts: dict[str, int] = field(default_factory=dict)
    # Compatibility views retained for callers that only need to know whether
    # a fingerprint has ever been admitted.
    action_hashes: set[str] = field(default_factory=set)
    tool_action_hashes: set[str] = field(default_factory=set)
    observation_kinds: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)

    def admit_action(self, code: str, *, tool_actions: list[ToolAction]) -> str | None:
        if self.termination_reason:
            return self.termination_reason
        if self.execute_rounds >= self.max_execute_rounds:
            self.termination_reason = "max_execute_rounds"
            return self.termination_reason
        if self.tool_calls + len(tool_actions) > self.max_tool_calls:
            self.termination_reason = "max_tool_calls"
            return self.termination_reason
        digest = hashlib.sha256(" ".join(code.split()).encode()).hexdigest()
        if self.action_attempts.get(digest, 0) >= 2:
            return "repeated_action"
        if any(
            action_hash is not None and self.tool_action_attempts.get(action_hash, 0) >= 2
            for _name, action_hash, _discovery in tool_actions
        ):
            return "repeated_action"
        fallback_requests = sum(name == "query_public_endpoint" for name, _hash, _discovery in tool_actions)
        endpoint_discoveries = sum(discovery for _name, _hash, discovery in tool_actions)
        if self.fallback_requests + fallback_requests > self.max_fallback_requests:
            self.termination_reason = "fallback_exhausted"
            return self.termination_reason
        if self.endpoint_discoveries + endpoint_discoveries > self.max_endpoint_discoveries:
            self.termination_reason = "fallback_exhausted"
            return self.termination_reason
        self.action_attempts[digest] = self.action_attempts.get(digest, 0) + 1
        self.action_hashes.add(digest)
        for _name, action_hash, _discovery in tool_actions:
            if action_hash is not None:
                self.tool_action_attempts[action_hash] = self.tool_action_attempts.get(action_hash, 0) + 1
                self.tool_action_hashes.add(action_hash)
        self.execute_rounds += 1
        self.tool_calls += len(tool_actions)
        self.fallback_requests += fallback_requests
        self.endpoint_discoveries += endpoint_discoveries
        return None

    def record_policy_rejection(self) -> int:
        self.policy_rejections += 1
        if self.policy_rejections >= self.max_policy_rejections:
            self.termination_reason = "policy_violation_limit"
        return max(0, self.max_policy_rejections - self.policy_rejections)

    def record_response_format_failure(self, reason: str) -> int:
        """Count malformed/truncated model responses across one A1 run."""
        self.response_format_failures += 1
        self.events.append(
            {
                "event": "response_format_failure",
                "reason": reason,
                "count": self.response_format_failures,
            }
        )
        if self.response_format_failures >= self.max_response_format_failures:
            self.termination_reason = reason
        return max(0, self.max_response_format_failures - self.response_format_failures)

    def record_generated_code_failure(self, reason: str = "invalid_generated_code") -> int:
        """Bound retries for code rejected before execution."""
        self.generated_code_failures += 1
        self.events.append(
            {
                "event": "generated_code_failure",
                "reason": reason,
                "count": self.generated_code_failures,
            }
        )
        if self.generated_code_failures >= self.max_generated_code_failures:
            self.termination_reason = reason
        return max(0, self.max_generated_code_failures - self.generated_code_failures)

    def record_observation(self, kind: str | None) -> None:
        if kind:
            self.observation_kinds.append(kind)
        if kind in {
            "authentication_required",
            "empty_result",
            "endpoint_provenance_unverified",
            "internal_error",
            "invalid_arguments",
            "not_found",
            "policy_blocked",
            "rate_limited",
            "timeout",
            "transient_network",
            "upstream_contract_changed",
            "upstream_error",
        }:
            self.consecutive_no_evidence += 1
        else:
            self.consecutive_no_evidence = 0
        if self.consecutive_no_evidence >= self.max_consecutive_no_evidence:
            self.termination_reason = "evidence_insufficient"


def normalize_observation(text: str) -> tuple[str | None, str | None]:
    """Classify common observations and return model-facing evidence guidance."""
    lowered = text.lower()
    stripped = lowered.strip()
    structured_failure = bool(re.search(r"['\"]success['\"]\s*:\s*false", lowered) or stripped.startswith("error:"))

    def transport_error(pattern: str) -> bool:
        if structured_failure:
            return re.search(pattern, lowered) is not None
        return re.match(rf"(?:public endpoint returned\s+)?{pattern}", stripped) is not None

    explicit = re.search(r"['\"]failure_kind['\"]\s*:\s*['\"]([a-z_]+)['\"]", lowered)
    if explicit:
        kind = explicit.group(1)
        guidance = {
            "authentication_required": "Credentials were unavailable; use a public source or report the limitation.",
            "endpoint_provenance_unverified": "The endpoint was not verified; do not infer why it is unavailable.",
            "invalid_arguments": "Correct the call from the registered schema before retrying.",
            "not_found": "Only the specified resource was not found; do not infer service-wide failure.",
            "policy_blocked": "The rejected action produced no scientific evidence. Use an admitted native tool.",
            "rate_limited": "The request was rate limited; this is not a scientific result.",
            "timeout": "The operation timed out; this is not evidence of absence.",
            "transient_network": "A transient network failure produced no scientific evidence.",
            "upstream_error": "The upstream request failed; do not treat it as a scientific result.",
        }.get(kind, "The failed observation must not be treated as scientific evidence.")
        return kind, guidance
    if "policy_blocked" in lowered:
        return "policy_blocked", "The rejected action produced no scientific evidence. Use an admitted native tool."
    if "endpoint_provenance_unverified" in lowered:
        return "endpoint_provenance_unverified", "The endpoint was not verified; do not infer why it is unavailable."
    if transport_error(r"(?:http\s*)?404\b|not found"):
        return (
            "not_found",
            "Only the specified resource was not found; do not infer deprecation, authentication, or service-wide failure.",
        )
    if transport_error(r"(?:http\s*)?429\b|rate.?limit"):
        return "rate_limited", "The request was rate limited; this is not a scientific result."
    if transport_error(r"(?:request\s+|operation\s+)?(?:timed out|timeout)"):
        return "timeout", "The operation timed out; this is not evidence of absence."
    if transport_error(r"(?:http\s*)?(?:401|403)\b|authentication required|missing credentials"):
        return "authentication_required", "Credentials were unavailable; use a public source or report the limitation."
    if structured_failure:
        return "internal_error", "The tool failed; its output must not be treated as scientific evidence."
    if re.search(r"['\"](?:result|results|records)['\"]\s*:\s*(?:\[\]|\{\}|none|null)", lowered):
        return "empty_result", "This query returned no records; absence of evidence is not evidence of absence."
    return None, None


def solution_invariant_violation(solution: str, observation_kinds: list[str]) -> str | None:
    """Return a correction when a final answer overstates a known failed observation."""
    lowered = solution.lower()
    kinds = set(observation_kinds)
    if "not_found" in kinds and re.search(
        r"\b(?:api|endpoint|service)\b.{0,80}\b(?:deprecated|decommissioned|requires? authentication|unavailable)\b",
        lowered,
        re.DOTALL,
    ):
        return "A 404 supports only that the specified resource was not found; remove unsupported causal explanations."
    if "empty_result" in kinds and re.search(
        r"\b(?:proves?|confirms?|demonstrates?)\b.{0,40}\b(?:no|none|absence)\b", lowered
    ):
        return "An empty query result cannot prove that the underlying evidence does not exist."
    if kinds & {"internal_error", "timeout", "policy_blocked", "endpoint_provenance_unverified"} and re.search(
        r"\b(?:the (?:data|results?) (?:show|demonstrate|confirm)|therefore definitively)\b",
        lowered,
    ):
        return "A failed or rejected action cannot support a definitive scientific conclusion."
    return None
