from __future__ import annotations

import math
import os
from dataclasses import dataclass


DEFAULT_MAX_ITERATIONS = 36
DEFAULT_TASK_TIMEOUT_SECONDS = 3600.0


def _read_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _read_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative number") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return value


@dataclass(frozen=True, slots=True)
class RuntimeLimits:
    """The two runner-level budgets shared by all OmniAgent script entrypoints."""

    max_iterations: int = DEFAULT_MAX_ITERATIONS
    task_timeout_seconds: float = DEFAULT_TASK_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if not math.isfinite(self.task_timeout_seconds) or self.task_timeout_seconds < 0:
            raise ValueError("task_timeout_seconds must be a non-negative number")

    @classmethod
    def from_environment(cls) -> RuntimeLimits:
        return cls(
            max_iterations=_read_int("OMNIAGENT_MAX_ITERATIONS", DEFAULT_MAX_ITERATIONS),
            task_timeout_seconds=_read_float(
                "OMNIAGENT_TASK_TIMEOUT_SECONDS", DEFAULT_TASK_TIMEOUT_SECONDS
            ),
        )
