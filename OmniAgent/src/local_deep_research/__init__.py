"""
Local Deep Research - AI-powered research assistant

A powerful AI research system with iterative analysis capabilities
and multiple search engines integration.
"""

from typing import Any

__version__ = "0.1.0"
__all__ = ["AdvancedSearchSystem"]


def __getattr__(name: str) -> Any:
    """Load the legacy search system only when its public class is requested."""
    if name == "AdvancedSearchSystem":
        from .search_system import AdvancedSearchSystem

        return AdvancedSearchSystem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
