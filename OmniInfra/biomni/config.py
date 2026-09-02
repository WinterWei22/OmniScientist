"""
Biomni Configuration Management

Simple configuration class for centralizing common settings.
Maintains full backward compatibility with existing code.
"""

import os
from dataclasses import dataclass
from pathlib import Path

# ``Qwen3.8`` is the project-facing alias requested for tool-side LLM calls.
# DashScope currently exposes concrete IDs such as ``qwen3.8-27b`` rather than
# the literal alias.  The provider resolves the alias at request time; keeping
# the alias here makes it possible to switch the concrete deployment centrally.
DEFAULT_TOOL_LLM = "Qwen3.8"
DEFAULT_TOOL_SOURCE = "Qwen"


@dataclass
class BiomniConfig:
    """Central configuration for Biomni agent.

    All settings are optional and have sensible defaults.
    API keys are read from environment variables or ignored local key files
    so credentials do not need to be committed to source control.

    Usage:
        # Create config with defaults
        config = BiomniConfig()

        # Override specific settings
        config = BiomniConfig(llm="gpt-4", timeout_seconds=1200)

        # Modify after creation
        config.path = "./custom_data"
    """

    # Data and execution settings
    path: str = "./data"
    timeout_seconds: int = 600

    # LLM settings (API keys still from environment)
    llm: str = "qwen3.8-max"
    # LLM used by tools that need to interpret natural-language API requests.
    # This is separate from ``llm`` so existing Claude A1 runs remain opt-in
    # compatible while all eligible tools use the Qwen3.8 deployment.
    tool_llm: str = DEFAULT_TOOL_LLM
    tool_source: str | None = DEFAULT_TOOL_SOURCE
    temperature: float = 0.7
    max_output_tokens: int = 8192
    # No local context-window budget is enforced unless explicitly configured.
    # The provider remains responsible for its actual model context limit.
    context_window_tokens: int | None = None
    token_safety_margin: int = 512

    # Tool settings
    use_tool_retriever: bool = True
    network_policy: str = "controlled"

    # Data licensing settings
    commercial_mode: bool = False  # If True, excludes non-commercial datasets

    # Custom model settings (for custom LLM serving)
    base_url: str | None = None
    api_key: str | None = None  # Only for custom models, not provider API keys

    # LLM source (auto-detected if None)
    source: str | None = (
        None  # Options include OpenAI, AzureOpenAI, Anthropic, Ollama, Gemini, Bedrock, Groq, MiniMax, Qwen, Custom
    )

    # Third-party integrations
    protocols_io_access_token: str | None = None
    openalex_api_key: str | None = None

    def __post_init__(self):
        """Load any environment variable overrides if they exist."""
        # Check for environment variable overrides (optional)
        # Support both old and new names for backwards compatibility
        if os.getenv("BIOMNI_PATH") or os.getenv("BIOMNI_DATA_PATH"):
            self.path = os.getenv("BIOMNI_PATH") or os.getenv("BIOMNI_DATA_PATH")
        if os.getenv("BIOMNI_TIMEOUT_SECONDS"):
            self.timeout_seconds = int(os.getenv("BIOMNI_TIMEOUT_SECONDS"))
        if os.getenv("BIOMNI_LLM") or os.getenv("BIOMNI_LLM_MODEL"):
            self.llm = os.getenv("BIOMNI_LLM") or os.getenv("BIOMNI_LLM_MODEL")
        if os.getenv("BIOMNI_TOOL_LLM") or os.getenv("BIOMNI_TOOL_LLM_MODEL"):
            self.tool_llm = os.getenv("BIOMNI_TOOL_LLM") or os.getenv("BIOMNI_TOOL_LLM_MODEL")
        if os.getenv("BIOMNI_TOOL_SOURCE"):
            self.tool_source = os.getenv("BIOMNI_TOOL_SOURCE")
        if os.getenv("BIOMNI_USE_TOOL_RETRIEVER"):
            self.use_tool_retriever = os.getenv("BIOMNI_USE_TOOL_RETRIEVER").lower() == "true"
        if os.getenv("BIOMNI_NETWORK_POLICY"):
            self.network_policy = os.getenv("BIOMNI_NETWORK_POLICY", "controlled").strip().lower()
        if os.getenv("BIOMNI_COMMERCIAL_MODE"):
            self.commercial_mode = os.getenv("BIOMNI_COMMERCIAL_MODE").lower() == "true"
        if os.getenv("BIOMNI_TEMPERATURE"):
            self.temperature = float(os.getenv("BIOMNI_TEMPERATURE"))
        if os.getenv("BIOMNI_MAX_OUTPUT_TOKENS"):
            self.max_output_tokens = int(os.getenv("BIOMNI_MAX_OUTPUT_TOKENS"))
        if os.getenv("BIOMNI_CONTEXT_WINDOW_TOKENS"):
            self.context_window_tokens = int(os.getenv("BIOMNI_CONTEXT_WINDOW_TOKENS"))
        if os.getenv("BIOMNI_TOKEN_SAFETY_MARGIN"):
            self.token_safety_margin = int(os.getenv("BIOMNI_TOKEN_SAFETY_MARGIN"))
        if os.getenv("BIOMNI_CUSTOM_BASE_URL"):
            self.base_url = os.getenv("BIOMNI_CUSTOM_BASE_URL")
        if os.getenv("BIOMNI_CUSTOM_API_KEY"):
            self.api_key = os.getenv("BIOMNI_CUSTOM_API_KEY")
        if os.getenv("BIOMNI_SOURCE"):
            self.source = os.getenv("BIOMNI_SOURCE")

        # Protocols.io access token (prefer specific env vars)
        env_token = os.getenv("PROTOCOLS_IO_ACCESS_TOKEN") or os.getenv("BIOMNI_PROTOCOLS_IO_ACCESS_TOKEN")
        if env_token:
            self.protocols_io_access_token = env_token

        # OpenAlex API key: an explicit environment override takes precedence
        # over a programmatic value or the ignored local default key file.
        env_openalex_key = os.getenv("OPENALEX_API_KEY", "").strip()
        if env_openalex_key:
            self.openalex_api_key = env_openalex_key
        elif not self.openalex_api_key:
            key_file = Path(os.getenv("OPENALEX_API_KEY_FILE", "openalex_apikey.txt")).expanduser()
            try:
                if key_file.is_file():
                    self.openalex_api_key = key_file.read_text(encoding="utf-8").strip() or None
            except OSError:
                # query_scholar can still use OpenAlex's anonymous free quota.
                self.openalex_api_key = None

    def to_dict(self) -> dict:
        """Convert config to dictionary for easy access."""
        return {
            "path": self.path,
            "timeout_seconds": self.timeout_seconds,
            "llm": self.llm,
            "tool_llm": self.tool_llm,
            "tool_source": self.tool_source,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "context_window_tokens": self.context_window_tokens,
            "token_safety_margin": self.token_safety_margin,
            "use_tool_retriever": self.use_tool_retriever,
            "network_policy": self.network_policy,
            "commercial_mode": self.commercial_mode,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "source": self.source,
        }


# Global default config instance (optional, for convenience)
default_config = BiomniConfig()
