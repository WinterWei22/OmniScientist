from __future__ import annotations

import os
import tomllib
from pathlib import Path
from types import SimpleNamespace

from langchain_openai import ChatOpenAI

from local_deep_research.qwen_provider import QwenAPIProvider

PROJECT_ROOT = Path(__file__).resolve().parent


def _load_secrets(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("rb") as file:
        payload = tomllib.load(file)
    return payload if isinstance(payload, dict) else {}


def _section(name: str) -> dict:
    value = secrets.get(name, {})
    return value if isinstance(value, dict) else {}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


CONFIG_PATH = Path(
    os.getenv(
        "OMNIAGENT_CONFIG_PATH",
        str(PROJECT_ROOT / "_settings" / ".secrets.toml"),
    )
)
secrets = _load_secrets(CONFIG_PATH)
embedding = _section("embedding")
model = _section("model")
biomni_mcp = _section("biomni_mcp")


settings = SimpleNamespace(
    quick=SimpleNamespace(iteration=2, questions_per_iteration=2),
    detailed=SimpleNamespace(iteration=2, questions_per_iteration=2),
    embedding_api_key=os.getenv("EMBEDDING_API_KEY", str(embedding.get("api_key", ""))),
    embedding_cache=os.getenv("EMBEDDING_CACHE", str(embedding.get("cache", ""))),
    default_model=os.getenv("BAILIAN_MODEL", str(model.get("default", "qwen3.8-max"))),
    biomni_mcp=SimpleNamespace(
        enabled=_env_bool(
            "BIOMNI_MCP_ENABLED", bool(biomni_mcp.get("enabled", True))
        ),
        transport=os.getenv(
            "BIOMNI_MCP_TRANSPORT",
            str(biomni_mcp.get("transport", "streamable_http")),
        ),
        url=os.getenv(
            "BIOMNI_MCP_URL",
            str(biomni_mcp.get("url", "http://127.0.0.1:18000/mcp")),
        ),
        tool_prefix=str(biomni_mcp.get("tool_prefix", "biomni")),
        expose_internal_tools=bool(
            biomni_mcp.get("expose_internal_tools", True)
        ),
        default_kg_path=os.getenv(
            "BIOMNI_DEFAULT_KG_PATH", str(biomni_mcp.get("default_kg_path", ""))
        ),
    ),
)

# Check and set OpenAI configuration
# If openai is not configured, try to use closeai as fallback
if (
    "openai" in secrets
    and secrets["openai"].get("api_key")
    and secrets["openai"].get("api_key") != "your-openai-api-key-here"
):
    endpoint_openai_api_base_url = secrets["openai"]["api_base"]
    endpoint_openai_api_key = secrets["openai"]["api_key"]
elif (
    "closeai" in secrets
    and secrets["closeai"].get("api_key")
    and secrets["closeai"].get("api_key") != "your-closeai-api-key-here"
):
    # Use closeai as fallback for openai
    endpoint_openai_api_base_url = secrets["closeai"]["api_base"]
    endpoint_openai_api_key = secrets["closeai"]["api_key"]
else:
    # Default to openai configuration (may fail if not configured)
    endpoint_openai_api_base_url = secrets.get("openai", {}).get("api_base", "")
    endpoint_openai_api_key = secrets.get("openai", {}).get("api_key", "")

# Check and set DeepSeek configuration
# If deepseek is not configured, try to use closeai as fallback
if (
    "deepseek" in secrets
    and secrets["deepseek"].get("api_key")
    and secrets["deepseek"].get("api_key") != "your-deepseek-api-key-here"
):
    deepseek__openai_api_base_url = secrets["deepseek"]["api_base"]
    deepseek_openai_api_key = secrets["deepseek"]["api_key"]
elif (
    "closeai" in secrets
    and secrets["closeai"].get("api_key")
    and secrets["closeai"].get("api_key") != "your-closeai-api-key-here"
):
    # Use closeai as fallback for deepseek
    deepseek__openai_api_base_url = secrets["closeai"]["api_base"]
    deepseek_openai_api_key = secrets["closeai"]["api_key"]
else:
    # Default to deepseek configuration (may fail if not configured)
    deepseek__openai_api_base_url = secrets.get("deepseek", {}).get("api_base", "")
    deepseek_openai_api_key = secrets.get("deepseek", {}).get("api_key", "")

mcp_url = os.getenv("MCP_URL", str(_section("mcp").get("server_url", "")))

template_config = _section("template")
template_embedding_api_base_url = str(template_config.get("api_base", ""))
template_embedding_api_key = str(template_config.get("api_key", ""))

qwen_config = _section("qwen")
qwen_provider = QwenAPIProvider.from_environment(qwen_config)
qwen_provider_name = qwen_provider.settings.provider
qwen_openai_api_base_url = qwen_provider.settings.base_url
qwen_openai_api_key = qwen_provider.settings.api_key
qwen_model_name = qwen_provider.settings.model
qwen_enable_thinking = qwen_provider.settings.enable_thinking
qwen_max_tokens = qwen_provider.settings.max_tokens


def get_gpt4_1() -> ChatOpenAI:
    """
    Get GPT-4 1 model configuration.

    Returns:
        Configured ChatOpenAI instance for GPT-4 1
    """
    return ChatOpenAI(
        model="gpt-4.1",
        api_key=endpoint_openai_api_key,
        openai_api_base=endpoint_openai_api_base_url,
        temperature=0.2,
        top_p=0.9,
        max_tokens=32000,
    )


def get_gpt4_1_mini() -> ChatOpenAI:
    """
    Get GPT-4 1 mini model configuration.

    Returns:
        Configured ChatOpenAI instance for GPT-4 1 mini
    """
    return ChatOpenAI(
        model="gpt-4.1-mini",
        api_key=endpoint_openai_api_key,
        openai_api_base=endpoint_openai_api_base_url,
        temperature=0.2,
        top_p=0.9,
        max_tokens=32000,
    )


def get_claude_openai() -> ChatOpenAI:
    """
    Get Claude model configuration through OpenAI-compatible API.

    Returns:
        Configured ChatOpenAI instance for Claude
    """
    return ChatOpenAI(
        model="claude-3-opus-20240229",
        api_key=endpoint_openai_api_key,
        openai_api_base=endpoint_openai_api_base_url,
        temperature=0.2,
        top_p=0.9,
        max_tokens=32000,
    )


def get_deepseek_r1() -> ChatOpenAI:
    """
    Get DeepSeek R1 reasoning model configuration.

    Returns:
        Configured ChatOpenAI instance for DeepSeek R1
    """
    return ChatOpenAI(
        model="deepseek-reasoner",
        api_key=deepseek_openai_api_key,
        openai_api_base=deepseek__openai_api_base_url,
        temperature=0.2,
        top_p=0.9,
        max_tokens=32000,
    )


def get_deepseek_v3() -> ChatOpenAI:
    """
    Get DeepSeek V3 chat model configuration.

    Returns:
        Configured ChatOpenAI instance for DeepSeek V3
    """
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=deepseek_openai_api_key,
        openai_api_base=deepseek__openai_api_base_url,
        temperature=0.2,
        top_p=0.9,
        max_tokens=32000,
    )

def get_qwen_model() -> ChatOpenAI:
    """Build Qwen using the provider selected by QWEN_PROVIDER."""
    return qwen_provider.create_chat_model()


def get_qwen_siliconflow() -> ChatOpenAI:
    """Backward-compatible alias for the generic Qwen provider."""
    return get_qwen_model()
