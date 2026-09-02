import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

from langchain_core.language_models.chat_models import BaseChatModel

if TYPE_CHECKING:
    from biomni.config import BiomniConfig

SourceType = Literal[
    "OpenAI",
    "AzureOpenAI",
    "Anthropic",
    "Ollama",
    "Gemini",
    "Bedrock",
    "Groq",
    "MiniMax",
    "Qwen",
    "Custom",
]
ALLOWED_SOURCES: set[str] = set(SourceType.__args__)


def resolve_provider_model(model: str, source: str) -> str:
    """Resolve project aliases to provider model IDs.

    ``Qwen3.8`` is intentionally a stable Biomni alias.  DashScope's
    compatible endpoint currently publishes concrete IDs (the default below
    is ``qwen3.8-max``).  Deployments can select another permitted Qwen3.8
    variant with ``BIOMNI_QWEN38_MODEL`` without changing tool code.
    """
    if source == "Qwen" and model.strip().lower() in {"qwen3.8", "qwen-3.8"}:
        return os.getenv("BIOMNI_QWEN38_MODEL", "qwen3.8-max")
    return model


def _load_minimax_credentials() -> tuple[str | None, str | None]:
    """Load MiniMax credentials from env or a local minimax_apikey.txt file."""
    api_key = os.getenv("MINIMAX_API_KEY")
    base_url = os.getenv("MINIMAX_BASE_URL")

    if api_key and base_url:
        return api_key, base_url

    credential_path = Path("minimax_apikey.txt")
    if credential_path.exists():
        for raw_line in credential_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if "=" in line:
                key, value = [part.strip() for part in line.split("=", 1)]
                key = key.upper()
                if key in {"MINIMAX_API_KEY", "API_KEY"} and not api_key:
                    api_key = value
                elif key in {"MINIMAX_BASE_URL", "BASE_URL"} and not base_url:
                    base_url = value
            elif line.startswith(("http://", "https://")) and not base_url:
                base_url = line
            elif not api_key:
                api_key = line

    return api_key, base_url


def _load_qwen_credentials() -> tuple[str | None, str | None]:
    """Load Qwen credentials from env or a local qwen_apikey.txt file."""
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
    base_url = os.getenv("DASHSCOPE_BASE_URL") or os.getenv("QWEN_BASE_URL")

    if api_key and base_url:
        return api_key, base_url

    credential_path = Path("qwen_apikey.txt")
    if credential_path.exists():
        for raw_line in credential_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if "=" in line:
                key, value = [part.strip() for part in line.split("=", 1)]
                key = key.upper()
                if key in {"DASHSCOPE_API_KEY", "QWEN_API_KEY", "API_KEY"} and not api_key:
                    api_key = value
                elif key in {"DASHSCOPE_BASE_URL", "QWEN_BASE_URL", "BASE_URL"} and not base_url:
                    base_url = value
            elif line.startswith(("http://", "https://")) and not base_url:
                base_url = line
            elif not api_key:
                api_key = line

    return api_key, base_url


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    stop_sequences: list[str] | None = None,
    source: SourceType | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    config: Optional["BiomniConfig"] = None,
    max_tokens: int | None = None,
) -> BaseChatModel:
    """
    Get a language model instance based on the specified model name and source.
    This function supports models from OpenAI, Azure OpenAI, Anthropic, Ollama, Gemini, Bedrock, MiniMax, Qwen, and custom model serving.
    Args:
        model (str): The model name to use
        temperature (float): Temperature setting for generation
        stop_sequences (list): Sequences that will stop generation
        max_tokens (int): Maximum output tokens. Provider defaults are used when omitted.
        source (str): Source provider: "OpenAI", "AzureOpenAI", "Anthropic", "Ollama", "Gemini", "Bedrock", "MiniMax", "Qwen", or "Custom"
                      If None, will attempt to auto-detect from model name
        base_url (str): The base URL for custom model serving (e.g., "http://localhost:8000/v1"), default is None
        api_key (str): The API key for the custom llm
        config (BiomniConfig): Optional configuration object. If provided, unspecified parameters will use config values
    """
    # Use config values for any unspecified parameters
    if config is not None:
        if model is None:
            model = config.llm_model
        if temperature is None:
            temperature = config.temperature
        if max_tokens is None:
            max_tokens = config.max_output_tokens
        if source is None:
            source = config.source
        if base_url is None:
            base_url = config.base_url
        if api_key is None:
            api_key = config.api_key or "EMPTY"

    # Use defaults if still not specified
    if model is None:
        model = "claude-3-5-sonnet-20241022"
    if temperature is None:
        temperature = 0.7
    if api_key is None:
        api_key = "EMPTY"
    # Auto-detect source from model name if not specified
    if source is None:
        env_source = os.getenv("LLM_SOURCE")
        if env_source in ALLOWED_SOURCES:
            source = env_source
        else:
            if model[:7] == "claude-":
                source = "Anthropic"
            elif model[:7] == "gpt-oss":
                source = "Ollama"
            elif model[:4] == "gpt-":
                source = "OpenAI"
            elif model.startswith("azure-"):
                source = "AzureOpenAI"
            elif model[:7] == "gemini-":
                source = "Gemini"
            elif "groq" in model.lower():
                source = "Groq"
            elif model.lower().startswith("minimax-"):
                source = "MiniMax"
            elif model.lower().startswith("qwen"):
                source = "Qwen"
            elif base_url is not None:
                source = "Custom"
            elif "/" in model or any(
                name in model.lower()
                for name in [
                    "llama",
                    "mistral",
                    "qwen",
                    "gemma",
                    "phi",
                    "dolphin",
                    "orca",
                    "vicuna",
                    "deepseek",
                ]
            ):
                source = "Ollama"
            elif model.startswith(
                ("anthropic.claude-", "amazon.titan-", "meta.llama-", "mistral.", "cohere.", "ai21.", "us.")
            ):
                source = "Bedrock"
            else:
                raise ValueError("Unable to determine model source. Please specify 'source' parameter.")

    # Create appropriate model based on source
    # Keep the user-facing alias in config while sending the concrete provider
    # ID to DashScope.
    model = resolve_provider_model(model, source)

    if source == "OpenAI":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for OpenAI models. Install with: pip install langchain-openai"
            )
        # Newer OpenAI models (e.g., gpt-5-*) require the Responses API and may reject
        # legacy Chat Completions parameters like `stop`. Force Responses API when
        # using gpt-5 models to avoid 400 errors such as: "Unsupported parameter: 'stop'".
        use_responses = model.startswith("gpt-5")

        if use_responses:
            # Define a minimal subclass that drops the `stop` field when using the
            # Responses API, since certain models (gpt-5-*) reject it entirely.
            class _ChatOpenAIResponsesNoStop(ChatOpenAI):
                def _get_request_payload(self, input_, *, stop=None, **kwargs):  # type: ignore[override]
                    payload = super()._get_request_payload(input_, stop=stop, **kwargs)
                    try:
                        # If this call will use the Responses API, drop `stop` to avoid 400s.
                        if hasattr(self, "_use_responses_api") and self._use_responses_api(payload):  # type: ignore[attr-defined]
                            payload.pop("stop", None)
                            # Also drop temperature for gpt-5 models as they only support default value
                            payload.pop("temperature", None)
                    except Exception:
                        # Be conservative: if anything goes wrong, still remove `stop` and `temperature`.
                        payload.pop("stop", None)
                        payload.pop("temperature", None)
                    return payload

            return _ChatOpenAIResponsesNoStop(
                model=model,
                temperature=1,  # Set to default value for gpt-5, will be removed in payload
                max_tokens=max_tokens,
                stop_sequences=stop_sequences,
                use_responses_api=True,
                output_version="v0",
            )
        else:
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stop_sequences=stop_sequences,
            )

    elif source == "AzureOpenAI":
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for Azure OpenAI models. Install with: pip install langchain-openai"
            )
        API_VERSION = "2024-12-01-preview"
        model = model.replace("azure-", "")
        return AzureChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
            azure_deployment=model,
            openai_api_version=API_VERSION,
            temperature=temperature,
        )

    elif source == "Anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-anthropic package is required for Anthropic models. Install with: pip install langchain-anthropic"
            )

        # Ensure ANTHROPIC_API_KEY is loaded from bash_profile if not in environment
        if not os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import subprocess

                result = subprocess.run(
                    ["bash", "-c", "source ~/.bash_profile 2>/dev/null && echo $ANTHROPIC_API_KEY"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.stdout.strip():
                    os.environ["ANTHROPIC_API_KEY"] = result.stdout.strip()
                    print("✓ Loaded ANTHROPIC_API_KEY from ~/.bash_profile")
            except Exception as e:
                print(f"Note: Could not load ANTHROPIC_API_KEY from bash_profile: {e}")

        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens or 8192,
            stop_sequences=stop_sequences,
        )

    elif source == "Gemini":
        # If you want to use ChatGoogleGenerativeAI, you need to pass the stop sequences upon invoking the model.
        # return ChatGoogleGenerativeAI(
        #     model=model,
        #     temperature=temperature,
        #     google_api_key=api_key,
        # )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for Gemini models. Install with: pip install langchain-openai"
            )
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            stop_sequences=stop_sequences,
        )

    elif source == "Groq":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for Groq models. Install with: pip install langchain-openai"
            )
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            stop_sequences=stop_sequences,
        )

    elif source == "MiniMax":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for MiniMax models. Install with: pip install langchain-openai"
            )

        minimax_api_key, minimax_base_url = _load_minimax_credentials()
        if api_key and api_key != "EMPTY":
            minimax_api_key = api_key
        if base_url is not None:
            minimax_base_url = base_url
        if minimax_base_url is None:
            minimax_base_url = "https://api.minimaxi.com/v1"
        if not minimax_api_key:
            raise ValueError(
                "MiniMax API key not found. Set MINIMAX_API_KEY or create minimax_apikey.txt with the API key."
            )

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens or 8192,
            stop_sequences=stop_sequences,
            api_key=minimax_api_key,
            base_url=minimax_base_url,
        )

    elif source == "Qwen":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for Qwen models. Install with: pip install langchain-openai"
            )

        qwen_api_key, qwen_base_url = _load_qwen_credentials()
        if api_key and api_key != "EMPTY":
            qwen_api_key = api_key
        if base_url is not None:
            qwen_base_url = base_url
        if qwen_base_url is None:
            qwen_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if not qwen_api_key:
            raise ValueError(
                "Qwen API key not found. Set DASHSCOPE_API_KEY or QWEN_API_KEY, "
                "or create qwen_apikey.txt with the API key."
            )

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens or 8192,
            stop_sequences=stop_sequences,
            api_key=qwen_api_key,
            base_url=qwen_base_url,
        )

    elif source == "Ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-ollama package is required for Ollama models. Install with: pip install langchain-ollama"
            )
        return ChatOllama(
            model=model,
            temperature=temperature,
        )

    elif source == "Bedrock":
        try:
            from langchain_aws import ChatBedrock
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-aws package is required for Bedrock models. Install with: pip install langchain-aws"
            )
        return ChatBedrock(
            model=model,
            temperature=temperature,
            stop_sequences=stop_sequences,
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )

    elif source == "Custom":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for custom models. Install with: pip install langchain-openai"
            )
        # Custom LLM serving such as SGLang. Must expose an openai compatible API.
        assert base_url is not None, "base_url must be provided for customly served LLMs"
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens or 8192,
            stop_sequences=stop_sequences,
            base_url=base_url,
            api_key=api_key,
        )
        return llm

    else:
        raise ValueError(
            f"Invalid source: {source}. Valid options are 'OpenAI', 'AzureOpenAI', 'Anthropic', 'Gemini', 'Groq', 'MiniMax', 'Qwen', 'Bedrock', 'Ollama', or 'Custom'"
        )
