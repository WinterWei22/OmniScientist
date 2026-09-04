from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI


_TRUE_VALUES = {"1", "true", "yes", "on"}
_BAILIAN_PROVIDER_NAMES = {"aliyun", "bailian", "dashscope"}
_LOCAL_PROVIDER_NAMES = {"local", "local_vllm", "openai_compatible"}
DEFAULT_BAILIAN_MAX_TOKENS = 81960

_BAILIAN_PUBLIC_BASE_URLS = {
    "beijing": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "singapore": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "virginia": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
}

_BAILIAN_WORKSPACE_HOSTS = {
    "beijing": "{workspace_id}.cn-beijing.maas.aliyuncs.com",
    "singapore": "{workspace_id}.ap-southeast-1.maas.aliyuncs.com",
    "tokyo": "{workspace_id}.ap-northeast-1.maas.aliyuncs.com",
}


def _env_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _TRUE_VALUES


def _env_int(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    parsed = int(value)
    if parsed < 1:
        raise ValueError("token and retry limits must be positive")
    return parsed


def _env_float(value: Any, default: float) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _first(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return default


@dataclass(frozen=True, slots=True)
class QwenProviderSettings:
    provider: str
    model: str
    base_url: str
    api_key: str = field(repr=False)
    enable_thinking: bool = False
    thinking_budget: int | None = None
    max_tokens: int = 4096
    temperature: float = 0.2
    top_p: float = 0.9
    timeout_seconds: float = 300.0
    max_retries: int = 3

    def public_summary(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_configured": bool(self.api_key),
            "enable_thinking": self.enable_thinking,
            "thinking_budget": self.thinking_budget,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
        }


class QwenAPIProvider:
    """Build a Qwen LangChain client for Alibaba Bailian or local vLLM."""

    def __init__(self, settings: QwenProviderSettings) -> None:
        self.settings = settings

    @classmethod
    def from_environment(
        cls,
        fallback: Mapping[str, Any] | None = None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> QwenAPIProvider:
        values = os.environ if environ is None else environ
        fallback = fallback or {}
        provider_name = str(
            _first(values.get("QWEN_PROVIDER"), fallback.get("provider"), default="bailian")
        ).strip().lower()

        if provider_name in _BAILIAN_PROVIDER_NAMES:
            settings = cls._bailian_settings(values, fallback)
        elif provider_name in _LOCAL_PROVIDER_NAMES:
            settings = cls._local_settings(values, fallback)
        else:
            allowed = sorted(_BAILIAN_PROVIDER_NAMES | _LOCAL_PROVIDER_NAMES)
            raise ValueError(
                f"Unsupported QWEN_PROVIDER={provider_name!r}; choose one of {allowed}"
            )
        return cls(settings)

    @staticmethod
    def _bailian_settings(
        values: Mapping[str, str], fallback: Mapping[str, Any]
    ) -> QwenProviderSettings:
        region = str(
            _first(values.get("BAILIAN_REGION"), fallback.get("bailian_region"), default="beijing")
        ).strip().lower()
        workspace_id = str(
            _first(
                values.get("BAILIAN_WORKSPACE_ID"),
                fallback.get("bailian_workspace_id"),
                default="",
            )
        ).strip()
        explicit_base_url = str(
            _first(
                values.get("BAILIAN_BASE_URL"),
                fallback.get("bailian_base_url"),
                default="",
            )
        ).strip()

        if explicit_base_url:
            base_url = explicit_base_url.rstrip("/")
        elif workspace_id:
            host_template = _BAILIAN_WORKSPACE_HOSTS.get(region)
            if host_template is None:
                raise ValueError(
                    f"BAILIAN_WORKSPACE_ID is not supported for region {region!r}; "
                    "set BAILIAN_BASE_URL explicitly"
                )
            host = host_template.format(workspace_id=workspace_id)
            base_url = f"https://{host}/compatible-mode/v1"
        else:
            try:
                base_url = _BAILIAN_PUBLIC_BASE_URLS[region]
            except KeyError as exc:
                raise ValueError(
                    f"No public Bailian URL is configured for region {region!r}; "
                    "set BAILIAN_WORKSPACE_ID or BAILIAN_BASE_URL"
                ) from exc

        if not base_url.startswith("https://"):
            raise ValueError("Bailian Base URL must use HTTPS")

        thinking_budget_raw = _first(
            values.get("BAILIAN_THINKING_BUDGET"),
            fallback.get("bailian_thinking_budget"),
        )
        thinking_budget = (
            _env_int(thinking_budget_raw, 1) if thinking_budget_raw not in (None, "") else None
        )

        return QwenProviderSettings(
            provider="bailian",
            model=str(
                _first(
                    values.get("BAILIAN_MODEL"),
                    values.get("BIOMNI_BAILIAN_MODEL"),
                    fallback.get("bailian_model"),
                    default="qwen3.8-max",
                )
            ),
            base_url=base_url,
            api_key=str(values.get("DASHSCOPE_API_KEY", "")).strip(),
            enable_thinking=_env_bool(
                _first(
                    values.get("BAILIAN_ENABLE_THINKING"),
                    fallback.get("bailian_enable_thinking"),
                ),
                False,
            ),
            thinking_budget=thinking_budget,
            max_tokens=_env_int(
                _first(
                    values.get("BAILIAN_MAX_TOKENS"),
                    fallback.get("bailian_max_tokens"),
                ),
                DEFAULT_BAILIAN_MAX_TOKENS,
            ),
            temperature=_env_float(
                _first(
                    values.get("BAILIAN_TEMPERATURE"),
                    fallback.get("bailian_temperature"),
                ),
                0.2,
            ),
            top_p=_env_float(
                _first(values.get("BAILIAN_TOP_P"), fallback.get("bailian_top_p")),
                0.9,
            ),
            timeout_seconds=_env_float(
                _first(
                    values.get("BAILIAN_TIMEOUT_SECONDS"),
                    fallback.get("bailian_timeout_seconds"),
                ),
                300.0,
            ),
            max_retries=_env_int(
                _first(
                    values.get("BAILIAN_MAX_RETRIES"),
                    fallback.get("bailian_max_retries"),
                ),
                3,
            ),
        )

    @staticmethod
    def _local_settings(
        values: Mapping[str, str], fallback: Mapping[str, Any]
    ) -> QwenProviderSettings:
        return QwenProviderSettings(
            provider="local_vllm",
            model=str(
                _first(
                    values.get("QWEN_MODEL"),
                    fallback.get("model"),
                    default="Qwen/Qwen3.5-27B",
                )
            ),
            base_url=str(
                _first(
                    values.get("QWEN_OPENAI_API_BASE_URL"),
                    fallback.get("api_base"),
                    default="http://127.0.0.1:8000/v1",
                )
            ).rstrip("/"),
            api_key=str(
                _first(
                    values.get("QWEN_OPENAI_API_KEY"),
                    fallback.get("api_key"),
                    default="EMPTY",
                )
            ),
            enable_thinking=_env_bool(
                _first(values.get("QWEN_ENABLE_THINKING"), fallback.get("enable_thinking")),
                False,
            ),
            max_tokens=_env_int(
                _first(values.get("QWEN_MAX_TOKENS"), fallback.get("max_tokens")),
                4096,
            ),
            temperature=_env_float(
                _first(values.get("QWEN_TEMPERATURE"), fallback.get("temperature")),
                0.2,
            ),
            top_p=_env_float(
                _first(values.get("QWEN_TOP_P"), fallback.get("top_p")),
                0.9,
            ),
            timeout_seconds=_env_float(
                _first(values.get("QWEN_TIMEOUT_SECONDS"), fallback.get("timeout_seconds")),
                300.0,
            ),
            max_retries=_env_int(
                _first(values.get("QWEN_MAX_RETRIES"), fallback.get("max_retries")),
                3,
            ),
        )

    def create_chat_model(self) -> ChatOpenAI:
        settings = self.settings
        if settings.provider == "bailian" and not settings.api_key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY is not configured. Create a Bailian API Key and "
                "expose it only through the server environment."
            )

        extra_body: dict[str, Any] = {
            "enable_thinking": settings.enable_thinking,
        }
        if settings.enable_thinking and settings.thinking_budget is not None:
            extra_body["thinking_budget"] = settings.thinking_budget

        return ChatOpenAI(
            model=settings.model,
            api_key=settings.api_key or "EMPTY",
            base_url=settings.base_url,
            temperature=settings.temperature,
            top_p=settings.top_p,
            max_tokens=settings.max_tokens,
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
            extra_body=extra_body,
        )
