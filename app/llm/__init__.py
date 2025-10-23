from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from config import settings

logger = logging.getLogger("chat.llm")


class LLMConfigurationError(RuntimeError):
    """Raised when the configured language model cannot be initialised."""


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    """Return a cached chat model instance based on environment configuration."""
    provider = (settings.llm_provider or "").strip().lower()
    temperature = settings.llm_temperature
    model_name = settings.llm_model

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - import guard
            raise LLMConfigurationError(
                "langchain-openai 패키지가 필요합니다. requirements.txt를 확인하세요."
            ) from exc

        if not settings.openai_api_key:
            raise LLMConfigurationError(
                "OPENAI_API_KEY가 설정되지 않았습니다. 환경 변수를 확인하세요."
            )

        kwargs = {
            "model": model_name,
            "temperature": temperature,
            "api_key": settings.openai_api_key,
        }
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url

        logger.info("Initialising ChatOpenAI model=%s", model_name)
        return ChatOpenAI(**kwargs)

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            try:
                from langchain_community.chat_models import ChatOllama  # type: ignore
                logger.warning(
                    "Using deprecated ChatOllama from langchain-community; "
                    "consider installing langchain-ollama>=0.1.0.",
                )
            except ImportError as exc2:
                raise LLMConfigurationError(
                    "langchain-ollama 패키지가 필요합니다. requirements.txt를 확인하세요."
                ) from exc2

        logger.info(
            "Initialising ChatOllama model=%s base_url=%s",
            model_name,
            settings.ollama_base_url,
        )
        return ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=settings.ollama_base_url,
        )

    raise LLMConfigurationError(
        f"지원하지 않는 LLM 제공자입니다: {settings.llm_provider!r}. "
        "LLM_PROVIDER 환경 변수를 'openai' 또는 'ollama'로 설정하세요."
    )
