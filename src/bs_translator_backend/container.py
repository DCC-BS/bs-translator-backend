from dependency_injector import containers, providers
from llama_index.core.llms import LLM

from bs_translator_backend.models.app_config import AppConfig
from bs_translator_backend.services.custom_llms.qwen3 import QwenVllm
from bs_translator_backend.services.llm_facade import LLMFacade
from bs_translator_backend.services.translation_service import TranslationService


class Container(containers.DeclarativeContainer):
    app_config: providers.Singleton[AppConfig] = providers.Singleton(AppConfig)

    llm: providers.Singleton[LLM] = providers.Singleton(
        QwenVllm,
        config=app_config,
    )

    llm_facade: providers.Singleton[LLMFacade] = providers.Singleton(
        LLMFacade,
        llm=llm,
    )

    translation_service: providers.Singleton[TranslationService] = providers.Singleton(
        TranslationService,
        llm_facade=llm_facade,
    )
