from dependency_injector import containers, providers
from llama_index.core.llms import LLM

from bs_translator_backend.models.app_config import AppConfig
from bs_translator_backend.services.custom_llms.qwen3 import QwenVllm
from bs_translator_backend.services.document_conversion_service import DocumentConversionService
from bs_translator_backend.services.llm_facade import LLMFacade
from bs_translator_backend.services.text_chunk_service import TextChunkService
from bs_translator_backend.services.translation_service import TranslationService
from bs_translator_backend.services.usage_tracking_service import UsageTrackingService


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

    text_chunk_service: providers.Singleton[TextChunkService] = providers.Singleton(
        TextChunkService,
        max_tokens=6000,
    )

    translation_service: providers.Singleton[TranslationService] = providers.Singleton(
        TranslationService,
        llm_facade=llm_facade,
        text_chunk_service=text_chunk_service,
    )

    document_conversion_service: providers.Singleton[DocumentConversionService] = providers.Singleton(
        DocumentConversionService
    )

    usage_tracking_service: providers.Singleton[UsageTrackingService] = providers.Singleton(
        UsageTrackingService,
        config=app_config,
    )
