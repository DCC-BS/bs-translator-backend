from dependency_injector import containers, providers

from bs_translator_backend.services.document_conversion_service import DocumentConversionService
from bs_translator_backend.services.text_chunk_service import TextChunkService
from bs_translator_backend.services.transcription_service import TranscriptionService
from bs_translator_backend.services.translation_service import TranslationService
from bs_translator_backend.services.usage_tracking_service import UsageTrackingService
from bs_translator_backend.utils.app_config import AppConfig


class Container(containers.DeclarativeContainer):
    app_config = providers.Object(AppConfig.from_env())

    text_chunk_service: providers.Singleton[TextChunkService] = providers.Singleton(
        TextChunkService,
        max_tokens=6000,
    )

    document_conversion_service: providers.Factory[DocumentConversionService] = providers.Factory(
        DocumentConversionService, config=app_config
    )

    translation_service: providers.Singleton[TranslationService] = providers.Singleton(
        TranslationService,
        app_config=app_config,
        text_chunk_service=text_chunk_service,
        conversion_service_factory=document_conversion_service.provider,
    )

    usage_tracking_service: providers.Singleton[UsageTrackingService] = providers.Singleton(
        UsageTrackingService,
        config=app_config,
    )

    transcription_service = providers.Singleton(
        TranscriptionService,
        config=app_config,
    )
