#!/usr/bin/env python3
"""
Example script demonstrating how to use image overlay functionality
to display translated text on images at bbox locations.

Usage:
    python examples/image_overlay_demo.py
"""

import asyncio
from pathlib import Path

from fastapi import UploadFile
from starlette.datastructures import Headers

from bs_translator_backend.models.app_config import AppConfig
from bs_translator_backend.models.language import Language
from bs_translator_backend.models.translation_config import TranslationConfig
from bs_translator_backend.services.custom_llms.qwen3 import QwenVllm
from bs_translator_backend.services.document_conversion_service import DocumentConversionService
from bs_translator_backend.services.llm_facade import LLMFacade
from bs_translator_backend.services.text_chunk_service import TextChunkService
from bs_translator_backend.services.translation_service import TranslationService
from bs_translator_backend.utils.image_overlay import (
    create_side_by_side_comparison,
    overlay_translations_on_image,
)
from bs_translator_backend.utils.load_env import load_env


async def demonstrate_image_overlay() -> None:
    """Demonstrate image overlay functionality."""
    print("🚀 Starting image overlay demonstration...")

    # Load environment and configuration
    load_env()
    app_config = AppConfig.from_env()

    # Initialize services
    conversion_service = DocumentConversionService(app_config)
    llm = QwenVllm(app_config)
    llm_facade = LLMFacade(llm)
    text_chunk_service = TextChunkService()
    translation_service = TranslationService(llm_facade, text_chunk_service, conversion_service)

    # Path to your test file
    pdf_path = Path("./tests/assets/Pocketbook.pdf")
    if not pdf_path.exists():
        print(f"❌ Test file not found: {pdf_path}")
        print("Please ensure you have test assets available.")
        return

    print(f"📄 Processing file: {pdf_path}")

    try:
        with open(pdf_path, "rb") as file:
            headers = Headers({"content-type": "application/pdf"})
            upload_file = UploadFile(file=file, filename=pdf_path.name, headers=headers)

            # Configure translation (German to English)
            translate_config = TranslationConfig(
                source_language=Language.DE, target_language=Language.EN
            )

            print("🔄 Translating document...")

            # Collect translation entries
            translation_entries = []
            async for entry in translation_service.translate_image(upload_file, translate_config):
                print(f"📝 Translated: '{entry.original}' -> '{entry.translated}'")
                print(
                    f"📍 BBox: ({entry.bbox.left:.1f}, {entry.bbox.top:.1f}) to ({entry.bbox.right:.1f}, {entry.bbox.bottom:.1f})"
                )
                translation_entries.append(entry)

            if not translation_entries:
                print("⚠️  No translations found.")
                return

            print(f"✅ Found {len(translation_entries)} translation entries")

            # Create output directory
            output_dir = Path("./output")
            output_dir.mkdir(exist_ok=True)

            # Method 1: Simple overlay
            print("🎨 Creating overlay image...")
            try:
                # Note: This assumes you have an image representation of your PDF
                # You might need to convert PDF to image first or use a different approach
                image_path = "./tests/assets/ReportView.png"  # Use existing test image

                _ = overlay_translations_on_image(
                    image_path,
                    translation_entries,
                    output_path=output_dir / "translated_overlay.png",
                    font_size=16,
                    text_color="red",
                    background_color="white",
                    background_opacity=200,
                )
                print(f"✅ Overlay saved to: {output_dir / 'translated_overlay.png'}")

            except Exception as e:
                print(f"⚠️  Could not create overlay: {e}")

            # Method 2: Side-by-side comparison
            print("🔀 Creating side-by-side comparison...")
            try:
                _ = create_side_by_side_comparison(
                    image_path,
                    translation_entries,
                    output_path=output_dir / "side_by_side_comparison.png",
                    font_size=14,
                    text_color="blue",
                    background_color="yellow",
                    background_opacity=180,
                )
                print(f"✅ Comparison saved to: {output_dir / 'side_by_side_comparison.png'}")

            except Exception as e:
                print(f"⚠️  Could not create comparison: {e}")

            # Method 3: Custom styling example
            print("🎨 Creating custom styled overlay...")
            try:
                _ = overlay_translations_on_image(
                    image_path,
                    translation_entries,
                    output_path=output_dir / "custom_styled.png",
                    font_size=18,
                    text_color="white",
                    background_color="black",
                    background_opacity=150,
                )
                print(f"✅ Custom styled overlay saved to: {output_dir / 'custom_styled.png'}")

            except Exception as e:
                print(f"⚠️  Could not create custom overlay: {e}")

            print("\n🎉 Demonstration complete!")
            print(f"📁 Check the '{output_dir}' directory for generated images.")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")


if __name__ == "__main__":
    asyncio.run(demonstrate_image_overlay())
