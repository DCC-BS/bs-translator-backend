"""
Image Overlay Utilities

This module provides utilities for overlaying translated text on images
at specified bounding box locations.
"""

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from bs_translator_backend.models.conversion_result import ConversionImageTextEntry


def overlay_translations_on_image(
    image_data: bytes | str | Path,
    translations: list[ConversionImageTextEntry],
    output_path: str | Path | None = None,
    font_size: int = 12,
    text_color: str = "red",
    background_color: str = "white",
    background_opacity: int = 200,
) -> Image.Image:
    """
    Overlay translated text on an image at bbox locations.

    Args:
        image_data: Image data as bytes, file path, or Path object
        translations: List of translation entries with bbox coordinates
        output_path: Optional path to save the result image
        font_size: Font size for the overlay text
        text_color: Color of the overlay text
        background_color: Background color for text overlay
        background_opacity: Opacity of the text background (0-255)

    Returns:
        PIL Image object with overlaid translations
    """
    # Load the image
    if isinstance(image_data, str | Path):
        image = Image.open(image_data)
    else:
        image = Image.open(BytesIO(image_data))

    # Convert to RGBA for transparency support
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Create a transparent overlay
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Try to load a better font, fall back to default if not available
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size
        )
    except OSError:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

    for entry in translations:
        if not entry.translated.strip():
            continue

        bbox = entry.bbox

        # Calculate text position and size
        text_bbox = draw.textbbox((0, 0), entry.translated, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Position text at the top-left of the bbox
        x = int(bbox.left)
        y = int(bbox.top)

        # Draw semi-transparent background rectangle
        bg_padding = 2
        draw.rectangle(
            [
                x - bg_padding,
                y - bg_padding,
                x + text_width + bg_padding,
                y + text_height + bg_padding,
            ],
            fill=(*_hex_to_rgb(background_color), background_opacity),
        )

        # Draw the translated text
        draw.text((x, y), entry.translated, fill=text_color, font=font)

    # Composite the overlay onto the original image
    result = Image.alpha_composite(image, overlay)

    # Convert back to RGB if needed
    if result.mode == "RGBA":
        result = result.convert("RGB")

    # Save if output path provided
    if output_path:
        result.save(output_path)

    return result


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color string to RGB tuple."""
    if hex_color.startswith("#"):
        hex_color = hex_color[1:]

    # Handle common color names
    color_map = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
    }

    if hex_color.lower() in color_map:
        return color_map[hex_color.lower()]

    # Convert hex to RGB
    if len(hex_color) == 6:
        return (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16))
    elif len(hex_color) == 3:
        return (int(hex_color[0], 16) * 17, int(hex_color[1], 16) * 17, int(hex_color[2], 16) * 17)
    else:
        return (0, 0, 0)  # Default to black


def create_side_by_side_comparison(
    original_image: bytes | str | Path,
    translations: list[ConversionImageTextEntry],
    output_path: str | Path | None = None,
    **overlay_kwargs: object,
) -> Image.Image:
    """
    Create a side-by-side comparison of original and translated image.

    Args:
        original_image: Original image data
        translations: List of translation entries
        output_path: Optional path to save the result
        **overlay_kwargs: Additional arguments for overlay_translations_on_image

    Returns:
        PIL Image object with side-by-side comparison
    """
    # Load original image
    if isinstance(original_image, str | Path):
        orig = Image.open(original_image)
    else:
        orig = Image.open(BytesIO(original_image))

    # Create translated version
    translated = overlay_translations_on_image(original_image, translations, **overlay_kwargs)

    # Ensure both images have the same height
    if orig.height != translated.height:
        # Resize to match the smaller height
        target_height = min(orig.height, translated.height)
        orig = orig.resize((int(orig.width * target_height / orig.height), target_height))
        translated = translated.resize((
            int(translated.width * target_height / translated.height),
            target_height,
        ))

    # Create side-by-side image
    total_width = orig.width + translated.width
    result = Image.new("RGB", (total_width, orig.height), "white")

    # Paste images side by side
    result.paste(orig, (0, 0))
    result.paste(translated, (orig.width, 0))

    # Save if output path provided
    if output_path:
        result.save(output_path)

    return result
