import tempfile
from contextlib import suppress
from pathlib import Path

import cv2
import easyocr  # type: ignore[import-untyped]
import numpy as np
from fastapi import UploadFile
from PIL import Image


class OCRError(Exception):
    """Raised when OCR processing fails or no text is found."""

    def __init__(self, message: str = "No text could be extracted from the image"):
        super().__init__(message)

    @classmethod
    def invalid_dimensions(cls) -> "OCRError":
        """Create an OCRError for invalid image dimensions."""
        return cls("Image has invalid dimensions")

    @classmethod
    def load_failed(cls, error: Exception) -> "OCRError":
        """Create an OCRError for image loading failure."""
        return cls(f"Failed to load image: {error!s}")

    @classmethod
    def save_failed(cls) -> "OCRError":
        """Create an OCRError for image save failure."""
        return cls("Failed to save preprocessed image")


class ImageReaderService:
    """
    Service for reading text from images and returning Markdown-formatted output using OCR.
    """

    def __init__(self, languages: list[str] | None = None):
        """
        Initialize the EasyOCR reader with CPU-only mode.

        Args:
            languages: List of language codes for OCR. Defaults to ['en'] if None.
        """
        if languages is None:
            languages = ["en", "de"]
        # Force CPU usage to avoid CUDA memory issues
        self.reader = easyocr.Reader(languages, gpu=False)

    def _preprocess_image(self, image_path: str) -> str:
        """
        Preprocess the image to handle problematic dimensions and aspect ratios.

        Args:
            image_path: Path to the input image

        Returns:
            str: Path to the preprocessed image

        Raises:
            OCRError: If image cannot be loaded or processed
        """
        # Load image
        try:
            image = cv2.imread(image_path)
        except Exception:
            image = None

        if image is None or image.size == 0:
            # Try with PIL if cv2 fails
            try:
                pil_image = Image.open(image_path)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            except Exception as e:
                raise OCRError.load_failed(e) from e

        height, width = image.shape[:2]

        # Validate image has valid dimensions
        if height <= 0 or width <= 0:
            raise OCRError.invalid_dimensions()

        # Check for problematic dimensions
        min_dimension = 10
        max_dimension = 4000
        max_aspect_ratio = 20  # Maximum width/height or height/width ratio

        # Fix very small dimensions
        if height < min_dimension or width < min_dimension:
            scale_factor = max(min_dimension / height, min_dimension / width)
            new_width = max(int(width * scale_factor), min_dimension)
            new_height = max(int(height * scale_factor), min_dimension)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            height, width = new_height, new_width

        # Fix very large dimensions
        if height > max_dimension or width > max_dimension:
            scale_factor = min(max_dimension / height, max_dimension / width)
            new_width = max(int(width * scale_factor), min_dimension)
            new_height = max(int(height * scale_factor), min_dimension)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            height, width = new_height, new_width

        # Fix extreme aspect ratios
        aspect_ratio = max(width / height, height / width)
        if aspect_ratio > max_aspect_ratio:
            if width > height:
                # Very wide image - pad height
                new_height = max(int(width / max_aspect_ratio), min_dimension)
                padding = (new_height - height) // 2
                image = cv2.copyMakeBorder(
                    image, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255]
                )
            else:
                # Very tall image - pad width
                new_width = max(int(height / max_aspect_ratio), min_dimension)
                padding = (new_width - width) // 2
                image = cv2.copyMakeBorder(
                    image, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255]
                )

        # Save the preprocessed image
        processed_path = image_path.replace(".", "_processed.")
        success = cv2.imwrite(processed_path, image)
        if not success:
            raise OCRError.save_failed()

        return processed_path

    def read_image(self, file: UploadFile) -> str:
        """
        Reads the image content and returns it as markdown using OCR.

        Args:
            file (UploadFile): The uploaded image file.

        Returns:
            str: Markdown-formatted text extracted from the image.

        Raises:
            OCRError: If OCR processing fails or no text is found.
        """
        file_content = file.file.read()

        # Get the file extension from the filename or default to .png
        file_extension = Path(file.filename).suffix if file.filename else ".png"

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        processed_path = None
        try:
            # Preprocess the image to handle dimension issues
            processed_path = self._preprocess_image(tmp_path)

            # Use EasyOCR to extract text from the preprocessed image
            # EasyOCR returns a list where each element is [bbox, text, confidence]
            results = self.reader.readtext(processed_path)

            # Extract text from OCR results (second element of each result)
            extracted_texts: list[str] = []
            for result in results:
                # EasyOCR result format: [bbox, text, confidence]
                if isinstance(result, list | tuple) and len(result) > 1:
                    text = str(result[1]).strip()
                    if text:  # Only add non-empty text
                        extracted_texts.append(text)

            if not extracted_texts:
                raise OCRError()

            # Join all extracted text with newlines and return as markdown
            return "\n".join(extracted_texts)

        finally:
            # Clean up the temporary files
            with suppress(Exception):
                Path(tmp_path).unlink()
            if processed_path:
                with suppress(Exception):
                    Path(processed_path).unlink()
