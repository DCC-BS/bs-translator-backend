import os
import shutil
import tempfile
import zipfile

from lxml import etree as ET

from bs_translator_backend.translator.config import TranslationConfig
from bs_translator_backend.translator.base_translator import BaseTranslator


class DocxTranslator(BaseTranslator):
    """Translator for DOCX files"""

    def __init__(self):
        super().__init__()
        self.namespaces = {
            "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
            "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
            "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
            "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
            "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
            "v": "urn:schemas-microsoft-com:vml",
        }
        for prefix, uri in self.namespaces.items():
            ET.register_namespace(prefix, uri)

    def translate(
        self, input_path: str, output_path: str, config: TranslationConfig
    ) -> None:
        """Translates a DOCX file"""
        temp_dir = tempfile.mkdtemp()
        try:
            self._process_docx(input_path, output_path, temp_dir, config)
        finally:
            shutil.rmtree(temp_dir)

    def _process_docx(
        self,
        input_path: str,
        output_path: str,
        temp_dir: str,
        config: TranslationConfig,
    ) -> None:
        """Processes the DOCX file"""
        with zipfile.ZipFile(input_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        xml_files = ["word/document.xml"] + [
            f"word/{name}"
            for name in os.listdir(os.path.join(temp_dir, "word"))
            if name.startswith(("header", "footer")) and name.endswith(".xml")
        ]

        for xml_file in xml_files:
            xml_path = os.path.join(temp_dir, xml_file)
            if os.path.exists(xml_path):
                self._process_xml(xml_path, config)

        self._create_output_docx(temp_dir, output_path)

    def _create_output_docx(self, temp_dir: str, output_docx_path: str) -> None:
        """Rezip the contents into a new .docx file"""
        with zipfile.ZipFile(output_docx_path, "w", zipfile.ZIP_DEFLATED) as docx:
            for foldername, subfolders, filenames in os.walk(temp_dir):
                for filename in filenames:
                    file_path = os.path.join(foldername, filename)
                    # Get the path relative to the temporary directory
                    archive_name = os.path.relpath(file_path, temp_dir)
                    docx.write(file_path, archive_name)

    def _get_run_properties(self, elem):
        """Get the formatting properties of a text run"""
        parent_run = elem.getparent()  # Get the parent 'w:r' element
        if parent_run is not None:
            props = parent_run.find(".//w:rPr", namespaces=self.namespaces)
            return ET.tostring(props) if props is not None else None
        return None

    def _process_xml(self, xml_path: str, config: TranslationConfig) -> None:
        """
        Parses and translates text within an XML file while preserving the structure.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        previous_translation = ""

        # Find all paragraphs
        for paragraph in root.findall(".//w:p", namespaces=self.namespaces):
            current_text = []
            current_format = None
            current_elem = None

            # Iterate through text elements in the paragraph
            for elem in paragraph.findall(".//w:t", namespaces=self.namespaces):
                if not elem.text or not elem.text.strip():
                    continue

                elem_format = self._get_run_properties(elem)

                # If this element has the same formatting as previous elements, combine them
                if elem_format == current_format:
                    current_text.append(elem.text)
                    elem.text = ""  # Clear the current element's text
                else:
                    # Translate and clear accumulated text if we have any
                    if current_text and current_elem is not None:
                        combined_text = "".join(current_text)
                        current_elem.text = self.translate_text(
                            text=combined_text, config=config
                        )
                        previous_translation = current_elem.text
                        config.context = previous_translation

                    # Start new accumulation
                    current_text = [elem.text]
                    current_format = elem_format
                    current_elem = elem

            # Handle the last group of text
            if current_text and current_elem is not None:
                combined_text = "".join(current_text)
                current_elem.text = self.translate_text(
                    text=combined_text, config=config
                )

        # Write the modified XML back to file
        tree.write(xml_path, xml_declaration=True, encoding="UTF-8", method="xml")
