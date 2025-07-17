# import math
# from typing import Tuple
# import fitz
# from bs_translator_backend.translator.config import TranslationConfig
# from bs_translator_backendtranslator.base_translator import BaseTranslator

# from docling.datamodel.base_models import InputFormat, BoundingBox
# from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
# from docling.document_converter import DocumentConverter, PdfFormatOption
# from docling.datamodel.document import TextItem
# from collections import defaultdict

# pipeline_options = PdfPipelineOptions(do_table_structure=True)
# pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

# doc_converter = DocumentConverter(
#     format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
# )


# class PdfTranslator(BaseTranslator):
#     """Translator for PDF files"""

#     # def translate_text(self, text: str, config: TranslationConfig) -> str:
#     #     return text + " " + text[: int(0.1 * len(text))]

#     def translate(
#         self, input_path: str, output_path: str, translation_config: TranslationConfig
#     ) -> None:
#         doc = fitz.open(input_path)
#         new_fitz_doc = fitz.open()
#         translation_config.context = ""

#         try:
#             result = doc_converter.convert(input_path, max_num_pages=60)
#             current_page_num = 0
#             current_fitz_page = new_fitz_doc.new_page(
#                 width=doc[0].rect.width, height=doc[0].rect.height
#             )
#             total_additional_v_space = 0

#             for item, level in result.document.iterate_items():
#                 page_num = item.prov[0].page_no - 1

#                 if page_num != current_page_num:
#                     # Initialize new page
#                     current_page_num = page_num
#                     current_fitz_page = new_fitz_doc.new_page(
#                         width=doc[page_num].rect.width, height=doc[page_num].rect.height
#                     )

#                 if isinstance(item, TextItem):
#                     bbox: BoundingBox = item.prov[0].bbox
#                     translated_text = self.translate_text(item.text, translation_config)
#                     page_height = result.pages[page_num].size.height
#                     rect = bbox.to_top_left_origin(page_height=page_height).as_tuple()
#                     rect = fitz.Rect(rect)

#                     font, fontsize, fontcolor, alignment, line_spacing = self._get_fonts_in_rect(
#                         doc[page_num], rect
#                     )
#                     fontsize = math.floor(fontsize) - 2
#                     text_writer = fitz.TextWriter(
#                         current_fitz_page.rect, color=fontcolor
#                     )

#                     # Shift rect down according to the previously moved elements
#                     rect.y0 += total_additional_v_space
#                     rect.y1 += total_additional_v_space

#                     not_written_lines = text_writer.fill_textbox(
#                         rect,
#                         translated_text,
#                         font=font,
#                         fontsize=fontsize,
#                         align=alignment,
#                     )
#                     text_writer.write_text(current_fitz_page)
                    
#                     if not_written_lines:
#                         additional_v_space = 0
#                         not_written_text = ""
#                         for text, text_length in not_written_lines:
#                             additional_v_space += line_spacing + fontsize + line_spacing
#                             not_written_text += text
                        
#                         if additional_v_space > 0:
#                             additional_v_space -= line_spacing # Remove the last line spacing
#                             additional_v_space += font.ascender * 1.2
#                         new_rect = fitz.Rect((rect.x0, text_writer.last_point.y, rect.x1, text_writer.last_point.y+additional_v_space))
#                         # Write the lines that are missing
#                         try:
#                             text_writer.fill_textbox(
#                                 new_rect,
#                                 not_written_text,
#                                 font=font,
#                                 fontsize=fontsize,
#                                 align=alignment
#                             )
#                         except Exception as e:
#                             print(e)
#                         text_writer.write_text(current_fitz_page)

#                     # print("Warning, those lines were not written: ", not_written_lines)

#             for page_num in range(doc.page_count):
#                 page = doc[page_num]
#                 text_dict = page.get_text("dict")
#                 blocks = text_dict["blocks"]
#                 current_page = new_fitz_doc[page_num]
#                 for block in blocks:
#                     if block["type"] == 1:  # Image block
#                         current_page.insert_image(
#                             rect=fitz.Rect(block["bbox"]),
#                             stream=block["image"],
#                             overlay=False,
#                         )

#             text_writer.write_text(current_fitz_page)
#             new_fitz_doc.save(output_path)

#         except Exception as e:
#             print(e)
#             raise e
#         finally:
#             doc.close()
#             new_fitz_doc.close()

#     def _get_fonts_in_rect(self, page: fitz.Page, rect: fitz.Rect) -> Tuple:
#         """
#         Extract fonts used within a specified rectangle on a PDF page, and determine the overall text alignment.

#         Args:
#             page: fitz.Page object - The PDF page to analyze
#             rect: fitz.Rect object - The rectangle area to check for fonts

#         Returns:
#             tuple: The most common font as fitz.Font, average font size, most common font color, majority text alignment, average line spacing
#         """
#         try:
#             from collections import Counter

#             text_dict = page.get_text("dict", clip=rect)
#             left_boundary = rect.x0
#             right_boundary = rect.x1
#             rect_width = right_boundary - left_boundary

#             alignments = []
#             font_properties = {
#                 "font_names": [],
#                 "font_sizes": [],
#                 "font_colors": [],
#                 "font_flags": [],
#                 "font_ascender": [],
#                 "font_descender": [],
#                 "font_linespacing": []
#             }

#             # Thresholds and tolerances
#             full_width_threshold = 0.98  # 98% of rectangle width
#             margin_tolerance = 5  # Tolerance for margin differences
#             small_margin = 10  # Threshold for considering a margin small

#             for block in text_dict.get("blocks", []):
#                 for line in block.get("lines", []):
#                     if not line.get("spans"):
#                         continue
#                     line_x0 = line["bbox"][0]
#                     line_x1 = line["bbox"][2]
#                     line_y0 = line["bbox"][1]
#                     line_y1 = line["bbox"][3]

#                     left_margin = line_x0 - left_boundary
#                     right_margin = right_boundary - line_x1

#                     line_width = line_x1 - line_x0
#                     line_height = line_y1 - line_y0
#                     width_ratio = line_width / rect_width
#                     margin_diff = abs(left_margin - right_margin)

#                     # Collect font properties from this line
#                     for span in line.get("spans", []):
#                         if "font" in span:
#                             font_name = span["font"]
#                             font_size = span["size"]
#                             font_color = span["color"]
#                             font_flags = span["flags"]
#                             font_ascender = span["ascender"]
#                             font_descender = span["descender"]

#                             font_properties["font_names"].append(font_name)
#                             font_properties["font_sizes"].append(font_size)
#                             font_properties["font_colors"].append(font_color)
#                             font_properties["font_flags"].append(font_flags)
#                             font_properties["font_ascender"].append(font_ascender)
#                             font_properties["font_descender"].append(font_descender)

#                             line_spacing = line_height - font_size
#                             font_properties["font_linespacing"].append(line_spacing)

#                     # Determine alignment
#                     if width_ratio >= full_width_threshold:
#                         alignment = 3  # JUSTIFIED
#                     elif margin_diff < margin_tolerance:
#                         alignment = 1  # CENTER
#                     elif left_margin < small_margin and right_margin > small_margin:
#                         alignment = 0  # LEFT
#                     elif right_margin < small_margin and left_margin > small_margin:
#                         alignment = 2  # RIGHT
#                     else:
#                         # Default to LEFT if unclear
#                         alignment = 0

#                     alignments.append(alignment)

#             # Determine majority alignment
#             alignment_counter = Counter(alignments)
#             majority_alignment = alignment_counter.most_common(1)[0][0]

#             # Determine most common font properties
#             font_name_counter = Counter(font_properties["font_names"])
#             most_common_font_name = font_name_counter.most_common(1)[0][0]

#             font_flags_counter = Counter(font_properties["font_flags"])
#             most_common_font_flags = font_flags_counter.most_common(1)[0][0]

#             # Average font size
#             average_font_size = sum(font_properties["font_sizes"]) / len(
#                 font_properties["font_sizes"]
#             )

#             # Most common font color
#             font_color_counter = Counter(font_properties["font_colors"])
#             most_common_font_color = font_color_counter.most_common(1)[0][0]

#             font_ascender_counter = Counter(font_properties["font_ascender"])
#             most_common_font_ascender = font_ascender_counter.most_common(1)[0][0]

#             font_descender_counter = Counter(font_properties["font_descender"])
#             most_common_font_descender = font_descender_counter.most_common(1)[0][0]

#             average_line_spacing = sum(font_properties["font_linespacing"]) / len(font_properties["font_linespacing"])

#             # Convert font color to RGB tuple if necessary
#             if isinstance(most_common_font_color, int):
#                 r = ((most_common_font_color >> 16) & 0xFF) / 255
#                 g = ((most_common_font_color >> 8) & 0xFF) / 255
#                 b = (most_common_font_color & 0xFF) / 255
#                 most_common_font_color = (r, g, b)

#             # Get fallback font
#             fallback_font = self._get_fallback_font(
#                 most_common_font_name, most_common_font_flags, most_common_font_ascender, most_common_font_descender
#             )

#             return (
#                 fallback_font,
#                 average_font_size,
#                 most_common_font_color,
#                 majority_alignment,
#                 average_line_spacing
#             )

#         except Exception as e:
#             print(f"Error extracting fonts: {str(e)}")
#             raise e

#     def _create_translation_context(
#         self, current_context: str, new_translation: str, max_context_length: int = 1000
#     ) -> str:
#         """
#         Creates a translation context by combining current context with new translation,
#         respecting the maximum context length and maintaining sentence integrity.

#         Args:
#             current_context (str): Existing context string
#             new_translation (str): New translation to be added
#             max_context_length (int): Maximum allowed length of the combined context

#         Returns:
#             str: Updated context string within max length constraints
#         """
#         # Add space between contexts if needed
#         separator = " " if current_context and new_translation else ""
#         combined = current_context + separator + new_translation

#         # If combined length is within limit, return the full context
#         if len(combined) <= max_context_length:
#             return combined

#         # If new translation alone exceeds limit, truncate it from the start of a sentence
#         if len(new_translation) > max_context_length:
#             # Find the first sentence boundary after max_context_length characters from the end
#             for punct in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
#                 start_pos = new_translation[-max_context_length:].find(punct)
#                 if start_pos != -1:
#                     return new_translation[-(max_context_length - start_pos - 2) :]
#             return new_translation[
#                 -max_context_length:
#             ]  # Fallback if no sentence boundary found

#         # Otherwise, trim from the beginning while preserving complete sentences
#         excess = len(combined) - max_context_length
#         truncated = combined[excess:]

#         # Find the start of the first complete sentence
#         for punct in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
#             start_pos = truncated.find(punct)
#             if start_pos != -1:
#                 return truncated[start_pos + 2 :]

#         return truncated  # Fallback if no sentence boundary found

#     def _get_docling_bboxes(self, input_path):
#         bboxes_per_page = defaultdict(list)
#         result = doc_converter.convert(input_path, max_num_pages=60)
#         for item, level in result.document.iterate_items():
#             if isinstance(item, TextItem):
#                 page_num = item.prov[0].page_no - 1
#                 page_height = result.pages[page_num].size.height
#                 bbox: BoundingBox = item.prov[0].bbox
#                 rect = bbox.to_top_left_origin(page_height=page_height).as_tuple()
#                 bboxes_per_page[page_num].append(rect)

#     def _get_fallback_font(self, font_name: str, font_flags, ascender: float, descender: float) -> fitz.Font:
#             # Default fallback font family is Helvetica
#             base_font = "helvetica"

#             # Check if the original font has serif characteristics
#             if "times" in font_name.lower() or font_flags & 1:  # serif flag is 1
#                 base_font = "times-roman"
#             elif "courier" in font_name.lower() or font_flags & 32:  # mono flag is 32
#                 base_font = "courier"

#             # Determine style variations
#             is_bold = font_flags & 2 != 0  # bold flag is 2
#             is_italic = font_flags & 4 != 0  # italic flag is 4

#             # Create font instance with style variations
#             if is_bold and is_italic:
#                 if base_font == "times-roman":
#                     font = fitz.Font("times", is_bold=1, is_italic=1)
#                 else:
#                     font = fitz.Font(base_font, is_bold=1, is_italic=1)
#             elif is_bold:
#                 font = fitz.Font(base_font, is_bold=1)
#             elif is_italic:
#                 if base_font == "times-roman":
#                     font = fitz.Font("times", is_italic=1)
#                 else:
#                     font = fitz.Font(base_font, is_italic=1)
#             else:
#                 font = fitz.Font(base_font)

#             return font
