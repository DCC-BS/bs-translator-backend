from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ConversionStatus(str, Enum):
    """Status of document conversion."""

    PENDING = "pending"
    STARTED = "started"
    FAILURE = "failure"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    SKIPPED = "skipped"


class DoclingComponentType(str, Enum):
    """Type of docling component that generated an error."""

    DOCUMENT_BACKEND = "document_backend"
    MODEL = "model"
    DOC_ASSEMBLER = "doc_assembler"
    USER_INPUT = "user_input"


class ErrorItem(BaseModel):
    """Error item from docling processing."""

    component_type: DoclingComponentType
    module_name: str
    error_message: str


class ProfilingScope(str, Enum):
    """Scope of profiling measurements."""

    PAGE = "page"
    DOCUMENT = "document"


class ProfilingItem(BaseModel):
    """Profiling information for processing steps."""

    scope: ProfilingScope
    count: int = 0
    times: list[float] = []
    start_timestamps: list[datetime] = []


class BoundingBox(BaseModel):
    """Bounding box coordinates."""

    left: float = Field(alias="l")
    top: float = Field(alias="t")
    right: float = Field(alias="r")
    bottom: float = Field(alias="b")
    coord_origin: str = "TOPLEFT"

    model_config = ConfigDict(populate_by_name=True)


class Size(BaseModel):
    """Size dimensions."""

    width: float
    height: float


class ImageRef(BaseModel):
    """Reference to an image."""

    mimetype: str
    dpi: int
    size: Size
    uri: str


class DocumentOrigin(BaseModel):
    """Origin information of the document."""

    mimetype: str
    binary_hash: int
    filename: str
    uri: str | None = None


class PageItem(BaseModel):
    """Page information."""

    size: Size
    image: ImageRef | None = None
    page_no: int


class ProvenanceItem(BaseModel):
    """Provenance information for document elements."""

    page_no: int
    bbox: BoundingBox
    charspan: tuple[int, int]


class RefItem(BaseModel):
    """Reference to another item in the document."""

    ref: str = Field(alias="$ref")

    model_config = ConfigDict(populate_by_name=True)


class Formatting(BaseModel):
    """Text formatting information."""

    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikethrough: bool = False
    script: str = "baseline"  # baseline, sub, super


class ContentLayer(str, Enum):
    """Content layer of document elements."""

    BODY = "body"
    FURNITURE = "furniture"
    BACKGROUND = "background"
    INVISIBLE = "invisible"
    NOTES = "notes"


class GroupLabel(str, Enum):
    """Labels for group items."""

    UNSPECIFIED = "unspecified"
    LIST = "list"
    ORDERED_LIST = "ordered_list"
    CHAPTER = "chapter"
    SECTION = "section"
    SHEET = "sheet"
    SLIDE = "slide"
    FORM_AREA = "form_area"
    KEY_VALUE_AREA = "key_value_area"
    COMMENT_SECTION = "comment_section"
    INLINE = "inline"
    PICTURE_AREA = "picture_area"


class Script(str, Enum):
    """Text script position."""

    BASELINE = "baseline"
    SUB = "sub"
    SUPER = "super"


class CodeLanguageLabel(str, Enum):
    """Programming language labels for code blocks."""

    ADA = "Ada"
    AWK = "Awk"
    BASH = "Bash"
    BC = "bc"
    C = "C"
    CSHARP = "C#"
    CPP = "C++"
    CMAKE = "CMake"
    COBOL = "COBOL"
    CSS = "CSS"
    CEYLON = "Ceylon"
    CLOJURE = "Clojure"
    CRYSTAL = "Crystal"
    CUDA = "Cuda"
    CYTHON = "Cython"
    D = "D"
    DART = "Dart"
    DC = "dc"
    DOCKERFILE = "Dockerfile"
    ELIXIR = "Elixir"
    ERLANG = "Erlang"
    FORTRAN = "FORTRAN"
    FORTH = "Forth"
    GO = "Go"
    HTML = "HTML"
    HASKELL = "Haskell"
    HAXE = "Haxe"
    JAVA = "Java"
    JAVASCRIPT = "JavaScript"
    JULIA = "Julia"
    KOTLIN = "Kotlin"
    LISP = "Lisp"
    LUA = "Lua"
    MATLAB = "Matlab"
    MOONSCRIPT = "MoonScript"
    NIM = "Nim"
    OCAML = "OCaml"
    OBJECTIVEC = "ObjectiveC"
    OCTAVE = "Octave"
    PHP = "PHP"
    PASCAL = "Pascal"
    PERL = "Perl"
    PROLOG = "Prolog"
    PYTHON = "Python"
    RACKET = "Racket"
    RUBY = "Ruby"
    RUST = "Rust"
    SML = "SML"
    SQL = "SQL"
    SCALA = "Scala"
    SCHEME = "Scheme"
    SWIFT = "Swift"
    TYPESCRIPT = "TypeScript"
    UNKNOWN = "unknown"
    VISUAL_BASIC = "VisualBasic"
    XML = "XML"
    YAML = "YAML"


# Base classes for document items
class BaseItem(BaseModel):
    """Base class for all document items."""

    model_config = ConfigDict(extra="forbid")

    self_ref: str
    parent: RefItem | None = None
    children: list[RefItem] = Field(default_factory=list)
    content_layer: ContentLayer = ContentLayer.BODY


class TextBaseItem(BaseItem):
    """Base class for text-containing items."""

    orig: str
    text: str
    formatting: Formatting | None = None
    hyperlink: str | None = None


class GroupItem(BaseItem):
    """Group item in document structure."""

    name: str = "group"
    label: GroupLabel = GroupLabel.UNSPECIFIED


class ListGroup(BaseItem):
    """List group item."""

    name: str = "group"
    label: str = "list"


class InlineGroup(BaseItem):
    """Inline group item."""

    name: str = "group"
    label: str = "inline"


class TitleItem(TextBaseItem):
    """Document title item."""

    label: str = "title"
    prov: list[ProvenanceItem] = Field(default_factory=list)


class SectionHeaderItem(TextBaseItem):
    """Section header item."""

    label: str = "section_header"
    level: int = Field(default=1, ge=1, le=100)
    prov: list[ProvenanceItem] = Field(default_factory=list)


class TextItem(TextBaseItem):
    """Regular text item."""

    label: str  # caption, checkbox_selected, checkbox_unselected, etc.
    prov: list[ProvenanceItem] = Field(default_factory=list)


class ListItem(TextBaseItem):
    """List item."""

    label: str = "list_item"
    enumerated: bool = False
    marker: str = "-"
    prov: list[ProvenanceItem] = Field(default_factory=list)


class CodeItem(TextBaseItem):
    """Code block item."""

    label: str = "code"
    code_language: CodeLanguageLabel = CodeLanguageLabel.UNKNOWN
    captions: list[RefItem] = Field(default_factory=list)
    references: list[RefItem] = Field(default_factory=list)
    footnotes: list[RefItem] = Field(default_factory=list)
    image: ImageRef | None = None
    prov: list[ProvenanceItem] = Field(default_factory=list)


class FormulaItem(TextBaseItem):
    """Mathematical formula item."""

    label: str = "formula"
    prov: list[ProvenanceItem] = Field(default_factory=list)


class TableCell(BaseModel):
    """Table cell data."""

    bbox: BoundingBox | None = None
    row_span: int = 1
    col_span: int = 1
    start_row_offset_idx: int
    end_row_offset_idx: int
    start_col_offset_idx: int
    end_col_offset_idx: int
    text: str
    column_header: bool = False
    row_header: bool = False
    row_section: bool = False


class TableData(BaseModel):
    """Table structure data."""

    table_cells: list[TableCell] = Field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0

    @property
    def grid(self) -> list[list[TableCell | None]]:
        """Compute grid from table cells."""
        if not self.table_cells:
            return []

        # Create empty grid
        grid: list[list[TableCell | None]] = [
            [None for _ in range(self.num_cols)] for _ in range(self.num_rows)
        ]

        # Fill grid with cells
        for cell in self.table_cells:
            for row in range(cell.start_row_offset_idx, cell.end_row_offset_idx):
                for col in range(cell.start_col_offset_idx, cell.end_col_offset_idx):
                    if 0 <= row < self.num_rows and 0 <= col < self.num_cols:
                        grid[row][col] = cell

        return grid


class DescriptionAnnotation(BaseModel):
    """Description annotation for images."""

    kind: str = "description"
    text: str
    provenance: str


class MiscAnnotation(BaseModel):
    """Miscellaneous annotation."""

    kind: str = "misc"
    content: dict[str, Any]


class TableItem(BaseItem):
    """Table item."""

    label: str = "table"  # or "document_index"
    prov: list[ProvenanceItem] = Field(default_factory=list)
    captions: list[RefItem] = Field(default_factory=list)
    references: list[RefItem] = Field(default_factory=list)
    footnotes: list[RefItem] = Field(default_factory=list)
    image: ImageRef | None = None
    data: TableData
    annotations: list[DescriptionAnnotation | MiscAnnotation] = Field(default_factory=list)


class PictureItem(BaseItem):
    """Picture item."""

    label: str = "picture"  # or "chart"
    prov: list[ProvenanceItem] = Field(default_factory=list)
    captions: list[RefItem] = Field(default_factory=list)
    references: list[RefItem] = Field(default_factory=list)
    footnotes: list[RefItem] = Field(default_factory=list)
    image: ImageRef | None = None
    annotations: list[DescriptionAnnotation | MiscAnnotation] = Field(default_factory=list)


class GraphCellLabel(str, Enum):
    """Graph cell label types."""

    UNSPECIFIED = "unspecified"
    KEY = "key"
    VALUE = "value"
    CHECKBOX = "checkbox"


class GraphLinkLabel(str, Enum):
    """Graph link label types."""

    UNSPECIFIED = "unspecified"
    TO_VALUE = "to_value"
    TO_KEY = "to_key"
    TO_PARENT = "to_parent"
    TO_CHILD = "to_child"


class GraphCell(BaseModel):
    """Graph cell data."""

    label: GraphCellLabel
    cell_id: int
    text: str
    orig: str
    prov: ProvenanceItem | None = None
    item_ref: RefItem | None = None


class GraphLink(BaseModel):
    """Graph link data."""

    label: GraphLinkLabel
    source_cell_id: int
    target_cell_id: int


class GraphData(BaseModel):
    """Graph structure data."""

    cells: list[GraphCell] = Field(default_factory=list)
    links: list[GraphLink] = Field(default_factory=list)


class KeyValueItem(BaseItem):
    """Key-value item."""

    label: str = "key_value_region"
    prov: list[ProvenanceItem] = Field(default_factory=list)
    captions: list[RefItem] = Field(default_factory=list)
    references: list[RefItem] = Field(default_factory=list)
    footnotes: list[RefItem] = Field(default_factory=list)
    image: ImageRef | None = None
    graph: GraphData


class FormItem(BaseItem):
    """Form item."""

    label: str = "form"
    captions: list[RefItem] = Field(default_factory=list)
    references: list[RefItem] = Field(default_factory=list)
    footnotes: list[RefItem] = Field(default_factory=list)
    image: ImageRef | None = None
    graph: GraphData


class DoclingDocument(BaseModel):
    """The main document structure from Docling."""

    model_config = ConfigDict(extra="allow")

    schema_name: str = "DoclingDocument"
    version: str = "1.5.0"
    name: str
    origin: DocumentOrigin | None = None

    # Content structure - full implementation
    furniture: GroupItem = Field(
        default_factory=lambda: GroupItem(
            self_ref="#/furniture",
            children=[],
            content_layer=ContentLayer.FURNITURE,
            name="_root_",
            label=GroupLabel.UNSPECIFIED,
        ),
        deprecated=True,
    )
    body: GroupItem = Field(
        default_factory=lambda: GroupItem(
            self_ref="#/body",
            children=[],
            content_layer=ContentLayer.BODY,
            name="_root_",
            label=GroupLabel.UNSPECIFIED,
        )
    )
    groups: list[ListGroup | InlineGroup | GroupItem] = Field(default_factory=list)
    texts: list[TitleItem | SectionHeaderItem | ListItem | CodeItem | FormulaItem | TextItem] = (
        Field(default_factory=list)
    )
    pictures: list[PictureItem] = Field(default_factory=list)
    tables: list[TableItem] = Field(default_factory=list)
    key_value_items: list[KeyValueItem] = Field(default_factory=list)
    form_items: list[FormItem] = Field(default_factory=list)
    pages: dict[str, PageItem] = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    """Document response content."""

    filename: str
    md_content: str | None = None
    json_content: DoclingDocument | dict[str, Any] | None = None
    html_content: str | None = None
    text_content: str | None = None
    doctags_content: str | None = None

    def model_post_init(self, __context: Any) -> None:
        """Post-init processing to ensure json_content is properly typed."""
        if (
            isinstance(self.json_content, dict)
            and self.json_content.get("schema_name") == "DoclingDocument"
        ):
            # Try to parse as DoclingDocument if it has the right structure
            import contextlib

            with contextlib.suppress(Exception):
                self.json_content = DoclingDocument.model_validate(self.json_content)


class ConvertDocumentResponse(BaseModel):
    """Complete response from Docling conversion service."""

    document: DocumentResponse
    status: ConversionStatus
    errors: list[ErrorItem] = []
    processing_time: float
    timings: dict[str, ProfilingItem] = {}


# Alias for backward compatibility with existing service code
DoclingResponse = ConvertDocumentResponse
