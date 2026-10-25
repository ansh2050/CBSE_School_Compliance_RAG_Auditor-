import io
import warnings
from typing import List, Dict, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import Config

try:
    import fitz  # PyMuPDF
    # Suppress PyMuPDF's own deprecation chatter at import time
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="fitz")
except Exception:
    fitz = None

try:
    from PIL import Image, ImageDraw
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None

import base64


def _image_to_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Supported image extensions for direct image upload
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class PDFProcessor:
    """
    Processes PDFs and images and produces text chunks with page-level
    bounding boxes and a page image for precise citation highlighting.

    Strategy:
    - PDF (native): Use PyMuPDF to extract text blocks with bboxes.
    - PDF (scanned): If a page has little text, fall back to Tesseract OCR.
    - Image (JPG/PNG/etc.): Open with PIL, run Tesseract OCR if available,
      otherwise store the image with a note that text extraction requires
      Tesseract.
    - Chunk text by concatenating contiguous elements up to chunk_size.
    - Per-chunk metadata: source filename, page number, bboxes, page image.
    """

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
        )

        # Check Tesseract ONCE at init time and remember the result.
        # Never print the warning inside a per-page loop.
        self.tesseract_available = False
        self._tesseract_status_message = ""

        if pytesseract is not None:
            try:
                ver = pytesseract.get_tesseract_version()
                self.tesseract_available = True
                self._tesseract_status_message = f"Tesseract {ver} available."
            except Exception:
                self.tesseract_available = False
                self._tesseract_status_message = (
                    "Tesseract OCR binary not found in PATH. "
                    "Scanned PDFs and image uploads will not have text extracted. "
                    "Install with: brew install tesseract"
                )
        else:
            self._tesseract_status_message = (
                "pytesseract package not installed. "
                "Scanned PDFs and image uploads will not have text extracted."
            )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _render_page_image(self, page, zoom: int = 2):
        """Render a PyMuPDF page to a PIL Image."""
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img

    def _extract_native_blocks(self, page, zoom: int = 2) -> List[Dict[str, Any]]:
        """Extract text blocks and scaled bboxes from a PyMuPDF page."""
        blocks = page.get_text("blocks")
        elements: List[Dict[str, Any]] = []
        for b in blocks:
            x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
            text = text.strip()
            if not text:
                continue
            sx0, sy0, sx1, sy1 = x0 * zoom, y0 * zoom, x1 * zoom, y1 * zoom
            elements.append({"text": text, "bbox": (sx0, sy0, sx1, sy1)})
        return elements

    def _extract_ocr_elements(self, image) -> List[Dict[str, Any]]:
        """Run Tesseract OCR and return elements with text and bbox in image coords."""
        if not self.tesseract_available or pytesseract is None:
            return []
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        except Exception as e:
            raise RuntimeError(str(e))

        elements: List[Dict[str, Any]] = []
        n = len(data.get("text", []))
        for i in range(n):
            text = data["text"][i].strip()
            if not text:
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            elements.append({"text": text, "bbox": (x, y, x + w, y + h)})
        return elements

    def _chunk_elements(
        self, elements: List[Dict[str, Any]]
    ) -> List[Tuple[str, List[Tuple[float, float, float, float]]]]:
        """Group contiguous elements into text chunks and collect bboxes."""
        chunks: List[Tuple[str, List[Tuple[float, float, float, float]]]] = []
        current_text: List[str] = []
        current_bboxes: List[Tuple[float, float, float, float]] = []

        for el in elements:
            txt = el["text"]
            bbox = el["bbox"]
            if current_text and (
                len(" ".join(current_text)) + len(txt) > Config.CHUNK_SIZE
            ):
                chunks.append((" ".join(current_text), current_bboxes))
                current_text = []
                current_bboxes = []
            current_text.append(txt)
            current_bboxes.append(bbox)

        if current_text:
            chunks.append((" ".join(current_text), current_bboxes))
        return chunks

    # ------------------------------------------------------------------ #
    # Public: PDF processing
    # ------------------------------------------------------------------ #

    def process_document(self, pdf_file) -> List[Document]:
        """
        Process a PDF Streamlit UploadedFile and return LangChain Documents.
        Warns about missing Tesseract ONCE (via returned warning in metadata),
        never in a per-page loop.
        """
        try:
            raw = pdf_file.read()
            if fitz is None:
                raise RuntimeError("PyMuPDF (fitz) is required for PDF processing.")
            if Image is None:
                raise RuntimeError("Pillow is required for rendering page images.")

            doc = fitz.open(stream=raw, filetype="pdf")
            all_documents: List[Document] = []
            # Track whether we already noted the OCR limitation for this document
            ocr_warning_emitted = False

            for page_number in range(len(doc)):
                page = doc[page_number]
                img = self._render_page_image(page, zoom=2)
                img_bytes = _image_to_bytes(img)
                img_w, img_h = img.width, img.height

                elements = self._extract_native_blocks(page)
                total_chars = sum(len(el["text"]) for el in elements)

                if total_chars < 50:
                    # Page appears to be scanned/image-only
                    if self.tesseract_available:
                        try:
                            elements = self._extract_ocr_elements(img)
                        except Exception as ocr_e:
                            print(f"Tesseract OCR failed on page {page_number+1}: {ocr_e}")
                    elif not ocr_warning_emitted:
                        # Emit the warning only ONCE for the whole document
                        print(
                            f"[processor] Page {page_number+1} appears scanned but "
                            f"Tesseract is unavailable: {self._tesseract_status_message}"
                        )
                        ocr_warning_emitted = True

                if not elements:
                    continue

                chunks = self._chunk_elements(elements)
                for idx, (chunk_text, bboxes) in enumerate(chunks):
                    metadata = {
                        "source": pdf_file.name,
                        "page": page_number + 1,
                        "chunk_id": idx,
                        "bboxes": bboxes,
                        "page_image": base64.b64encode(img_bytes).decode("utf-8"),
                        "page_image_width": img_w,
                        "page_image_height": img_h,
                        "file_type": "pdf",
                        "extraction_status": "ocr" if not elements else "native",
                    }
                    all_documents.append(Document(page_content=chunk_text, metadata=metadata))

            return all_documents

        except Exception as e:
            print(f"Error processing PDF {getattr(pdf_file, 'name', '<memory>')}: {str(e)}")
            return []

    # ------------------------------------------------------------------ #
    # Public: Image processing (JPG, PNG, etc.)
    # ------------------------------------------------------------------ #

    def process_image_file(self, image_file) -> List[Document]:
        """
        Process a JPG/PNG/BMP/TIFF Streamlit UploadedFile.

        - If Tesseract is available: runs OCR to extract text with bboxes.
        - If not: stores the image as a Document with an extraction_status
          of 'no_ocr' and a placeholder text so the embedding pipeline still
          works. The UI will surface the limitation to the user.

        Returns a list of LangChain Document objects (one per OCR chunk or
        one placeholder document if OCR is unavailable).
        """
        if Image is None:
            print("Pillow is required to process image files.")
            return []

        try:
            raw = image_file.read()
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            img_bytes = _image_to_bytes(img)
            img_w, img_h = img.width, img.height
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")

            base_metadata = {
                "source": image_file.name,
                "page": 1,
                "file_type": "image",
                "page_image": img_b64,
                "page_image_width": img_w,
                "page_image_height": img_h,
            }

            # ── With Tesseract: extract text ──────────────────────────────
            if self.tesseract_available:
                elements = self._extract_ocr_elements(img)
                if not elements:
                    # OCR ran but found no text (blank or logo-only image)
                    meta = {
                        **base_metadata,
                        "chunk_id": 0,
                        "bboxes": [],
                        "extraction_status": "ocr_empty",
                    }
                    return [
                        Document(
                            page_content=(
                                f"[Image: {image_file.name}] "
                                "OCR ran but could not extract any text. "
                                "The image may contain graphics or handwriting only."
                            ),
                            metadata=meta,
                        )
                    ]

                chunks = self._chunk_elements(elements)
                docs: List[Document] = []
                for idx, (chunk_text, bboxes) in enumerate(chunks):
                    meta = {
                        **base_metadata,
                        "chunk_id": idx,
                        "bboxes": bboxes,
                        "extraction_status": "ocr",
                    }
                    docs.append(Document(page_content=chunk_text, metadata=meta))
                return docs

            # ── Without Tesseract: return placeholder ─────────────────────
            else:
                meta = {
                    **base_metadata,
                    "chunk_id": 0,
                    "bboxes": [],
                    "extraction_status": "no_ocr",
                }
                return [
                    Document(
                        page_content=(
                            f"[Image uploaded: {image_file.name}] "
                            "Text could not be extracted because Tesseract OCR is not installed. "
                            "Install Tesseract to enable text extraction from images: "
                            "brew install tesseract (macOS) or "
                            "apt install tesseract-ocr (Linux)."
                        ),
                        metadata=meta,
                    )
                ]

        except Exception as e:
            print(f"Error processing image {getattr(image_file, 'name', '<memory>')}: {str(e)}")
            return []

    # ------------------------------------------------------------------ #
    # Public: unified dispatcher
    # ------------------------------------------------------------------ #

    def process_file(self, uploaded_file) -> tuple[List[Document], str]:
        """
        Dispatch to the correct processor based on file extension.

        Returns (documents, warning_message).
        warning_message is empty string if all is fine, or a user-facing
        warning to surface in the UI (e.g. Tesseract not available).
        """
        name = getattr(uploaded_file, "name", "").lower()
        ext = "." + name.rsplit(".", 1)[-1] if "." in name else ""

        warning = ""

        if ext == ".pdf":
            docs = self.process_document(uploaded_file)
        elif ext in IMAGE_EXTENSIONS:
            docs = self.process_image_file(uploaded_file)
            if not self.tesseract_available:
                warning = (
                    "⚠️ Tesseract OCR is not installed — text could not be extracted from the image. "
                    "Install it with: `brew install tesseract`"
                )
        else:
            warning = f"Unsupported file type: {ext}. Supported: PDF, JPG, PNG, BMP, TIFF."
            docs = []

        return docs, warning