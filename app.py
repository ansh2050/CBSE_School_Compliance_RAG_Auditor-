"""
app.py — CBSE School Compliance Auditor
Phase 1: reliable baseline lifecycle + renamed UI (all existing chat intact)
"""
import streamlit as st
import time
import json
from langchain_core.documents import Document

from src.processor import PDFProcessor
from src.embedding import EmbeddingManager
from src.chat import ChatManager
from src.config import Config
from src.baseline import load_baseline, get_baseline_retriever

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CBSE School Compliance Auditor",
    page_icon="🏫",
    layout="wide",
)

# ── Custom CSS — keep dark theme, improve spacing & hierarchy ────────────────
st.markdown(
    """
    <style>
    /* ---------- Global typography ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1117 0%, #1a1d29 100%);
        border-right: 1px solid rgba(255,255,255,0.07);
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #c9d1d9;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 1.2rem 0 0.4rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }

    /* ---------- Baseline status badge ---------- */
    .baseline-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .badge-ready   { background: rgba(46,160,67,0.18); color: #3fb950; border: 1px solid rgba(46,160,67,0.4); }
    .badge-missing { background: rgba(248,81,73,0.15);  color: #f85149; border: 1px solid rgba(248,81,73,0.4); }
    .badge-warn    { background: rgba(210,153,34,0.15); color: #d9a92a; border: 1px solid rgba(210,153,34,0.4); }

    /* ---------- Baseline info card ---------- */
    .baseline-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.8rem;
        color: #8b949e;
        line-height: 1.6;
    }

    /* ---------- Main panel landing ---------- */
    .landing-card {
        background: linear-gradient(135deg, rgba(31,111,235,0.08) 0%, rgba(139,92,246,0.06) 100%);
        border: 1px solid rgba(31,111,235,0.2);
        border-radius: 12px;
        padding: 2.5rem 2rem;
        text-align: center;
        max-width: 720px;
        margin: 3rem auto;
    }
    .landing-card h2 { color: #c9d1d9; margin-bottom: 0.5rem; }
    .landing-card p  { color: #8b949e; font-size: 0.9rem; line-height: 1.7; }

    /* ---------- App header ---------- */
    .app-header {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        padding: 0.5rem 0 1.5rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.07);
        margin-bottom: 1.5rem;
    }
    .app-header .title { font-size: 1.5rem; font-weight: 700; color: #e6edf3; }
    .app-header .subtitle { font-size: 0.82rem; color: #8b949e; margin-top: 0.15rem; }

    /* ---------- Source expander tweak ---------- */
    .streamlit-expanderHeader { font-size: 0.82rem; }

    /* ---------- Disclaimer footer ---------- */
    .disclaimer {
        font-size: 0.72rem;
        color: #6e7681;
        border-top: 1px solid rgba(255,255,255,0.06);
        padding-top: 0.8rem;
        margin-top: 1.5rem;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session state initializer ────────────────────────────────────────────────
def initialize_session_state() -> bool:
    """
    Initialize the Streamlit session state with the required components.
    Creates instances of the processor, embedding manager, and chat manager.
    Returns False if the API key is missing.
    """
    if "processor" not in st.session_state:
        st.session_state.processor = PDFProcessor()

    if "embedding_manager" not in st.session_state:
        st.session_state.embedding_manager = EmbeddingManager()

    if "chat_manager" not in st.session_state:
        if not Config.is_valid():
            st.error("Missing API key. Please check your .env file or Space Secrets.")
            return False
        st.session_state.chat_manager = ChatManager(Config.GOOGLE_API_KEY)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "documents" not in st.session_state:
        st.session_state.documents = []

    # Phase 1 additions
    if "baseline_loaded" not in st.session_state:
        st.session_state.baseline_loaded = False

    return True


# ── Document processing (unchanged logic, renamed label) ────────────────────
def process_documents(uploaded_files):
    """
    Process the uploaded audit document(s) into session-scoped chunks.
    Supports PDF, JPG, PNG and other image formats via process_file().
    These chunks are NEVER added to the permanent CBSE baseline.
    """
    try:
        with st.spinner("Processing document …"):
            all_documents = []
            ui_warnings = []

            for file in uploaded_files:
                docs, warning = st.session_state.processor.process_file(file)
                all_documents.extend(docs)
                if warning:
                    ui_warnings.append(warning)

            st.session_state.documents = all_documents

            # Surface any per-file warnings in the UI (not as console spam)
            for w in ui_warnings:
                st.warning(w)

            if not all_documents:
                st.warning(
                    "No text chunks were extracted from the uploaded file. "
                    "For scanned PDFs or images, install Tesseract: `brew install tesseract`"
                )
                return False

            success = st.session_state.embedding_manager.create_embeddings(all_documents)

            if success:
                st.session_state.chat_manager.set_retriever(
                    st.session_state.embedding_manager.retriever
                )
                st.success(
                    f"✅ Processed {len(all_documents)} chunks from uploaded file."
                )
                return True
            else:
                st.error("Failed to create session embeddings.")
                return False

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return False


# ── Sidebar: baseline status panel ──────────────────────────────────────────
def render_baseline_status(vs, manifest, status_msg: str):
    """Render the CBSE Baseline Status panel in the sidebar."""
    st.markdown("### 📚 CBSE Baseline")

    if vs is not None:
        # Ready
        st.markdown(
            '<span class="baseline-badge badge-ready">🟢 Ready</span>',
            unsafe_allow_html=True,
        )
        built_at = manifest.get("built_at", "")[:10] if manifest else "unknown"
        n_sources = len(manifest.get("source_inventory", [])) if manifest else 0
        index_ver = manifest.get("index_version", "—") if manifest else "—"
        st.markdown(
            f"""
            <div class="baseline-card">
            <b>Version:</b> {index_ver}<br>
            <b>Sources:</b> {n_sources} reference document(s)<br>
            <b>Built:</b> {built_at}<br>
            <b>Model:</b> {manifest.get('embedding_model','—') if manifest else '—'}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "⚠️ A loaded index does not certify that the collection is exhaustive "
            "or current. Always verify findings against official CBSE sources."
        )
        st.session_state.baseline_loaded = True

    elif "Incompatible" in status_msg or "⚠️" in status_msg:
        st.markdown(
            '<span class="baseline-badge badge-warn">⚠️ Incompatible</span>',
            unsafe_allow_html=True,
        )
        st.warning(status_msg)
        st.session_state.baseline_loaded = False
    else:
        st.markdown(
            '<span class="baseline-badge badge-missing">🔴 Missing</span>',
            unsafe_allow_html=True,
        )
        st.error(status_msg)
        with st.expander("How to build the baseline"):
            st.code("python scripts/build_baseline.py", language="bash")
        st.session_state.baseline_loaded = False


# ── Sidebar: document upload controls ───────────────────────────────────────
def render_upload_sidebar():
    """Render the upload section and action buttons in the sidebar."""
    st.markdown("### 📄 Audit Document")

    uploaded_files = st.file_uploader(
        "Upload a school document",
        type=["pdf", "jpg", "jpeg", "png", "bmp", "tiff"],
        accept_multiple_files=False,
        key="audit_file_uploader",
        help="Supported formats: PDF, JPG, PNG, BMP, TIFF",
    )

    if uploaded_files:
        # Show file info with type badge
        fsize_kb = round(uploaded_files.size / 1024, 1)
        ext = uploaded_files.name.rsplit(".", 1)[-1].upper() if "." in uploaded_files.name else "FILE"
        ext_colors = {"PDF": "#e74c3c", "JPG": "#3498db", "JPEG": "#3498db",
                      "PNG": "#2ecc71", "BMP": "#9b59b6", "TIFF": "#f39c12"}
        badge_color = ext_colors.get(ext, "#7f8c8d")
        st.markdown(
            f"""
            <div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.4rem;'>
                <span style='background:{badge_color};color:white;padding:0.1rem 0.45rem;
                    border-radius:4px;font-size:0.7rem;font-weight:700;'>{ext}</span>
                <span style='font-size:0.82rem;color:#c9d1d9;'>{uploaded_files.name}</span>
                <span style='font-size:0.75rem;color:#6e7681;margin-left:auto;'>{fsize_kb} KB</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Show Tesseract status for image files
        if ext in ("JPG", "JPEG", "PNG", "BMP", "TIFF"):
            if st.session_state.processor.tesseract_available:
                st.caption("🟢 Tesseract OCR available — text will be extracted.")
            else:
                st.caption(
                    "🟡 Tesseract not installed — image will upload but text won't be extracted. "
                    "Run `brew install tesseract` to enable OCR."
                )

        if st.button("⚙️ Process Document", use_container_width=True):
            process_documents([uploaded_files])

    # Chunk stats + reset controls
    if st.session_state.documents:
        n_chunks = len(st.session_state.documents)
        st.success(f"📝 {n_chunks} session chunks in memory")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.chat_manager.reset_conversation()
                st.rerun()
        with col2:
            if st.button("❌ Remove Document", use_container_width=True):
                st.session_state.documents = []
                if hasattr(st.session_state.embedding_manager, "clear_embeddings"):
                    st.session_state.embedding_manager.clear_embeddings()
                st.session_state.messages = []
                st.session_state.chat_manager.reset_conversation()
                st.rerun()


# ── Main panel: landing state ────────────────────────────────────────────────
def render_landing():
    st.markdown(
        """
        <div class="landing-card">
            <h2>🏫 CBSE School Compliance Auditor</h2>
            <p>
                Upload a school document — such as a <b>fee receipt, admission letter,
                TC demand, academic plan, or infrastructure certificate</b> — and ask
                questions or run a full audit against curated CBSE reference material.
            </p>
            <p>
                Findings are evidence-based review outputs with source citations.<br>
                They are <b>not</b> official CBSE decisions, legal advice, or regulatory certifications.
            </p>
            <p style="margin-top:1.2rem; color:#6e7681; font-size:0.8rem;">
                ← Upload a PDF in the sidebar to begin.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Main panel: chat + sources ────────────────────────────────────────────────
def render_chat():
    """Display chat history and handle new user input."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if query := st.chat_input("Ask a compliance question about the uploaded document …"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        if not st.session_state.documents:
            with st.chat_message("assistant"):
                st.write(
                    "Please upload and process a school document first using the sidebar."
                )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Please upload and process a school document first using the sidebar.",
                }
            )
            return

        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and generating response …"):
                if (
                    hasattr(st.session_state.embedding_manager, "retriever")
                    and st.session_state.embedding_manager.retriever
                ):
                    response = st.session_state.chat_manager.generate_response(query, [])
                else:
                    relevant_docs = st.session_state.embedding_manager.search(query)
                    response = st.session_state.chat_manager.generate_response(
                        query, relevant_docs
                    )

                answer = ""
                sources = []

                if isinstance(response, str):
                    try:
                        parsed = json.loads(response)
                        if isinstance(parsed, dict) and (
                            "answer" in parsed or "sources" in parsed
                        ):
                            response = parsed
                    except Exception:
                        pass

                if isinstance(response, dict):
                    answer = response.get("answer", "")
                    sources = response.get("sources", []) or []
                else:
                    answer = str(response)

                st.write(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

                # Deduplicated source expanders
                unique: dict = {}
                for src in sources:
                    if not isinstance(src, dict):
                        continue
                    md = src.get("metadata", {}) or {}
                    key = (md.get("source"), md.get("page"), md.get("chunk_id"))
                    snippet = src.get("text")
                    if key in unique:
                        if snippet and snippet not in unique[key]["snippets"]:
                            unique[key]["snippets"].append(snippet)
                    else:
                        unique[key] = {
                            "metadata": md,
                            "snippets": [snippet] if snippet else [],
                        }

                for i, ((source_name, page, chunk_id), entry) in enumerate(
                    unique.items()
                ):
                    md = entry["metadata"]
                    snippets = entry["snippets"]
                    page_image_b64 = md.get("page_image")
                    bboxes = md.get("bboxes", []) or []
                    img_w = md.get("page_image_width")
                    img_h = md.get("page_image_height")

                    header = (
                        f"Source {i+1}: {source_name or 'unknown'} "
                        f"(page {page or '?'})"
                    )
                    with st.expander(header, expanded=False):
                        if snippets:
                            for s in snippets:
                                st.markdown(f"- {s}")

                        if page_image_b64 and img_w and img_h:
                            boxes_html = ""
                            for bbox in bboxes:
                                try:
                                    x0, y0, x1, y1 = bbox
                                    left = (x0 / img_w) * 100 if img_w else 0
                                    top = (y0 / img_h) * 100 if img_h else 0
                                    width = ((x1 - x0) / img_w) * 100 if img_w else 0
                                    height = ((y1 - y0) / img_h) * 100 if img_h else 0
                                    boxes_html += (
                                        f"<div style='position:absolute;"
                                        f"left:{left}%;top:{top}%;width:{width}%;"
                                        f"height:{height}%;border:3px solid "
                                        f"rgba(255,0,0,0.8);box-sizing:border-box;'>"
                                        f"</div>"
                                    )
                                except Exception:
                                    continue

                            html = f"""
                            <div style='position:relative;display:inline-block;border:1px solid #eee'>
                              <img src='data:image/png;base64,{page_image_b64}'
                                   style='max-width:600px;height:auto;display:block' />
                              <div style='position:absolute;left:0;top:0;width:100%;
                                          height:100%;pointer-events:none;'>
                                {boxes_html}
                              </div>
                            </div>
                            """
                            st.components.v1.html(html, height=450)
                        else:
                            st.markdown(f"**Metadata:** {md}")


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    if not initialize_session_state():
        return

    # ── Load the permanent CBSE baseline (cached per manifest mtime) ─────
    # The cache_key is based on the manifest file's modification time.
    # This ensures @st.cache_resource re-runs load_baseline() whenever
    # the baseline is rebuilt, instead of returning a stale cached failure.
    manifest_path = Config.BASELINE_MANIFEST
    if manifest_path.exists():
        cache_key = str(manifest_path.stat().st_mtime)
    else:
        cache_key = "missing"

    baseline_vs, baseline_manifest, baseline_status = load_baseline(cache_key)

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            """
            <div style='padding:0.4rem 0 1rem 0;'>
                <span style='font-size:1.1rem;font-weight:700;color:#e6edf3;'>
                    🏫 Compliance Auditor
                </span>
                <div style='font-size:0.72rem;color:#6e7681;margin-top:0.2rem;'>
                    CBSE School Document Review Aid
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_baseline_status(baseline_vs, baseline_manifest, baseline_status)
        st.divider()
        render_upload_sidebar()
        st.divider()

        # Disclaimer
        st.markdown(
            """
            <div class="disclaimer">
            Results are evidence-based review findings only — <b>not</b> official CBSE
            decisions, legal advice, or compliance certificates. Always verify with
            qualified professionals and official CBSE sources.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Main panel header ─────────────────────────────────────────────────
    st.markdown(
        """
        <div class="app-header">
            <span style="font-size:2rem;">🏫</span>
            <div>
                <div class="title">CBSE School Compliance Auditor</div>
                <div class="subtitle">
                    Evidence-grounded review of school documents against CBSE reference material
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Baseline unavailable — disable chat ───────────────────────────────
    if baseline_vs is None:
        st.error(
            "⛔ The CBSE baseline is unavailable. "
            "Audit and Q&A features are disabled until the baseline is built and loaded. "
            "Run `python scripts/build_baseline.py` from the project root, then restart the app."
        )
        render_landing()
        return

    # ── Landing or chat view ──────────────────────────────────────────────
    if not st.session_state.documents:
        render_landing()
    else:
        render_chat()


if __name__ == "__main__":
    main()