# CBSE School Compliance Auditor

A domain-specific **Retrieval-Augmented Generation (RAG)** application that audits school documents against curated CBSE (Central Board of Secondary Education) reference material. Upload a school document — fee receipt, admission letter, TC demand, academic plan, or infrastructure certificate — and get evidence-grounded compliance findings with source citations.

Built with LangChain, Google Gemini, FAISS, PyMuPDF, and Streamlit.

---

## System Architecture

```
                          ┌──────────────────────────────────────────┐
                          │           Streamlit UI (app.py)          │
                          │   Chat • Source Highlights • Telemetry   │
                          └────────────────┬───────────────────────┘
                                           │ user query
                                           ▼
                 ┌─────────────────────────────────────────────────────┐
                 │        Episodic Memory Cache (SQLite)               │
                 │        src/episodic_memory.py                       │
                 │                                                     │
                 │  ┌───────────────────────────────────────────────┐  │
                 │  │  Precondition Verification on every lookup:   │  │
                 │  │    • baseline_version  (manifest SHA-256)     │  │
                 │  │    • document_hash     (uploaded doc SHA-256) │  │
                 │  │    • ttl               (wall-clock expiry)   │  │
                 │  │                                               │  │
                 │  │  STALE_MEMORY_INVALIDATED if any check fails  │  │
                 │  └───────────────────────────────────────────────┘  │
                 └────────────────┬──────────────────────────────────┘
                                  │ cache miss
                                  ▼
          ┌────────────────────────────────────────────────────────────┐
          │          CRAG Precondition Retrieval Gate                  │
          │          src/retrieval_gate.py                             │
          │                                                            │
          │  Phase 1 — Deterministic checks:                           │
          │    • supersession_status from baseline manifest             │
          │    • keyword / lexical overlap scoring                      │
          │    • minimum content-length threshold                      │
          │                                                            │
          │  Phase 2 — LLM-assisted structured evaluation:             │
          │    • Pydantic-validated ChunkEvaluation per chunk           │
          │    • Fast deterministic fallback (never crashes)            │
          │                                                            │
          │  Actions: proceed │ filter │ rewrite_query │               │
          │           insufficient_evidence                            │
          └────────┬───────────────────────────────┬──────────────────┘
                   │                               │
                   ▼                               ▼
   ┌───────────────────────────┐   ┌───────────────────────────────────┐
   │  Dual-Source FAISS Index  │   │  User Upload Session Index        │
   │                           │   │  (src/embedding.py)               │
   │  ┌─────────────────────┐  │   │                                   │
   │  │ CBSE Baseline Index │  │   │  FAISS index built per-session    │
   │  │ (baseline/)         │  │   │  from uploaded PDFs/images        │
   │  │                     │  │   │                                   │
   │  │ 5 reference docs    │  │   │  PyMuPDF bounding-box extraction  │
   │  │ SHA-256 verified    │  │   │  + Tesseract OCR fallback         │
   │  │ manifest.json       │  │   │                                   │
   │  └─────────────────────┘  │   └───────────────────────────────────┘
   │                           │
   │  5-Guard Integrity Chain: │
   │   1. Directory existence  │
   │   2. Manifest existence   │
   │   3. Manifest parsing     │
   │   4. SHA-256 checksums    │
   │   5. Embedding model      │
   │      compatibility        │
   └───────────────────────────┘
                   │
                   ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │              ChatManager + Gemini LLM (src/chat.py)              │
   │                                                                   │
   │  Query Flow:                                                      │
   │   1. Episodic memory lookup → cache hit? return immediately       │
   │   2. CRAG gate filtering    → reject stale / irrelevant chunks    │
   │   3. LLM generation         → chain or direct fallback            │
   │   4. Episodic memory store  → persist verified trajectory         │
   │                                                                   │
   │  Graceful fallback:  chain → direct LLM → error message          │
   └───────────────────────────────────────────────────────────────────┘
```

---

## Key Engineering Features

### 5-Guard SHA-256 Baseline Integrity Chain

The permanent CBSE baseline FAISS index (`baseline/`) is protected by a 5-guard validation chain in [`src/baseline.py`](src/baseline.py):

| Guard | Check | Failure Mode |
|-------|-------|--------------|
| 1 | Baseline directory exists | UI shows "Missing" status |
| 2 | `manifest.json` exists | UI shows rebuild instructions |
| 3 | `manifest.json` parses as valid JSON | Surfaces parse error |
| 4 | SHA-256 checksums of `.faiss` and `.pkl` match manifest | Reports corruption/tampering |
| 5 | Embedding model name matches `Config.EMBEDDING_MODEL` | Reports model mismatch |

### Corrective RAG (CRAG) Precondition Retrieval Gate

[`src/retrieval_gate.py`](src/retrieval_gate.py) implements a two-phase retrieval filter with Pydantic v2 schemas:

- **Phase 1 — Deterministic:** supersession status, lexical overlap, content-length thresholds
- **Phase 2 — LLM-assisted:** structured `ChunkEvaluation` scoring with automatic deterministic fallback
- **Output:** `RetrievalGateResult` with action (`proceed` / `filter` / `rewrite_query` / `insufficient_evidence`), passed/rejected chunk IDs, and per-chunk evaluations

### Precondition-Verified Episodic Memory

[`src/episodic_memory.py`](src/episodic_memory.py) is a SQLite-backed cache of past query→response trajectories:

- **Lookup:** exact query hash match + cosine similarity over `all-MiniLM-L6-v2` embeddings
- **Precondition verification on every lookup:**
  - `baseline_version` — SHA-256 of current `manifest.json`
  - `document_hash` — SHA-256 of the uploaded document content
  - `ttl` — wall-clock expiry (default: 1 hour)
- **Invalidation:** if any precondition fails, the episode is logged as `STALE_MEMORY_INVALIDATED` and skipped
- **Storage:** episodes are persisted only after the CRAG retrieval gate passes

### PyMuPDF Bounding-Box Citations

[`src/processor.py`](src/processor.py) extracts text blocks from PDFs with per-block bounding-box coordinates, enabling the UI to render red highlight rectangles directly on the rendered page image — showing exactly which text region was cited in the response.

---

## Features

- **CBSE Baseline Index:** Permanent, offline-built FAISS vector store of 5 CBSE reference documents covering academic subjects, assessment & attendance rules, fees & admissions, lab infrastructure, and a verification matrix.
- **Audit Document Upload:** Upload any school document (PDF or image). It is processed into session-scoped chunks and semantically compared against the CBSE baseline.
- **CRAG Retrieval Gate:** Pydantic-validated retrieval filtering that rejects stale or irrelevant chunks before they reach the LLM.
- **Episodic Memory Cache:** SQLite-backed cache with precondition-verified invalidation — returns instantly on cache hit, automatically invalidates when the baseline or uploaded documents change.
- **Source Citations with Highlights:** Answers include expandable source cards showing the exact page image with red bounding boxes around the cited text region.
- **OCR Support:** Scanned PDFs and image uploads (JPG, PNG, BMP, TIFF) are processed via Tesseract OCR if installed, with graceful fallback and clear UI messaging.
- **Conversational Memory:** Multi-turn Q&A with context retention via LangChain `ConversationBufferMemory`.
- **LLM Integration:** Answers generated by Google Gemini (`gemma-3-27b-it`) via `ChatGoogleGenerativeAI`.
- **Baseline Lifecycle:** Manifest versioning — the UI automatically detects when the baseline is rebuilt and refreshes the cached index.

---

## Project Structure

```
.
├── app.py                          # Streamlit UI & main application
├── src/
│   ├── config.py                   # Centralised config (API keys, feature flags, paths)
│   ├── processor.py                # PDF + image extraction with OCR & bounding-box support
│   ├── embedding.py                # Session embedding creation & FAISS retriever
│   ├── chat.py                     # Chat manager with gate + memory integration
│   ├── baseline.py                 # Loader for the permanent CBSE baseline (5-guard chain)
│   ├── retrieval_gate.py           # CRAG Precondition Retrieval Gate (Pydantic schemas)
│   └── episodic_memory.py          # SQLite Episodic Memory with precondition verification
├── scripts/
│   └── build_baseline.py           # Offline script: indexes CBSE docs → baseline/
├── documents/                      # CBSE reference Markdown documents (knowledge base)
│   ├── CBSE_Academic_Subjects_Structure.md
│   ├── CBSE_Assessment_Attendance_Rules.md
│   ├── CBSE_Fees_Admissions_Operational_Compliance.md
│   ├── CBSE_Lab_Infrastructure_Mandates.md
│   └── User_Upload_Verification_Matrix.md
├── baseline/                       # Auto-generated FAISS index artifacts
│   ├── cbse_index.faiss
│   ├── cbse_index.pkl
│   └── manifest.json
├── requirements.txt
├── Dockerfile
└── .env                            # API keys (not committed)
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/ansh2050/CBSE_School_Compliance_RAG_Auditor-.git
cd <project-folder>
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your-gemini-api-key-here

# Optional overrides (defaults shown)
MODEL_NAME=gemma-3-27b-it
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=4
LLM_TEMPERATURE=0.7
MAX_TOKENS=2048

# Feature flags (all enabled by default)
ENABLE_CRAG_GATE=true
ENABLE_EPISODIC_MEMORY=true
EPISODIC_TTL_SECONDS=3600
```

### 5. (Optional) Install Tesseract for scanned PDFs & image uploads

```bash
brew install tesseract        # macOS
# sudo apt install tesseract-ocr  # Ubuntu/Debian
```

### 6. Build the CBSE Baseline Index

Run **once** before starting the app. Re-run whenever documents in `documents/` are updated:

```bash
python scripts/build_baseline.py
```

This reads all `.md` files from `documents/`, chunks and embeds them using `sentence-transformers/all-MiniLM-L6-v2`, and saves the FAISS index to `baseline/`.

### 7. Run the app

```bash
streamlit run app.py
```

---

## Usage

1. The sidebar shows the **CBSE Baseline Status** — ensure it shows 🟢 **Ready** before proceeding.
2. Upload a school document (PDF, JPG, PNG, BMP, or TIFF) using the sidebar uploader.
3. Click **⚙️ Process Document** to extract and embed the document into session memory.
4. Ask compliance questions in the chat input — e.g. *"Does this fee receipt comply with CBSE fee transparency norms?"*
5. The assistant responds with evidence-grounded findings and expandable source citations with highlighted page images.
6. Use **🗑️ Clear Chat** to reset the conversation, or **❌ Remove Document** to start fresh with a new upload.

---

## CBSE Reference Documents

The `documents/` folder contains curated Markdown summaries of CBSE policies that form the compliance knowledge base:

| File | Covers |
|---|---|
| `CBSE_Academic_Subjects_Structure.md` | Subject offerings, stream rules, elective policies |
| `CBSE_Assessment_Attendance_Rules.md` | Attendance thresholds, exam eligibility, grading norms |
| `CBSE_Fees_Admissions_Operational_Compliance.md` | Fee transparency, admission norms, operational rules |
| `CBSE_Lab_Infrastructure_Mandates.md` | Lab requirements, infrastructure standards |
| `User_Upload_Verification_Matrix.md` | Document-type verification checklist |

---

## Disclaimer

> Findings produced by this tool are **evidence-based review outputs only**. They are **not** official CBSE decisions, legal advice, or regulatory certifications. Always verify compliance findings against official CBSE circulars and consult qualified professionals.

---

## Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| LLM | Google Gemini (`gemma-3-27b-it`) via LangChain |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS (dual-source: baseline + session) |
| Retrieval Gate | Corrective RAG with Pydantic v2 schemas |
| Episodic Memory | SQLite with precondition-verified invalidation |
| PDF Parsing | PyMuPDF (`fitz`) with bounding-box extraction |
| OCR | Tesseract via `pytesseract` (optional) |
| Memory | LangChain `ConversationBufferMemory` |
| Config | `python-dotenv` + `src/config.py` |
