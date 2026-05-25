# Medi-Simple — AI-Powered Medical Report Assistant

An end-to-end RAG system that turns complex clinical PDFs into an interactive patient-facing Q&A tool with a structured health dashboard.

Upload a blood report → get a risk-scored dashboard with abnormal flags → ask follow-up questions in plain English.

**Stack:** Python · FastAPI · LangChain · FAISS · Groq (Llama 3.3 70B) · HuggingFace Embeddings · Pydantic

---

## Architecture

```
PDF Upload
    │
    ▼
┌─────────────────────┐
│  PDF Processor       │  PyMuPDF (fitz) — extracts text with page boundaries
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Summary Engine      │  LLM + Pydantic structured output → ClinicalSummary
│                     │  Extracts: patient name, abnormal/normal labs,
│                     │  risk level (LOW/MODERATE/HIGH), key findings
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Medical Chunker     │  Custom line-aware splitter (not naive sentence-window)
│                     │  Keeps lab result rows intact across chunk boundaries
│                     │  Falls back to RecursiveCharacterTextSplitter for long lines
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Contextualizer      │  LLM enriches each chunk with a 1-2 sentence context
│                     │  using the global patient summary — so retrieval hits
│                     │  are meaningful even when the raw chunk is just numbers
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Vector Store        │  FAISS index with HuggingFace embeddings
│                     │  (sentence-transformers/all-MiniLM-L6-v2)
│                     │  Embeds enriched text (context + raw chunk)
│                     │  Persists to disk for session reuse
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Chat Engine         │  Session-based conversational Q&A
│                     │  Retrieves top-5 chunks per question
│                     │  Maintains last 4 interactions for context
│                     │  Guardrails: no diagnosis, no treatment advice
└─────────────────────┘
```

## Key Design Decisions

**Why custom chunking instead of LangChain's default splitters?**

Medical reports have a specific structure: lab results are typically one value per line (`Hemoglobin  14.2 g/dL  12.0-16.0`). Naive sentence-window chunking splits these mid-row, which corrupts the data during retrieval. `MedicalChunker` splits on line breaks first to keep lab rows intact, only falling back to `RecursiveCharacterTextSplitter` when a single line exceeds the chunk size.

**Why contextualize chunks before embedding?**

A raw chunk like `"14.2 g/dL  12.0-16.0  NORMAL"` is meaningless without context. The `Contextualizer` uses the global patient summary to generate a semantic label for each chunk (e.g., *"Complete Blood Count results for the patient, showing Hemoglobin levels"*). This label is prepended to the raw text before embedding, which dramatically improves retrieval relevance.

**Why Pydantic structured output for the summary?**

The `SummaryEngine` uses LangChain's `.with_structured_output(ClinicalSummary)` to force the LLM into a typed schema: `LabResult` objects with `test_name`, `value`, `unit`, `reference_range`, and `status` (HIGH/LOW/NORMAL). This makes the dashboard deterministic — the frontend gets structured JSON, not free-text that needs parsing.

## API

```
POST /upload     — Upload a PDF, returns structured dashboard JSON + builds vector index
POST /chat       — Send a question with session_id, returns context-aware answer
GET  /           — Serves the frontend
```

## Setup

```bash
# Clone and install
git clone https://github.com/chaitanya8008/medi-simple.git
cd medi-simple
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your GROQ_API_KEY and HUGGINGFACEHUB_API_TOKEN

# Run
uvicorn backend.main:app --reload
```

## Project Structure

```
medi-simple/
├── backend/
│   ├── main.py              # FastAPI app, upload + chat endpoints
│   ├── pdf_processor.py     # PyMuPDF text extraction with page markers
│   ├── summary_engine.py    # LLM → Pydantic ClinicalSummary
│   ├── contextualizer.py    # MedicalChunker + LLM chunk enrichment
│   ├── vector_store.py      # FAISS build/save/load with HF embeddings
│   ├── chat_engine.py       # Session-based RAG Q&A with guardrails
│   └── schemas.py           # Pydantic models (ClinicalSummary, LabResult)
├── public/
│   └── index.html           # Frontend
├── .env.example
├── requirements.txt
└── README.md
```
