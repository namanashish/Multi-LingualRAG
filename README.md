# Multi-LingualRAG
# MultiLingual RAG — Government AI Assistant

A multilingual voice-to-RAG pipeline for Indian government citizen services. Speak in Hindi (or other Indic languages), get answers from government documents in your language.

---

## Project Vision

Build a government app where a farmer, shopkeeper, or daily wage worker can **speak in Hindi** and either:
- Get **information** from government rulebooks and scheme documents (RAG pipeline)
- **Apply** for government schemes by filling forms through voice (Agentic pipeline)

---

## Architecture Overview

```
Audio Input (Hindi)
      ↓
faster-whisper (CUDA)         — Speech to text
      ↓
IndicTrans2 (CPU)             — Hindi → English translation
      ↓
flan-t5 cleaner (CPU)         — Translate residual Indic proper nouns
      ↓
Router 1 (bart-large-mnli)    — RAG or Agent?
      ↓
    ┌─────────────────────────────────────┐
    │                                     │
 Router 2                           LangGraph Agent
 Vector DB or Web?                  Form fill workflow
    │
    ↓
Query Transformation              — Rewrite, expand, HyDE
    ↓
Hybrid Retrieval                  — BM25 + vector search + RRF
    ↓
Reranker                          — Cross-encoder reranking
    ↓
Context Compression               — Filter redundant chunks
    ↓
Prompt Construction               — System + context + question
    ↓
LLM Generation                    — Groq / Llama 3
    ↓
Guardrails                        — Hallucination, PII, safety
    ↓
Final Response (in user language)
```

---

## Document Ingestion Pipeline

```
Government PDFs (digital + scanned)
      ↓
parser.py
  ├── Plain text extraction     (PyMuPDF)
  ├── Section header detection  (regex patterns)
  ├── List/bullet extraction    (regex patterns)
  ├── Table extraction          (pdfplumber → natural language)
  └── OCR for scanned pages     (Tesseract hin+eng)
      ↓
Rich metadata per chunk
  (source, page, type, section, chunk_id)
      ↓
Chunker (RecursiveCharacterTextSplitter)
  chunk_size=1000, overlap=200
      ↓
Embeddings (all-MiniLM-L6-v2)
      ↓
ChromaDB (local vector store)
```

---

## Tech Stack

| Component | Tool | Version |
|---|---|---|
| Python | CPython | 3.11.9 |
| CUDA | PyTorch wheel | cu121 |
| Speech to text | faster-whisper | 1.2.1 |
| Translation | IndicTrans2 (ai4bharat) | indic-en-1B |
| Translation toolkit | IndicTransToolkit | 1.1.1 |
| LLM cleanup | flan-t5-base | via transformers |
| Intent router | facebook/bart-large-mnli | via transformers |
| Transformers | HuggingFace transformers | 4.40.0 (pinned) |
| PDF parsing | PyMuPDF (fitz) | 1.27.x |
| Table extraction | pdfplumber | latest |
| OCR | pytesseract + Tesseract | 5.x |
| Chunking | langchain-text-splitters | 1.1.2 |
| Embeddings | sentence-transformers | 2.7.0 (pinned) |
| Vector store | ChromaDB | 1.5.9 |
| LangChain | langchain + community | 1.3.2 |

> ⚠️ See [ISSUES.md](./ISSUES.md) for critical dependency pinning requirements — especially the `transformers` version conflict between faster-whisper, IndicTrans2, and sentence-transformers.

---

## Project Structure

```
MulitLingualRAG/
│
├── audio/                        ← audio input files
│
├── docs/                         ← government PDF documents
│
├── src/                          ← all Python modules
│   ├── parser.py                 ← document parser (PyMuPDF + pdfplumber + OCR)
│   ├── cleaner.py                ← flan-t5 Indic term cleanup
│   ├── router.py                 ← intent classifier (RAG vs agent)
│   └── rag.py                    ← RAG pipeline (ChromaDB + retrieval)
│
├── notebooks/
│   └── main.ipynb                ← main notebook (orchestrates everything)
│
├── IndicTrans2/                  ← cloned AI4Bharat repo
├── IndicTransToolkit/            ← cloned VarunGumma repo
├── raprod/                       ← Python 3.11 virtual environment
│
├── README.md
└── ISSUES.md                     ← all problems and fixes
```

---

## Setup Guide

### Prerequisites

- Windows 10/11 (64-bit)
- NVIDIA GPU with CUDA driver
- Python 3.11 (specifically — 3.12+ not supported by ML stack)
- Git
- ffmpeg (in PATH)
- Tesseract OCR 5.x installed at `C:\Program Files\Tesseract-OCR\`
- Poppler for Windows extracted to `C:\poppler\`


## What's Built

| Module | Status | Description |
|---|---|---|
| faster-whisper transcription | ✅ Done | Hindi audio → Hindi text |
| IndicTrans2 translation | ✅ Done | Hindi → English |
| flan-t5 cleaner | ✅ Done | Residual Indic terms → English |
| Intent Router (Router 1) | ✅ Done | RAG vs agent classification |
| Document parser | ✅ Done | Text, tables, lists, headers, OCR |
| ChromaDB ingestion | ✅ Done | Chunking + embedding + storage |
| RAG retrieval | 🔄 In progress | Similarity search working, improving quality |
| Router 2 | ⏳ To build | Vector DB vs web search |
| Hybrid retrieval | ⏳ To build | BM25 + vector + RRF |
| Reranker | ⏳ To build | Cross-encoder reranking |
| LLM generation | ⏳ To build | Groq / Llama 3 |
| Guardrails | ⏳ To build | Hallucination + PII checks |
| LangGraph agent | ⏳ To build | Form fill workflow |
| Voice output (TTS) | ⏳ To build | Response in user language |

---

## Known Issues

- **Duplicate retrieval results** — MMR retrieval will fix this
- **Q&A chunking** — current fixed-size chunker mixes multiple Q&A pairs, smart Q-based splitter needed
- **Embedding model** — switching to `multi-qa-MiniLM-L6-cos-v1` will improve Q&A retrieval accuracy
- **Poppler dependency** — OCR requires Poppler on Windows, not cross-platform out of the box
- **Dependency conflicts** — see [ISSUES.md](./ISSUES.md) for full details

---

## Next Steps

1. Fix retrieval quality — smart chunker + MMR + better embedding model
2. Add Router 2 — decide between vector DB and web search
3. Add BM25 hybrid retrieval
4. Add reranker
5. Connect Groq LLM for generation
6. Add guardrails
7. Build LangGraph agent for form filling
8. Add TTS for voice output
