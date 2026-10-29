# HR Policy Assistant (HRBot)

An enterprise-ready, agentic Retrieval-Augmented Generation (RAG) assistant that accurately answers employee questions from HR policy PDFs with verifiable inline citations and automatic human escalation.

Built with **LangGraph**, **ChromaDB**, **Streamlit**, and **LangSmith**.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
  - [1. Ingest Policy Documents](#1-ingest-policy-documents)
  - [2. Launch the Streamlit Web Application](#2-launch-the-streamlit-web-application)
  - [3. Query via Command Line Interface (CLI)](#3-query-via-command-line-interface-cli)
  - [4. Debug Retrieval Performance](#4-debug-retrieval-performance)
- [Threshold Calibration & Evaluation](#threshold-calibration--evaluation)
  - [Calibrating Confidence Thresholds](#calibrating-confidence-thresholds)
  - [Running the Golden Evaluation Suite](#running-the-golden-evaluation-suite)
- [Running Tests](#running-tests)
- [Configuration Reference](#configuration-reference)
- [License](#license)

---

## Overview

Navigating dense company policy handbooks (leave policies, remote work, health insurance, travel reimbursements) is often time-consuming for employees and HR teams alike. Standard LLM chatbots risk hallucinating corporate policies or providing outdated answers.

**HRBot** solves this with an agentic, guardrailed RAG architecture:
1. **Verifiable Citations**: Every answer directly maps assertions to page numbers and exact document excerpts (`[1]`, `[2]`).
2. **Confidence Grading & Escalation**: Retrieved chunks are scored against a calibrated relevance threshold. If policy documents lack sufficient relevance or confidence, the assistant gracefully escalates to the HR team with contact details instead of guessing.
3. **Local & Cost-Effective Embeddings**: Employs local HuggingFace embeddings (`BAAI/bge-small-en-v1.5`), eliminating vector embedding API costs and external data exposure.
4. **End-to-End Evaluation & Observability**: Integrated with LangSmith tracing and an automated evaluation suite testing hit rate, citation presence, correct escalation, correctness, and groundedness.

---

## Key Features

- **Agentic LangGraph Workflow**: A stateful graph governing retrieval, confidence grading, answer generation, citation extraction, and fallback escalation.
- **Fast, Precise PDF Ingestion**: Extracts text using [PyMuPDF](https://github.com/pymupdf/PyMuPDF) (`fitz`), divides text into semantic chunks with overlap, and generates deterministic chunk hashes for idempotent upserts in ChromaDB.
- **Hybrid Retrieval & MMR Support**: Standard similarity search and Maximal Marginal Relevance (MMR) to balance query relevance with chunk diversity.
- **Interactive Streamlit Web UI**: Includes instant example question chips, live system health checks (Chroma vector count & LLM reachability), and collapsible source inspector panes.
- **CLI & Diagnostics**: Built-in CLI commands to ask questions, inspect top-$k$ retrieved chunks with similarity scores, and run smoke tests.
- **Automated Threshold Calibration**: Analyzes separation between answerable and out-of-scope queries to recommend optimal decision boundaries.

---

## Project Structure

```text
HRAssistant/
├── app/
│   └── streamlit_app.py         # Streamlit chat application & UI
├── data/
│   ├── chroma/                  # Persistent ChromaDB vector database directory
│   └── policies/                # Directory containing HR policy PDF handbooks
├── docs/
│   └── eval_log.md              # Historical log of evaluation experiments and metrics
├── eval/
│   ├── calibration_questions.jsonl  # In-scope and out-of-scope query pairs
│   ├── evaluators.py            # LangSmith custom evaluators (groundedness, hit rate, etc.)
│   ├── golden.jsonl             # Ground-truth QA golden dataset
│   ├── run_eval.py              # Automated evaluation runner against LangSmith
│   ├── thresholds.yaml          # Target evaluation pass/fail thresholds
│   └── upload_dataset.py        # LangSmith dataset synchronization utility
├── src/
│   └── hrbot/
│       ├── ask.py               # CLI interface for asking questions
│       ├── config.py            # Pydantic Settings configuration & env loader
│       ├── embeddings.py        # HuggingFace BGE embedding model loader
│       ├── graph.py             # LangGraph state machine, nodes, and citation parser
│       ├── ingest.py            # PDF text extraction, chunking, and Chroma indexing
│       ├── llm.py               # LLM client initialization and health checks
│       ├── prompts.py           # Strict system prompts and context formatting
│       ├── retrieval_debug.py   # CLI tool to inspect retrieved chunks and scores
│       ├── retriever.py         # Vectorstore queries and scoring logic
│       └── smoke.py             # LLM connectivity and latency verification
├── tests/                       # Unit and integration test suite
├── tools/
│   └── calibrate_threshold.py   # Statistical threshold calibration utility
├── .env.example                 # Environment variables configuration template
├── pyproject.toml               # Project dependencies and build configuration
└── uv.lock                      # Deterministic lockfile
```

---

## Prerequisites

- **Python**: `>= 3.11`
- **Package Manager**: [uv](https://github.com/astral-sh/uv) (recommended) or standard `pip`
- **Groq API Key**: For fast generation LLM inference (e.g. `qwen/qwen3.8-27b` or `llama-3.3-70b-versatile`)
- *(Optional)* **LangSmith API Key**: For execution tracing and evaluation tracking

---

## Installation & Setup

### 1. Clone Repository & Setup Environment

Using `uv` (recommended):
```bash
git clone <repo-url>
cd HRAssistant
uv sync
```

Or using standard `venv` and `pip`:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -e ".[dev]"
```

### 2. Configure Environment Variables

Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```

Open `.env` and fill in your credentials and desired model settings:
```ini
GROQ_API_KEY=gsk_your_groq_api_key_here
LLM_MODEL=qwen/qwen3.8-27b
HR_CONTACT_EMAIL=hr@yourcompany.com

# Optional: LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=hr-policy-bot
```

### 3. Verify Connection

Run the smoke test to confirm your LLM endpoint is reachable:
```bash
uv run python -m hrbot.smoke
```

---

## Usage Guide

### 1. Ingest Policy Documents

Place your company policy PDFs (e.g., `Employee_Handbook_2025.pdf`) into `data/policies/`.

Run the ingestion pipeline to parse, chunk, embed, and index documents into ChromaDB:

```bash
# Ingest with existing collection retention
uv run python -m hrbot.ingest --path data/policies

# Or reset and rebuild the collection from scratch
uv run python -m hrbot.ingest --path data/policies --reset
```

### 2. Launch the Streamlit Web Application

Start the interactive web UI:

```bash
uv run streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser. You can:
- Ask questions in natural language.
- Click pre-configured example prompts in the sidebar.
- Expand the **Sources** drawer under each answer to review cited document titles, page numbers, and exact excerpts.
- View real-time system status indicators (number of indexed chunks and LLM health).

### 3. Query via Command Line Interface (CLI)

You can ask questions directly from your terminal:

```bash
uv run python -m hrbot.ask "How many days of paid annual leave do full-time employees receive?"
```

Example output:
```text
Question: How many days of paid annual leave do full-time employees receive?
Route: answer
Run ID: d4f944ae-16bf-4632-bd99-7023c0a4ef41

Answer:
Full-time employees receive 20 days of paid annual leave per year, accrued semi-monthly [1].

Sources:
  [1] Company Salaried Employee Handbook 2025.pdf (page 14): Full-time employees accrue paid annual leave at a rate of 1.67 days per month (20 days annually)...
```

If you ask an out-of-scope question:
```bash
uv run python -m hrbot.ask "Can I bring my pet iguana to the client site?"
```
```text
Question: Can I bring my pet iguana to the client site?
Route: escalate

Answer:
I couldn't find a confident answer to your question in the HR policy documents. Please contact HR directly at hr@yourcompany.com for assistance.

Sources:
  (none)
```

### 4. Debug Retrieval Performance

To inspect the raw similarity scores, sources, and chunk texts returned for any query:

```bash
# Retrieve top-5 chunks using cosine similarity
uv run python -m hrbot.retrieval_debug "What is the daily meal allowance?" -k 5

# Retrieve using Maximal Marginal Relevance (MMR)
uv run python -m hrbot.retrieval_debug "What is the daily meal allowance?" -k 5 --mmr
```

---

## Threshold Calibration & Evaluation

### Calibrating Confidence Thresholds

To separate valid HR queries from queries that should be escalated to human HR, run the calibration tool against `eval/calibration_questions.jsonl`:

```bash
uv run python tools/calibrate_threshold.py
```

This tool calculates:
- Score distribution for answerable vs. unanswerable questions (mean, median, 25th/75th percentiles).
- Top-$k$ source and page recall.
- Suggested `RELEVANCE_THRESHOLD` value to update in your `.env`.

### Running the Golden Evaluation Suite

Run the full evaluation pipeline using LangSmith and our golden dataset (`eval/golden.jsonl`):

```bash
uv run python eval/run_eval.py --experiment "bge-qwen27b-eval"
```

The evaluator verifies quality metrics against target baselines defined in `eval/thresholds.yaml`:

| Metric | Target Threshold | Description |
|---|---|---|
| `retrieval_hit` | $\ge 0.85$ | Ground truth source document and page found in top-$k$ chunks |
| `citation_present` | $\ge 0.90$ | Generated answers contain valid `[n]` citation markers |
| `correct_escalation` | $\ge 0.85$ | Out-of-scope questions route correctly to human HR escalation |
| `correctness` | $\ge 0.70$ | Answer matches ground-truth facts (evaluated via LLM judge) |
| `groundedness` | $\ge 0.80$ | Claims in answer are strictly supported by context chunks |

Results are automatically appended to `docs/eval_log.md`.

---

## Running Tests

Execute the unit test suite:

```bash
# Run standard offline unit tests
uv run pytest

# Run with coverage report
uv run pytest --cov=src/hrbot

# Code linting and formatting check
uv run ruff check .
```

---

## Configuration Reference

Key parameters configured in `.env` (managed via `hrbot.config.Settings`):

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | Base URL for LLM inference |
| `GROQ_API_KEY` | *(Required)* | Groq API authentication token |
| `LLM_MODEL` | *(Required)* | Generation LLM identifier (e.g. `qwen/qwen3.8-27b`) |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Hugging Face model for vector embeddings |
| `CHROMA_DIR` | `data/chroma` | Path to persistent Chroma storage directory |
| `COLLECTION_NAME` | `hr_policies` | Chroma collection name |
| `CHUNK_SIZE` | `600` | Target character count per text chunk |
| `CHUNK_OVERLAP` | `150` | Overlapping character count between consecutive chunks |
| `TOP_K` | `3` | Number of context chunks passed to generator |
| `RELEVANCE_THRESHOLD`| `0.68` | Cosine similarity cutoff for answer vs. escalate routing |
| `HR_CONTACT_EMAIL` | `hr@yourcompany.com` | Fallback contact email provided on escalation |
| `LANGSMITH_TRACING` | `true` | Enables/disables LangSmith execution tracing |
| `EVAL_JUDGE_MODEL` | `openai/gpt-oss-120b` | Model used as LLM judge in evaluation pipeline |

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

