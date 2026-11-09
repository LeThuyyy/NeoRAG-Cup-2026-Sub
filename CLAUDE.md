# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Contestant repository for **NeoRAG Cup 2026** (organized by Team AI – ProPTIT). The task is to build a Retrieval-Augmented Generation (RAG) system answering Vietnamese questions about the S2B community (ProPTIT, MPC/Multimedia, GCC) based on `data/corpus.jsonl` (118 passages with integer `id`s).

All user-facing text, prompts, code comments, and CLI outputs are in Vietnamese. Maintain Vietnamese for new prompts and outputs.

### Competition rules & constraints (`docs/rules.md`)
- **LLM**: The **only** allowed generative LLM is `claude-haiku-5-5`, accessed through the organizers' OpenAI-compatible gateway (`OPENAI_BASE_URL=https://proptit.dekthe.dev/v1`). All generative tasks (answer generation, query rewriting, LLM reranking, agents) must use this model.
- **Embeddings & Rerankers**: `bge-m3` via the gateway, or local non-generative open-source models (via `sentence-transformers`, cross-encoders).
- **Gateway constraints**:
  - $5 hard budget per key for the entire competition; 100 req/min, 8 concurrent requests.
  - `claude-haiku-5-5` only accepts the default temperature; do **not** pass explicit `temperature` values (keep `temperature=None`).
  - Calls are cached in `.cache/*.sqlite` (`pipeline.sqlite` for pipeline runs, `judge.sqlite` for evaluation). Reruns with identical inputs consume no budget.
- **Hard restrictions**:
  - Do not edit `configs/eval.toml`, `src/neorag/evaluate.py`, or `src/neorag/metrics/` (official evaluation engine).
  - Do not hardcode answers or `retrieved_ids` for test questions.
  - Do not inject prompt injection / prompt gaming strings aimed at the LLM judge into generated answers.

## Common commands

### Environment setup
```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
python scripts/install.py            # or pip install -r requirements.txt
python scripts/install.py --local    # optional: adds local sentence-transformers with CUDA autodetect
copy .env.example .env               # fill OPENAI_API_KEY with competition key
```

### Running & evaluating pipelines
```bash
# Run pipeline and evaluate on train set (126 questions with ground truth)
neorag run train

# Fast evaluation on train without LLM judge (retrieval + lexical metrics only, no judge cost)
neorag run train --no-judge

# Run custom pipeline module
neorag run train --pipeline neorag.pipelines.baseline_sub6:BaselinePipeline --no-judge

# Generate submission predictions for public test (creates outputs/public_test_submission.zip)
neorag run public_test [--pipeline module.path:ClassName]

# Validate predictions file format against rules
neorag validate <split> [--predictions path]

# Evaluate an existing predictions file on train
neorag evaluate train [--predictions path] [--no-judge]
```
Note: `<split>` can be `train`, `public_test`, or `private_test` (maps to `data/<split>/questions.jsonl`).

### Testing & code quality
```bash
# Run all tests (offline, using mock LLM/Embedder from tests/conftest.py)
pytest

# Run a specific test file or single test
pytest tests/test_pipeline.py
pytest tests/test_pipeline.py::test_baseline_runs_end_to_end
pytest -k "test_baseline"

# Code formatting and linting (line length: 100)
ruff check .
ruff format .
```

### Demo chatbot
```powershell
# Windows PowerShell
python -m streamlit run app/chat.py
# With custom pipeline:
$env:NEORAG_PIPELINE="neorag.pipelines.baseline_sub6:BaselinePipeline"; python -m streamlit run app/chat.py
```

## High-level architecture

### 1. Pipeline Contract (`src/neorag/pipelines/base.py`)
All pipelines inherit from `RAGPipeline`:
- `prepare()`: Optional hook for preloading models (e.g. local reranker/embedder).
- `index(corpus: list[Document])`: Called once with the 118 corpus passages (`id`, `club`, `topic`, `text`).
- `run(query: str) -> RAGOutput(answer=str, retrieved_ids=list[int])`: Answers a query and returns retrieved document IDs (sorted by relevance, parent document IDs only).

Pipelines are loaded dynamically via `"module.path:ClassName"` strings. The default is `neorag.pipelines.baseline:BaselinePipeline`.

### 2. Execution & Evaluation Separation
- **Runner (`src/neorag/predict.py`)**: Runs the pipeline over queries, records per-query `latency_ms` (max 120s limit), deduplicates `retrieved_ids`, truncates them to `max_retrieved = 10` (configured in `eval.toml`), and outputs `<split>_pred.jsonl` and `<split>_submission.zip`.
- **Evaluator (`src/neorag/evaluate.py`)**: Reads the predictions JSONL and computes metrics without importing pipeline code.
- **Validation (`src/neorag/data.py:validate_predictions`)**: Validates IDs, maximum answer length (2,000 characters), non-empty answers, and bounds.

### 3. Scoring System (`configs/eval.toml`)
- **Benchmark Score = 0.5 × Retrieval + 0.5 × Generation** (range 0 to 1).
- **Retrieval Score**: Mean of Recall@5, MRR@5, and NDCG@5 (`src/neorag/metrics/retrieval.py`).
- **Generation Score**: Mean of:
  - Lexical: String Presence (`key_facts` substring hits), ROUGE-L.
  - Judge LLM: Groundedness (evaluated *only* against the text of the submitted `retrieved_ids`!), Response Relevancy, and `(1 - Noise Sensitivity)`.
  - Abstention: About 7–9% of questions are unanswerable (`answerable: false`). The model must decline to answer (e.g. "Xin lỗi, tôi không tìm thấy thông tin này trong tài liệu."). Correct abstention gives full generation points on unanswerable questions.

### 4. LLM & Embeddings Clients
- `LLMClient` (`src/neorag/llm.py`): Wraps `openai.OpenAI` client with exponential backoff retries and local SQLite caching.
- `LocalEmbedder` & `LocalCrossEncoder` (`src/neorag/embeddings.py`): Thread-safe local models via `sentence-transformers` for running dense embeddings or cross-encoder rerankers locally without API costs.

### 5. Outputs
All output artifacts are stored in `outputs/`:
- `<split>_pred.jsonl`: Predictions with `id`, `answer`, `retrieved_ids`, `latency_ms`.
- `<split>_submission.zip`: The archive uploaded to [aichallenge.ptit.edu.vn](https://aichallenge.ptit.edu.vn/).
- `eval_<pred-stem>/report.md`: Summary markdown report.
- `eval_<pred-stem>/report.json`: Machine-readable score summary.
- `eval_<pred-stem>/per_query.jsonl`: Individual query scores across all metrics (primary resource for error analysis).
