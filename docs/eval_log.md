# Evaluation Log

Records notable evaluation runs on `hr-policy-golden` (`eval/golden.jsonl`).

| Date | Experiment | Change Made | Config (`LLM_MODEL`, `EVAL_JUDGE_MODEL`, `CHUNK_SIZE`, `TOP_K`, `RELEVANCE_THRESHOLD`) | `retrieval_hit` | `citation_present` | `correct_escalation` | `correctness` | `groundedness` |
|---|---|---|---|---|---|---|---|---|

| 2026-10-28 | `chunk-600` | CHUNK_SIZE 1000 -> 600 | `llm=qwen/qwen3.8-27b`, `judge=openai/gpt-oss-120b`, `chunk=600`, `k=5`, `threshold=0.68` | 1.00 | 1.00 | 0.97 | 0.88 | 0.82 |
