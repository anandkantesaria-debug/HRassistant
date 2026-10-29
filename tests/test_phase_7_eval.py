from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from eval.evaluators import (
    _invoke_judge,
    build_evaluators,
    citation_present,
    correct_escalation,
    correctness,
    groundedness,
    retrieval_hit,
)
from eval.run_eval import (
    append_eval_log,
    build_experiment_metadata,
    load_thresholds,
    run_evaluation,
    validate_eval_models,
)
from eval.run_eval import (
    main as run_eval_main,
)
from eval.upload_dataset import (
    DEFAULT_DATASET_NAME,
    count_needs_review,
    load_golden_items,
)
from eval.upload_dataset import (
    main as upload_dataset_main,
)
from hrbot.config import Settings
from hrbot.retriever import RetrievedChunk


class FakeJudgeChatModel(BaseChatModel):
    """Offline fake LLM judge for unit testing correctness and groundedness."""

    responses: list[str]
    calls: list[list[BaseMessage]]

    def __init__(self, responses: list[str]) -> None:
        super().__init__(responses=responses, calls=[])
        self.responses = list(responses)
        self.calls = []

    @property
    def _llm_type(self) -> str:
        return "fake-judge-llm"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(messages)
        idx = min(len(self.calls) - 1, len(self.responses) - 1)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.responses[idx]))]
        )


@dataclass
class _FakeDataset:
    id: str
    name: str


@dataclass
class _FakeExample:
    id: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]


class FakeLangSmithClient:
    """In-memory fake LangSmith client for testing upload_dataset."""

    def __init__(self) -> None:
        self.datasets: dict[str, _FakeDataset] = {}
        self.examples: dict[str, list[_FakeExample]] = {}
        self.updated_ids: list[str] = []

    def has_dataset(self, *, dataset_name: str) -> bool:
        return dataset_name in self.datasets

    def create_dataset(self, *, dataset_name: str, description: str = "") -> _FakeDataset:
        ds = _FakeDataset(id=f"ds-{len(self.datasets) + 1}", name=dataset_name)
        self.datasets[dataset_name] = ds
        self.examples[ds.id] = []
        return ds

    def read_dataset(self, *, dataset_name: str) -> _FakeDataset:
        return self.datasets[dataset_name]

    def list_examples(self, *, dataset_id: str) -> list[_FakeExample]:
        return list(self.examples.get(dataset_id, []))

    def create_examples(
        self,
        *,
        inputs: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
        dataset_id: str,
    ) -> None:
        bucket = self.examples.setdefault(dataset_id, [])
        for inp, out in zip(inputs, outputs, strict=True):
            bucket.append(
                _FakeExample(
                    id=f"ex-{len(bucket) + 1}",
                    inputs=inp,
                    outputs=out,
                )
            )

    def update_example(
        self,
        *,
        example_id: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
    ) -> None:
        self.updated_ids.append(example_id)
        for bucket in self.examples.values():
            for ex in bucket:
                if ex.id == example_id:
                    ex.inputs = inputs
                    ex.outputs = outputs


def _make_settings(
    llm_model: str = "qwen/qwen3.8-27b",
    judge_model: str | None = "gemini-2.5-flash",
    concurrency: int = 2,
) -> Settings:
    return Settings(
        GROQ_API_KEY="test-groq-key",
        GOOGLE_API_KEY="test-google-key",
        LLM_MODEL=llm_model,
        EVAL_JUDGE_MODEL=judge_model,
        CHUNK_SIZE=1000,
        CHUNK_OVERLAP=150,
        TOP_K=5,
        RELEVANCE_THRESHOLD=0.68,
        EVAL_MAX_CONCURRENCY=concurrency,
        LANGSMITH_TRACING=False,
        _env_file=None,
    )


def _write_ready_golden_file(tmp_path: Path) -> Path:
    golden_path = tmp_path / "golden_ready.jsonl"
    rows = [
        {
            "question": "How many unused vacation days can I carry over?",
            "reference_answer": "Up to 5 unused vacation days.",
            "expected_source": "Company Salaried Employee Handbook 2025.pdf",
            "expected_page": 43,
            "answerable": True,
        },
        {
            "question": "What is the employee stock option vesting cliff?",
            "reference_answer": "Not covered in HR policies; escalate to HR.",
            "expected_source": None,
            "expected_page": None,
            "answerable": False,
        },
    ]
    golden_path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    return golden_path


def test_ac_7_1_golden_dataset_schema_counts_and_variety() -> None:
    """AC-7.1: eval/golden.jsonl has >= 30 items, required fields, >= 25% unanswerable, and tricky/paraphrased items."""
    golden_path = Path("eval/golden.jsonl")
    assert golden_path.exists()

    items = load_golden_items(golden_path)
    assert len(items) >= 30

    required_keys = {
        "question",
        "reference_answer",
        "expected_source",
        "expected_page",
        "answerable",
    }
    for row in items:
        assert required_keys.issubset(row.keys())
        assert isinstance(row["question"], str) and row["question"].strip()
        assert isinstance(row["reference_answer"], str) and row["reference_answer"].strip()
        assert isinstance(row["answerable"], bool)
        if row["answerable"]:
            assert isinstance(row["expected_source"], str) and row["expected_source"].endswith(".pdf")
            assert isinstance(row["expected_page"], int) and row["expected_page"] >= 1
        else:
            assert row["expected_source"] is None
            assert row["expected_page"] is None
        if "needs_review" in row:
            assert isinstance(row["needs_review"], bool)

    unanswerable = [r for r in items if r["answerable"] is False]
    assert len(unanswerable) / len(items) >= 0.25

    all_questions_lower = " ".join(r["question"].lower() for r in items)
    assert "who must" in all_questions_lower or "approve" in all_questions_lower or "authorize" in all_questions_lower
    assert "how many" in all_questions_lower or "percentage" in all_questions_lower


def test_ac_7_2_upload_dataset_creates_or_updates_langsmith_dataset(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """AC-7.2: `python -m eval.upload_dataset` creates or updates the LangSmith dataset `hr-policy-golden`."""
    fake_client = FakeLangSmithClient()
    settings = _make_settings()

    # First run: creates dataset and all examples
    rc1 = upload_dataset_main(
        ["--path", "eval/golden.jsonl"],
        client=fake_client,
        settings=settings,
    )
    assert rc1 == 0
    out1 = capsys.readouterr().out
    assert DEFAULT_DATASET_NAME in out1
    assert "created" in out1

    ds = fake_client.read_dataset(dataset_name=DEFAULT_DATASET_NAME)
    assert len(fake_client.examples[ds.id]) >= 30

    # Second run: updates existing examples idempotently without duplicating
    rc2 = upload_dataset_main(
        ["--path", "eval/golden.jsonl"],
        client=fake_client,
        settings=settings,
    )
    assert rc2 == 0
    out2 = capsys.readouterr().out
    assert "updated" in out2
    assert len(fake_client.examples[ds.id]) == len(load_golden_items("eval/golden.jsonl"))
    assert len(fake_client.updated_ids) == len(load_golden_items("eval/golden.jsonl"))


def test_ac_7_3_run_eval_cli_invokes_langsmith_evaluate_and_respects_needs_review_guard(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """AC-7.3: `python -m eval.run_eval --experiment-prefix <name>` runs graph with LangSmith evaluate (after review flags cleared)."""
    settings = _make_settings()

    # 1. With needs_review=true on a golden file, refuses to run experiment
    unreviewed_golden = tmp_path / "golden_unreviewed.jsonl"
    unreviewed_golden.write_text(
        json.dumps(
            {
                "question": "How many unused vacation days can I carry over?",
                "reference_answer": "Up to 5 unused vacation days.",
                "expected_source": "Company Salaried Employee Handbook 2025.pdf",
                "expected_page": 43,
                "answerable": True,
                "needs_review": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert count_needs_review(load_golden_items(unreviewed_golden)) == 1
    rc_blocked = run_eval_main(
        [
            "--experiment-prefix",
            "baseline-blocked",
            "--golden-path",
            str(unreviewed_golden),
        ],
        settings=settings,
    )
    assert rc_blocked == 1
    err_blocked = capsys.readouterr().err
    assert "needs_review" in err_blocked

    # 2. Once needs_review flags are cleared, invokes LangSmith `evaluate` with graph target_fn
    ready_golden = _write_ready_golden_file(tmp_path)
    captured_call: dict[str, Any] = {}

    fake_gen_llm = FakeJudgeChatModel(["Employees may carry over up to 5 vacation days [1]."])
    fake_judge_llm = FakeJudgeChatModel(["SCORE: 1\nREASON: Matches reference and context."])
    sample_chunk = RetrievedChunk(
        text="Employees may carry over up to 5 vacation days into the following year.",
        source="Company Salaried Employee Handbook 2025.pdf",
        page=43,
        score=0.82,
    )

    def fake_evaluate(
        target: Any,
        *,
        data: str,
        evaluators: list[Any],
        experiment_prefix: str,
        metadata: dict[str, Any],
        max_concurrency: int,
    ) -> list[dict[str, Any]]:
        captured_call["data"] = data
        captured_call["experiment_prefix"] = experiment_prefix
        captured_call["metadata"] = metadata
        captured_call["max_concurrency"] = max_concurrency
        captured_call["evaluator_names"] = [e.__name__ for e in evaluators]

        ex_output = target({"question": "How many unused vacation days can I carry over?"})
        captured_call["target_output"] = ex_output

        run_obj = {"outputs": ex_output}
        example_obj = {
            "inputs": {"question": "How many unused vacation days can I carry over?"},
            "outputs": {
                "reference_answer": "Up to 5 unused vacation days.",
                "expected_source": "Company Salaried Employee Handbook 2025.pdf",
                "expected_page": 43,
                "answerable": True,
            },
        }
        eval_results = [ev(run_obj, example_obj) for ev in evaluators]
        return [{"evaluation_results": {"results": eval_results}}]

    rc_ok = run_eval_main(
        [
            "--experiment-prefix",
            "baseline-v1",
            "--golden-path",
            str(ready_golden),
        ],
        settings=settings,
        evaluate_fn=fake_evaluate,
        llm=fake_gen_llm,
        judge_llm=fake_judge_llm,
        retriever_fn=lambda q, k: [sample_chunk],
    )
    assert rc_ok == 0
    assert captured_call["data"] == DEFAULT_DATASET_NAME
    assert captured_call["experiment_prefix"] == "baseline-v1"
    assert captured_call["target_output"]["route"] == "answer"
    assert len(captured_call["target_output"]["citations"]) == 1
    assert captured_call["evaluator_names"] == [
        "retrieval_hit",
        "citation_present",
        "correct_escalation",
        "correctness",
        "groundedness",
    ]


def test_ac_7_4_evaluators_code_and_llm_judges() -> None:
    """AC-7.4: Tests retrieval_hit, citation_present, correct_escalation, correctness, and groundedness."""
    answerable_example = {
        "inputs": {"question": "How many vacation days can I carry over?"},
        "outputs": {
            "reference_answer": "Up to 5 vacation days can be carried over.",
            "expected_source": "Company Salaried Employee Handbook 2025.pdf",
            "expected_page": 43,
            "answerable": True,
        },
    }
    unanswerable_example = {
        "inputs": {"question": "Do we get pet insurance?"},
        "outputs": {
            "reference_answer": "Not covered in the HR policy handbook; escalate to HR.",
            "expected_source": None,
            "expected_page": None,
            "answerable": False,
        },
    }

    good_run = {
        "outputs": {
            "answer": "You can carry over up to 5 vacation days [1].",
            "route": "answer",
            "citations": [
                {
                    "n": 1,
                    "source": "Company Salaried Employee Handbook 2025.pdf",
                    "page": 43,
                    "snippet": "Employees may carry over up to 5 vacation days.",
                }
            ],
            "chunks": [
                {
                    "text": "Employees may carry over up to 5 vacation days.",
                    "source": "Company Salaried Employee Handbook 2025.pdf",
                    "page": 43,
                    "score": 0.81,
                }
            ],
            "citations_missing": False,
        }
    }
    bad_run = {
        "outputs": {
            "answer": "You can carry over 20 days.",
            "route": "escalate",
            "citations": [],
            "chunks": [{"text": "Other text", "source": "other.pdf", "page": 1, "score": 0.4}],
            "citations_missing": True,
        }
    }
    escalated_run = {
        "outputs": {
            "answer": "I couldn't find a confident answer to your question in the HR policy documents.",
            "route": "escalate",
            "citations": [],
            "chunks": [],
            "citations_missing": False,
        }
    }

    # 1. retrieval_hit (answerable only)
    assert retrieval_hit(good_run, answerable_example)["score"] == 1.0
    assert retrieval_hit(bad_run, answerable_example)["score"] == 0.0
    assert retrieval_hit(escalated_run, unanswerable_example)["score"] is None

    # 2. citation_present (answerable only)
    assert citation_present(good_run, answerable_example)["score"] == 1.0
    assert citation_present(bad_run, answerable_example)["score"] == 0.0
    assert citation_present(escalated_run, unanswerable_example)["score"] is None

    # 3. correct_escalation
    assert correct_escalation(good_run, answerable_example)["score"] == 1.0
    assert correct_escalation(bad_run, answerable_example)["score"] == 0.0
    assert correct_escalation(escalated_run, unanswerable_example)["score"] == 1.0
    assert correct_escalation(good_run, unanswerable_example)["score"] == 0.0

    # 4. correctness (LLM judge)
    judge_pass = FakeJudgeChatModel(["SCORE: 1\nREASON: Answer states 5 vacation days accurately."])
    judge_fail = FakeJudgeChatModel(["SCORE: 0\nREASON: Answer states 20 days instead of 5."])
    res_corr_pass = correctness(good_run, answerable_example, judge_llm=judge_pass)
    res_corr_fail = correctness(bad_run, answerable_example, judge_llm=judge_fail)
    assert res_corr_pass["score"] == 1.0
    assert "5 vacation days" in res_corr_pass["comment"]
    assert res_corr_fail["score"] == 0.0

    # 5. groundedness (LLM judge)
    judge_grounded = FakeJudgeChatModel(["SCORE: 1\nREASON: Claim of 5 days is directly in chunk [1]."])
    judge_ungrounded = FakeJudgeChatModel(["SCORE: 0\nREASON: 20 days is not in the retrieved context."])
    assert groundedness(good_run, answerable_example, judge_llm=judge_grounded)["score"] == 1.0
    assert groundedness(bad_run, answerable_example, judge_llm=judge_ungrounded)["score"] == 0.0

    evaluators = build_evaluators(judge_llm=judge_pass)
    assert len(evaluators) == 5

    # 6. Judge invocation retries with exponential backoff on transient 429 rate-limit errors
    attempts = 0
    backoff_delays: list[float] = []

    def flaky_judge(_msgs: list[BaseMessage]) -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("HTTP 429 Too Many Requests")
        return "SCORE: 1\nREASON: Recovered after rate limit."

    score, reason = _invoke_judge(
        [],
        judge_llm=flaky_judge,
        backoff_fn=backoff_delays.append,
    )
    assert score == 1.0
    assert reason == "Recovered after rate limit."
    assert attempts == 3
    assert backoff_delays == [1.0, 2.0]

    # 7. Google 429 errors with "Please retry in Ns" honor the server-suggested delay
    google_attempts = 0
    google_delays: list[float] = []

    def google_rate_limited_judge(_msgs: list[BaseMessage]) -> str:
        nonlocal google_attempts
        google_attempts += 1
        if google_attempts == 1:
            raise RuntimeError(
                "Error code: 429 - RESOURCE_EXHAUSTED ... Please retry in 28.504239895s."
            )
        return "SCORE: 0\nREASON: Wrong number."

    score_g, _ = _invoke_judge(
        [],
        judge_llm=google_rate_limited_judge,
        backoff_fn=google_delays.append,
    )
    assert score_g == 0.0
    assert google_attempts == 2
    assert google_delays == [pytest.approx(29.504239895)]


def test_ac_7_5_explicit_model_validation_and_experiment_metadata() -> None:
    """AC-7.5: LLM_MODEL and EVAL_JUDGE_MODEL must be explicit model IDs, and metadata records both + chunk_size, k, threshold."""
    valid_settings = _make_settings(
        llm_model="qwen/qwen3.8-27b",
        judge_model="gemini-2.5-flash",
    )
    gen_id, judge_id = validate_eval_models(valid_settings)
    assert gen_id == "qwen/qwen3.8-27b"
    assert judge_id == "gemini-2.5-flash"

    metadata = build_experiment_metadata(valid_settings)
    assert metadata["llm_model"] == "qwen/qwen3.8-27b"
    assert metadata["eval_judge_model"] == "gemini-2.5-flash"
    assert metadata["chunk_size"] == 1000
    assert metadata["k"] == 5
    assert metadata["threshold"] == 0.68

    for bad_alias in ("auto", "default", "free", "combo/free", "router/default", None):
        with pytest.raises(ValueError, match="explicit model ID"):
            validate_eval_models(
                _make_settings(llm_model=bad_alias or "unused", judge_model="gemini-2.5-pro")
            )
        with pytest.raises(ValueError, match="explicit model ID"):
            validate_eval_models(
                _make_settings(llm_model="gemini-2.5-flash", judge_model=bad_alias)
            )


def test_ac_7_6_summary_table_and_threshold_exit_codes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """AC-7.6: Run prints summary table of mean scores and exits non-zero if any score is below eval/thresholds.yaml."""
    thresholds = load_thresholds("eval/thresholds.yaml")
    assert thresholds == {
        "retrieval_hit": 0.85,
        "correct_escalation": 0.85,
        "correctness": 0.70,
        "groundedness": 0.80,
    }

    ready_golden = _write_ready_golden_file(tmp_path)
    settings = _make_settings()

    def _make_fake_evaluate(scores: dict[str, float]) -> Any:
        def _runner(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            return [
                {
                    "evaluation_results": {
                        "results": [
                            {"key": k, "score": v}
                            for k, v in scores.items()
                        ]
                    }
                }
            ]

        return _runner

    # Passing scores -> exit code 0
    rc_pass = run_eval_main(
        ["--experiment-prefix", "pass-run", "--golden-path", str(ready_golden)],
        settings=settings,
        evaluate_fn=_make_fake_evaluate(
            {
                "retrieval_hit": 0.90,
                "citation_present": 0.95,
                "correct_escalation": 0.90,
                "correctness": 0.80,
                "groundedness": 0.88,
            }
        ),
    )
    assert rc_pass == 0
    out_pass = capsys.readouterr().out
    assert "retrieval_hit" in out_pass
    assert "0.9000" in out_pass
    assert "PASS" in out_pass

    # Failing score (correctness 0.60 < 0.70) -> exit code 1
    rc_fail = run_eval_main(
        ["--experiment-prefix", "fail-run", "--golden-path", str(ready_golden)],
        settings=settings,
        evaluate_fn=_make_fake_evaluate(
            {
                "retrieval_hit": 0.90,
                "citation_present": 0.95,
                "correct_escalation": 0.90,
                "correctness": 0.60,
                "groundedness": 0.88,
            }
        ),
    )
    assert rc_fail == 1
    captured_fail = capsys.readouterr()
    assert "FAIL" in captured_fail.out
    assert "correctness: 0.6000 < 0.70" in captured_fail.err


def test_ac_7_7_eval_log_markdown_recording(tmp_path: Path) -> None:
    """AC-7.7: docs/eval_log.md exists and records date, change made, config, and scores per notable run."""
    repo_log = Path("docs/eval_log.md")
    assert repo_log.exists()
    header_content = repo_log.read_text(encoding="utf-8")
    assert "Date" in header_content
    assert "Change Made" in header_content
    assert "Config" in header_content

    temp_log = tmp_path / "eval_log.md"
    settings = _make_settings(llm_model="gemini-2.5-flash", judge_model="gemini-2.5-pro")
    row = append_eval_log(
        experiment_prefix="baseline-v1",
        change_made="Initial baseline with CHUNK_SIZE=1000",
        settings=settings,
        mean_scores={
            "retrieval_hit": 0.92,
            "citation_present": 0.88,
            "correct_escalation": 0.94,
            "correctness": 0.79,
            "groundedness": 0.91,
        },
        log_path=temp_log,
        run_date="2026-10-25",
    )
    assert "2026-10-25" in row
    assert "baseline-v1" in row
    assert "Initial baseline with CHUNK_SIZE=1000" in row
    assert "gemini-2.5-flash" in row
    assert "0.92" in row
    assert temp_log.read_text(encoding="utf-8").endswith(row)


def test_ac_7_8_configurable_concurrency_default_two(tmp_path: Path) -> None:
    """AC-7.8: Evaluation concurrency is configurable and defaults to 2."""
    ready_golden = _write_ready_golden_file(tmp_path)
    default_settings = Settings(
        GROQ_API_KEY="test-groq-key",
        GOOGLE_API_KEY="test-google-key",
        LLM_MODEL="qwen/qwen3.8-27b",
        EVAL_JUDGE_MODEL="gemini-2.5-flash",
        LANGSMITH_TRACING=False,
        _env_file=None,
    )
    assert default_settings.EVAL_MAX_CONCURRENCY == 2

    observed_concurrencies: list[int] = []

    def fake_evaluate(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        observed_concurrencies.append(kwargs["max_concurrency"])
        return [
            {
                "evaluation_results": {
                    "results": [
                        {"key": "retrieval_hit", "score": 1.0},
                        {"key": "citation_present", "score": 1.0},
                        {"key": "correct_escalation", "score": 1.0},
                        {"key": "correctness", "score": 1.0},
                        {"key": "groundedness", "score": 1.0},
                    ]
                }
            }
        ]

    summary_default = run_evaluation(
        "concurrency-default",
        golden_path=ready_golden,
        settings=default_settings,
        evaluate_fn=fake_evaluate,
    )
    assert summary_default.max_concurrency == 2

    summary_custom = run_evaluation(
        "concurrency-custom",
        golden_path=ready_golden,
        max_concurrency=4,
        settings=default_settings,
        evaluate_fn=fake_evaluate,
    )
    assert summary_custom.max_concurrency == 4
    assert observed_concurrencies == [2, 4]
