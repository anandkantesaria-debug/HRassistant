from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from langsmith import Client
from pydantic import ValidationError

from hrbot.config import Settings, setup_environment

DEFAULT_GOLDEN_PATH = Path("eval/golden.jsonl")
DEFAULT_DATASET_NAME = "hr-policy-golden"
REQUIRED_FIELDS = {
    "question",
    "reference_answer",
    "expected_source",
    "expected_page",
    "answerable",
}


def _resolve_settings(settings: Settings | None = None) -> Settings:
    if settings is not None:
        return settings
    try:
        return Settings()
    except ValidationError:
        return Settings(
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY") or "unused",
            LLM_MODEL=os.environ.get("LLM_MODEL") or "unused",
        )


def load_golden_items(path: str | Path = DEFAULT_GOLDEN_PATH) -> list[dict[str, Any]]:
    """Load and validate golden evaluation items from a JSONL file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Golden dataset file not found: {file_path}")

    items: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = REQUIRED_FIELDS - set(row.keys())
            if missing:
                raise ValueError(
                    f"Line {line_no} in {file_path} is missing required fields: {sorted(missing)}"
                )
            if not isinstance(row["answerable"], bool):
                raise TypeError(
                    f"Line {line_no} in {file_path}: 'answerable' must be a boolean."
                )
            items.append(row)

    if not items:
        raise ValueError(f"No items found in golden dataset file: {file_path}")

    return items


def count_needs_review(items: list[dict[str, Any]]) -> int:
    """Return the number of golden dataset rows still marked with needs_review=True."""
    return sum(1 for item in items if bool(item.get("needs_review", False)))


def _format_example_payload(item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = {"question": str(item["question"])}
    outputs = {
        "reference_answer": str(item["reference_answer"]),
        "expected_source": item["expected_source"],
        "expected_page": item["expected_page"],
        "answerable": bool(item["answerable"]),
    }
    return inputs, outputs


def upload_dataset(
    path: str | Path = DEFAULT_GOLDEN_PATH,
    dataset_name: str = DEFAULT_DATASET_NAME,
    *,
    client: Any | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Create or update the LangSmith dataset `hr-policy-golden` from `eval/golden.jsonl`."""
    items = load_golden_items(path)
    resolved_settings = _resolve_settings(settings)
    setup_environment(resolved_settings)

    ls_client = (
        client
        if client is not None
        else Client(api_key=resolved_settings.LANGSMITH_API_KEY)
    )

    dataset_exists = False
    if hasattr(ls_client, "has_dataset"):
        dataset_exists = bool(ls_client.has_dataset(dataset_name=dataset_name))
    else:
        try:
            ls_client.read_dataset(dataset_name=dataset_name)
            dataset_exists = True
        except Exception:  # noqa: BLE001
            dataset_exists = False

    if dataset_exists:
        dataset = ls_client.read_dataset(dataset_name=dataset_name)
        action = "updated"
    else:
        dataset = ls_client.create_dataset(
            dataset_name=dataset_name,
            description="Golden evaluation dataset for the HR Policy RAG Chatbot.",
        )
        action = "created"

    dataset_id = getattr(dataset, "id", None)

    existing_by_question: dict[str, Any] = {}
    if dataset_exists and hasattr(ls_client, "list_examples"):
        for ex in ls_client.list_examples(dataset_id=dataset_id):
            ex_inputs = getattr(ex, "inputs", None) or {}
            q_text = ex_inputs.get("question")
            if isinstance(q_text, str):
                existing_by_question[q_text] = ex

    created_count = 0
    updated_count = 0
    new_inputs: list[dict[str, Any]] = []
    new_outputs: list[dict[str, Any]] = []

    for item in items:
        inputs, outputs = _format_example_payload(item)
        question_text = inputs["question"]
        existing_ex = existing_by_question.get(question_text)
        if existing_ex is not None and hasattr(ls_client, "update_example"):
            ls_client.update_example(
                example_id=existing_ex.id,
                inputs=inputs,
                outputs=outputs,
            )
            updated_count += 1
        else:
            new_inputs.append(inputs)
            new_outputs.append(outputs)

    if new_inputs:
        if hasattr(ls_client, "create_examples"):
            ls_client.create_examples(
                inputs=new_inputs,
                outputs=new_outputs,
                dataset_id=dataset_id,
            )
        else:
            for inp, out in zip(new_inputs, new_outputs, strict=True):
                ls_client.create_example(
                    inputs=inp,
                    outputs=out,
                    dataset_id=dataset_id,
                )
        created_count = len(new_inputs)

    return {
        "dataset_name": dataset_name,
        "dataset_id": str(dataset_id) if dataset_id is not None else None,
        "action": action,
        "total_items": len(items),
        "created_examples": created_count,
        "updated_examples": updated_count,
        "needs_review_count": count_needs_review(items),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or update the LangSmith golden evaluation dataset."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_GOLDEN_PATH,
        help="Path to the golden JSONL dataset file (default: eval/golden.jsonl).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="LangSmith dataset name (default: hr-policy-golden).",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    client: Any | None = None,
    settings: Settings | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = upload_dataset(
            path=args.path,
            dataset_name=args.dataset_name,
            client=client,
            settings=settings,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error uploading dataset: {exc}", file=sys.stderr)
        return 1

    print(
        f"Dataset '{summary['dataset_name']}' ({summary['action']}): "
        f"{summary['total_items']} items "
        f"(created={summary['created_examples']}, updated={summary['updated_examples']}, "
        f"needs_review={summary['needs_review_count']})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
