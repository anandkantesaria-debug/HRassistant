from __future__ import annotations

import argparse
import sys
import uuid
from collections.abc import Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from hrbot.config import Settings
from hrbot.graph import RetrieverFn, ask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask an HR policy question and print the grounded answer with citations."
    )
    parser.add_argument(
        "question",
        type=str,
        help="The HR policy question to ask.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional UUID for the LangSmith run.",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    settings: Settings | None = None,
    llm: BaseChatModel | Callable[[list[BaseMessage]], Any] | None = None,
    retriever_fn: RetrieverFn | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    run_id = args.run_id or str(uuid.uuid4())
    result = ask(
        question=args.question,
        run_id=run_id,
        settings=settings,
        llm=llm,
        retriever_fn=retriever_fn,
    )

    print(f"Question: {args.question}")
    print(f"Route: {result.route}")
    print(f"Run ID: {run_id}")
    print("\nAnswer:")
    print(result.answer)

    if result.error:
        print(f"\nError: {result.error}")

    print("\nSources:")
    if result.citations:
        for cit in result.citations:
            print(f"  [{cit.n}] {cit.source} (page {cit.page}): {cit.snippet}")
    else:
        print("  (none)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

