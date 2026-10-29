from __future__ import annotations

import argparse
import sys

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from hrbot.config import Settings
from hrbot.retriever import retrieve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Debug retrieval by printing the top-k chunks with scores, sources, and pages."
    )
    parser.add_argument(
        "question",
        type=str,
        help="Question to search against the HR policy index.",
    )
    parser.add_argument(
        "-k",
        "--k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5).",
    )
    parser.add_argument(
        "--mmr",
        action="store_true",
        help="Enable Maximal Marginal Relevance (MMR) search.",
    )
    return parser


def main(
    argv: list[str] | None = None,
    settings: Settings | None = None,
    embeddings: Embeddings | None = None,
    vectorstore: Chroma | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    chunks = retrieve(
        query=args.question,
        k=args.k,
        use_mmr=args.mmr,
        settings=settings,
        embeddings=embeddings,
        vectorstore=vectorstore,
    )

    print(f"Query: {args.question}")
    print(f"Retrieved {len(chunks)} chunk(s):")
    for idx, chunk in enumerate(chunks, start=1):
        print(
            f"[{idx}] score={chunk.score:.4f} | source={chunk.source} | page={chunk.page}"
        )
        print(f"    {chunk.text}")
        print("-" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

