from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from hrbot.retriever import RetrievedChunk

SYSTEM_PROMPT = (
    "You are an HR policy assistant. Follow these rules strictly:\n"
    "1. Answer ONLY from the numbered context passages provided below ([1], [2], etc.). "
    "Never use outside or general knowledge.\n"
    "2. If the provided context does not contain the answer, say plainly that the "
    "provided HR policy context does not contain the answer.\n"
    "3. Keep your answers short, direct, and factual.\n"
    "4. Cite every claim using bracketed markers like [1] or [2] that match the "
    "numbering of the context passages."
)

USER_PROMPT_TEMPLATE = (
    "Numbered Policy Context:\n"
    "{context}\n\n"
    "Employee Question: {question}\n\n"
    "Answer concisely using only the numbered context above and cite sources with [n]:"
)


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks as 1-based numbered blocks [1]..[k]."""
    if not chunks:
        return "(No context passages provided.)"

    blocks: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{idx}] (Source: {chunk.source}, Page: {chunk.page})\n{chunk.text.strip()}"
        )
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[BaseMessage]:
    """Build the system and user messages for the LLM."""
    context_str = format_context(chunks)
    user_content = USER_PROMPT_TEMPLATE.format(
        context=context_str,
        question=question.strip(),
    )
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]
