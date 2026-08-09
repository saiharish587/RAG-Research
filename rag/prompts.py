"""Prompt templates shared by every pipeline arm.

Defined in one place because prompt parity is an experimental requirement, not
a style preference. In v1 the no-RAG arm used a bare ``"Question: {q}\\nAnswer:"``
while all three RAG arms received a persona and an explicit abstention
instruction. That difference confounds the variable under study: a measured
gap between no-RAG and RAG conflated *"does retrieval help?"* with *"does a
better prompt help?"*, and the two cannot be separated after the fact.

The templates below differ **only** in the presence of a retrieved-context
block. Persona, task framing and abstention instruction are byte-identical
across arms, so the sole systematic difference between conditions is retrieval
itself.
"""

from __future__ import annotations

from typing import Sequence

__all__ = [
    "SYSTEM_PERSONA",
    "ABSTENTION_INSTRUCTION",
    "build_answer_prompt",
    "build_rewrite_prompt",
    "build_routing_prompt",
    "build_subquery_prompt",
    "format_context",
]

# Identical across every arm.
SYSTEM_PERSONA = "You are a helpful research assistant."

# Identical across every arm. The wording deliberately avoids the word
# "context" so the same sentence is meaningful with and without retrieval.
ABSTENTION_INSTRUCTION = (
    "If you do not have enough information to answer, state that you do not know."
)

CONTEXT_SEPARATOR = "\n\n"


def format_context(chunks: Sequence[str]) -> str:
    """Join retrieved chunks into a single context block."""
    return CONTEXT_SEPARATOR.join(chunks)


def build_answer_prompt(query: str, retrieved_context: Sequence[str] | None = None) -> str:
    """The answer-generation prompt for every arm.

    With ``retrieved_context`` empty or ``None`` this produces the no-RAG
    prompt, which is the RAG prompt minus only the context block.
    """
    if retrieved_context:
        return (
            f"{SYSTEM_PERSONA} Use the following pieces of context to answer the question. "
            f"{ABSTENTION_INSTRUCTION}\n\n"
            f"Context:\n{format_context(retrieved_context)}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
    return (
        f"{SYSTEM_PERSONA} Answer the question. "
        f"{ABSTENTION_INSTRUCTION}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


def build_rewrite_prompt(query: str) -> str:
    """Query-rewriting prompt used by the advanced arm."""
    return (
        "You are a search query optimizer. Given a user query, output a search-friendly "
        "rephrased version of it. Do not include introductory text or markdown formatting. "
        "Output only the optimized query.\n\n"
        f"Original Query: {query}\n"
        "Optimized Query:"
    )


def build_routing_prompt(query: str) -> str:
    """Routing prompt used by the modular arm.

    The instruction demands a single bare token so the reply can be parsed
    exactly rather than by substring search.
    """
    return (
        "Determine if answering the following user question requires retrieving information "
        "from internal files/documents. Answer with the single word 'yes' or 'no' and nothing "
        "else.\n\n"
        f"Question: {query}\n"
        "Requires Retrieval:"
    )


def build_subquery_prompt(query: str, n: int = 2) -> str:
    """Sub-query generation prompt used by the modular arm.

    ``n`` is interpolated rather than hard-coded so the prompt can never
    disagree with the number of sub-queries the code actually keeps -- in v1
    the prompt asked for two, the code sliced two, and the paper claimed three.
    """
    return (
        f"Generate exactly {_spell(n)} search queries related to the following user question. "
        "Separate them by a newline. Do not output numbers or markdown. Just the queries.\n\n"
        f"Question: {query}\n"
        "Queries:"
    )


_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def _spell(n: int) -> str:
    return _NUMBER_WORDS.get(n, str(n))
