"""Naive RAG: single dense retrieval, then generation.

The simplest retrieval condition -- one embedding search, top-k chunks, one
generation call. No query rewriting, no reranking, no routing.
"""

from __future__ import annotations

from typing import Any

from rag.prompts import build_answer_prompt


class NaiveRAGPipeline:
    """Retrieves top-k chunks by dense similarity and answers from them."""

    name = "naive"

    def __init__(self, db_manager, generator, top_k: int = 3):
        self.db_manager = db_manager
        self.generator = generator
        self.top_k = top_k

    def run(self, query: str) -> dict[str, Any]:
        search_results = self.db_manager.search(query, top_k=self.top_k)
        retrieved_contexts = [r["chunk"]["text"] for r in search_results]

        prompt = build_answer_prompt(query, retrieved_contexts)
        result = self.generator.generate(prompt)
        result["retrieved_context"] = retrieved_contexts
        return result
