"""Tests for Oracle RAG pipeline."""

import pytest
from rag.oracle.retriever import OracleRAGPipeline


class DummyGenerator:
    """Mock generator returning canned responses."""

    def __init__(self, response: str = "Oracle answer."):
        self.response_text = response
        self.calls = []

    def generate(self, prompt: str, system_prompt: str | None = None) -> dict:
        self.calls.append((prompt, system_prompt))
        return {
            "response": self.response_text,
            "error": False,
            "error_message": None,
            "prompt_tokens": 40,
            "generation_tokens": 15,
            "latency": 0.05,
            "load_duration": 0.01,
            "prompt_eval_speed": 800.0,
            "generation_speed": 300.0,
        }


def test_oracle_pipeline_execution():
    generator = DummyGenerator("Expected output based on gold context.")
    pipeline = OracleRAGPipeline(generator=generator)

    query = "What causes process synchronization issues?"
    gold_context = "Race conditions occur when multiple processes access shared data concurrently."

    result = pipeline.run(query, ground_truth_context=gold_context)

    assert result["response"] == "Expected output based on gold context."
    assert result["error"] is False
    assert result["retrieved_context"] == [gold_context]
    assert result["n_retrieved"] == 1
    assert result["n_llm_calls"] == 1
    assert result["latency"] > 0
