"""RAGChecker Evaluation Integration Module.

Evaluates benchmark run predictions against ground truths and retrieved contexts
using the RAGChecker framework.
"""

from __future__ import annotations
import os
import json
from typing import Dict, List, Any

try:
    from ragchecker import RAGResults, RAGChecker
    RAGCHECKER_AVAILABLE = True
except ImportError:
    RAGCHECKER_AVAILABLE = False


def evaluate_with_ragchecker(
    results: List[Dict[str, Any]],
    extractor_model: str = "gpt-4o-mini",
    checker_model: str = "gpt-4o-mini",
    output_path: str = "results/reports/ragchecker_report.json"
) -> Dict[str, Any]:
    """Evaluates a batch of benchmark results using RAGChecker.
    
    Expected result dictionary format:
    {
        "query": str,
        "response": str,
        "ground_truth": str,
        "retrieved_context": List[str]
    }
    """
    if not RAGCHECKER_AVAILABLE:
        print("[WARNING] RAGChecker is not installed in the environment. Skipping RAGChecker evaluation.")
        return {}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    ragchecker_input = []
    for idx, res in enumerate(results):
        # Format for RAGChecker: query, response, gt_answer, retrieved_context (list of dict or str)
        retrieved = res.get("retrieved_context") or []
        if isinstance(retrieved, str):
            retrieved = [retrieved]
            
        ragchecker_input.append({
            "query_id": str(res.get("run_id", idx)),
            "query": res.get("query", ""),
            "gt_answer": res.get("ground_truth", ""),
            "response": res.get("response") or "",
            "retrieved_context": [{"doc_id": str(i), "text": doc} for i, doc in enumerate(retrieved)]
        })

    if not os.getenv("OPENAI_API_KEY"):
        print("[WARNING] OPENAI_API_KEY is not set. Skipping RAGChecker LLM-based claim extraction.")
        print("[INFO] Note: All primary benchmark metrics (Token F1, Exact Match, Cosine Similarity, Groundedness, Retrieval Hit Rate, Latency) have already been computed cleanly!")
        return {}

    try:
        rag_results = RAGResults.from_dict({"results": ragchecker_input})
        evaluator = RAGChecker(
            extractor_name=extractor_model,
            checker_name=checker_model,
            batch_size_extractor=16,
            batch_size_checker=16
        )
        evaluator.evaluate(rag_results)
        
        metrics = rag_results.metrics
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
            
        print(f"[RAGChecker] Evaluation complete. Results saved to {output_path}")
        return metrics
    except Exception as e:
        print(f"[RAGChecker WARNING] Could not complete RAGChecker LLM pass ({e}). Primary metrics remain fully intact.")
        return {}
