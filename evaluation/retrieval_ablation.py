"""Retrieval-only ablation of the Advanced RAG arm, replayed on the logged rewrites.

The Advanced arm has three retrieval stages: query rewriting (written by the
evaluated model), hybrid dense+sparse retrieval, and cross-encoder reranking.
Retrieval itself is deterministic, so each stage can be switched on or off and
replayed on the exact rewritten queries logged in August, with no LLM calls.
The results are therefore directly comparable with the paper's Advanced and
Naive rows, independent of the Ollama server version.

Configurations (query used; retrieval; reranking):

    naive          original;  dense top-3;           none
    rewrite_only   rewrite;   dense top-3;           none
    no_rewrite     original;  hybrid top-12;         rerank -> top-3
    no_rerank      rewrite;   hybrid top-12 -> top-3; none
    full           rewrite;   hybrid top-12;         rerank -> top-3   (the Advanced arm)

The ``full`` configuration must reproduce the logged Advanced top-3 passages;
its agreement rate is reported as a validity check.

Gold documents are matched by each chunk's source title (``chunks.npy``
metadata), not by text before the first colon.

Writes ``results/csv/retrieval_ablation.csv`` (one row per model, configuration
and question) and ``results/csv/retrieval_ablation_summary.csv``.

Usage (from the repository root)::

    python -m evaluation.retrieval_ablation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import load_config, setup_vector_db  # noqa: E402
from rag.advanced.retriever import AdvancedRAGPipeline  # noqa: E402

RESULTS_CSV = ROOT / "results" / "csv" / "benchmark_results.csv"
EVAL_SET = ROOT / "data" / "benchmark" / "eval_set.json"
OUT_ROWS = ROOT / "results" / "csv" / "retrieval_ablation.csv"
OUT_SUMMARY = ROOT / "results" / "csv" / "retrieval_ablation_summary.csv"

MODELS = {
    "hf.co/LiquidAI/LFM2-350M-GGUF:Q4_K_M": "350M",
    "hf.co/LiquidAI/LFM2-700M-GGUF:Q4_K_M": "700M",
}
CONFIGS = {
    # name: (use_rewrite, sparse, rerank)
    "naive": (False, None, False),
    "rewrite_only": (True, None, False),
    "no_rewrite": (False, True, True),
    "no_rerank": (True, True, False),
    "full": (True, True, True),
}
K = 3


def doc_metrics(sources: list[str], gold: set[str]) -> dict:
    """Document-level metrics over the top-K unique source titles."""
    unique = list(dict.fromkeys(sources))[:K]
    hits = [1 if s in gold else 0 for s in unique] + [0] * (K - len(unique))
    first = next((i + 1 for i, h in enumerate(hits) if h), None)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(gold), K)))
    return {
        "hit_at_3": float(any(hits)),
        "mrr_at_3": 1.0 / first if first else 0.0,
        "recall_at_3": len(set(unique) & gold) / len(gold) if gold else 0.0,
        "ndcg_at_3": sum(h / np.log2(i + 2) for i, h in enumerate(hits)) / idcg if idcg else 0.0,
    }


def main() -> None:
    config = load_config(str(ROOT / "configs" / "config.yaml"))
    db = setup_vector_db(config)
    retrieval = config.get("retrieval", {})
    pipe = AdvancedRAGPipeline(
        db_manager=db, generator=None, top_k=retrieval.get("top_k", K),
        rerank=True, rerank_model_name=retrieval.get("rerank_model", "BAAI/bge-reranker-base"),
    )
    text_to_source = {c["text"]: c["metadata"]["source"] for c in db.doc_chunks}

    eval_set = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    gold = {item["query"]: {sf[0] for sf in item["supporting_facts"]} for item in eval_set}
    order = {item["query"]: i + 1 for i, item in enumerate(eval_set)}

    logged = pd.read_csv(RESULTS_CSV, usecols=["query", "model_tag", "rag_type", "run_id", "rewritten_query"],
                         low_memory=False)
    logged = logged[(logged["rag_type"] == "advanced") & (logged["run_id"] == 1)]

    def retrieve(query: str, use_sparse, rerank: bool) -> list[str]:
        if use_sparse is None:
            results = db.search(query, top_k=K)
        else:
            pipe.sparse = use_sparse
            results = pipe.hybrid_search(query, top_k=pipe.rerank_top_n)
            if rerank:
                scores = pipe.reranker.predict([[query, r["chunk"]["text"]] for r in results])
                results = [r for _, r in sorted(zip(scores, results), key=lambda p: p[0], reverse=True)]
        return [r["chunk"]["text"] for r in results[:K]]

    rows, agreement = [], {}
    for _, rec in logged.iterrows():
        model = MODELS[rec["model_tag"]]
        query = rec["query"]
        rewrite = rec["rewritten_query"] if isinstance(rec["rewritten_query"], str) and rec["rewritten_query"] else query
        q_idx = order[query]
        logged_top = json.loads(
            (ROOT / "results" / "raw" / f"{rec['model_tag'].replace('/', '_').replace(':', '_')}_advanced_q{q_idx}_run1.json")
            .read_text(encoding="utf-8"))["pipeline_result"]["retrieved_context"]
        for name, (use_rewrite, use_sparse, rerank) in CONFIGS.items():
            texts = retrieve(rewrite if use_rewrite else query, use_sparse, rerank)
            if name == "full":
                agreement.setdefault(model, []).append(texts == logged_top)
            sources = [text_to_source.get(t, f"__unknown_{i}") for i, t in enumerate(texts)]
            rows.append({"model": model, "config": name, "question_index": q_idx,
                         **doc_metrics(sources, gold[query])})
        if q_idx % 100 == 0:
            print(f"{model}: {q_idx}/600")

    df = pd.DataFrame(rows).sort_values(["model", "config", "question_index"])
    df.to_csv(OUT_ROWS, index=False)

    summary = []
    for model, sub in df.groupby("model"):
        base = sub[sub["config"] == "naive"].set_index("question_index")
        full = sub[sub["config"] == "full"].set_index("question_index")
        for name, part in sub.groupby("config"):
            part = part.set_index("question_index")
            row = {"model": model, "config": name, "n_questions": len(part)}
            for col in ("hit_at_3", "mrr_at_3", "recall_at_3", "ndcg_at_3"):
                row[col] = round(part[col].mean(), 4)
            for ref_name, ref in (("naive", base), ("full", full)):
                lost = int(((ref["hit_at_3"] == 1) & (part["hit_at_3"] == 0)).sum())
                gained = int(((ref["hit_at_3"] == 0) & (part["hit_at_3"] == 1)).sum())
                row[f"vs_{ref_name}_lost"] = lost
                row[f"vs_{ref_name}_gained"] = gained
                row[f"vs_{ref_name}_mcnemar_p"] = (
                    float(f"{stats.binomtest(lost, lost + gained, 0.5).pvalue:.6g}") if lost + gained else 1.0)
                d = part["mrr_at_3"] - ref["mrr_at_3"]
                row[f"vs_{ref_name}_mrr_diff"] = round(d.mean(), 4)
                row[f"vs_{ref_name}_mrr_p"] = float(f"{stats.ttest_rel(part['mrr_at_3'], ref['mrr_at_3']).pvalue:.6g}") if d.abs().sum() else 1.0
            if name == "full":
                row["full_matches_logged_top3"] = round(float(np.mean(agreement[model])), 4)
            summary.append(row)
    out = pd.DataFrame(summary)
    out.to_csv(OUT_SUMMARY, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
