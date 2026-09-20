"""
compute_ir_metrics.py - Compute standard IR metrics (MRR@3, Hit@1, Hit@3, Recall@3, nDCG@3)
from raw benchmark logs with strict document-title ground-truth matching.

Evaluates across all N=600 canonical HotpotQA evaluation questions for all 4 context-augmented arms:
- Oracle RAG (100% gold injected reference)
- Naive RAG (Single-stage FAISS retrieval)
- Advanced RAG (Query rewrite + BGE cross-encoder rerank)
- Modular RAG (Dynamic routing + Sub-queries + RRF fusion)

Exports comprehensive results to results/csv/ir_benchmark_metrics.csv.
"""

import os
import re
import json
import numpy as np
import pandas as pd


def extract_doc_title(chunk_text: str) -> str:
    """Extract document title prefix from passage text."""
    if not chunk_text:
        return ""
    m = re.match(r'^([^:\n\r]+):', chunk_text)
    if m:
        return m.group(1).strip().lower()
    return chunk_text.split('\n')[0].strip().lower()


def compute_query_ir_metrics(retrieved_chunks: list, gold_titles: set, k: int = 3):
    """
    Compute standard IR metrics for a single query:
    - MRR@k: Mean Reciprocal Rank of the first retrieved gold passage
    - Hit@1: Binary indicator if Rank 1 passage is a gold passage
    - Hit@k: Binary indicator if any top-k passage is a gold passage
    - Recall@k: Fraction of gold supporting passages retrieved in top-k
    - nDCG@k: Normalized Discounted Cumulative Gain at rank k with binary relevance
    """
    if not gold_titles:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    top_chunks = retrieved_chunks[:k]
    hits = [1 if extract_doc_title(c) in gold_titles else 0 for c in top_chunks]
    
    # Pad to length k if fewer retrieved
    while len(hits) < k:
        hits.append(0)

    # 1. MRR@k
    first_rank = next((idx + 1 for idx, h in enumerate(hits) if h == 1), None)
    mrr = 1.0 / first_rank if first_rank else 0.0

    # 2. Hit@1
    hit1 = float(hits[0])

    # 3. Hit@k
    hit_k = 1.0 if sum(hits) > 0 else 0.0

    # 4. Recall@k (unique gold titles covered in top-k)
    matched_gold_titles = set(extract_doc_title(c) for c in top_chunks if extract_doc_title(c) in gold_titles)
    recall_k = len(matched_gold_titles) / len(gold_titles)

    # 5. nDCG@k
    dcg = sum(hits[i] / np.log2(i + 2) for i in range(k))
    ideal_hits = [1] * min(len(gold_titles), k)
    idcg = sum(ideal_hits[i] / np.log2(i + 2) for i in range(len(ideal_hits)))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return mrr, hit1, hit_k, recall_k, ndcg


def main():
    eval_set_path = "data/benchmark/eval_set.json"
    raw_logs_dir = "results/raw"
    output_csv = "results/csv/ir_benchmark_metrics.csv"

    if not os.path.exists(eval_set_path):
        raise FileNotFoundError(f"Evaluation set not found at {eval_set_path}")

    # Load canonical ground-truth mapping
    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    print(f"Loaded {len(eval_set)} evaluation questions from {eval_set_path}")

    gold_titles_map = {}
    for item in eval_set:
        q = item["query"].strip()
        titles = set(sf[0].strip().lower() for sf in item.get("supporting_facts", []))
        if not titles and item.get("gold_context"):
            for gc in item["gold_context"]:
                if ":" in gc:
                    titles.add(gc.split(":", 1)[0].strip().lower())
        gold_titles_map[q] = titles

    models = [
        ("hf.co_LiquidAI_LFM2-350M-GGUF_Q4_K_M", "LFM2 350M"),
        ("hf.co_LiquidAI_LFM2-700M-GGUF_Q4_K_M", "LFM2 700M"),
    ]
    arms = ["oracle", "naive", "advanced", "modular"]

    all_results = []
    summary_rows = []

    for model_key, model_display in models:
        for arm in arms:
            mrrs, hit1s, hit3s, recalls, ndcgs = [], [], [], [], []
            active_mrrs, active_hit1s, active_hit3s, active_recalls, active_ndcgs = [], [], [], [], []
            active_count = 0

            for q_idx in range(1, len(eval_set) + 1):
                fname = f"{model_key}_{arm}_q{q_idx}_run1.json"
                fpath = os.path.join(raw_logs_dir, fname)

                if not os.path.exists(fpath):
                    continue

                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                query = data.get("query", "").strip()
                g_titles = gold_titles_map.get(query, set())
                retrieved = data.get("pipeline_result", {}).get("retrieved_context", [])
                n_retrieved = len(retrieved)

                mrr, h1, h3, rec, ndcg = compute_query_ir_metrics(retrieved, g_titles, k=3)

                mrrs.append(mrr)
                hit1s.append(h1)
                hit3s.append(h3)
                recalls.append(rec)
                ndcgs.append(ndcg)

                if n_retrieved > 0:
                    active_count += 1
                    active_mrrs.append(mrr)
                    active_hit1s.append(h1)
                    active_hit3s.append(h3)
                    active_recalls.append(rec)
                    active_ndcgs.append(ndcg)

                all_results.append({
                    "model": model_display,
                    "rag_type": arm,
                    "question_id": f"q{q_idx}",
                    "query": query,
                    "n_retrieved": n_retrieved,
                    "mrr_at_3": mrr,
                    "hit_at_1": h1,
                    "hit_at_3": h3,
                    "recall_at_3": rec,
                    "ndcg_at_3": ndcg
                })

            n_eval = len(mrrs)
            summary_rows.append({
                "Model": model_display,
                "RAG Arm": arm.capitalize() + " RAG",
                "Evaluated Queries (N)": f"{n_eval} (Active: {active_count})",
                "MRR@3 (Mean +/- SD)": f"{np.mean(mrrs):.4f} +/- {np.std(mrrs):.4f}",
                "Hit@1 (Precision@1)": f"{np.mean(hit1s)*100:.2f}%",
                "Hit@3 (Success@3)": f"{np.mean(hit3s)*100:.2f}%",
                "Recall@3": f"{np.mean(recalls)*100:.2f}%",
                "nDCG@3 (Mean +/- SD)": f"{np.mean(ndcgs):.4f} +/- {np.std(ndcgs):.4f}",
            })

            # For modular RAG, also show conditional active stats
            if arm == "modular" and active_count > 0:
                summary_rows.append({
                    "Model": model_display,
                    "RAG Arm": "Modular RAG (Active Only)",
                    "Evaluated Queries (N)": f"{active_count} / {n_eval}",
                    "MRR@3 (Mean +/- SD)": f"{np.mean(active_mrrs):.4f} +/- {np.std(active_mrrs):.4f}",
                    "Hit@1 (Precision@1)": f"{np.mean(active_hit1s)*100:.2f}%",
                    "Hit@3 (Success@3)": f"{np.mean(active_hit3s)*100:.2f}%",
                    "Recall@3": f"{np.mean(active_recalls)*100:.2f}%",
                    "nDCG@3 (Mean +/- SD)": f"{np.mean(active_ndcgs):.4f} +/- {np.std(active_ndcgs):.4f}",
                })

    # Save per-query detailed CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pd.DataFrame(all_results).to_csv(output_csv, index=False)
    print(f"\n[SUCCESS] Detailed per-query IR metrics saved to: {output_csv}")

    # Display clean summary table
    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 105)
    print("ALL-ARM STANDARD INFORMATION RETRIEVAL (IR) METRICS SUMMARY (N=600)")
    print("=" * 105)
    print(summary_df.to_string(index=False))
    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()
