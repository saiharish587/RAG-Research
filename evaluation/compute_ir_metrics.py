"""
compute_ir_metrics.py - Compute standard Document-Level Information Retrieval (IR) metrics
(Document-Level MRR@3, Hit@1, Hit@3, Document-Level Recall@3, and Document-Level nDCG@3)
from raw benchmark logs with strict document-title ground-truth matching.

Methodological Design & Rigor:
------------------------------
1. Retrieval Unit: Unique Document Title.
   The full retrieved list of chunks is first deduplicated by document title
   (preserving the earliest rank position of each unique document).
   The top-k (k=3) unique retrieved documents are then evaluated.
   This guarantees that "Doc@3" strictly represents the top 3 unique retrieved documents.

2. Document-Level Recall@3:
   (Number of unique gold supporting document titles in Top-3 Unique Retrieved Documents) / (Total unique gold titles for question).

3. Document-Level nDCG@3:
   Computed with binary document relevance across the top-3 unique documents,
   benchmarked against ideal DCG (IDCG) based on the true unique gold document count.

4. Statistical Dispersion:
   Reports Mean and Sample Standard Deviation (ddof=1) across the benchmark population.

5. Missing Run Policy:
   Missing/failed logs are explicitly accounted for and scored as zero retrieval reward
   (rather than silently excluded), ensuring N=600 population completeness.

Evaluates all 4 context-augmented arms across LFM2-350M and LFM2-700M:
- Oracle RAG (Gold injected reference)
- Naive RAG (Single-stage FAISS retrieval)
- Advanced RAG (Query rewrite + BGE cross-encoder rerank)
- Modular RAG (Dynamic routing + Sub-queries + RRF fusion)
"""

import os
import re
import json
import numpy as np
import pandas as pd


def extract_doc_title(chunk_text: str) -> str:
    """Extract canonical document title prefix from passage text."""
    if not chunk_text:
        return ""
    # HotpotQA documents begin with 'Title: ...'
    m = re.match(r'^([^:\n\r]+):', chunk_text)
    if m:
        return m.group(1).strip().lower()
    return chunk_text.split('\n')[0].strip().lower()


def compute_document_level_ir_metrics(retrieved_chunks: list, gold_titles: set, k: int = 3):
    """
    Compute standard document-level IR metrics for a single query:
    1. Deduplicate all retrieved chunks by document title preserving earliest rank.
    2. Evaluate the top-k unique documents against gold supporting document titles.
    
    Returns:
    - doc_mrr_at_k: Reciprocal rank of the first retrieved gold document
    - doc_hit_at_1: 1.0 if Rank 1 unique document is a gold document, else 0.0
    - doc_hit_at_k: 1.0 if any unique document in top-k is a gold document, else 0.0
    - doc_recall_at_k: Fraction of unique gold document titles retrieved in top-k unique documents
    - doc_ndcg_at_k: nDCG at rank k over the top-k unique documents with binary relevance
    """
    if not gold_titles:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Step 1: Deduplicate entire retrieved list by document title, preserving earliest rank
    unique_doc_titles = []
    seen = set()
    for chunk in retrieved_chunks:
        title = extract_doc_title(chunk)
        if title and title not in seen:
            seen.add(title)
            unique_doc_titles.append(title)

    # Step 2: Take top-k unique documents
    top_docs = unique_doc_titles[:k]
    doc_hits = [1 if t in gold_titles else 0 for t in top_docs]

    # Pad to length k if fewer than k unique documents retrieved
    while len(doc_hits) < k:
        doc_hits.append(0)

    # 1. Doc-MRR@k (earliest rank among unique retrieved documents)
    first_rank = next((idx + 1 for idx, h in enumerate(doc_hits) if h == 1), None)
    mrr = 1.0 / first_rank if first_rank else 0.0

    # 2. Doc-Hit@1
    hit1 = float(doc_hits[0]) if doc_hits else 0.0

    # 3. Doc-Hit@k
    hit_k = 1.0 if sum(doc_hits) > 0 else 0.0

    # 4. Doc-Recall@k (unique gold titles covered in top-k unique docs)
    matched_gold_titles = set(top_docs) & gold_titles
    recall_k = len(matched_gold_titles) / len(gold_titles)

    # 5. Doc-nDCG@k over unique document ranks
    dcg = sum(doc_hits[i] / np.log2(i + 2) for i in range(k))
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

    total_questions = len(eval_set)
    print(f"Loaded {total_questions} canonical evaluation questions from {eval_set_path}")

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
            missing_count = 0

            for q_idx in range(1, total_questions + 1):
                fname = f"{model_key}_{arm}_q{q_idx}_run1.json"
                fpath = os.path.join(raw_logs_dir, fname)

                # Question query from eval_set
                eval_item = eval_set[q_idx - 1]
                query = eval_item["query"].strip()
                g_titles = gold_titles_map.get(query, set())

                if not os.path.exists(fpath):
                    # Explicit accounting for missing log: scored as 0 reward, not silently skipped
                    missing_count += 1
                    mrr, h1, h3, rec, ndcg = 0.0, 0.0, 0.0, 0.0, 0.0
                    n_retrieved = 0
                else:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    retrieved = data.get("pipeline_result", {}).get("retrieved_context", [])
                    n_retrieved = len(retrieved)
                    mrr, h1, h3, rec, ndcg = compute_document_level_ir_metrics(retrieved, g_titles, k=3)

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
                    "log_exists": (n_retrieved > 0 or os.path.exists(fpath)),
                    "doc_mrr_at_3": mrr,
                    "doc_hit_at_1": h1,
                    "doc_hit_at_3": h3,
                    "doc_recall_at_3": rec,
                    "doc_ndcg_at_3": ndcg
                })

            # Full Population N=600 (Sample SD ddof=1)
            sd_mrr = np.std(mrrs, ddof=1) if len(mrrs) > 1 else 0.0
            sd_ndcg = np.std(ndcgs, ddof=1) if len(ndcgs) > 1 else 0.0

            summary_rows.append({
                "Model": model_display,
                "RAG Arm": arm.capitalize() + " RAG",
                "Total Benchmark Qs (N)": total_questions,
                "Missing Logs": missing_count,
                "Active Retrievals": active_count,
                "Doc-MRR@3 (Mean +/- SD)": f"{np.mean(mrrs):.4f} +/- {sd_mrr:.4f}",
                "Doc-Hit@1 (Precision@1)": f"{np.mean(hit1s)*100:.2f}%",
                "Doc-Hit@3 (Success@3)": f"{np.mean(hit3s)*100:.2f}%",
                "Doc-Recall@3": f"{np.mean(recalls)*100:.2f}%",
                "Doc-nDCG@3 (Mean +/- SD)": f"{np.mean(ndcgs):.4f} +/- {sd_ndcg:.4f}",
            })

            # Conditional active stats for modular RAG
            if arm == "modular" and active_count > 0:
                act_sd_mrr = np.std(active_mrrs, ddof=1) if len(active_mrrs) > 1 else 0.0
                act_sd_ndcg = np.std(active_ndcgs, ddof=1) if len(active_ndcgs) > 1 else 0.0
                summary_rows.append({
                    "Model": model_display,
                    "RAG Arm": "Modular RAG (Active Only)",
                    "Total Benchmark Qs (N)": f"{active_count} / {total_questions}",
                    "Missing Logs": 0,
                    "Active Retrievals": active_count,
                    "Doc-MRR@3 (Mean +/- SD)": f"{np.mean(active_mrrs):.4f} +/- {act_sd_mrr:.4f}",
                    "Doc-Hit@1 (Precision@1)": f"{np.mean(active_hit1s)*100:.2f}%",
                    "Doc-Hit@3 (Success@3)": f"{np.mean(active_hit3s)*100:.2f}%",
                    "Doc-Recall@3": f"{np.mean(active_recalls)*100:.2f}%",
                    "Doc-nDCG@3 (Mean +/- SD)": f"{np.mean(active_ndcgs):.4f} +/- {act_sd_ndcg:.4f}",
                })

    # Save per-query detailed CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pd.DataFrame(all_results).to_csv(output_csv, index=False)
    print(f"\n[SUCCESS] Document-level IR metrics saved to: {output_csv}")

    # Display clean summary table
    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 125)
    print("STRICT DOCUMENT-LEVEL INFORMATION RETRIEVAL (IR) METRICS SUMMARY (N=600)")
    print("=" * 125)
    print(summary_df.to_string(index=False))
    print("=" * 125 + "\n")


if __name__ == "__main__":
    main()
