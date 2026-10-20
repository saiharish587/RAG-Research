"""Token precision, token recall, and paired tests for the SLM-RAG benchmark.

Reads ``results/csv/benchmark_results.csv`` and writes two files:

``results/csv/token_precision_recall_summary.csv``
    Per model and RAG arm: Token F1 (recomputed), Token recall, Token precision,
    and Exact Match, averaged over all runs.
``results/csv/paired_tests_recall_em.csv``
    Question-level paired comparisons (each question's runs averaged first, then
    compared across questions) for Token recall and Exact Match.

Degenerate runs have no exact-match value in the CSV. They are scored as 0, the
same way they count as failures for Token F1.

Usage (from the repository root)::

    python -m evaluation.token_recall_analysis
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.metrics import tokenise  # noqa: E402

RESULTS_CSV = ROOT / "results" / "csv" / "benchmark_results.csv"
SUMMARY_CSV = ROOT / "results" / "csv" / "token_precision_recall_summary.csv"
PAIRED_CSV = ROOT / "results" / "csv" / "paired_tests_recall_em.csv"

M350 = "hf.co/LiquidAI/LFM2-350M-GGUF:Q4_K_M"
M700 = "hf.co/LiquidAI/LFM2-700M-GGUF:Q4_K_M"

# (label, (model, arm), (model, arm))
COMPARISONS = [
    ("350M vs 700M, Naive", (M350, "naive"), (M700, "naive")),
    ("350M vs 700M, Oracle", (M350, "oracle"), (M700, "oracle")),
    ("350M Naive vs Advanced", (M350, "naive"), (M350, "advanced")),
]
PAIRED_METRICS = ["token_recall", "exact_match"]


def token_prf(response: str, ground_truth: str) -> tuple[float, float, float]:
    """Token precision, recall, and F1, matching ``evaluation.metrics.token_f1``."""
    pred = tokenise(response)
    gold = tokenise(ground_truth)
    if not pred or not gold:
        return 0.0, 0.0, 0.0
    overlap = sum((Counter(pred) & Counter(gold)).values())
    if overlap == 0:
        return 0.0, 0.0, 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return precision, recall, 2 * precision * recall / (precision + recall)


def load_scored_runs() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV, low_memory=False)
    df["response"] = df["response"].fillna("")
    df["ground_truth"] = df["ground_truth"].fillna("")

    scores = [token_prf(r, g) for r, g in zip(df["response"], df["ground_truth"])]
    df["token_precision"], df["token_recall"], df["token_f1_recomputed"] = zip(*scores)

    df["exact_match"] = pd.to_numeric(df["exact_match"], errors="coerce").fillna(0.0)

    stored = pd.to_numeric(df["token_f1"], errors="coerce")
    max_gap = float(np.nanmax(np.abs(df["token_f1_recomputed"] - stored)))
    print(f"Runs: {len(df)}  |  max |recomputed F1 - stored token_f1| = {max_gap:.2e}")
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["model_tag", "rag_type"])
        .agg(
            n=("token_recall", "size"),
            token_f1_mean=("token_f1_recomputed", "mean"),
            token_recall_mean=("token_recall", "mean"),
            token_recall_sd=("token_recall", "std"),
            token_precision_mean=("token_precision", "mean"),
            exact_match_mean=("exact_match", "mean"),
        )
        .round(4)
        .reset_index()
    )
    return summary


def paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    # One value per question: the mean over its repeated runs.
    per_question = df.groupby(["model_tag", "rag_type", "query"])[PAIRED_METRICS].mean()

    rows = []
    for label, a, b in COMPARISONS:
        for metric in PAIRED_METRICS:
            x = per_question.xs(a, level=["model_tag", "rag_type"])[metric]
            y = per_question.xs(b, level=["model_tag", "rag_type"])[metric]
            d = (x - y).dropna()
            n = len(d)
            dbar = d.mean()
            se = d.std(ddof=1) / np.sqrt(n)
            tcrit = stats.t.ppf(0.975, n - 1)
            t_stat, p_t = stats.ttest_rel(x.loc[d.index], y.loc[d.index])
            try:
                _, p_w = stats.wilcoxon(d)
            except ValueError:
                p_w = float("nan")
            rows.append(
                {
                    "comparison": label,
                    "metric": metric,
                    "n_questions": n,
                    "mean_a": round(x.mean(), 4),
                    "mean_b": round(y.mean(), 4),
                    "mean_diff_a_minus_b": round(dbar, 4),
                    "ci95_low": round(dbar - tcrit * se, 4),
                    "ci95_high": round(dbar + tcrit * se, 4),
                    "t": round(float(t_stat), 3),
                    "p_t": float(p_t),
                    "p_wilcoxon": float(p_w),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    df = load_scored_runs()

    summary = summarise(df)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nWrote {SUMMARY_CSV.relative_to(ROOT)}")
    print(summary.to_string(index=False))

    paired = paired_tests(df)
    paired.to_csv(PAIRED_CSV, index=False)
    print(f"\nWrote {PAIRED_CSV.relative_to(ROOT)}")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
