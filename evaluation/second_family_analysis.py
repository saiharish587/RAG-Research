"""Second model family (SmolLM2-135M and SmolLM2-360M): the same analyses as for LFM2.

Reads ``results/csv/benchmark_results_smollm2.csv`` and the SmolLM2 raw logs, and
writes:

``results/csv/smollm2_summary.csv``
    Per model and arm: Token F1, EM, token precision and recall, cosine,
    groundedness, answer-term hit rate, latency, generated tokens, refusal rate,
    failure counts, and repeat consistency.
``results/csv/smollm2_evidence.csv``
    One row per statistic: the scale comparison (smaller minus larger model) on
    every metric, wins, sign and permutation tests, the length-matched gap,
    Advanced vs Naive, router behaviour, and document-level retrieval.

Conventions match the LFM2 analyses: failed runs (degenerate or truncated)
count as zero for Token F1 and EM, and tests compare question-level means over
the 600 questions.

Usage (from the repository root)::

    python -m evaluation.second_family_analysis
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.metrics import tokenise  # noqa: E402
from evaluation.revision_analyses import (  # noqa: E402
    BOOT_SEED, N_BOOT, N_LENGTH_BINS, N_PERM, Evidence, binned_gap, length_bins,
    ols_intercept, paired,
)

RESULTS_CSV = ROOT / "results" / "csv" / "benchmark_results_smollm2.csv"
EVAL_SET = ROOT / "data" / "benchmark" / "eval_set.json"
CHUNKS = ROOT / "vector_db" / "faiss_index" / "chunks.npy"
SUMMARY_CSV = ROOT / "results" / "csv" / "smollm2_summary.csv"
EVIDENCE_CSV = ROOT / "results" / "csv" / "smollm2_evidence.csv"

SMALL, LARGE = "135M", "360M"
MODELS = {
    "hf.co/bartowski/SmolLM2-135M-Instruct-GGUF:Q4_K_M": SMALL,
    "hf.co/bartowski/SmolLM2-360M-Instruct-GGUF:Q4_K_M": LARGE,
}
ARMS = ["no_rag", "oracle", "naive", "advanced", "modular"]
NUMERIC = ["token_f1", "exact_match", "cosine", "groundedness", "retrieval_hit_rate",
           "latency", "generation_tokens", "is_refusal"]


def load() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV, low_memory=False)
    df["model"] = df["model_tag"].map(MODELS)
    df["response"] = df["response"].fillna("")
    df["ground_truth"] = df["ground_truth"].fillna("")
    df["is_refusal"] = df["is_refusal"].astype(str).str.lower().isin(["true", "1", "1.0"]).astype(float)
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["token_f1"] = df["token_f1"].fillna(0.0)
    df["exact_match"] = df["exact_match"].fillna(0.0)

    prec, rec = [], []
    for resp, truth in zip(df["response"], df["ground_truth"]):
        p, g = tokenise(resp), tokenise(truth)
        ov = sum((Counter(p) & Counter(g)).values()) if p and g else 0
        prec.append(ov / len(p) if ov else 0.0)
        rec.append(ov / len(g) if ov else 0.0)
    df["token_precision"], df["token_recall"] = prec, rec
    return df


def summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, arm), sub in df.groupby(["model", "rag_type"]):
        distinct = sub.groupby("query")["response"].nunique()
        rows.append({
            "model": model, "rag_type": arm, "runs": len(sub),
            "token_f1_mean": sub["token_f1"].mean(), "token_f1_sd": sub["token_f1"].std(),
            "exact_match": sub["exact_match"].mean(),
            "token_precision": sub["token_precision"].mean(), "token_recall": sub["token_recall"].mean(),
            "cosine": sub["cosine"].mean(), "groundedness": sub["groundedness"].mean(),
            "answer_term_hit_rate": sub["retrieval_hit_rate"].mean(),
            "latency_mean": sub["latency"].mean(), "latency_sd": sub["latency"].std(),
            "latency_median": sub["latency"].median(),
            "generation_tokens_mean": sub["generation_tokens"].mean(),
            "generation_tokens_sd": sub["generation_tokens"].std(),
            "refusal_rate": sub["is_refusal"].mean(),
            "failed_runs": int((sub["status"] != "ok").sum()),
            "questions_with_identical_repeats": int((distinct == 1).sum()),
        })
    out = pd.DataFrame(rows)
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(4)
    return out


def question_level(df: pd.DataFrame) -> pd.DataFrame:
    cols = NUMERIC + ["token_precision", "token_recall"]
    return df.groupby(["model", "rag_type", "query"])[cols].mean().reset_index()


def series(q, model, arm, col):
    return q[(q["model"] == model) & (q["rag_type"] == arm)].set_index("query")[col]


def scale_comparison(q: pd.DataFrame, ev: Evidence) -> None:
    for arm in ARMS:
        for metric in ("token_f1", "exact_match", "token_precision", "token_recall", "cosine", "groundedness"):
            a, b = series(q, SMALL, arm, metric), series(q, LARGE, arm, metric)
            if a.notna().sum() and b.notna().sum():
                ev.add_test(f"scale_{metric}", paired(a, b), arm=arm)


def wins_and_permutation(q: pd.DataFrame, ev: Evidence, arm: str) -> None:
    a, b = series(q, SMALL, arm, "token_f1"), series(q, LARGE, arm, "token_f1")
    la, lb = series(q, SMALL, arm, "generation_tokens"), series(q, LARGE, arm, "generation_tokens")
    aw, bw = a > b, b > a
    ev.add("wins", "questions_small_higher", int(aw.sum()), arm=arm)
    ev.add("wins", "questions_large_higher", int(bw.sum()), arm=arm)
    ev.add("wins", "questions_tied", int((~aw & ~bw).sum()), arm=arm)
    ev.add("wins", "sign_test_p", stats.binomtest(int(aw.sum()), int(aw.sum() + bw.sum()), 0.5).pvalue, arm=arm)
    ev.add("wins", "mean_margin_on_small_wins", (a - b)[aw].mean(), arm=arm)
    ev.add("wins", "mean_margin_on_large_wins", (b - a)[bw].mean(), arm=arm)
    ev.add("wins", "tokens_small_on_small_wins", la[aw].mean(), arm=arm)
    ev.add("wins", "tokens_large_on_small_wins", lb[aw].mean(), arm=arm)
    ev.add("wins", "tokens_small_on_large_wins", la[bw].mean(), arm=arm)
    ev.add("wins", "tokens_large_on_large_wins", lb[bw].mean(), arm=arm)
    d = (a - b).to_numpy()
    rng = np.random.default_rng(BOOT_SEED)
    exceed = 0
    for start in range(0, N_PERM, 10_000):
        signs = rng.choice([-1.0, 1.0], size=(min(10_000, N_PERM - start), len(d)))
        exceed += int((np.abs((signs * d).mean(axis=1)) >= abs(d.mean())).sum())
    ev.add("wins", "permutation_p", (exceed + 1) / (N_PERM + 1), arm=arm)
    ra, rb = series(q, SMALL, arm, "token_recall"), series(q, LARGE, arm, "token_recall")
    ev.add("wins", "share_full_recall_small", (ra >= 1.0).mean(), arm=arm)
    ev.add("wins", "share_full_recall_large", (rb >= 1.0).mean(), arm=arm)
    ea, eb = series(q, SMALL, arm, "exact_match"), series(q, LARGE, arm, "exact_match")
    ev.add("wins", "questions_exact_small_only", int(((ea > 0) & (eb == 0)).sum()), arm=arm)
    ev.add("wins", "questions_exact_large_only", int(((eb > 0) & (ea == 0)).sum()), arm=arm)


def length_matched(q: pd.DataFrame, ev: Evidence, arm: str) -> None:
    fa, fb = series(q, SMALL, arm, "token_f1"), series(q, LARGE, arm, "token_f1")
    la, lb = series(q, SMALL, arm, "generation_tokens"), series(q, LARGE, arm, "generation_tokens")
    common = fa.index.intersection(fb.index)
    fa, fb, la, lb = (s.loc[common].to_numpy() for s in (fa, fb, la, lb))
    la, lb = np.clip(la, 1, None), np.clip(lb, 1, None)
    edges = np.unique(np.quantile(np.concatenate([la, lb]), np.linspace(0, 1, N_LENGTH_BINS + 1)))
    nb = len(edges) - 1
    ba, bb = length_bins(la, edges), length_bins(lb, edges)
    x, y = np.log(la) - np.log(lb), fa - fb
    rng = np.random.default_rng(BOOT_SEED)
    boot_b, boot_r = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        boot_b.append(binned_gap(fa[i], fb[i], ba[i], bb[i], nb))
        boot_r.append(ols_intercept(x[i], y[i])[0])
    ev.add("length_matched_gap", "raw_gap_small_minus_large", y.mean(), arm=arm)
    ev.add("length_matched_gap", "binned_gap", binned_gap(fa, fb, ba, bb, nb), arm=arm)
    ev.add("length_matched_gap", "binned_ci95_low", np.nanpercentile(boot_b, 2.5), arm=arm)
    ev.add("length_matched_gap", "binned_ci95_high", np.nanpercentile(boot_b, 97.5), arm=arm)
    ev.add("length_matched_gap", "regression_gap_at_equal_length", ols_intercept(x, y)[0], arm=arm)
    ev.add("length_matched_gap", "regression_ci95_low", np.nanpercentile(boot_r, 2.5), arm=arm)
    ev.add("length_matched_gap", "regression_ci95_high", np.nanpercentile(boot_r, 97.5), arm=arm)


def arm_comparisons(df: pd.DataFrame, q: pd.DataFrame, ev: Evidence) -> None:
    for model in (SMALL, LARGE):
        ev.add_test("advanced_minus_naive_f1",
                    paired(series(q, model, "advanced", "token_f1"), series(q, model, "naive", "token_f1")), model)
        ev.add_test("oracle_minus_naive_f1",
                    paired(series(q, model, "oracle", "token_f1"), series(q, model, "naive", "token_f1")), model)
        ev.add_test("naive_minus_no_rag_f1",
                    paired(series(q, model, "naive", "token_f1"), series(q, model, "no_rag", "token_f1")), model)
        mod = df[(df["model"] == model) & (df["rag_type"] == "modular")]
        route = mod.groupby("query")["route"].agg(lambda s: s.mode().iat[0])
        declined = route.index[route == "no_rag"]
        ev.add("router", "runs_routed_away", int((mod["route"] == "no_rag").sum()), model)
        ev.add("router", "questions_routed_away", len(declined), model)
        no_rag, naive = series(q, model, "no_rag", "token_f1").loc[declined], series(q, model, "naive", "token_f1").loc[declined]
        ev.add("router", "no_rag_f1_on_declined", no_rag.mean(), model)
        ev.add("router", "naive_f1_on_declined", naive.mean(), model)
        ev.add("router", "retrieval_helped", int((naive > no_rag).sum()), model)
        ev.add("router", "retrieval_hurt", int((no_rag > naive).sum()), model)
        raw = mod["router_raw"].fillna("").astype(str).str.strip().str.lower()
        ev.add("router", "replies_starting_no", int(raw.str.match(r"^no\b").sum()), model)
        ev.add("router", "replies_starting_yes", int(raw.str.match(r"^yes\b").sum()), model)


def document_retrieval(ev: Evidence) -> None:
    """Document-level Hit@3 and MRR@3 from run-1 logs, matching chunks by source title."""
    chunks = np.load(CHUNKS, allow_pickle=True)
    source = {c["text"]: c["metadata"]["source"] for c in chunks}
    eval_set = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    for tag, model in MODELS.items():
        safe = tag.replace("/", "_").replace(":", "_")
        for arm in ("naive", "advanced", "modular"):
            hits, mrrs, active = [], [], 0
            for i, item in enumerate(eval_set, 1):
                gold = {sf[0] for sf in item["supporting_facts"]}
                log = ROOT / "results" / "raw" / f"{safe}_{arm}_q{i}_run1.json"
                ctx = json.loads(log.read_text(encoding="utf-8"))["pipeline_result"]["retrieved_context"]
                active += bool(ctx)
                titles = list(dict.fromkeys(source.get(t, f"_{j}") for j, t in enumerate(ctx)))[:3]
                h = [t in gold for t in titles]
                hits.append(float(any(h)))
                mrrs.append(next((1 / (k + 1) for k, v in enumerate(h) if v), 0.0))
            ev.add("document_retrieval", "hit_at_3", np.mean(hits), model, arm)
            ev.add("document_retrieval", "mrr_at_3", np.mean(mrrs), model, arm)
            ev.add("document_retrieval", "questions_with_retrieval", active, model, arm)


def main() -> None:
    df = load()
    print(f"Runs: {len(df)}")
    summ = summary(df)
    summ.to_csv(SUMMARY_CSV, index=False)
    print(summ.to_string(index=False))

    ev = Evidence()
    q = question_level(df)
    scale_comparison(q, ev)
    for arm in ("naive", "oracle"):
        wins_and_permutation(q, ev, arm)
        length_matched(q, ev, arm)
    arm_comparisons(df, q, ev)
    document_retrieval(ev)
    out = pd.DataFrame(ev.rows)
    out.to_csv(EVIDENCE_CSV, index=False)
    print(f"\nWrote {SUMMARY_CSV.name} and {EVIDENCE_CSV.name} ({len(out)} rows)")


if __name__ == "__main__":
    main()
