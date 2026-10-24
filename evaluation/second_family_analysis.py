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
    the refusal-free scale test, Advanced vs Naive, router behaviour, rewrite
    diagnostics, per-stage latency, and document-level retrieval.

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

from evaluation.compute_ir_metrics import compute_document_level_ir_metrics  # noqa: E402
from evaluation.metrics import tokenise  # noqa: E402
from evaluation.retrieval_ablation import doc_metrics  # noqa: E402
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
    # Wilcoxon statistic (zero differences excluded) and standardised paired effect.
    ev.add("wins", "wilcoxon_W", stats.wilcoxon(d[d != 0]).statistic, arm=arm)
    ev.add("wins", "d_z_large_minus_small", -d.mean() / d.std(ddof=1), arm=arm)
    ev.add("wins", "median_question_difference", float(np.median(d)), arm=arm)
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
        for metric in ("token_recall", "exact_match"):
            ev.add_test(f"advanced_minus_naive_{metric}",
                        paired(series(q, model, "advanced", metric), series(q, model, "naive", metric)), model)
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
    """Document-level IR metrics at rank 3 from run-1 logs.

    Gold documents are matched by each chunk's source title (``chunks.npy``);
    ``document_retrieval_colon_rule`` repeats the metrics with the title taken
    as the text before the first colon, the rule used for the LFM2 IR table.
    """
    chunks = np.load(CHUNKS, allow_pickle=True)
    source = {c["text"]: c["metadata"]["source"] for c in chunks}
    eval_set = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    names = ("mrr_at_3", "hit_at_1", "hit_at_3", "recall_at_3", "ndcg_at_3")
    per_q = {}
    for tag, model in MODELS.items():
        safe = tag.replace("/", "_").replace(":", "_")
        for arm in ("oracle", "naive", "advanced", "modular"):
            rows = {"source": [], "colon": []}
            active = []
            for i, item in enumerate(eval_set, 1):
                gold = {sf[0] for sf in item["supporting_facts"]}
                log = ROOT / "results" / "raw" / f"{safe}_{arm}_q{i}_run1.json"
                ctx = json.loads(log.read_text(encoding="utf-8"))["pipeline_result"]["retrieved_context"]
                active.append(bool(ctx))
                m = doc_metrics([source.get(c, f"_{j}") for j, c in enumerate(ctx)], gold)
                m["hit_at_1"] = float(bool(ctx) and source.get(ctx[0]) in gold)
                rows["source"].append([m[n] for n in names])
                rows["colon"].append(compute_document_level_ir_metrics(ctx, {g.strip().lower() for g in gold}, k=3))
            active = np.array(active)
            for rule, values in rows.items():
                m = np.array(values, dtype=float)
                analysis = "document_retrieval" if rule == "source" else "document_retrieval_colon_rule"
                for k, name in enumerate(names):
                    ev.add(analysis, name, m[:, k].mean(), model, arm)
                if arm == "modular" and active.any():
                    for k, name in enumerate(names):
                        ev.add(analysis, f"{name}_active_only", m[active, k].mean(), model, arm)
                per_q[(rule, model, arm)] = pd.DataFrame(m, columns=names)
            ev.add("document_retrieval", "questions_with_retrieval", int(active.sum()), model, arm)

    # Advanced versus Naive retrieval, and the effect of each model's rewrite,
    # under both title-matching rules.
    for rule, suffix in (("source", ""), ("colon", "_colon_rule")):
        for model in (SMALL, LARGE):
            naive, adv = per_q[(rule, model, "naive")], per_q[(rule, model, "advanced")]
            lost = int(((naive["hit_at_3"] == 1) & (adv["hit_at_3"] == 0)).sum())
            gained = int(((naive["hit_at_3"] == 0) & (adv["hit_at_3"] == 1)).sum())
            name = f"advanced_vs_naive_retrieval{suffix}"
            ev.add(name, "gold_doc_lost_by_advanced", lost, model)
            ev.add(name, "gold_doc_gained_by_advanced", gained, model)
            ev.add(name, "mcnemar_exact_p", stats.binomtest(lost, lost + gained, 0.5).pvalue if lost + gained else 1.0, model)
            ev.add_test(f"advanced_vs_naive_retrieval_mrr_naive_minus_advanced{suffix}",
                        paired(naive["mrr_at_3"], adv["mrr_at_3"]), model)
        a, b = per_q[(rule, SMALL, "advanced")], per_q[(rule, LARGE, "advanced")]
        only_large = int(((b["hit_at_3"] == 1) & (a["hit_at_3"] == 0)).sum())
        only_small = int(((a["hit_at_3"] == 1) & (b["hit_at_3"] == 0)).sum())
        name = f"advanced_rewrite_effect{suffix}"
        ev.add(name, "gold_doc_found_with_large_rewrite_only", only_large, arm="advanced")
        ev.add(name, "gold_doc_found_with_small_rewrite_only", only_small, arm="advanced")
        if only_large + only_small:
            ev.add(name, "mcnemar_exact_p", stats.binomtest(only_large, only_large + only_small, 0.5).pvalue, arm="advanced")
        ev.add_test(f"advanced_rewrite_effect_mrr_large_minus_small{suffix}", paired(b["mrr_at_3"], a["mrr_at_3"]), arm="advanced")


def rewrite_diagnostics(df: pd.DataFrame, ev: Evidence) -> None:
    """Length and form of the rewritten queries (all runs, and run 1 as used in the ablation)."""
    adv = df[df["rag_type"] == "advanced"]
    for model, sub in adv.groupby("model"):
        for label, part in (("all_runs", sub), ("run1", sub[sub["run_id"] == 1])):
            rw = part["rewritten_query"].fillna("").astype(str)
            words = rw.str.split().str.len()
            ev.add("rewrite_diagnostics", f"mean_words_original_{label}", part["query"].str.split().str.len().mean(), model, "advanced")
            ev.add("rewrite_diagnostics", f"mean_words_rewritten_{label}", words.mean(), model, "advanced")
            ev.add("rewrite_diagnostics", f"median_words_rewritten_{label}", words.median(), model, "advanced")
            ev.add("rewrite_diagnostics", f"share_with_line_break_{label}", rw.str.contains("\n", regex=False).mean(), model, "advanced")
            ev.add("rewrite_diagnostics", f"share_unchanged_{label}", (rw.str.strip() == part["query"].str.strip()).mean(), model, "advanced")
        same = sub.groupby("query")["rewritten_query"].nunique()
        ev.add("rewrite_diagnostics", "questions_with_identical_rewrite_across_runs", int((same == 1).sum()), model, "advanced")


def stage_latency(df: pd.DataFrame, ev: Evidence) -> None:
    for (model, arm), sub in df.groupby(["model", "rag_type"]):
        ev.add("latency", "mean_s", sub["latency"].mean(), model, arm)
        ev.add("latency", "median_s", sub["latency"].median(), model, arm)
        for stage in ("rewrite", "retrieve", "rerank", "generate", "route", "subquery", "fuse"):
            values = pd.to_numeric(sub[f"stage_{stage}_s"], errors="coerce").dropna()
            if len(values):
                ev.add("latency_stage", f"mean_{stage}_s", values.mean(), model, arm)
                ev.add("latency_stage", f"median_{stage}_s", values.median(), model, arm)


def refusal_free_scale_test(q: pd.DataFrame, ev: Evidence) -> None:
    """Naive RAG scale comparison restricted to questions neither model ever refused."""
    ra, rb = series(q, SMALL, "naive", "is_refusal"), series(q, LARGE, "naive", "is_refusal")
    keep = ra.index[(ra == 0) & (rb.reindex(ra.index) == 0)]
    ev.add("scale_test_naive_f1_no_refusal", "questions_refused_by_small", int((ra > 0).sum()), arm="naive")
    ev.add("scale_test_naive_f1_no_refusal", "questions_refused_by_large", int((rb > 0).sum()), arm="naive")
    ev.add_test("scale_test_naive_f1_no_refusal",
                paired(series(q, SMALL, "naive", "token_f1").loc[keep], series(q, LARGE, "naive", "token_f1").loc[keep]),
                arm="naive")


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
    refusal_free_scale_test(q, ev)
    rewrite_diagnostics(df, ev)
    stage_latency(df, ev)
    document_retrieval(ev)
    out = pd.DataFrame(ev.rows)
    out.to_csv(EVIDENCE_CSV, index=False)
    print(f"\nWrote {SUMMARY_CSV.name} and {EVIDENCE_CSV.name} ({len(out)} rows)")


if __name__ == "__main__":
    main()
