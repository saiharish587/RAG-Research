"""Evidence for the claims revised in response to the TMLR reviews.

Recomputes, from the released results, every statistic the revised manuscript
cites in support of its explanations: repeat identity, refusal rates, answer
lengths, the length-matched scale comparison, the Oracle and non-refusal scale
tests, metric agreement, the Advanced-versus-Naive retrieval loss, query-rewrite
diagnostics, latency ordering, generation-length limits, and the degenerate and
long-running runs.

Reads ``results/csv/benchmark_results.csv``, ``results/csv/ir_benchmark_metrics.csv``,
``data/benchmark/eval_set.json`` and ``data/raw/hotpot_dev_distractor_v1.json``.
Writes ``results/csv/revision_evidence.csv`` with one row per statistic.

Conventions match the paper: degenerate runs count as zero for Token F1 and
exact match, and inferential tests compare question-level means across the 600
questions (paired t-test with a t-based 95% CI, plus a Wilcoxon signed-rank test).

Usage (from the repository root)::

    python -m evaluation.revision_analyses
"""

from __future__ import annotations

import json
import math
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
IR_CSV = ROOT / "results" / "csv" / "ir_benchmark_metrics.csv"
EVAL_SET = ROOT / "data" / "benchmark" / "eval_set.json"
RAW_DEV = ROOT / "data" / "raw" / "hotpot_dev_distractor_v1.json"
EVIDENCE_CSV = ROOT / "results" / "csv" / "revision_evidence.csv"

MODELS = {
    "hf.co/LiquidAI/LFM2-350M-GGUF:Q4_K_M": "350M",
    "hf.co/LiquidAI/LFM2-700M-GGUF:Q4_K_M": "700M",
}
IR_MODELS = {"LFM2 350M": "350M", "LFM2 700M": "700M"}
ARMS = ["no_rag", "oracle", "naive", "advanced", "modular"]

COLUMNS = [
    "query", "response", "ground_truth", "model_tag", "rag_type", "run_id",
    "status", "is_refusal", "generation_tokens", "total_generation_tokens",
    "prompt_tokens", "latency", "token_f1", "exact_match", "cosine",
    "groundedness", "rewritten_query", "n_llm_calls", "route", "retrieval_hit_rate",
    "stage_rewrite_s", "stage_retrieve_s", "stage_rerank_s", "stage_generate_s",
]
NUMERIC = [
    "generation_tokens", "total_generation_tokens", "prompt_tokens", "latency",
    "token_f1", "exact_match", "cosine", "groundedness", "is_refusal",
    "retrieval_hit_rate", "stage_rewrite_s", "stage_retrieve_s", "stage_rerank_s",
    "stage_generate_s",
]

N_BOOT = 2000
N_PERM = 100_000
BOOT_SEED = 42
N_LENGTH_BINS = 10
LONG_RUN_S = 60.0
CAP_TOKENS = 512


class Evidence:
    """Collects one row per statistic."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, analysis: str, statistic: str, value, model: str = "", arm: str = "") -> None:
        if isinstance(value, (np.floating, float)):
            # Six significant figures, so small p-values are not stored as 0.0.
            value = float(f"{float(value):.6g}")
        elif isinstance(value, (np.integer, np.bool_)):
            value = value.item()
        self.rows.append(
            {"analysis": analysis, "model": model, "arm": arm, "statistic": statistic, "value": value}
        )

    def add_test(self, analysis: str, result: dict, model: str = "", arm: str = "") -> None:
        for key, value in result.items():
            self.add(analysis, key, value, model=model, arm=arm)


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def load_runs() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV, usecols=COLUMNS, low_memory=False)
    df["model"] = df["model_tag"].map(MODELS)
    for col in ("response", "ground_truth", "rewritten_query"):
        df[col] = df[col].fillna("")
    df["is_refusal"] = df["is_refusal"].astype(str).str.lower().isin(["true", "1", "1.0"]).astype(float)
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Degenerate runs have no stored F1 or EM; they count as zero, as in the paper's tables.
    df["token_f1"] = df["token_f1"].fillna(0.0)
    df["exact_match"] = df["exact_match"].fillna(0.0)
    return df


def question_level(df: pd.DataFrame) -> pd.DataFrame:
    """One row per model, arm and question: numeric means plus run-1 text."""
    keys = ["model", "rag_type", "query"]
    means = df.groupby(keys)[NUMERIC].mean()
    text = (
        df.sort_values("run_id")
        .groupby(keys)[["response", "ground_truth", "rewritten_query", "status"]]
        .first()
    )
    q = means.join(text).reset_index()

    precision, recall, words, overlap, gold = [], [], [], [], []
    for response, truth in zip(q["response"], q["ground_truth"]):
        pred, ref = tokenise(response), tokenise(truth)
        common = sum((Counter(pred) & Counter(ref)).values()) if pred and ref else 0
        words.append(len(pred))
        gold.append(len(ref))
        overlap.append(common)
        precision.append(common / len(pred) if common else 0.0)
        recall.append(common / len(ref) if common else 0.0)
    q["token_precision"] = precision
    q["token_recall"] = recall
    q["response_words"] = words
    q["gold_words"] = gold
    q["overlap_words"] = overlap
    return q


def series(q: pd.DataFrame, model: str, arm: str, col: str) -> pd.Series:
    return q[(q["model"] == model) & (q["rag_type"] == arm)].set_index("query")[col]


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------

def paired(x: pd.Series, y: pd.Series) -> dict:
    """Paired comparison of x minus y over the questions both share."""
    d = (x - y).dropna()
    x, y = x.loc[d.index], y.loc[d.index]
    n = len(d)
    dbar = d.mean()
    half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / math.sqrt(n)
    t_stat, p_t = stats.ttest_rel(x, y)
    try:
        p_w = stats.wilcoxon(d).pvalue
    except ValueError:
        p_w = float("nan")
    return {
        "n_questions": n,
        "mean_a": x.mean(),
        "mean_b": y.mean(),
        "mean_diff_a_minus_b": dbar,
        "ci95_low": dbar - half,
        "ci95_high": dbar + half,
        "t": t_stat,
        "p_t": p_t,
        "p_wilcoxon": p_w,
    }


def length_bins(lengths: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Right-closed bins matching pandas.cut(include_lowest=True)."""
    return np.clip(np.searchsorted(edges, lengths, side="left") - 1, 0, len(edges) - 2)


def binned_gap(f_a, f_b, bin_a, bin_b, n_bins: int) -> float:
    """Within-bin mean difference a minus b, weighted by bin size."""
    weighted, total = 0.0, 0
    for k in range(n_bins):
        in_a, in_b = bin_a == k, bin_b == k
        if in_a.any() and in_b.any():
            w = int(in_a.sum() + in_b.sum())
            weighted += w * (f_a[in_a].mean() - f_b[in_b].mean())
            total += w
    return weighted / total if total else float("nan")


def ols_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(beta[0]), float(beta[1])


# ----------------------------------------------------------------------
# Analyses
# ----------------------------------------------------------------------

def repeat_identity(df: pd.DataFrame, ev: Evidence) -> bool:
    distinct = df.groupby(["model", "rag_type", "query"])["response"].nunique()
    ev.add("repeat_identity", "model_arm_question_groups", len(distinct))
    ev.add("repeat_identity", "groups_with_identical_text_across_runs", int((distinct == 1).sum()))
    ev.add("repeat_identity", "distinct_generations", int(distinct.sum()))
    return bool((distinct == 1).all())


def refusal_rates(df: pd.DataFrame, ev: Evidence) -> None:
    for (model, arm), sub in df.groupby(["model", "rag_type"]):
        ev.add("refusal_rate", "runs", len(sub), model, arm)
        ev.add("refusal_rate", "refusals", int(sub["is_refusal"].sum()), model, arm)
        ev.add("refusal_rate", "rate", sub["is_refusal"].mean(), model, arm)


def answer_lengths(q: pd.DataFrame, ev: Evidence) -> None:
    gold = q.drop_duplicates("query")["gold_words"]
    ev.add("gold_answer_length", "mean_words", gold.mean())
    ev.add("gold_answer_length", "median_words", gold.median())
    ev.add("gold_answer_length", "share_at_most_3_words", (gold <= 3).mean())
    for arm in ("naive", "oracle"):
        for model in ("350M", "700M"):
            sub = q[(q["model"] == model) & (q["rag_type"] == arm)]
            ok = sub[sub["status"] == "ok"]
            ev.add("response_length", "mean_generation_tokens", sub["generation_tokens"].mean(), model, arm)
            ev.add("response_length", "median_generation_tokens", sub["generation_tokens"].median(), model, arm)
            ev.add("response_length", "mean_generation_tokens_ok_only", ok["generation_tokens"].mean(), model, arm)
            ev.add("response_length", "mean_response_words", sub["response_words"].mean(), model, arm)
            ev.add("response_length", "mean_overlap_words", sub["overlap_words"].mean(), model, arm)
            ev.add("response_length", "mean_token_precision", sub["token_precision"].mean(), model, arm)
            ev.add("response_length", "mean_token_recall", sub["token_recall"].mean(), model, arm)
        a = q[(q["model"] == "350M") & (q["rag_type"] == arm)]
        b = q[(q["model"] == "700M") & (q["rag_type"] == arm)]
        ev.add("response_length", "ratio_700_to_350_response_words", b["response_words"].mean() / a["response_words"].mean(), arm=arm)
        ev.add("response_length", "ratio_700_to_350_overlap_words", b["overlap_words"].mean() / a["overlap_words"].mean(), arm=arm)


def wins(q: pd.DataFrame, ev: Evidence, arm: str = "naive") -> None:
    f_a, f_b = series(q, "350M", arm, "token_f1"), series(q, "700M", arm, "token_f1")
    l_a, l_b = series(q, "350M", arm, "generation_tokens"), series(q, "700M", arm, "generation_tokens")
    a_wins, b_wins = f_a > f_b, f_b > f_a
    ev.add("wins", "questions_350M_higher", int(a_wins.sum()), arm=arm)
    ev.add("wins", "questions_700M_higher", int(b_wins.sum()), arm=arm)
    ev.add("wins", "questions_tied", int((~a_wins & ~b_wins).sum()), arm=arm)
    ev.add("wins", "sign_test_p", stats.binomtest(int(a_wins.sum()), int(a_wins.sum() + b_wins.sum()), 0.5).pvalue, arm=arm)
    ev.add("wins", "median_question_difference", (f_a - f_b).median(), arm=arm)

    # Sign-flip permutation test on the question-level differences.
    d = (f_a - f_b).to_numpy()
    observed = abs(d.mean())
    rng = np.random.default_rng(BOOT_SEED)
    exceed = 0
    for start in range(0, N_PERM, 10_000):
        size = min(10_000, N_PERM - start)
        signs = rng.choice([-1.0, 1.0], size=(size, len(d)))
        exceed += int((np.abs((signs * d).mean(axis=1)) >= observed).sum())
    ev.add("wins", "permutation_draws", N_PERM, arm=arm)
    ev.add("wins", "permutation_draws_at_least_observed", exceed, arm=arm)
    ev.add("wins", "permutation_p", (exceed + 1) / (N_PERM + 1), arm=arm)

    # Lexical proxies for a correct answer that pull in opposite directions.
    em_a, em_b = series(q, "350M", arm, "exact_match"), series(q, "700M", arm, "exact_match")
    ev.add("wins", "questions_exact_350M_only", int(((em_a > 0) & (em_b == 0)).sum()), arm=arm)
    ev.add("wins", "questions_exact_700M_only", int(((em_b > 0) & (em_a == 0)).sum()), arm=arm)
    rec_a, rec_b = series(q, "350M", arm, "token_recall"), series(q, "700M", arm, "token_recall")
    ev.add("wins", "share_full_recall_350M", (rec_a >= 1.0).mean(), arm=arm)
    ev.add("wins", "share_full_recall_700M", (rec_b >= 1.0).mean(), arm=arm)
    ev.add("wins", "mean_margin_on_350M_wins", (f_a - f_b)[a_wins].mean(), arm=arm)
    ev.add("wins", "mean_margin_on_700M_wins", (f_b - f_a)[b_wins].mean(), arm=arm)
    ev.add("wins", "tokens_350M_on_350M_wins", l_a[a_wins].mean(), arm=arm)
    ev.add("wins", "tokens_700M_on_350M_wins", l_b[a_wins].mean(), arm=arm)
    ev.add("wins", "tokens_350M_on_700M_wins", l_a[b_wins].mean(), arm=arm)
    ev.add("wins", "tokens_700M_on_700M_wins", l_b[b_wins].mean(), arm=arm)


def length_matched_gap(q: pd.DataFrame, ev: Evidence, arm: str) -> None:
    f_a, f_b = series(q, "350M", arm, "token_f1"), series(q, "700M", arm, "token_f1")
    l_a, l_b = series(q, "350M", arm, "generation_tokens"), series(q, "700M", arm, "generation_tokens")
    common = f_a.index.intersection(f_b.index)
    f_a, f_b = f_a.loc[common].to_numpy(), f_b.loc[common].to_numpy()
    l_a, l_b = l_a.loc[common].to_numpy(), l_b.loc[common].to_numpy()
    n = len(common)

    edges = np.unique(np.quantile(np.concatenate([l_a, l_b]), np.linspace(0, 1, N_LENGTH_BINS + 1)))
    n_bins = len(edges) - 1
    bin_a, bin_b = length_bins(l_a, edges), length_bins(l_b, edges)
    log_ratio = np.log(l_a) - np.log(l_b)
    diff = f_a - f_b

    raw = diff.mean()
    matched = binned_gap(f_a, f_b, bin_a, bin_b, n_bins)
    intercept, slope = ols_intercept(log_ratio, diff)

    rng = np.random.default_rng(BOOT_SEED)
    boot_raw, boot_matched, boot_intercept = [], [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        boot_raw.append(diff[idx].mean())
        boot_matched.append(binned_gap(f_a[idx], f_b[idx], bin_a[idx], bin_b[idx], n_bins))
        boot_intercept.append(ols_intercept(log_ratio[idx], diff[idx])[0])

    def ci(values):
        return np.nanpercentile(values, 2.5), np.nanpercentile(values, 97.5)

    ev.add("length_matched_gap", "n_questions", n, arm=arm)
    ev.add("length_matched_gap", "length_bins", n_bins, arm=arm)
    ev.add("length_matched_gap", "raw_gap_350M_minus_700M", raw, arm=arm)
    lo, hi = ci(boot_raw)
    ev.add("length_matched_gap", "raw_gap_boot_ci95_low", lo, arm=arm)
    ev.add("length_matched_gap", "raw_gap_boot_ci95_high", hi, arm=arm)
    ev.add("length_matched_gap", "binned_gap_350M_minus_700M", matched, arm=arm)
    lo, hi = ci(boot_matched)
    ev.add("length_matched_gap", "binned_gap_boot_ci95_low", lo, arm=arm)
    ev.add("length_matched_gap", "binned_gap_boot_ci95_high", hi, arm=arm)
    ev.add("length_matched_gap", "regression_gap_at_equal_length", intercept, arm=arm)
    lo, hi = ci(boot_intercept)
    ev.add("length_matched_gap", "regression_boot_ci95_low", lo, arm=arm)
    ev.add("length_matched_gap", "regression_boot_ci95_high", hi, arm=arm)
    ev.add("length_matched_gap", "regression_slope_on_log_length_ratio", slope, arm=arm)


def scale_tests(q: pd.DataFrame, ev: Evidence) -> None:
    # Oracle: gold passages only, so no distractors in the prompt.
    ev.add_test(
        "scale_test_oracle_f1",
        paired(series(q, "350M", "oracle", "token_f1"), series(q, "700M", "oracle", "token_f1")),
        arm="oracle",
    )
    # Naive, restricted to questions neither model refused.
    ref_a = series(q, "350M", "naive", "is_refusal")
    ref_b = series(q, "700M", "naive", "is_refusal")
    keep = ref_a.index[(ref_a == 0) & (ref_b.reindex(ref_a.index) == 0)]
    ev.add("scale_test_naive_f1_no_refusal", "questions_refused_by_350M", int((ref_a > 0).sum()), arm="naive")
    ev.add("scale_test_naive_f1_no_refusal", "questions_refused_by_700M", int((ref_b > 0).sum()), arm="naive")
    ev.add_test(
        "scale_test_naive_f1_no_refusal",
        paired(series(q, "350M", "naive", "token_f1").loc[keep], series(q, "700M", "naive", "token_f1").loc[keep]),
        arm="naive",
    )
    # Gold context versus vector retrieval, per model.
    for model in ("350M", "700M"):
        ev.add_test(
            "oracle_minus_naive_f1",
            paired(series(q, model, "oracle", "token_f1"), series(q, model, "naive", "token_f1")),
            model=model,
        )
        ev.add_test(
            "advanced_minus_naive_f1",
            paired(series(q, model, "advanced", "token_f1"), series(q, model, "naive", "token_f1")),
            model=model,
        )
    # 350M minus 700M Token F1 in the arms not covered above.
    for arm in ("advanced", "modular"):
        ev.add_test(
            "scale_test_f1",
            paired(series(q, "350M", arm, "token_f1"), series(q, "700M", arm, "token_f1")),
            arm=arm,
        )
    for arm in ("naive", "oracle"):
        for model in ("350M", "700M"):
            ev.add("full_recall_share", "share_questions_recall_1",
                   (series(q, model, arm, "token_recall") >= 1.0).mean(), model, arm)


def router_need(df: pd.DataFrame, q: pd.DataFrame, ev: Evidence) -> None:
    """On questions the Modular router sent away from retrieval, would retrieval have helped?"""
    route = df[df["rag_type"] == "modular"].groupby(["model", "query"])["route"].first()
    for model in ("350M", "700M"):
        declined = route.loc[model][route.loc[model] == "no_rag"].index
        no_rag = series(q, model, "no_rag", "token_f1").loc[declined]
        naive = series(q, model, "naive", "token_f1").loc[declined]
        ev.add("router_need", "questions_routed_away_from_retrieval", len(declined), model, "modular")
        ev.add("router_need", "no_rag_f1_on_those", no_rag.mean(), model, "modular")
        ev.add("router_need", "naive_f1_on_those", naive.mean(), model, "modular")
        ev.add("router_need", "questions_retrieval_helped", int((naive > no_rag).sum()), model, "modular")
        ev.add("router_need", "questions_retrieval_hurt", int((no_rag > naive).sum()), model, "modular")


def metric_agreement(q: pd.DataFrame, ev: Evidence, arm: str = "naive") -> None:
    for metric in ("token_f1", "exact_match", "token_precision", "token_recall", "cosine", "groundedness"):
        ev.add_test(
            f"metric_agreement_{metric}",
            paired(series(q, "350M", arm, metric), series(q, "700M", arm, metric)),
            arm=arm,
        )


def retrieval_loss(ev: Evidence) -> None:
    ir = pd.read_csv(IR_CSV)
    ir["model"] = ir["model"].map(IR_MODELS)
    for model in ("350M", "700M"):
        sub = ir[ir["model"] == model]
        naive = sub[sub["rag_type"] == "naive"].set_index("question_id")
        adv = sub[sub["rag_type"] == "advanced"].set_index("question_id")
        joined = naive.join(adv, lsuffix="_naive", rsuffix="_adv", how="inner")
        lost = int(((joined["doc_hit_at_3_naive"] == 1) & (joined["doc_hit_at_3_adv"] == 0)).sum())
        gained = int(((joined["doc_hit_at_3_naive"] == 0) & (joined["doc_hit_at_3_adv"] == 1)).sum())
        ev.add("advanced_vs_naive_retrieval", "questions", len(joined), model)
        ev.add("advanced_vs_naive_retrieval", "gold_doc_lost_by_advanced", lost, model)
        ev.add("advanced_vs_naive_retrieval", "gold_doc_gained_by_advanced", gained, model)
        if lost + gained:
            ev.add("advanced_vs_naive_retrieval", "mcnemar_exact_p",
                   stats.binomtest(lost, lost + gained, 0.5).pvalue, model)
        mrr = paired(joined["doc_mrr_at_3_naive"], joined["doc_mrr_at_3_adv"])
        for key in ("mean_a", "mean_b", "mean_diff_a_minus_b", "ci95_low", "ci95_high", "p_t"):
            ev.add("advanced_vs_naive_retrieval", f"mrr_naive_vs_advanced_{key}", mrr[key], model)
        ev.add("gold_share_of_top3", "mean_gold_docs_in_top3", (naive["doc_recall_at_3"] * 2).mean(), model, "naive")

        active = sub[(sub["rag_type"] == "modular") & (sub["n_retrieved"] > 0)]
        ev.add("modular_active_ir", "questions", len(active), model, "modular")
        for col in ("doc_mrr_at_3", "doc_hit_at_1", "doc_hit_at_3", "doc_recall_at_3", "doc_ndcg_at_3"):
            ev.add("modular_active_ir", col, active[col].mean(), model, "modular")

    # The retrieval components are identical across scales; only the rewritten
    # query differs, because each model writes its own rewrite.
    adv = ir[ir["rag_type"] == "advanced"]
    a = adv[adv["model"] == "350M"].set_index("question_id")
    b = adv[adv["model"] == "700M"].set_index("question_id")
    joined = a.join(b, lsuffix="_350M", rsuffix="_700M", how="inner")
    only_700 = int(((joined["doc_hit_at_3_700M"] == 1) & (joined["doc_hit_at_3_350M"] == 0)).sum())
    only_350 = int(((joined["doc_hit_at_3_350M"] == 1) & (joined["doc_hit_at_3_700M"] == 0)).sum())
    ev.add("advanced_rewrite_effect", "gold_doc_found_with_700M_rewrite_only", only_700, arm="advanced")
    ev.add("advanced_rewrite_effect", "gold_doc_found_with_350M_rewrite_only", only_350, arm="advanced")
    if only_700 + only_350:
        ev.add("advanced_rewrite_effect", "mcnemar_exact_p",
               stats.binomtest(only_700, only_700 + only_350, 0.5).pvalue, arm="advanced")
    ev.add_test("advanced_rewrite_effect_mrr_700M_minus_350M",
                paired(joined["doc_mrr_at_3_700M"], joined["doc_mrr_at_3_350M"]), arm="advanced")


def rewrite_diagnostics(q: pd.DataFrame, ev: Evidence) -> None:
    adv = q[q["rag_type"] == "advanced"]
    for model, sub in adv.groupby("model"):
        rewritten = sub["rewritten_query"]
        original_words = sub["query"].str.split().str.len()
        rewritten_words = rewritten.str.split().str.len()
        ev.add("rewrite_diagnostics", "questions", len(sub), model, "advanced")
        ev.add("rewrite_diagnostics", "mean_words_original", original_words.mean(), model, "advanced")
        ev.add("rewrite_diagnostics", "mean_words_rewritten", rewritten_words.mean(), model, "advanced")
        ev.add("rewrite_diagnostics", "median_words_rewritten", rewritten_words.median(), model, "advanced")
        ev.add("rewrite_diagnostics", "share_with_line_break", rewritten.str.contains("\n").mean(), model, "advanced")
        ev.add("rewrite_diagnostics", "share_unchanged",
               (rewritten.str.strip() == sub["query"].str.strip()).mean(), model, "advanced")


def latency_and_limits(df: pd.DataFrame, ev: Evidence) -> None:
    for (model, arm), sub in df.groupby(["model", "rag_type"]):
        ev.add("latency", "mean_s", sub["latency"].mean(), model, arm)
        ev.add("latency", "median_s", sub["latency"].median(), model, arm)
        for stage in ("rewrite", "retrieve", "rerank", "generate"):
            values = sub[f"stage_{stage}_s"].dropna()
            if len(values):
                ev.add("latency_stage", f"mean_{stage}_s", values.mean(), model, arm)

        hit = sub["retrieval_hit_rate"].dropna()
        ev.add("table1_hit_rate", "defined_runs", len(hit), model, arm)
        if len(hit):
            ev.add("table1_hit_rate", "mean", hit.mean(), model, arm)
        if arm == "modular":
            routed = sub[sub["route"] == "rag"]
            ev.add("table1_modular_active", "runs_routed_to_retrieval", len(routed), model, arm)
            ev.add("table1_modular_active", "hit_rate_defined_runs", int(routed["retrieval_hit_rate"].notna().sum()), model, arm)
            ev.add("table1_modular_active", "hit_rate_mean", routed["retrieval_hit_rate"].mean(), model, arm)
            ev.add("table1_modular_active", "groundedness_defined_runs", int(routed["groundedness"].notna().sum()), model, arm)
            ev.add("table1_modular_active", "groundedness_mean", routed["groundedness"].mean(), model, arm)
        ok = sub[sub["status"] == "ok"]
        ev.add("generation_limit", "max_tokens_in_one_call_ok_runs", ok["generation_tokens"].max(), model, arm)
        ev.add("generation_limit", f"calls_over_{CAP_TOKENS}_tokens_ok_runs",
               int((ok["generation_tokens"] > CAP_TOKENS).sum()), model, arm)
        ev.add("generation_limit", "max_prompt_tokens", sub["prompt_tokens"].max(), model, arm)

        degenerate = sub[sub["status"] == "degenerate"]
        if len(degenerate):
            ev.add("degenerate_runs", "runs", len(degenerate), model, arm)
            ev.add("degenerate_runs", "questions", degenerate["query"].nunique(), model, arm)
            ev.add("degenerate_runs", "generation_tokens_per_run_min", degenerate["total_generation_tokens"].min(), model, arm)
            ev.add("degenerate_runs", "generation_tokens_per_run_max", degenerate["total_generation_tokens"].max(), model, arm)
            ev.add("degenerate_runs", "generation_tokens_all_runs", degenerate["total_generation_tokens"].sum(), model, arm)
            ev.add("degenerate_runs", "llm_calls_per_run_max", degenerate["n_llm_calls"].max(), model, arm)
            ev.add("degenerate_runs", "mean_latency_s", degenerate["latency"].mean(), model, arm)

        long = sub[sub["latency"] > LONG_RUN_S]
        if len(long):
            rest = sub[sub["latency"] <= LONG_RUN_S]
            ev.add("long_runs", "runs_over_60s", len(long), model, arm)
            ev.add("long_runs", "questions", long["query"].nunique(), model, arm)
            ev.add("long_runs", "statuses", "|".join(sorted(long["status"].astype(str).unique())), model, arm)
            ev.add("long_runs", "mean_latency_s", long["latency"].mean(), model, arm)
            ev.add("long_runs", "max_total_generation_tokens", long["total_generation_tokens"].max(), model, arm)
            ev.add("long_runs", "mean_latency_other_runs_s", rest["latency"].mean(), model, arm)


def oracle_context(ev: Evidence) -> None:
    eval_set = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    sizes = [len(item["gold_context"]) if isinstance(item["gold_context"], list) else 1 for item in eval_set]
    all_sizes = [len(item["all_context"]) if isinstance(item["all_context"], list) else 1 for item in eval_set]
    ev.add("oracle_context", "questions", len(eval_set))
    ev.add("oracle_context", "gold_passages_min", min(sizes))
    ev.add("oracle_context", "gold_passages_max", max(sizes))
    ev.add("oracle_context", "distractor_setting_passages_min", min(all_sizes))
    ev.add("oracle_context", "distractor_setting_passages_max", max(all_sizes))


def corpus(ev: Evidence) -> None:
    """Corpus size, and titles lost to filename collisions in ingest_hotpotqa.py."""
    raw = json.loads(RAW_DEV.read_text(encoding="utf-8"))
    eval_set = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    order, seen = [], set()
    for item in raw:
        for title, _ in item["context"]:
            if title not in seen:
                seen.add(title)
                order.append(title)
    # Same sanitiser as ingest_hotpotqa.py. File names are case-insensitive on the
    # Windows host, so a later title overwrites any earlier file whose sanitised
    # name differs only in case.
    last_writer = {}
    for title in order:
        last_writer["".join(c if c.isalnum() else "_" for c in title).lower()] = title
    lost = set(order) - set(last_writer.values())
    gold = {sf[0] for item in eval_set for sf in item["supporting_facts"]}
    ev.add("corpus", "questions_in_full_dev_split", len(raw))
    ev.add("corpus", "unique_titles", len(order))
    ev.add("corpus", "files_after_sanitising", len(last_writer))
    ev.add("corpus", "titles_lost_to_collisions", len(lost))
    ev.add("corpus", "lost_titles_that_are_gold_in_eval_set", len(lost & gold))


def main() -> None:
    ev = Evidence()
    df = load_runs()
    print(f"Runs: {len(df)}")

    identical = repeat_identity(df, ev)
    print(f"Repeats identical in every model/arm/question group: {identical}")

    q = question_level(df)
    refusal_rates(df, ev)
    answer_lengths(q, ev)
    wins(q, ev, "naive")
    for arm in ("naive", "oracle"):
        length_matched_gap(q, ev, arm)
    scale_tests(q, ev)
    metric_agreement(q, ev, "naive")
    router_need(df, q, ev)
    retrieval_loss(ev)
    rewrite_diagnostics(q, ev)
    latency_and_limits(df, ev)
    oracle_context(ev)
    corpus(ev)

    out = pd.DataFrame(ev.rows)
    out.to_csv(EVIDENCE_CSV, index=False)
    print(f"Wrote {EVIDENCE_CSV.relative_to(ROOT)} ({len(out)} rows)")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
