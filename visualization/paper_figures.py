"""Figures for the TMLR manuscript, generated from the released results.

Writes into TMLR_Journal_Submissions/:
    accuracy_vs_rag.png        Token F1 by arm for all four models (95% CI over questions)
    latency_vs_rag.png         Median latency by arm, one panel per host machine
    groundedness_vs_rag.png    Groundedness by arm for all four models
    precision_recall.png       Token precision vs recall (Naive and Oracle RAG)
    rag_architecture.png       Benchmark architecture diagram

Usage (from the repository root)::

    python -m visualization.paper_figures
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "TMLR_Journal_Submissions"
sys.path.insert(0, str(ROOT))

MODELS = [  # (label, csv, model_tag, colour, hatch)
    ("LFM2-350M", "benchmark_results.csv", "hf.co/LiquidAI/LFM2-350M-GGUF:Q4_K_M", "#2a78d6", ""),
    ("LFM2-700M", "benchmark_results.csv", "hf.co/LiquidAI/LFM2-700M-GGUF:Q4_K_M", "#eb6834", ""),
    ("SmolLM2-135M", "benchmark_results_smollm2.csv", "hf.co/bartowski/SmolLM2-135M-Instruct-GGUF:Q4_K_M", "#1baf7a", "//"),
    ("SmolLM2-360M", "benchmark_results_smollm2.csv", "hf.co/bartowski/SmolLM2-360M-Instruct-GGUF:Q4_K_M", "#eda100", "//"),
]
ARMS = [("no_rag", "No-RAG"), ("oracle", "Oracle"), ("naive", "Naive"), ("advanced", "Advanced"), ("modular", "Routed\nModular")]
INK, MUTED, GRID = "#1f1f1e", "#5f5e58", "#e4e3dc"

plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def load() -> pd.DataFrame:
    frames = []
    for csv in sorted({m[1] for m in MODELS}):
        cols = ["query", "model_tag", "rag_type", "token_f1", "groundedness", "latency", "route"]
        frames.append(pd.read_csv(ROOT / "results" / "csv" / csv, usecols=cols, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    df["model"] = df["model_tag"].map({m[2]: m[0] for m in MODELS})
    df["token_f1"] = pd.to_numeric(df["token_f1"], errors="coerce").fillna(0.0)
    df["groundedness"] = pd.to_numeric(df["groundedness"], errors="coerce")
    return df.dropna(subset=["model"])


def grouped_bars(ax, values, errors=None, ylabel=""):
    n = len(MODELS)
    width = 0.8 / n
    x = np.arange(len(ARMS))
    for i, (label, _, _, colour, hatch) in enumerate(MODELS):
        pos = x - 0.4 + width * (i + 0.5)
        vals = [values.get((label, arm), np.nan) for arm, _ in ARMS]
        err = None if errors is None else [errors.get((label, arm), 0) for arm, _ in ARMS]
        ax.bar(pos, vals, width * 0.9, color=colour, hatch=hatch, edgecolor="white", linewidth=0.6,
               yerr=err, error_kw={"ecolor": INK, "elinewidth": 0.7, "capsize": 1.5}, label=label)
    ax.set_xticks(x, [a[1] for a in ARMS])
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", length=0)


def fig_accuracy(df):
    q = df.groupby(["model", "rag_type", "query"])["token_f1"].mean()
    means, errs = {}, {}
    for (model, arm), s in q.groupby(level=[0, 1]):
        means[(model, arm)] = s.mean()
        errs[(model, arm)] = stats.t.ppf(0.975, len(s) - 1) * s.std(ddof=1) / np.sqrt(len(s))
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    grouped_bars(ax, means, errs, "Token F1 (mean, 95% CI)")
    ax.set_ylim(0, None)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.savefig(OUT / "accuracy_vs_rag.png")
    plt.close(fig)


def fig_groundedness(df):
    g = df[df["rag_type"] != "no_rag"].copy()
    # Modular groundedness is defined only on runs that retrieved.
    vals = g.groupby(["model", "rag_type"])["groundedness"].mean().to_dict()
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    grouped_bars(ax, vals, None, "Groundedness (mean)")
    ax.set_xlim(0.5, len(ARMS) - 0.5)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.savefig(OUT / "groundedness_vs_rag.png")
    plt.close(fig)


def fig_latency(df):
    med = df.groupby(["model", "rag_type"])["latency"].median().to_dict()
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 2.6), sharey=True)
    for ax, family, title in ((axes[0], "LFM2", "LFM2 host"), (axes[1], "SmolLM2", "SmolLM2 host")):
        models = [m for m in MODELS if m[0].startswith(family)]
        x = np.arange(len(ARMS))
        for i, (label, _, _, colour, hatch) in enumerate(models):
            pos = x - 0.4 + 0.4 * (i + 0.5)
            ax.bar(pos, [med.get((label, a), np.nan) for a, _ in ARMS], 0.36, color=colour, hatch=hatch,
                   edgecolor="white", linewidth=0.6, label=label)
        ax.set_xticks(x, [a[1].replace("\n", " ") for a in ARMS], rotation=45, ha="right")
        ax.set_title(title, fontsize=8, color=INK)
        ax.tick_params(axis="x", length=0)
        ax.legend(fontsize=6, loc="upper left")
    axes[0].set_ylabel("Latency (median, s)")
    fig.savefig(OUT / "latency_vs_rag.png")
    plt.close(fig)


def fig_precision_recall():
    lfm = pd.read_csv(ROOT / "results" / "csv" / "token_precision_recall_summary.csv")
    lfm["model"] = lfm["model_tag"].map({m[2]: m[0] for m in MODELS})
    lfm = lfm.rename(columns={"token_precision_mean": "token_precision", "token_recall_mean": "token_recall"})
    smol = pd.read_csv(ROOT / "results" / "csv" / "smollm2_summary.csv")
    smol["model"] = "SmolLM2-" + smol["model"]
    data = pd.concat([lfm[["model", "rag_type", "token_precision", "token_recall"]],
                      smol[["model", "rag_type", "token_precision", "token_recall"]]])
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6), sharex=True, sharey=True)
    for ax, (arm, name) in zip(axes, (("naive", "Naive RAG"), ("oracle", "Oracle RAG"))):
        for label, _, _, colour, _ in MODELS:
            r = data[(data["model"] == label) & (data["rag_type"] == arm)].iloc[0]
            marker = "o" if label.startswith("LFM2") else "s"
            ax.scatter(r["token_precision"], r["token_recall"], s=46, color=colour, marker=marker,
                       edgecolor="white", linewidth=1.5, zorder=3)
            ax.annotate(label, (r["token_precision"], r["token_recall"]), xytext=(5, 3),
                        textcoords="offset points", fontsize=6.5, color=INK)
        ax.set_title(name, fontsize=8, color=INK)
        ax.set_xlabel("Token precision")
        ax.grid(axis="both")
    axes[0].set_ylabel("Token recall")
    axes[0].set_xlim(0, 0.24)
    axes[0].set_ylim(0.25, 0.9)
    fig.savefig(OUT / "precision_recall.png")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc="#f2f1ec", ec=MUTED, size=7, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
                                fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
            weight=weight, wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=MUTED))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_xlim(-0.02, 1.03)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.00, 0.62, 0.17, 0.30,
        "HotpotQA Distractor\n(dev split)\n\n600 questions\n(first 600; 486 bridge,\n114 comparison)",
        size=6.3)
    box(ax, 0.00, 0.14, 0.17, 0.38,
        "Retrieval corpus\n66,581 passages\n(all 7,405 dev questions)\n\nbge-small-en-v1.5\nembeddings\n\nFAISS IndexFlatIP\n67,527 chunks", size=6.3)

    arms = [
        ("No-RAG", ["question only"]),
        ("Oracle RAG", ["2 gold passages", "(no distractors)"]),
        ("Naive RAG", ["dense top-3"]),
        ("Advanced RAG", ["rewrite (by model)", "dense + TF-IDF fusion (12)", "cross-encoder rerank, top-3"]),
        ("Routed Modular RAG", ["router: yes / no", "2 related queries", "RRF fusion, top-3"]),
    ]
    top, h, gap = 0.95, 0.155, 0.03
    for i, (name, steps) in enumerate(arms):
        y = top - (i + 1) * h - i * gap + gap
        box(ax, 0.23, y, 0.32, h, "", fc="#f7f6f2")
        ax.text(0.24, y + h - 0.025, name, fontsize=7, weight="bold", color=INK, va="top")
        ax.text(0.24, y + h - 0.06, "\n".join(steps), fontsize=6, color=INK, va="top")
        # Every arm receives the question; retrieval arms also search the corpus.
        arrow(ax, 0.17, 0.77, 0.23, y + h / 2 + 0.02)
        if name in ("Naive RAG", "Advanced RAG", "Modular RAG"):
            arrow(ax, 0.17, 0.33, 0.23, y + h / 2 - 0.02)
        arrow(ax, 0.55, y + h / 2, 0.60, 0.53)

    box(ax, 0.60, 0.36, 0.17, 0.34,
        "Local Ollama server\ngreedy decoding,\nseed 42\n\nLFM2-350M\nLFM2-700M\n\nSmolLM2-135M\nSmolLM2-360M", size=6.3)
    arrow(ax, 0.77, 0.53, 0.81, 0.53)
    box(ax, 0.81, 0.28, 0.20, 0.50,
        "Evaluation\n\nToken F1, EM\nToken precision, recall\nCosine similarity\nGroundedness\nDocument IR @3\n(MRR, Hit, Recall,\nnDCG)\nLatency, tokens,\nrefusals\n\nPaired tests over\n600 questions", size=5.8)
    ax.text(0.5, 0.01, "5 arms × 600 questions × 10 repetitions per model (60,000 runs per family)",
            ha="center", fontsize=6.5, color=MUTED)
    fig.savefig(OUT / "rag_architecture.png")
    plt.close(fig)


def main():
    df = load()
    fig_accuracy(df)
    fig_groundedness(df)
    fig_latency(df)
    fig_precision_recall()
    fig_architecture()
    print("Wrote figures to", OUT)


if __name__ == "__main__":
    main()
