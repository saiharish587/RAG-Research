# Legacy — v1 experimental record (archived 2026-08-08)

This directory preserves the **first iteration** of the benchmark exactly as it
was published, so it stays editable and citable while `v2` is developed.
Nothing here is deleted or rewritten. Git history additionally retains every
prior state of these files.

## Contents

| Path | What it is |
|---|---|
| `paper_v1.tex` | Main paper source, 2-author version (was `paper.tex`) |
| `paper_v1.pdf` | Compiled PDF of the above |
| `paper_v1_5author.tex` | 5-author variant with cleaner prose (was `RAG_Research/paper.tex`) |
| `results_v1/csv/benchmark_results.csv` | 4,680-row main result set |
| `results_v1/csv/pruned_benchmark_results.csv` | 468-row secondary run, never reported in the paper |
| `results_v1/graphs/*.png` | All five figures as published |
| `results_v1/benchmark_summary.md` | Generated summary table |

## Why v1 is superseded

An audit on 2026-08-08 confirmed that every number in the v1 results table is
**arithmetically correct** — all 36 cells reproduce exactly from
`benchmark_results.csv`, and no fabricated, mocked or simulated data exists
anywhere in the pipeline. The defects are in *measurement and inference*, one
layer above the arithmetic:

1. **870 empty responses were scored as `answer_accuracy = 0.0`**, making them
   indistinguishable from confidently wrong answers. This alone produces the
   "RAG Loop Stabilization Effect": Qwen3.5's reported accuracy is essentially
   its non-empty rate. On runs that returned text it scored 82–91%, and
   `no_rag` was the *best* configuration (90.6%), not the worst (6.97%).
2. **`hallucination_rate` was defined as `1 - answer_accuracy`**, so it carried
   no independent information and the hallucination figure was the accuracy
   figure mirrored.
3. **The "Reranking Efficiency Paradox" rests on a single row** of 390 at
   226.8 s. Under the median the effect reverses sign; Welch p = 0.33.
   Its stated mechanism (prompt shrinking 1500 → 500 tokens) is contradicted by
   the measured `prompt_tokens` (1085.6 vs 1079.9, a ~6-token difference).
4. **Latency timed only the final `generate()` call**, so query rewriting,
   reranking, routing and sub-query generation were free in the measurement.
5. **`temperature = 0.0` with a fixed seed** made the 30 repeats deterministic,
   so 4,680 rows represent ~156 distinct measurements, not 4,680.
6. **The modular router used a substring test** (`"no" in decision`), so
   "know" / "not" / "cannot" silently routed to `no_rag`.

`v2` addresses each of these. The v1 numbers remain valid as a record of what
the v1 code measured — they are simply not evidence for the claims the v1 paper
draws from them.

## Reproducing v1

The v1 code is reachable through Git history:

```bash
git log --oneline -- legacy/paper_v1.tex
git checkout <commit-before-v2> -- rag/ evaluation/
```
