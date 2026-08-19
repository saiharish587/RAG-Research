import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv(r"c:\Users\iamsa\Documents\RAG-Research\results\csv\benchmark_results.csv", low_memory=False)

# Step 1: Aggregate to question-level means (600 values per config)
q_agg = df.groupby(["model", "rag_type", "query"])["token_f1"].mean().reset_index()
q_agg.rename(columns={"token_f1": "q_mean_f1"}, inplace=True)

print("=" * 70)
print("PAIRED QUESTION-LEVEL STATISTICAL TESTS")
print("=" * 70)

# ---- Test 1: Naive vs Advanced RAG on LFM2-350M ----
print("\n--- Test 1: Naive vs Advanced RAG (LFM2-350M) ---")
naive_350 = q_agg[(q_agg["model"].str.contains("350")) & (q_agg["rag_type"] == "naive")].sort_values("query")
adv_350 = q_agg[(q_agg["model"].str.contains("350")) & (q_agg["rag_type"] == "advanced")].sort_values("query")

# Merge on query to ensure pairing
merged1 = naive_350.merge(adv_350, on="query", suffixes=("_naive", "_adv"))
d1 = merged1["q_mean_f1_naive"] - merged1["q_mean_f1_adv"]

print(f"  N (questions): {len(d1)}")
print(f"  Naive mean:    {merged1['q_mean_f1_naive'].mean():.4f}")
print(f"  Advanced mean: {merged1['q_mean_f1_adv'].mean():.4f}")
print(f"  Mean diff (d): {d1.mean():.4f}")
print(f"  Std diff:      {d1.std():.4f}")

# Paired t-test
t1, p1 = stats.ttest_rel(merged1["q_mean_f1_naive"], merged1["q_mean_f1_adv"])
print(f"  Paired t-test: t={t1:.4f}, p={p1:.6e}")

# 95% CI on the difference
se1 = d1.std() / np.sqrt(len(d1))
ci_low1 = d1.mean() - 1.96 * se1
ci_high1 = d1.mean() + 1.96 * se1
print(f"  95% CI on diff: [{ci_low1:.4f}, {ci_high1:.4f}]")

# Wilcoxon signed-rank (robustness)
w1, pw1 = stats.wilcoxon(d1, alternative="two-sided")
print(f"  Wilcoxon: W={w1:.1f}, p={pw1:.6e}")

# ---- Test 2: 350M vs 700M under Naive RAG (Scale Inversion) ----
print("\n--- Test 2: LFM2-350M vs LFM2-700M (Naive RAG) ---")
naive_700 = q_agg[(q_agg["model"].str.contains("700")) & (q_agg["rag_type"] == "naive")].sort_values("query")

merged2 = naive_350.merge(naive_700, on="query", suffixes=("_350", "_700"))
d2 = merged2["q_mean_f1_350"] - merged2["q_mean_f1_700"]

print(f"  N (questions): {len(d2)}")
print(f"  350M mean:     {merged2['q_mean_f1_350'].mean():.4f}")
print(f"  700M mean:     {merged2['q_mean_f1_700'].mean():.4f}")
print(f"  Mean diff (d): {d2.mean():.4f}")
print(f"  Std diff:      {d2.std():.4f}")

t2, p2 = stats.ttest_rel(merged2["q_mean_f1_350"], merged2["q_mean_f1_700"])
print(f"  Paired t-test: t={t2:.4f}, p={p2:.6e}")

se2 = d2.std() / np.sqrt(len(d2))
ci_low2 = d2.mean() - 1.96 * se2
ci_high2 = d2.mean() + 1.96 * se2
print(f"  95% CI on diff: [{ci_low2:.4f}, {ci_high2:.4f}]")

w2, pw2 = stats.wilcoxon(d2, alternative="two-sided")
print(f"  Wilcoxon: W={w2:.1f}, p={pw2:.6e}")

# ---- Latency median/IQR for 700M Oracle and Modular ----
print("\n--- Latency Median/IQR for 700M extreme configs ---")
for rag in ["oracle", "modular"]:
    subset = df[(df["model"].str.contains("700")) & (df["rag_type"] == rag)]
    lat = subset["latency"].dropna()
    print(f"  700M {rag}: median={lat.median():.3f}s, IQR=[{lat.quantile(0.25):.3f}, {lat.quantile(0.75):.3f}], p95={lat.quantile(0.95):.3f}s, p99={lat.quantile(0.99):.3f}s")

print("\n--- Latency Median/IQR for all configs ---")
for model_tag in ["350", "700"]:
    for rag in ["no_rag", "oracle", "naive", "advanced", "modular"]:
        subset = df[(df["model"].str.contains(model_tag)) & (df["rag_type"] == rag)]
        lat = subset["latency"].dropna()
        if len(lat) > 0:
            print(f"  LFM2-{model_tag}M {rag}: median={lat.median():.3f}s, IQR=[{lat.quantile(0.25):.3f}, {lat.quantile(0.75):.3f}]")
