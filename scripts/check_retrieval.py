"""Preflight: does this machine retrieve exactly the passages logged in the paper's runs?

Naive RAG retrieval does not depend on the language model, so the top-3 passages
for each question must match the logged LFM2 Naive runs. Runs the first N
questions (no LLM calls) and reports how many match. Expected: all of them.

Usage (from the repository root): python scripts/check_retrieval.py [N]
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from main import load_config, setup_vector_db  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
LOG = "results/raw/hf.co_LiquidAI_LFM2-350M-GGUF_Q4_K_M_naive_q{}_run1.json"

config = load_config()
db = setup_vector_db(config)
top_k = config.get("retrieval", {}).get("top_k", 3)
eval_set = json.load(open("data/benchmark/eval_set.json", encoding="utf-8"))[:N]

matches = 0
for i, item in enumerate(eval_set, 1):
    logged = json.load(open(LOG.format(i), encoding="utf-8"))["pipeline_result"]["retrieved_context"]
    now = [r["chunk"]["text"] for r in db.search(item["query"], top_k=top_k)]
    matches += now == logged
print(f"\nRetrieval identical to the logged runs on {matches}/{N} questions.")
sys.exit(0 if matches == N else 1)
