# Evaluating the Impact of RAG Pipeline Sophistication and Model Scale on Sub-1B Language Models: A Comparative Study

A rigorous, empirical benchmarking framework designed to evaluate whether increasing Retrieval-Augmented Generation (RAG) pipeline sophistication compensates for limited parametric capacity in sub-billion language models (SLMs), or whether context utilization remains a bottleneck even when retrieval improves.

---

## 📌 Research Overview

Small Language Models (SLMs) with fewer than 1 billion parameters possess limited capacity to memorize multi-hop factual knowledge in their weights. While Retrieval-Augmented Generation (RAG) is commonly deployed to supply external context, existing literature focuses primarily on multi-billion parameter models (e.g., 7B–70B+).

This project conducts a **60,000-run controlled empirical study** evaluating **Liquid AI's LFM2 architecture** across **5 distinct RAG architectural arms** on the **HotpotQA Distractor benchmark**.

### Core Research Questions
1. **Capacity vs. Retrieval**: Does RAG pipeline sophistication compensate for parameter constraints in sub-1B models?
2. **The Context Utilization Bottleneck**: When provided with perfect gold context, do sub-1B models successfully extract answers or hit a context processing ceiling?
3. **Scale Inversion in Sub-1B RAG**: Does doubling parameters from 350M to 700M improve context utilization under noisy RAG conditions?

---

## 🔬 Experimental Matrix & Protocol

The experiment evaluates a full factorial matrix across **600 multi-hop questions** from the **HotpotQA Distractor dev set**, repeated **10 times per configuration** with honest cost and load-duration accounting:

- **Models**:
  - `LFM2-350M` (`hf.co/LiquidAI/LFM2-350M-GGUF:Q4_K_M`)
  - `LFM2-700M` (`hf.co/LiquidAI/LFM2-700M-GGUF:Q4_K_M`)
- **RAG Pipeline Arms (5 Conditions)**:
  1. **No-RAG**: Direct LLM generation (zero context).
  2. **Oracle RAG**: Gold ground-truth evidence injection.
  3. **Naive RAG**: Vector search (`BAAI/bge-small-en-v1.5` + FAISS top-$k$).
  4. **Advanced RAG**: Query rewriting + Hybrid Retrieval + Cross-Encoder Reranking (`bge-reranker-base`).
  5. **Modular RAG**: Dynamic Query Routing + Multi-Query Fusion + Reciprocal Rank Fusion (RRF).
- **Total Executed Runs**: **2 Models × 5 Arms × 600 Questions × 10 Repetitions = 60,000 Runs**.

---

## 📊 Summary of Empirical Findings

| Model | RAG Pipeline Arm | Token F1 ($\mu$) | Exact Match (EM) | Cosine Sim. ($\mu$) | Groundedness ($\mu$) | Mean Latency (s) | Output Tokens |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LFM2-350M** | **No-RAG** | `0.0147` | `0.0000` | `0.5426` | N/A | `0.730s` | `160.5` |
| **LFM2-350M** | **Oracle RAG** | `0.0622` | `0.0033` | **`0.6260`** | `0.0826` | `0.675s` | `128.9` |
| **LFM2-350M** | **Naive RAG** | **`0.1107`** | **`0.0367`** | `0.5909` | `0.7277` | **`0.581s`** | **`67.2`** |
| **LFM2-350M** | **Advanced RAG** | `0.1017` | `0.0283` | `0.5924` | `0.6851` | `1.460s` | `72.6` |
| **LFM2-350M** | **Modular RAG** | `0.0218` | `0.0017` | `0.5469` | **`0.8067`** | `1.173s` | `156.2` |
| | | | | | | | |
| **LFM2-700M** | **No-RAG** | `0.0149` | `0.0000` | `0.5462` | N/A | `1.033s` | `161.2` |
| **LFM2-700M** | **Oracle RAG** | `0.0441` | `0.0000` | `0.6233` | `0.0772` | `1.164s` | `178.5` |
| **LFM2-700M** | **Naive RAG** | `0.0727` | `0.0033` | `0.5905` | `0.6791` | `0.779s` | `98.5` |
| **LFM2-700M** | **Advanced RAG** | `0.0755` | `0.0017` | `0.5914` | `0.6778` | `1.679s` | `92.2` |
| **LFM2-700M** | **Modular RAG** | `0.0157` | `0.0000` | `0.5467` | `0.6785` | `4.567s` | `159.6` |

### Key Paper Takeaways
1. **Naive RAG is Optimal for Sub-1B Models**:
   Naive RAG achieved the highest accuracy (**Token F1 = 11.07%**) while maintaining the lowest latency (**0.581s**). Adding query rewriting and reranking in Advanced RAG yields comparable quality (`10.17%` F1) at **2.5× the latency cost (`1.460s`)**.
2. **Sub-1B Scale Inversion**:
   Doubling model parameters from 350M to 700M does **not** improve RAG performance (`7.27%` F1 for 700M vs `11.07%` F1 for 350M under Naive RAG). Small models possess tighter attention focus for direct context extraction, whereas larger sub-1B models show higher sensitivity to distractor noise.
3. **Context Utilization Ceiling**:
   Even under **Oracle RAG** (100% gold context), sub-1B models reach only `6.22%` Token F1, proving that **in-context reasoning and answer extraction**—rather than retrieval quality—is the dominant bottleneck.

---

## 🏗️ Repository Architecture

```text
RAG-Research/
├── configs/
│   └── config.yaml             # Benchmark configuration (models, embedding, paths)
├── data/
│   ├── raw/                    # Raw HotpotQA Distractor dev set JSON
│   └── benchmark/
│       └── eval_set.json       # Cleaned 600-question evaluation set
├── documents/
│   └── hotpotqa_corpus/        # 66,581 extracted passage text files
├── vector_db/
│   └── faiss_index/            # FAISS vector database (index.faiss + chunks.npy)
├── rag/                        # RAG Pipeline Implementations
│   ├── base.py                 # Shared timing and run recorder
│   ├── prompts.py              # Parity prompt templates across all arms
│   ├── no_rag/                 # Baseline zero-retrieval arm
│   ├── oracle/                 # Gold evidence context injection arm
│   ├── naive/                  # Top-K vector retrieval arm
│   ├── advanced/               # Query rewrite + hybrid search + reranking arm
│   └── modular/                # Router + sub-query fusion + RRF arm
├── evaluation/
│   ├── evaluator.py            # Token F1, Exact Match, Cosine, Groundedness scoring
│   ├── metrics.py              # Pure metric calculation functions
│   ├── aggregate.py            # Honest per-query & per-run aggregation
│   └── ragchecker_eval.py      # RAGChecker diagnostic framework integration
├── results/
│   ├── csv/
│   │   └── benchmark_results.csv  # Full 60,000-run output dataset
│   ├── raw/                    # 60,000 individual JSON evidence files
│   └── graphs/                 # Publication-ready Seaborn plots
├── visualization/
│   └── visualize.py            # Automated plot generator
├── ingest_hotpotqa.py          # HotpotQA distractor dataset ingestion script
├── rebuild_index.py            # FAISS index builder with GPU CUDA support
├── verify_models.py            # Ollama model verification helper
├── main.py                     # Master benchmark driver pipeline
└── requirements.txt            # Project dependencies
```

---

## 🚀 How to Run

### 1. Requirements & Setup
- Python 3.12+
- Local Ollama runner running on GPU
- GPU Acceleration (CUDA supported for FAISS & sentence-transformers)

```bash
# Clone the repository
git clone https://github.com/saiharish587/RAG-Research.git
cd RAG-Research

# Create virtual environment and install dependencies
uv venv --python 3.12 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 2. Prepare Data & Rebuild FAISS Index
```bash
# Ingest HotpotQA Distractor dataset
python ingest_hotpotqa.py --sample-size 600

# Build FAISS Vector Index
python rebuild_index.py
```

### 3. Verify Local Models in Ollama
```bash
python verify_models.py
```

### 4. Execute Benchmark Runs
```bash
# Test-only mode (1 question, 1 run per configuration)
python main.py --test-only

# Full 60,000-run benchmark matrix
python main.py
```

---

## 📈 Visualizations

Generated plots are saved to `results/graphs/`:
- `accuracy_vs_rag.png`: Token F1 quality across RAG arms and model sizes.
- `latency_vs_rag.png`: Inference latency profile comparisons.
- `groundedness_vs_rag.png`: Context groundedness and hallucination analysis.
- `accuracy_vs_latency_tradeoff.png`: Quality vs. Latency cost-benefit scatter.
