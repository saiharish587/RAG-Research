# SLM RAG Benchmark

A benchmarking framework to evaluate the impact of Retrieval-Augmented Generation (RAG) pipeline sophistication and model scale on sub-1B language models.

## Project Title
**Evaluating the Impact of RAG Pipeline Sophistication and Model Scale on Sub-1B Language Models: A Comparative Study**

---

## Architecture Layout
```text
SLM-RAG-Benchmark/
├── app/                  # Web or desktop visualization application interfaces
├── configs/              # Global YAML configurations for runs, paths, and models
├── data/                 # Raw and processed datasets
│   ├── raw/
│   ├── processed/
│   ├── chunks/
│   └── benchmark/
├── documents/            # User-supplied books, PDFs, lecture notes
├── embeddings/           # Saved embedding cache files
├── vector_db/            # FAISS index storage files
├── models/               # Local model downloads or custom setup configurations
├── rag/                  # RAG implementations by sophistication
│   ├── no_rag/           # Direct LLM calls with zero retrieval context
│   ├── naive/            # Top-K context insertion
│   ├── advanced/         # Query rewriting, hybrid search, and reranking
│   └── modular/          # Routing, context fusion, and custom pipelines
├── evaluation/           # Ragas/evaluator setups and metrics scripts
├── experiments/          # Custom trial runners or tracking logs
├── results/              # Output files from runs
│   ├── csv/
│   ├── graphs/
│   └── reports/
├── visualization/        # Matplotlib/Plotly scripts for benchmark results
├── utils/                # General helpers
├── notebooks/            # Jupyter notebooks for interactive analysis
└── tests/                # Unit and integration test suites
```

---

## Experimental Setup
The project runs a matrix evaluation:
- **Models**: IBM Granite 3B (instruct), Qwen2.5 0.5B, Qwen2.5 1.5B (and other sub-1B models).
- **Pipelines**: No RAG, Naive RAG, Advanced RAG, Modular RAG.
- **Run Matrix**: 3 Models × 4 RAG Pipelines × 10 Runs = 120 experimental runs.

---

## How to Get Started
1. Run `python setup_skeleton.py` to create the project structure.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the evaluation entrypoint:
   ```bash
   python main.py --model qwen2.5:0.5b --rag no_rag --runs 10
   ```
