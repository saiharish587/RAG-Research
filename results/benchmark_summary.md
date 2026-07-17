# SLM-RAG Benchmarking Results Summary

This file summarizes the final performance metrics obtained from our 4,680-run comparative study testing three Small Language Models (SLMs) across four Retrieval-Augmented Generation (RAG) configurations on local consumer-grade GPU hardware.

---

## 1. Empirical Performance Matrix

| Model | Pipeline Configuration | Semantic Accuracy | Average Latency | Hallucination Rate |
| :--- | :--- | :---: | :---: | :---: |
| **Granite 350M** | No RAG | **87.03%** | 0.76s | 12.97% |
| | Naive RAG | 85.81% | **0.62s** | 14.19% |
| | Advanced RAG | 85.67% | 0.81s | 14.33% |
| | Modular RAG | 86.11% | 0.82s | 13.89% |
| **Qwen2.5 0.5B** | No RAG | 86.39% | 20.24s | 13.61% |
| | Naive RAG | 87.73% | 2.28s | 12.27% |
| | Advanced RAG | 86.47% | **1.71s** | 13.53% |
| | Modular RAG | **88.02%** | 1.87s | **11.98%** |
| **Qwen3.5 0.8B** | No RAG | 6.97% | 41.34s | 93.03% |
| | Naive RAG | 50.75% | 37.55s | 49.25% |
| | Advanced RAG | **57.02%** | **22.55s** | **42.98%** |
| | Modular RAG | 31.84% | 30.57s | 68.16% |

---

## 2. Key Empirical Insights

### A. The Qwen2.5 0.5B Sweet Spot
* **Observation**: Pairing Qwen2.5 0.5B with Modular RAG yields the highest overall accuracy (**88.02%**) and the lowest hallucination rate (**11.98%**).
* **Speed Profile**: It responds in **1.87 seconds**, representing the best performance-latency trade-off for consumer deployments.

### B. The Reranking Latency Paradox
* **Observation**: Advanced RAG is faster than Naive RAG on local hardware (e.g., Qwen2.5 0.5B: **1.71s vs. 2.28s**).
* **Explanation**: Although Advanced RAG adds cross-encoder reranking, it trims the prompt size down to 1 chunk ($500$ tokens) instead of 3 chunks ($1,500$ tokens). On low-VRAM GPUs, the time saved during prompt evaluation far outweighs the reranker overhead.

### C. Qwen3.5 0.8B Loop Stabilization
* **Observation**: Under direct completion (No RAG), Qwen3.5 0.8B gets stuck in an infinite whitespace loop, driving latency to **41.34s** and accuracy to **6.97%**.
* **Stabilizer**: Providing retrieved context (RAG) anchors the attention heads, breaking the loop, cutting latency in half, and recovering accuracy to **57.02%** (Advanced RAG).

---

## 3. Reference Files
* **Raw CSV Data**: [benchmark_results.csv](results/csv/benchmark_results.csv)
* **LaTeX Research Paper**: [paper.tex](paper.tex)
* **Compiled PDF Paper**: [paper.pdf](paper.pdf)
* **Visual Plots**: [results/graphs/](results/graphs/)
