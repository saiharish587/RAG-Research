import os
import json

def create_samples():
    # 1. Create documents directory
    os.makedirs("documents", exist_ok=True)
    os.makedirs("data/benchmark", exist_ok=True)

    # 2. Sample document 1: Backpropagation
    doc1_content = """Backpropagation, or backward propagation of errors, is a widely used algorithm in training artificial neural networks.
It calculates the gradient of the loss function with respect to the weights of the network for a single input-output example.
This calculation is done recursively backwards through the layers of the network, starting from the output layer.
By computing the gradients, backpropagation allows optimization algorithms like Gradient Descent to update the network weights, minimizing the error.
The mathematical foundation of backpropagation is the chain rule of calculus, which enables the derivative of a composite function to be computed efficiently.
Crucially, sub-1B parameter models often struggle to remember the exact mathematical derivation of backpropagation without external context.
"""
    
    # 3. Sample document 2: RAG Pipeline Sophistication
    doc2_content = """Retrieval-Augmented Generation (RAG) is a framework that improves language model outputs by retrieving relevant content from external sources.
Naive RAG is the simplest pipeline. It chunks documents, indexes them, performs a vector search on user queries, and appends the top-K chunks to the prompt.
Advanced RAG improves upon Naive RAG by adding sophisticated processing steps:
First, it uses query rewriting to optimize user search terms.
Second, it uses hybrid search (combining sparse keyword search like BM25/TF-IDF with dense vector search) to fetch diverse results.
Third, it applies a reranker model (such as a Cross-Encoder) to order retrieved chunks based on semantic relevance, filtering out noise.
Modular RAG introduces even more complexity. It features query routing to decide whether to search documents or answer directly, and context fusion to merge results from multiple search queries using Reciprocal Rank Fusion (RRF).
"""

    with open("documents/backpropagation_notes.txt", "w", encoding="utf-8") as f:
        f.write(doc1_content)
    print("Created documents/backpropagation_notes.txt")

    with open("documents/rag_framework_notes.txt", "w", encoding="utf-8") as f:
        f.write(doc2_content)
    print("Created documents/rag_framework_notes.txt")

    # 4. Create benchmark evaluation set
    eval_set = [
        {
            "query": "Explain what backpropagation is and how it computes gradients.",
            "ground_truth": "Backpropagation is an algorithm used to train neural networks by calculating the gradient of the loss function with respect to the weights. It does this recursively backwards through the layers starting from the output layer, using the chain rule of calculus."
        },
        {
            "query": "What optimization algorithm uses the gradients computed by backpropagation?",
            "ground_truth": "Optimization algorithms like Gradient Descent use the gradients calculated by backpropagation to update network weights and minimize errors."
        },
        {
            "query": "What are the three main improvements of Advanced RAG over Naive RAG?",
            "ground_truth": "Advanced RAG improves on Naive RAG by using query rewriting, hybrid search (combining keyword/sparse and vector/dense searches), and a reranker model (like a Cross-Encoder) to order context."
        },
        {
            "query": "How does Modular RAG handle context fusion and query routing?",
            "ground_truth": "Modular RAG uses query routing to decide if document retrieval is necessary, and context fusion to merge retrieval results from multiple queries using Reciprocal Rank Fusion (RRF)."
        }
    ]

    with open("data/benchmark/eval_set.json", "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=4)
    print("Created data/benchmark/eval_set.json")

if __name__ == "__main__":
    create_samples()
