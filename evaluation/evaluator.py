import numpy as np
from sentence_transformers import SentenceTransformer

class Evaluator:
    def __init__(self, embedding_model_name="BAAI/bge-small-en-v1.5", device="cpu"):
        print("Initializing Evaluator...")
        self.model = SentenceTransformer(embedding_model_name, device=device)

    def calculate_semantic_similarity(self, text1, text2):
        """Calculates cosine similarity between two texts using the embedding model."""
        if not text1 or not text2:
            return 0.0
        try:
            embeddings = self.model.encode([text1, text2], convert_to_numpy=True)
            norm1 = embeddings[0] / np.linalg.norm(embeddings[0])
            norm2 = embeddings[1] / np.linalg.norm(embeddings[1])
            return float(np.dot(norm1, norm2))
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return 0.0

    def calculate_exact_overlap_precision_recall(self, retrieved_context, ground_truth):
        """
        Calculates simple character/word overlap metrics for retrieved context.
        Returns:
            dict containing:
                - precision: Overlap precision
                - recall: Overlap recall
        """
        if not retrieved_context or not ground_truth:
            return {"precision": 0.0, "recall": 0.0}
            
        # Combine retrieved context to single string
        context_str = " ".join(retrieved_context).lower()
        gt_str = ground_truth.lower()
        
        # Tokenize by words
        context_words = set(context_str.split())
        gt_words = set(gt_str.split())
        
        if not gt_words or not context_words:
            return {"precision": 0.0, "recall": 0.0}
            
        intersection = context_words.intersection(gt_words)
        
        precision = len(intersection) / len(context_words) if len(context_words) > 0 else 0.0
        recall = len(intersection) / len(gt_words) if len(gt_words) > 0 else 0.0
        
        return {"precision": precision, "recall": recall}

    def evaluate_run(self, query, response, retrieved_context, ground_truth, stats):
        """
        Evaluates a single run of the benchmark.
        Returns:
            dict of all calculated metrics.
        """
        similarity = self.calculate_semantic_similarity(response, ground_truth)
        overlap = self.calculate_exact_overlap_precision_recall(retrieved_context, ground_truth)
        
        # Calculate context utilization (does the response contain words from retrieved context?)
        response_words = set(response.lower().split())
        context_words = set(" ".join(retrieved_context).lower().split()) if retrieved_context else set()
        
        context_utilization = 0.0
        if response_words and context_words:
            context_utilization = len(response_words.intersection(context_words)) / len(response_words)

        # Simple hallucination heuristic: 1 - semantic similarity to ground truth
        hallucination_rate = max(0.0, 1.0 - similarity)
        
        metrics = {
            "query": query,
            "response": response,
            "ground_truth": ground_truth,
            "answer_accuracy": similarity, # Proxy for semantic accuracy
            "context_utilization": context_utilization,
            "precision_at_k": overlap["precision"],
            "recall_at_k": overlap["recall"],
            "hallucination_rate": hallucination_rate,
            "latency": stats.get("latency", 0.0),
            "generation_tokens": stats.get("generation_tokens", 0),
            "prompt_tokens": stats.get("prompt_tokens", 0),
            "generation_speed": stats.get("generation_speed", 0.0),
            "prompt_eval_speed": stats.get("prompt_eval_speed", 0.0)
        }
        
        return metrics
