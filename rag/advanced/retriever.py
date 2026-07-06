import numpy as np
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

class AdvancedRAGPipeline:
    def __init__(self, db_manager, generator, top_k=3, rerank=True, rerank_model_name="BAAI/bge-reranker-base"):
        self.db_manager = db_manager
        self.generator = generator
        self.top_k = top_k
        self.rerank = rerank
        
        # Load cross-encoder reranker if requested
        self.reranker = None
        if self.rerank:
            try:
                print(f"Loading reranker model: {rerank_model_name}...")
                self.reranker = CrossEncoder(rerank_model_name)
            except Exception as e:
                print(f"Error loading reranker {rerank_model_name}: {e}. Disabling reranking.")
                self.rerank = False

    def rewrite_query(self, query):
        """Asks the model to rephrase the query for search optimization."""
        prompt = (
            "You are a search query optimizer. Given a user query, output a search-friendly "
            "rephrased version of it. Do not include introductory text or markdown formatting. "
            "Output only the optimized query.\n\n"
            f"Original Query: {query}\n"
            "Optimized Query:"
        )
        # We run this on the generator
        res = self.generator.generate(prompt)
        rewritten = res.get("response", "").strip()
        # Clean up any potential model prefixes
        if ":" in rewritten and len(rewritten.split(":")[0]) < 15:
            rewritten = "".join(rewritten.split(":")[1:]).strip()
        # Fallback if model errors or gives empty response
        if not rewritten or len(rewritten) < 3 or "ERROR" in rewritten:
            return query
        return rewritten

    def hybrid_search(self, query, top_k=10):
        """Performs BM25-like TF-IDF keyword search + FAISS dense search and fuses scores."""
        # 1. Dense search (get top_k * 2 candidates for fusion)
        dense_results = self.db_manager.search(query, top_k=top_k * 2)
        
        # 2. Sparse (TF-IDF) search
        chunks = self.db_manager.doc_chunks
        if not chunks:
            return dense_results
            
        corpus = [c["text"] for c in chunks]
        
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(corpus)
            query_vector = vectorizer.transform([query])
            
            # Calculate cosine similarities for TF-IDF
            sparse_scores = (tfidf_matrix * query_vector.T).toarray().flatten()
        except Exception as e:
            print(f"Sparse TF-IDF index error: {e}")
            return dense_results
            
        # Combine dense and sparse scores
        # We map chunk indices to compute combined scores
        dense_scores_dict = {}
        for r in dense_results:
            # Match chunk text to find index in corpus
            for idx, c in enumerate(chunks):
                if c["text"] == r["chunk"]["text"]:
                    dense_scores_dict[idx] = r["score"]
                    break
        
        hybrid_results = []
        for idx, c in enumerate(chunks):
            dense_score = dense_scores_dict.get(idx, 0.0)
            sparse_score = float(sparse_scores[idx])
            
            # Hybrid score formula (70% dense + 30% sparse)
            combined_score = 0.7 * dense_score + 0.3 * sparse_score
            
            if combined_score > 0:
                hybrid_results.append({
                    "chunk": c,
                    "score": combined_score
                })
                
        # Sort by combined score and return top candidates
        hybrid_results = sorted(hybrid_results, key=lambda x: x["score"], reverse=True)
        return hybrid_results[:top_k * 2]

    def run(self, query):
        """Runs the complete Advanced RAG pipeline: Query Rewrite -> Hybrid Search -> Rerank."""
        # 1. Query Rewriting
        optimized_query = self.rewrite_query(query)
        print(f"  [Advanced RAG] Query Rewritten: '{query}' -> '{optimized_query}'")
        
        # 2. Hybrid Search (retrieving double top_k candidates)
        candidates = self.hybrid_search(optimized_query, top_k=self.top_k)
        
        # 3. Rerank candidates if reranker is available
        retrieved_contexts = []
        if self.rerank and self.reranker and candidates:
            pairs = [[optimized_query, c["chunk"]["text"]] for c in candidates]
            rerank_scores = self.reranker.predict(pairs)
            
            for score, c in zip(rerank_scores, candidates):
                c["rerank_score"] = float(score)
                
            # Sort by rerank score
            candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
            retrieved_contexts = [c["chunk"]["text"] for c in candidates[:self.top_k]]
        else:
            retrieved_contexts = [c["chunk"]["text"] for c in candidates[:self.top_k]]
            
        context_str = "\n\n".join(retrieved_contexts)
        
        # 4. Prompt Generation
        prompt = (
            "You are a helpful research assistant. Use the following pieces of context to answer the question. "
            "If the context does not contain enough information to answer, state that you do not know.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        
        # 5. Generate
        result = self.generator.generate(prompt)
        result["retrieved_context"] = retrieved_contexts
        return result
