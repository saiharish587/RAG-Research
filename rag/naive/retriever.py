class NaiveRAGPipeline:
    def __init__(self, db_manager, generator, top_k=3):
        self.db_manager = db_manager
        self.generator = generator
        self.top_k = top_k

    def run(self, query):
        """Retrieves top-K context chunks, constructs the prompt, and runs generation."""
        # 1. Retrieve
        search_results = self.db_manager.search(query, top_k=self.top_k)
        
        # 2. Extract texts
        retrieved_contexts = [r["chunk"]["text"] for r in search_results]
        context_str = "\n\n".join(retrieved_contexts)
        
        # 3. Construct prompt
        prompt = (
            "You are a helpful research assistant. Use the following pieces of context to answer the question. "
            "If the context does not contain enough information to answer, state that you do not know.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        
        # 4. Generate
        result = self.generator.generate(prompt)
        result["retrieved_context"] = retrieved_contexts
        return result
