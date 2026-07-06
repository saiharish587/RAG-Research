class ModularRAGPipeline:
    def __init__(self, db_manager, generator, no_rag_pipeline, naive_rag_pipeline, top_k=3):
        self.db_manager = db_manager
        self.generator = generator
        self.no_rag_pipeline = no_rag_pipeline
        self.naive_rag_pipeline = naive_rag_pipeline
        self.top_k = top_k

    def route_query(self, query):
        """Routes query to RAG or No RAG based on model routing query."""
        prompt = (
            "Determine if answering the following user question requires retrieving information "
            "from internal files/documents. Respond with only 'yes' or 'no'.\n\n"
            f"Question: {query}\n"
            "Requires Retrieval:"
        )
        res = self.generator.generate(prompt)
        decision = res.get("response", "").strip().lower()
        if "no" in decision:
            return "no_rag"
        return "rag"

    def generate_sub_queries(self, query):
        """Generates two sub-queries to retrieve diverse aspects of the query."""
        prompt = (
            "Generate exactly two search queries related to the following user question. "
            "Separate them by a newline. Do not output numbers or markdown. Just the queries.\n\n"
            f"Question: {query}\n"
            "Queries:"
        )
        res = self.generator.generate(prompt)
        text = res.get("response", "").strip()
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        
        # Clean up lists if model outputs numbered list
        sub_queries = []
        for line in lines[:2]:
            cleaned = line
            # strip numbers like "1. ", "2. ", etc.
            if cleaned and cleaned[0].isdigit() and "." in cleaned[:3]:
                cleaned = cleaned.split(".", 1)[1].strip()
            sub_queries.append(cleaned)
            
        return sub_queries

    def reciprocal_rank_fusion(self, list_of_retrievals, k=60):
        """
        Combines multiple retrieval lists using Reciprocal Rank Fusion.
        list_of_retrievals: list of lists, where each sublist contains search result dicts.
        """
        rrf_scores = {}
        text_to_chunk_map = {}
        
        for retrieval_list in list_of_retrievals:
            for rank, r in enumerate(retrieval_list):
                chunk = r["chunk"]
                text = chunk["text"]
                text_to_chunk_map[text] = chunk
                
                # RRF formula: score += 1 / (k + rank)
                rrf_scores[text] = rrf_scores.get(text, 0.0) + 1.0 / (k + rank)
                
        # Sort chunks based on RRF scores
        sorted_texts = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        fused_results = []
        for text in sorted_texts[:self.top_k]:
            fused_results.append(text_to_chunk_map[text])
            
        return fused_results

    def run(self, query):
        """Runs the Modular RAG pipeline: Query Routing -> Sub-queries -> Retrieval Fusion."""
        # 1. Routing
        route = self.route_query(query)
        print(f"  [Modular RAG] Router directed query to: {route}")
        if route == "no_rag":
            return self.no_rag_pipeline.run(query)
            
        # 2. Sub-query Generation
        sub_queries = self.generate_sub_queries(query)
        all_queries = [query] + sub_queries
        print(f"  [Modular RAG] Target queries for fusion search: {all_queries}")
        
        # 3. Parallel Retrieval for all queries
        retrieval_lists = []
        for q in all_queries:
            results = self.db_manager.search(q, top_k=self.top_k * 2)
            retrieval_lists.append(results)
            
        # 4. Context Fusion using RRF
        fused_contexts = self.reciprocal_rank_fusion(retrieval_lists)
        retrieved_contexts = [c["text"] for c in fused_contexts]
        
        context_str = "\n\n".join(retrieved_contexts)
        
        # 5. Generation
        prompt = (
            "You are a helpful research assistant. Use the following pieces of context to answer the question. "
            "If the context does not contain enough information to answer, state that you do not know.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        
        result = self.generator.generate(prompt)
        result["retrieved_context"] = retrieved_contexts
        return result
