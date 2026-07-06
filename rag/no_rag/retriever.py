class NoRAGPipeline:
    def __init__(self, db_manager=None, generator=None):
        self.db_manager = db_manager
        self.generator = generator

    def run(self, query):
        """Runs direct generation without any context."""
        prompt = f"Question: {query}\nAnswer:"
        # Use generator to call Ollama
        result = self.generator.generate(prompt)
        result["retrieved_context"] = []
        return result
