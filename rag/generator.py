import time
import ollama

class Generator:
    def __init__(self, model_name="qwen2.5:0.5b"):
        self.model_name = model_name

    def generate(self, prompt, system_prompt=None):
        """
        Sends the prompt to the Ollama model and profiles the performance.
        Returns:
            dict containing:
                - response: The generated text string
                - prompt_tokens: Number of prompt tokens
                - generation_tokens: Number of generated tokens
                - latency: Total response time in seconds
                - load_duration: Model load time in seconds
                - prompt_eval_speed: Prompt tokens per second
                - generation_speed: Generated tokens per second
        """
        start_time = time.time()
        
        try:
            # We use ollama.generate (non-streaming)
            options = {
                "temperature": 0.0, # Zero temperature for reproducible benchmarking
                "seed": 42
            }
            
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                system=system_prompt if system_prompt else "",
                options=options
            )
            
            latency = time.time() - start_time
            
            # Extract metrics from Ollama response
            # Note: Ollama times are returned in nanoseconds
            total_duration = response.get("total_duration", 0) / 1e9
            load_duration = response.get("load_duration", 0) / 1e9
            prompt_tokens = response.get("prompt_eval_count", 0)
            prompt_eval_duration = response.get("prompt_eval_duration", 0) / 1e9
            generation_tokens = response.get("eval_count", 0)
            eval_duration = response.get("eval_duration", 0) / 1e9
            
            prompt_eval_speed = prompt_tokens / prompt_eval_duration if prompt_eval_duration > 0 else 0
            generation_speed = generation_tokens / eval_duration if eval_duration > 0 else 0
            
            return {
                "response": response.get("response", "").strip(),
                "prompt_tokens": prompt_tokens,
                "generation_tokens": generation_tokens,
                "latency": latency,
                "load_duration": load_duration,
                "prompt_eval_speed": prompt_eval_speed,
                "generation_speed": generation_speed,
                "raw_stats": response
            }
            
        except Exception as e:
            latency = time.time() - start_time
            print(f"Error during generation with model {self.model_name}: {e}")
            return {
                "response": f"ERROR: {str(e)}",
                "prompt_tokens": 0,
                "generation_tokens": 0,
                "latency": latency,
                "load_duration": 0.0,
                "prompt_eval_speed": 0.0,
                "generation_speed": 0.0,
                "raw_stats": {}
            }
