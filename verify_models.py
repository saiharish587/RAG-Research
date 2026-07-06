import ollama
import sys

def verify_models():
    models_to_test = [
        "hf.co/ibm-granite/granite-4.0-h-350m-GGUF:Q4_K_M",
        "qwen2.5:0.5b",
        "qwen3.5:0.8b"
    ]
    
    print("Testing Ollama API connection...")
    try:
        ollama_list = ollama.list()
        available_names = [m.get("model", m.get("name", "")) for m in ollama_list.get("models", [])]
        print(f"Available models in Ollama: {available_names}")
    except Exception as e:
        print(f"Failed to connect to local Ollama service: {e}")
        sys.exit(1)

    for model in models_to_test:
        print(f"\nVerifying model: {model}")
        if model not in available_names and not any(model in name for name in available_names):
            print(f"  [WARNING] Model {model} is not listed in Ollama. Attempting to pull...")
            try:
                ollama.pull(model)
                print(f"  Successfully pulled {model}")
            except Exception as e:
                print(f"  Failed to pull model: {e}")
                continue
        
        try:
            print("  Sending test prompt...")
            response = ollama.generate(
                model=model,
                prompt="Explain backpropagation in 1 sentence.",
                options={"temperature": 0.0}
            )
            print("  Response:")
            print(f"    {response.get('response', '').strip()}")
            print(f"  Speed: {response.get('eval_count', 0) / (response.get('eval_duration', 1) / 1e9):.2f} tok/s")
        except Exception as e:
            print(f"  Error during inference: {e}")

if __name__ == "__main__":
    verify_models()
