import os
import sys
import yaml
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db import VectorDBManager

def test():
    print("Loading config...")
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    db_config = config.get("embedding", {})
    db_manager = VectorDBManager(
        embedding_model_name=db_config.get("model_name", "BAAI/bge-small-en-v1.5"),
        device=db_config.get("device", "cpu")
    )
    
    print("Loading documents...")
    docs = db_manager.load_documents("notebooks")
    print(f"Loaded {len(docs)} documents.")
    
    print("Chunking...")
    chunks = db_manager.chunk_documents(docs)
    print(f"Generated {len(chunks)} chunks.")
    
    # Let's inspect the chunks
    print("Validating chunk text types...")
    for idx, c in enumerate(chunks):
        t = c.get("text")
        if t is None:
            print(f"Error: Chunk {idx} is None")
            continue
        if not isinstance(t, str):
            print(f"Error: Chunk {idx} type is {type(t)}: {repr(t)[:100]}")
            continue
            
    print("All checks completed. Now let's try encoding batch-by-batch to find the exact index...")
    texts = [c["text"] for c in chunks if isinstance(c.get("text"), str)]
    
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            db_manager.model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        except Exception as e:
            print(f"\nFailed on batch starting at index {i}!")
            print(f"Error details: {type(e).__name__}: {e}")
            print("Inspecting batch elements:")
            for j, item in enumerate(batch):
                print(f"Index {i+j} type: {type(item)}, len: {len(item)}")
                try:
                    # try encoding individually
                    db_manager.model.encode([item], show_progress_bar=False)
                except Exception as individual_error:
                    print(f"  --> ELEMENT {i+j} FAILS INDIVIDUALLY!")
                    print(f"  --> Value: {repr(item)}")
                    print(f"  --> Error: {individual_error}")
            sys.exit(1)
            
    print("All batches encoded successfully!")

if __name__ == "__main__":
    test()
