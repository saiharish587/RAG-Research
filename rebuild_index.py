import os
import shutil
import yaml
import warnings
warnings.filterwarnings("ignore")
from utils.db import VectorDBManager

def rebuild():
    # 1. Load config
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    db_config = config.get("embedding", {})
    db_path = config.get("vector_db", {}).get("path", "vector_db/faiss_index")
    documents_dir = config.get("vector_db", {}).get("documents_dir", "notebooks")
    
    print("=" * 60)
    print(f"Rebuilding FAISS index from documents directory: '{documents_dir}'")
    print("=" * 60)
    
    # 2. Remove old index if exists
    if os.path.exists(db_path):
        print(f"Removing old index at '{db_path}'...")
        shutil.rmtree(db_path)
        
    # 3. Initialize DB Manager
    db_manager = VectorDBManager(
        embedding_model_name=db_config.get("model_name", "BAAI/bge-small-en-v1.5"),
        device=db_config.get("device", "cpu")
    )
    
    # 4. Load documents
    docs = db_manager.load_documents(documents_dir)
    if not docs:
        print("[ERROR] No documents loaded.")
        return
        
    # 5. Chunk documents
    chunk_config = config.get("chunking", {})
    chunks = db_manager.chunk_documents(
        docs,
        chunk_size=chunk_config.get("chunk_size", 500),
        chunk_overlap=chunk_config.get("chunk_overlap", 50)
    )
    print(f"Total chunks generated: {len(chunks)}")
    
    # 6. Build and save index
    db_manager.build_index(chunks)
    db_manager.save_index(db_path)
    print("\n[SUCCESS] FAISS index rebuilt successfully.")

if __name__ == "__main__":
    rebuild()
