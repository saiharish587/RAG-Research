import os

def create_structure():
    # List of directories to create
    directories = [
        "app",
        "configs",
        "data/raw",
        "data/processed",
        "data/chunks",
        "data/benchmark",
        "documents",
        "embeddings",
        "vector_db",
        "models",
        "rag/no_rag",
        "rag/naive",
        "rag/advanced",
        "rag/modular",
        "evaluation",
        "experiments",
        "results/csv",
        "results/graphs",
        "results/reports",
        "visualization",
        "utils",
        "notebooks",
        "tests"
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  Created: {directory}")

    # List of package directories to initialize with __init__.py
    package_dirs = [
        "app",
        "rag",
        "rag/no_rag",
        "rag/naive",
        "rag/advanced",
        "rag/modular",
        "evaluation",
        "visualization",
        "utils"
    ]

    print("\nInitializing Python package structure (__init__.py)...")
    for pkg_dir in package_dirs:
        init_file = os.path.join(pkg_dir, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w", encoding="utf-8") as f:
                f.write(f'# Package initialization for {pkg_dir.replace("/", ".")}\n')
            print(f"  Created: {init_file}")
        else:
            print(f"  Exists: {init_file}")

    # Create dummy placeholder scripts for RAG modules
    rag_placeholders = {
        "rag/no_rag/retriever.py": "# No RAG implementation (direct queries to LLM)\n",
        "rag/naive/retriever.py": "# Naive RAG implementation (Top-K context injection)\n",
        "rag/advanced/retriever.py": "# Advanced RAG implementation (Query rewriting, hybrid search, reranking)\n",
        "rag/modular/retriever.py": "# Modular RAG implementation (Query routing, context fusion)\n",
        "utils/helpers.py": "# Shared helper functions (logging, file loading)\n"
    }

    print("\nCreating module placeholders...")
    for filepath, content in rag_placeholders.items():
        if not os.path.exists(filepath):
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  Created placeholder: {filepath}")
        else:
            print(f"  Exists: {filepath}")

    print("\nSkeleton setup completed successfully!")

if __name__ == "__main__":
    create_structure()
