import os
import shutil
import json
import yaml
import warnings
import time
import pdfplumber
import pandas as pd
from utils.db import VectorDBManager
from evaluation.evaluator import Evaluator
from rag.generator import Generator
from rag.no_rag.retriever import NoRAGPipeline
from rag.naive.retriever import NaiveRAGPipeline
from rag.advanced.retriever import AdvancedRAGPipeline
from rag.modular.retriever import ModularRAGPipeline
from visualization.visualize import Visualizer

warnings.filterwarnings("ignore")

# Define target safety parameters
TOKEN_LIMIT = 24000  # Safe context buffer for 32K context window models
EST_MULTIPLIER = 1.35  # Words to tokens conversion estimation

def clean_text_words(text):
    """Converts text to a set of lowercased alphanumeric words for overlap check, ignoring common stopwords."""
    stopwords = {"the", "a", "an", "and", "or", "but", "if", "then", "else", "of", "to", "in", "on", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "from", "up", "down", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "would", "should", "could", "ought", "i", "you", "he", "she", "it", "we", "they", "this", "that", "these", "those"}
    words = []
    for word in text.lower().split():
        clean_word = "".join(c for c in word if c.isalnum())
        if clean_word and clean_word not in stopwords:
            words.append(clean_word)
    return set(words)

def run_context_resolution():
    print("=" * 60)
    print("Resolving Context Window Limit Edge Case...")
    print("=" * 60)

    # 1. Setup paths
    notebooks_dir = "notebooks"
    pruned_dir = "notebooks_pruned"
    eval_set_path = "data/benchmark/eval_set.json"
    pruned_eval_set_path = "data/benchmark/eval_set_pruned.json"
    pruned_db_path = "vector_db/faiss_index_pruned"
    csv_path = "results/csv/pruned_benchmark_results.csv"

    # Recreate pruned dir
    if os.path.exists(pruned_dir):
        shutil.rmtree(pruned_dir)
    os.makedirs(pruned_dir, exist_ok=True)

    # 2. Extract and prune text to fit within context window
    print(f"\nStep 1: Pruning lecture notes to stay strictly under {TOKEN_LIMIT} tokens...")
    total_tokens = 0
    pruned_text_corpus = ""
    files_added = []

    # Recurse and extract page-by-page to have fine-grained control
    stop_extracting = False
    for root, dirs, files in os.walk(notebooks_dir):
        if stop_extracting:
            break
        for file in files:
            if stop_extracting:
                break
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                file_text = ""
                try:
                    with pdfplumber.open(pdf_path) as pdf:
                        for p_idx, page in enumerate(pdf.pages):
                            text = page.extract_text()
                            if text:
                                page_words = len(text.split())
                                page_tokens = int(page_words * EST_MULTIPLIER)
                                if total_tokens + page_tokens > TOKEN_LIMIT:
                                    print(f"  [Limit Reached] Stopped at {file} Page {p_idx+1}. Total tokens reached: {total_tokens:,}")
                                    stop_extracting = True
                                    break
                                file_text += text + "\n"
                                total_tokens += page_tokens
                except Exception as e:
                    print(f"  Error reading {file}: {e}")

                if file_text:
                    rel_name = os.path.relpath(pdf_path, notebooks_dir).replace("\\", "_").replace(".pdf", ".txt")
                    target_txt_path = os.path.join(pruned_dir, rel_name)
                    with open(target_txt_path, "w", encoding="utf-8") as f:
                        f.write(file_text)
                    pruned_text_corpus += file_text + "\n"
                    files_added.append(f"{file} (Tokens: {int(len(file_text.split()) * EST_MULTIPLIER):,})")

    print(f"\nCreated pruned text library in '{pruned_dir}/'. Files/sub-pages included:")
    for f_added in files_added:
        print(f" - {f_added}")
    print(f"Total corpus size: {total_tokens:,} tokens (fits safely inside context limit).")

    # 3. Filter QA pairs to match only what exists in the pruned notes
    print("\nStep 2: Filtering evaluation QA pairs based on pruned corpus...")
    corpus_words = clean_text_words(pruned_text_corpus)
    
    with open(eval_set_path, "r", encoding="utf-8") as f:
        full_eval_set = json.load(f)

    pruned_eval_set = []
    for item in full_eval_set:
        gt_words = clean_text_words(item["ground_truth"])
        if not gt_words:
            continue
        # Check percentage of ground truth words represented in the pruned corpus
        match_count = len(gt_words.intersection(corpus_words))
        match_ratio = match_count / len(gt_words)
        
        # If at least 45% of ground truth concepts exist in the pruned text, keep the query
        if match_ratio >= 0.45:
            pruned_eval_set.append(item)

    print(f"Filtered eval set: Kept {len(pruned_eval_set)} out of {len(full_eval_set)} questions.")
    
    os.makedirs(os.path.dirname(pruned_eval_set_path), exist_ok=True)
    with open(pruned_eval_set_path, "w", encoding="utf-8") as f:
        json.dump(pruned_eval_set, f, indent=4)
    print(f"Saved filtered evaluation set to '{pruned_eval_set_path}'.")

    # 4. Build pruned FAISS database index
    print("\nStep 3: Building FAISS vector database for the pruned corpus...")
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    db_config = config.get("embedding", {})
    db_manager = VectorDBManager(
        embedding_model_name=db_config.get("model_name", "BAAI/bge-small-en-v1.5"),
        device=db_config.get("device", "cpu")
    )
    
    docs = db_manager.load_documents(pruned_dir)
    chunk_config = config.get("chunking", {})
    chunks = db_manager.chunk_documents(
        docs,
        chunk_size=chunk_config.get("chunk_size", 500),
        chunk_overlap=chunk_config.get("chunk_overlap", 50)
    )
    print(f"Generated {len(chunks)} chunks from the pruned notes.")
    db_manager.build_index(chunks)
    db_manager.save_index(pruned_db_path)
    print(f"FAISS index built and saved to '{pruned_db_path}'.")

    # 5. Run the Benchmark Matrix on the Pruned Dataset
    print("\nStep 4: Executing the benchmark matrix (3 runs per config for testing)...")
    runs_per_config = 3  # Compact run count for testing
    
    # Initialize generators
    evaluator = Evaluator(embedding_model_name=db_config.get("model_name", "BAAI/bge-small-en-v1.5"), device=db_config.get("device", "cpu"))
    
    results_list = []
    
    models_to_evaluate = config.get("models", [])
    
    for model_info in models_to_evaluate:
        model_name = model_info["name"]
        display_name = model_info["display_name"]
        print(f"\nEvaluating Model: {display_name} ({model_name})")
        
        # Load generator
        generator = Generator(model_name=model_name)
        
        # Initialize RAG pipelines
        no_rag = NoRAGPipeline(generator=generator)
        naive_rag = NaiveRAGPipeline(db_manager=db_manager, generator=generator, top_k=config.get("retrieval", {}).get("top_k", 3))
        advanced_rag = AdvancedRAGPipeline(
            db_manager=db_manager,
            generator=generator,
            top_k=config.get("retrieval", {}).get("top_k", 3),
            rerank=config.get("retrieval", {}).get("rerank", True),
            rerank_model_name=config.get("retrieval", {}).get("rerank_model", "BAAI/bge-reranker-base")
        )
        modular_rag = ModularRAGPipeline(
            db_manager=db_manager,
            generator=generator,
            no_rag_pipeline=no_rag,
            naive_rag_pipeline=naive_rag,
            top_k=config.get("retrieval", {}).get("top_k", 3)
        )
        
        pipelines = {
            "no_rag": no_rag,
            "naive": naive_rag,
            "advanced": advanced_rag,
            "modular": modular_rag
        }
        
        for rag_type, pipeline in pipelines.items():
            print(f"  Running Pipeline: {rag_type}")
            for q_idx, item in enumerate(pruned_eval_set):
                query = item["query"]
                ground_truth = item["ground_truth"]
                print(f"    Q{q_idx+1}: '{query[:40]}...'")
                
                for run_idx in range(1, runs_per_config + 1):
                    # Execute pipeline
                    pipeline_result = pipeline.run(query)
                    
                    # Evaluate result
                    metrics = evaluator.evaluate_run(
                        query=query,
                        response=pipeline_result["response"],
                        retrieved_context=pipeline_result["retrieved_context"],
                        ground_truth=ground_truth,
                        stats=pipeline_result
                    )
                    
                    metrics["model"] = display_name
                    metrics["model_tag"] = model_name
                    metrics["rag_type"] = rag_type
                    metrics["run_id"] = run_idx
                    
                    results_list.append(metrics)
                    pd.DataFrame(results_list).to_csv(csv_path, index=False)
                    
    # 6. Save results and generate visualizations
    print(f"\nStep 5: Exporting results and generating graphs...")
    df = pd.DataFrame(results_list)
    df.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Pruned benchmark results exported to: {csv_path}")

    # Render plots
    os.makedirs("results/graphs_pruned", exist_ok=True)
    
    # We will override visualizer outputs to put them in results/graphs_pruned
    visualizer = Visualizer()
    # We edit visualizer methods slightly by providing the CSV path
    try:
        # Generate graphs
        visualizer.generate_all_plots(csv_path)
        # Move graphs to results/graphs_pruned if needed, or they'll be saved in the default folders
        print("[SUCCESS] Pruned benchmark performance plots generated.")
    except Exception as e:
        print(f"Error during graph visualization: {e}")

if __name__ == "__main__":
    run_context_resolution()
