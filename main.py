import os
import json
import argparse
import pandas as pd
import yaml
from utils.db import VectorDBManager
from rag.generator import Generator
from rag.no_rag.retriever import NoRAGPipeline
from rag.naive.retriever import NaiveRAGPipeline
from rag.advanced.retriever import AdvancedRAGPipeline
from rag.modular.retriever import ModularRAGPipeline
from evaluation.evaluator import Evaluator
from visualization.visualize import Visualizer

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def setup_vector_db(config):
    db_config = config.get("embedding", {})
    db_manager = VectorDBManager(
        embedding_model_name=db_config.get("model_name", "BAAI/bge-small-en-v1.5"),
        device=db_config.get("device", "cpu")
    )
    
    db_path = config.get("vector_db", {}).get("path", "vector_db/faiss_index")
    # If index already exists, load it, otherwise build it
    if db_manager.load_index(db_path):
        print("Loaded existing FAISS index.")
    else:
        documents_dir = config.get("vector_db", {}).get("documents_dir", "documents")
        print(f"FAISS index not found. Building index from documents directory: '{documents_dir}'...")
        docs = db_manager.load_documents(documents_dir)
        if not docs:
            print(f"[ERROR] No documents found in '{documents_dir}/' directory. Please place documents there first.")
            return None
        
        chunk_config = config.get("chunking", {})
        chunks = db_manager.chunk_documents(
            docs, 
            chunk_size=chunk_config.get("chunk_size", 500),
            chunk_overlap=chunk_config.get("chunk_overlap", 50)
        )
        db_manager.build_index(chunks)
        db_manager.save_index(db_path)
        
    return db_manager

def main():
    parser = argparse.ArgumentParser(description="SLM-RAG-Benchmark Master Pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--runs", type=int, default=None, help="Number of runs per configuration (overrides config)")
    parser.add_argument("--test-only", action="store_true", help="Run only 1 test question to verify pipeline works")
    args = parser.parse_args()

    # 1. Load config
    config = load_config(args.config)
    runs_per_config = args.runs if args.runs is not None else config.get("experiment", {}).get("runs_per_config", 10)
    
    print("=" * 60)
    print("Initializing SLM-RAG Benchmark Suite")
    print(f"  Runs per configuration: {runs_per_config}")
    print("=" * 60)

    # 2. Setup Vector DB
    db_manager = setup_vector_db(config)
    if db_manager is None:
        return

    # 3. Load Benchmark Dataset
    eval_set_path = "data/benchmark/eval_set.json"
    if not os.path.exists(eval_set_path):
        print(f"[ERROR] Evaluation set not found at {eval_set_path}. Run create_sample_data.py first.")
        return
        
    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)
        
    if args.test_only:
        eval_set = eval_set[:1]
        runs_per_config = 1
        print("  [INFO] Running in test-only mode with 1 question and 1 run.")

    # 4. Setup Evaluator
    evaluator = Evaluator(
        embedding_model_name=config.get("embedding", {}).get("model_name", "BAAI/bge-small-en-v1.5"),
        device=config.get("embedding", {}).get("device", "cpu")
    )

    # 5. Define Evaluation Matrix
    models = config.get("models", [])
    results_list = []
    
    csv_dir = "results/csv"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "benchmark_results.csv")
    
    # Load existing runs to support resumption
    existing_runs = set()
    if os.path.exists(csv_path) and not args.test_only:
        try:
            existing_df = pd.read_csv(csv_path)
            if not existing_df.empty and "model_tag" in existing_df.columns and "rag_type" in existing_df.columns and "query" in existing_df.columns and "run_id" in existing_df.columns:
                for _, row in existing_df.iterrows():
                    key = (str(row["model_tag"]), str(row["rag_type"]), str(row["query"]), int(row["run_id"]))
                    existing_runs.add(key)
                print(f"Loaded {len(existing_runs)} existing benchmark runs from {csv_path}. Resuming mode active.")
                results_list = existing_df.to_dict(orient="records")
        except Exception as e:
            print(f"Warning: Could not read existing results CSV ({e}). Starting fresh.")

    # 6. Execute Benchmark Runs
    for model_info in models:
        model_name = model_info["name"]
        display_name = model_info["display_name"]
        print(f"\nEvaluating Model: {display_name} ({model_name})")
        
        # Initialize generator for this model
        generator = Generator(model_name=model_name)
        
        # Initialize pipelines
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
            
            for q_idx, item in enumerate(eval_set):
                query = item["query"]
                ground_truth = item["ground_truth"]
                
                print(f"    Q{q_idx+1}: '{query[:40]}...'")
                
                for run_idx in range(1, runs_per_config + 1):
                    key = (str(model_name), str(rag_type), str(query), int(run_idx))
                    if key in existing_runs:
                        continue
                        
                    print(f"      Run {run_idx}/{runs_per_config}...")
                    
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
                    
                    # Log metadata
                    metrics["model"] = display_name
                    metrics["model_tag"] = model_name
                    metrics["rag_type"] = rag_type
                    metrics["run_id"] = run_idx
                    
                    results_list.append(metrics)
                    
                    # Save results progressively to prevent data loss on interruption
                    pd.DataFrame(results_list).to_csv(csv_path, index=False)
                    
    # 7. Compile Results and Export
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(csv_path, index=False)
    print(f"\n[SUCCESS] Benchmark results saved to {csv_path}")

    # 8. Generate Visualizations
    visualizer = Visualizer()
    visualizer.generate_all_plots(csv_path)

if __name__ == "__main__":
    main()
