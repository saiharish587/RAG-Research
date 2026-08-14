import os
import json
import urllib.request
import argparse
from typing import Dict, List, Any

HOTPOTQA_DEV_URL = "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/main/hotpot_dev_distractor_v1.json"

def download_hotpotqa(target_path: str = "data/raw/hotpot_dev_distractor_v1.json") -> List[Dict[str, Any]]:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(target_path):
        print(f"Loading cached HotpotQA from {target_path}...")
        with open(target_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    print(f"Downloading HotpotQA Distractor dev set from {HOTPOTQA_DEV_URL}...")
    req = urllib.request.urlopen(HOTPOTQA_DEV_URL)
    data = json.loads(req.read().decode("utf-8"))
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"Successfully saved {len(data)} items to {target_path}")
    return data

def process_hotpotqa(raw_data: List[Dict[str, Any]], sample_size: int = 600) -> None:
    os.makedirs("data/benchmark", exist_ok=True)
    os.makedirs("documents/hotpotqa_corpus", exist_ok=True)
    
    eval_set = []
    seen_titles = set()
    
    for idx, item in enumerate(raw_data):
        qid = item["_id"]
        query = item["question"]
        answer = item["answer"]
        supporting_facts = item["supporting_facts"] # list of [title, sentence_idx]
        context_paras = item["context"] # list of [title, [sentences]]
        
        # Build gold context text
        gold_titles = set(sf[0] for sf in supporting_facts)
        gold_passages = []
        all_passages = []
        
        for title, sentences in context_paras:
            text = f"{title}: " + " ".join(sentences)
            all_passages.append(text)
            if title in gold_titles:
                gold_passages.append(text)
                
            # Write to corpus documents for FAISS index build
            if title not in seen_titles:
                seen_titles.add(title)
                safe_title = "".join(c if c.isalnum() else "_" for c in title)
                doc_path = f"documents/hotpotqa_corpus/{safe_title}.txt"
                with open(doc_path, "w", encoding="utf-8") as df:
                    df.write(text)
                    
        if idx < sample_size:
            eval_set.append({
                "id": qid,
                "query": query,
                "ground_truth": answer,
                "gold_context": gold_passages,
                "all_context": all_passages,
                "supporting_facts": supporting_facts,
                "type": item.get("type", ""),
                "level": item.get("level", "")
            })
            
    eval_set_path = "data/benchmark/eval_set.json"
    with open(eval_set_path, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2)
        
    print(f"Created benchmark eval_set at {eval_set_path} with {len(eval_set)} questions.")
    print(f"Extracted {len(seen_titles)} unique document passages into documents/hotpotqa_corpus/.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=600, help="Number of questions in main eval set")
    args = parser.parse_args()
    
    raw = download_hotpotqa()
    process_hotpotqa(raw, sample_size=args.sample_size)
