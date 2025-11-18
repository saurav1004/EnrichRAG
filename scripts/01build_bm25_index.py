import os
import json
import yaml
import argparse
import sys
import multiprocessing as mp
from tqdm import tqdm
import pickle
from rank_bm25 import BM25Okapi

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# --- Top-level function for multiprocessing ---
def parallel_tokenize(text):
    """A simple top-level function for multiprocessing to call."""
    return text.split()

def main():
    print("--- Starting: Build BM25 Index (PARALLEL) ---")
    
    # 1. Load config to get default paths
    base_config_path = os.path.join(project_root, "configs", "base.yaml")
    try:
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {base_config_path} not found.")
        sys.exit(1)
    
    # 2. Setup Argparser
    parser = argparse.ArgumentParser(description="Build BM25 Index")
    parser.add_argument("--corpus_path", type=str, default=base_config.get('corpus_path'))
    parser.add_argument("--bm25_index_path", type=str, default=base_config.get('bm25_index_path'))
    args = parser.parse_args()
    
    if not all([args.corpus_path, args.bm25_index_path]):
        print("Error: 'corpus_path' or 'bm25_index_path' not defined in configs/base.yaml or args.")
        sys.exit(1)

    # 3. Load documents from JSONL
    doc_texts_for_bm25 = []
    print(f"Loading corpus for BM25 build: {args.corpus_path}")
    try:
        with open(args.corpus_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading corpus"):
                item = json.loads(line)
                doc_text = item.get('contents') or item.get('text', '')
                if doc_text:
                    doc_texts_for_bm25.append(doc_text)
    except FileNotFoundError:
        print(f"Error: Corpus file not found at {args.corpus_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON at line: {e}")
        # Continue with what we have
    
    if not doc_texts_for_bm25:
        print("No documents loaded. Exiting.")
        sys.exit(1)

    # 4. Parallel Tokenization
    print(f"Tokenizing {len(doc_texts_for_bm25)} documents using all CPU cores...")
    num_cores = mp.cpu_count()
    print(f"Using {num_cores} cores...")
    
    with mp.Pool(processes=num_cores) as pool:
        tokenized_corpus = list(tqdm(
            pool.imap(parallel_tokenize, doc_texts_for_bm25, chunksize=1000), 
            total=len(doc_texts_for_bm25),
            desc="Parallel Tokenizing"
        ))
    
    print("Tokenization complete. Initializing BM25 index...")
    bm25_index = BM25Okapi(tokenized_corpus)
    
    # 5. Save the index to disk
    print(f"Saving BM25 index to: {args.bm25_index_path}...")
    os.makedirs(os.path.dirname(args.bm25_index_path), exist_ok=True)
    with open(args.bm25_index_path, "wb") as f:
        pickle.dump(bm25_index, f)
    
    print("\n--- BM25 Index Build Complete ---")
    print(f"BM25 Index saved to: {args.bm25_index_path}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()