import os
import json
import yaml
import argparse
import sys
import multiprocessing as mp
from tqdm import tqdm
import pickle
import bm25s  
import Stemmer 

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def parallel_tokenize(text):
    """
    A simple top-level function for multiprocessing to call.
    We'll do stemming here too for consistency.
    """
    stemmer = Stemmer.Stemmer("english")
    return stemmer.stemWords(text.split())

def main():
    print("--- Starting: Build bm25s Index (PARALLEL) ---")
    
    base_config_path = os.path.join(project_root, "configs", "base.yaml")
    try:
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {base_config_path} not found.")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="Build bm25s Index")
    parser.add_argument("--corpus_path", type=str, default=base_config.get('corpus_path'))
    parser.add_argument("--bm25_index_path", type=str, default=base_config.get('bm25_index_path'), 
                        help="Path to save the bm25s index *directory*")
    args = parser.parse_args()
    
    if not all([args.corpus_path, args.bm25_index_path]):
        print("Error: 'corpus_path' or 'bm25_index_path' not defined.")
        sys.exit(1)
        
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
    
    if not doc_texts_for_bm25:
        print("No documents loaded. Exiting.")
        sys.exit(1)

    print(f"Tokenizing and stemming {len(doc_texts_for_bm25)} documents using all CPU cores...")
    num_cores = mp.cpu_count()
    print(f"Using {num_cores} cores...")
    
    with mp.Pool(processes=num_cores) as pool:
        tokenized_corpus = list(tqdm(
            pool.imap(parallel_tokenize, doc_texts_for_bm25, chunksize=1000), 
            total=len(doc_texts_for_bm25),
            desc="Parallel Tokenizing"
        ))
    
    print("Tokenization complete. Initializing bm25s index...")
    
    bm25_indexer = bm25s.BM25() # <-- CORRECTED
    
    print("Indexing... This may take some time.")
    bm25_indexer.index(tokenized_corpus)
    
    print(f"Saving bm25s index to: {args.bm25_index_path}...")
    os.makedirs(args.bm25_index_path, exist_ok=True)
    bm25_indexer.save(args.bm25_index_path)
    
    print("\n--- bm25s Index Build Complete ---")
    print(f"bm25s Index saved to: {args.bm25_index_path}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()