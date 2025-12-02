import os
import yaml
import argparse
import sys
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
import shutil
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

try:
    import txtai
except ImportError:
    print("Error: Could not import txtai. Please ensure it is installed.")
    sys.exit(1)

# Use a large number of workers to saturate the CPU cores.
# Leave some cores for the OS and other tasks.
NUM_WORKERS = 128

def build_partial_index(args):
    """
    Worker function executed by the multiprocessing pool.
    Builds a small txtai index from a given chunk of data.
    """
    worker_id, data_chunk, tmp_dir = args
    partial_index_path = os.path.join(tmp_dir, f"worker_{worker_id}")
    
    # Short sleep to stagger worker starts and prevent log clutter
    time.sleep(worker_id * 0.1)
    
    # Each worker creates and saves its own small index.
    try:
        embeddings = txtai.Embeddings({
            "keyword": True,
            "content": True,
            "bm25": {
                "stopwords": "english",
                "stemmer": "english",
                "tokenizer": "nltk.word_tokenize",
                "terms": True
            }
        })
        embeddings.index(data_chunk)
        embeddings.save(partial_index_path)
        return partial_index_path
    except Exception as e:
        print(f"ERROR in worker {worker_id}: {e}")
        return None

def main():
    print("--- Starting Phase 2: Build txtai Graph Index (PARALLEL) ---")

    parser = argparse.ArgumentParser(description="Build txtai index in parallel from Parquet graph files")
    default_base_path = os.path.join(project_root, "data", "graphs", "base")
    parser.add_argument("--nodes_path", type=str, default=os.path.join(default_base_path, "nodes.parquet"))
    parser.add_argument("--edges_path", type=str, default=os.path.join(default_base_path, "edges.parquet"))
    parser.add_argument("--index_path", type=str, default=os.path.join(default_base_path, "graph_index.txtai"))
    args = parser.parse_args()

    print("Preparing all data for indexing (this may take a few minutes)...")
    nodes_df = pd.read_parquet(args.nodes_path)
    edges_df = pd.read_parquet(args.edges_path)
    
    index_data = []
    for _, row in tqdm(nodes_df[nodes_df['type'] == 'entity'].iterrows(), total=nodes_df[nodes_df['type'] == 'entity'].shape[0], desc="Preparing Nodes"):
        index_data.append({"id": row['node_id'], "text": row['node_id'], "tags": "type:entity"})
    
    for i, row in tqdm(edges_df.iterrows(), total=edges_df.shape[0], desc="Preparing Edges"):
        text_rep = f"{row['subject']} {row['relation']} {row['object']}"
        index_data.append({"id": f"triple_{i}", "text": text_rep, "tags": "type:triple", "subject": row['subject'], "relation": row['relation'], "object": row['object'], "doc_id": row['doc_id']})

    if not index_data:
        print("No data to index. Exiting.")
        sys.exit(0)

    tmp_index_dir = os.path.join(default_base_path, "tmp_parallel_index")
    print(f"\n--- Starting Map Phase: Building {NUM_WORKERS} partial indexes in parallel in '{tmp_index_dir}' ---")
    if os.path.exists(tmp_index_dir):
        shutil.rmtree(tmp_index_dir)
    os.makedirs(tmp_index_dir)

    data_chunks = np.array_split(index_data, NUM_WORKERS)
    worker_args = [(i, chunk.tolist(), tmp_index_dir) for i, chunk in enumerate(data_chunks)]

    with mp.Pool(NUM_WORKERS) as pool:
        partial_paths = list(tqdm(pool.imap_unordered(build_partial_index, worker_args), total=NUM_WORKERS, desc="Building Partial Indexes"))

    print("\n--- Map Phase Complete ---")
    
    partial_paths = sorted([p for p in partial_paths if p])
    if not partial_paths:
        print("FATAL: All workers failed to build partial indexes. Aborting.")
        sys.exit(1)

    print(f"\n--- Starting Reduce Phase: Merging {len(partial_paths)} indexes ---")
    
    main_index_path = partial_paths.pop(0)
    main_index = txtai.Embeddings()
    print(f"Loading base index from {main_index_path}...")
    main_index.load(main_index_path)

    try:
        for path in tqdm(partial_paths, desc="Merging Indexes"):
            partial_index = txtai.Embeddings()
            partial_index.load(path)
            
            documents_to_upsert = partial_index.search("SELECT * FROM documents")
            
            main_index.upsert(documents_to_upsert)
        
        print(f"\nSaving final merged index to {args.index_path}...")
        main_index.save(args.index_path)
        print("--- Reduce Phase Complete ---")

    finally:
        print("Cleaning up temporary partial indexes...")
        shutil.rmtree(tmp_index_dir)
        print("Cleanup complete.")

    print(f"\n--- Parallel Indexing Complete ---")
    print(f"Final txtai graph index saved to: {args.index_path}")

if __name__ == "__main__":
    # Set start method to 'spawn' for CUDA and general multiprocessing safety
    mp.set_start_method("spawn", force=True)
    main()
