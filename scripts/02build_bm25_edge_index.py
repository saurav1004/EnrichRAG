import os
import json
import yaml
import argparse
import sys
import multiprocessing as mp
from tqdm import tqdm
import bm25s
import Stemmer
import time

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def parallel_tokenize(text):
    """A simple top-level function for multiprocessing to call."""
    stemmer = Stemmer.Stemmer("english")
    return stemmer.stemWords(str(text).split())

def main():
    print("--- Starting: Build bm25s EDGE Index (PARALLEL) ---")
    
    base_config_path = os.path.join(project_root, "configs", "base.yaml")
    try:
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {base_config_path} not found.")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="Build bm25s Edge Index")
    parser.add_argument("--graph_path", type=str, default=base_config.get('graph_path'))
    parser.add_argument("--edge_index_path", type=str, default=base_config.get('edge_index_path'), 
                        help="Path to save the bm25s edge index *directory*")
    args = parser.parse_args()
    
    if not all([args.graph_path, args.edge_index_path]):
        print("Error: 'graph_path' or 'edge_index_path' not defined in configs/base.yaml or args.")
        sys.exit(1)
        
    edge_list = [] # This will map index_id -> {"s": "sub", "r": "rel", "t": "obj"}
    edge_texts_for_bm25 = [] # This is what we will index
    
    print(f"Loading {args.graph_path} (this may take ~15-20 minutes)...")
    try:
        with open(args.graph_path, 'r') as f:
            graph_data = json.load(f) # Load the outer {'graph': ..., 'hyperedges': ...}
        
        print("Graph file loaded. Extracting edges...")
        
        if 'graph' not in graph_data or 'links' not in graph_data['graph']:
            print("Error: 'graph' or 'graph.links' key not found in graph.json.")
            sys.exit(1)
        
        # Iterate over the 'links' (edges) in memory
        for edge in tqdm(graph_data['graph']['links'], desc="Extracting Edges"):
            relation = edge.get('relation', '')
            
            # This is our filter: We only want meaningful OIE triples
            if relation and relation not in ['contains_entity'] and not relation.startswith('in_hyperedge'):
                s = edge.get('source') 
                t = edge.get('target')
                if s and t:
                    # Create the string to be indexed
                    edge_texts_for_bm25.append(f"{s} {relation} {t}")
                    # Create the object for the lookup map
                    edge_list.append({"s": s, "r": relation, "t": t})

    except FileNotFoundError:
        print(f"Error: Graph file not found at {args.graph_path}")
        sys.exit(1)
    except MemoryError:
        print("Error: Ran out of RAM while loading the 29GB graph.json.")
        sys.exit(1)
        
    if not edge_texts_for_bm25:
        print("No meaningful edges (triples) found in the graph. Check your 00_build_seed_graph.py script.")
        sys.exit(1)
        
    print(f"Extracted {len(edge_texts_for_bm25)} meaningful triples.")

    print(f"Tokenizing {len(edge_texts_for_bm25)} triples using all CPU cores...")
    num_cores = mp.cpu_count()
    print(f"Using {num_cores} cores...")
    
    with mp.Pool(processes=num_cores) as pool:
        tokenized_edges = list(tqdm(
            pool.imap(parallel_tokenize, edge_texts_for_bm25, chunksize=1000), 
            total=len(edge_texts_for_bm25),
            desc="Parallel Tokenizing"
        ))
    
    print("Tokenization complete. Initializing bm25s index...")
    
    bm25_edge_indexer = bm25s.BM25()
    
    # Index the tokenized triples
    print("Indexing triples... This may take some time.")
    bm25_edge_indexer.index(tokenized_edges)
    
    print(f"Saving bm25s edge index to: {args.edge_index_path}...")
    os.makedirs(args.edge_index_path, exist_ok=True)
    bm25_edge_indexer.save(args.edge_index_path)
    
    # Save the edge list that maps index IDs to the triple data
    edge_list_path = os.path.join(args.edge_index_path, "edge_list.json")
    with open(edge_list_path, 'w') as f:
        json.dump(edge_list, f)
    print(f"Edge list map saved to: {edge_list_path}")

    print("\n--- bm2s Edge Index Build Complete ---")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()