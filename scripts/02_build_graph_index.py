import os
import yaml
import argparse
import sys
import pandas as pd
from tqdm import tqdm

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

try:
    import txtai
except ImportError:
    print("Error: Could not import txtai. Please ensure it is installed.")
    sys.exit(1)

def main():
    print("--- Starting Phase 2: Build txtai Graph Index ---")

    # 1. Load config
    base_config_path = os.path.join(project_root, "configs", "base.yaml")
    try:
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {base_config_path} not found.")
        sys.exit(1)

    # 2. Setup Argparser
    parser = argparse.ArgumentParser(description="Build txtai index from Parquet graph files")
    # Define paths using the new structure
    default_base_path = os.path.join(project_root, "data", "graphs", "base")
    parser.add_argument("--nodes_path", type=str, 
                        default=os.path.join(default_base_path, "nodes.parquet"))
    parser.add_argument("--edges_path", type=str, 
                        default=os.path.join(default_base_path, "edges.parquet"))
    parser.add_argument("--index_path", type=str, 
                        default=os.path.join(default_base_path, "graph_index.txtai"))
    args = parser.parse_args()

    # 3. Load Parquet files
    print(f"Loading nodes from: {args.nodes_path}")
    try:
        nodes_df = pd.read_parquet(args.nodes_path)
    except Exception as e:
        print(f"Error loading nodes parquet file: {e}")
        sys.exit(1)

    print(f"Loading edges from: {args.edges_path}")
    try:
        edges_df = pd.read_parquet(args.edges_path)
    except Exception as e:
        print(f"Error loading edges parquet file: {e}")
        sys.exit(1)

    # 4. Prepare data for indexing
    # We will index both the entity nodes and the triples (edges)
    print("Preparing data for txtai indexing...")
    index_data = []

    # Add entity nodes to the index
    # We only index nodes of type 'entity'
    entity_nodes = nodes_df[nodes_df['type'] == 'entity']
    for _, row in tqdm(entity_nodes.iterrows(), total=entity_nodes.shape[0], desc="Processing Nodes"):
        # The document to index is the node_id itself.
        # We store the type as a tag.
        index_data.append({
            "id": row['node_id'],
            "text": row['node_id'],
            "tags": "type:entity"
        })

    # Add edges (triples) to the index
    for _, row in tqdm(edges_df.iterrows(), total=edges_df.shape[0], desc="Processing Edges"):
        # The text is a natural language representation of the triple
        text_representation = f"{row['subject']} {row['relation']} {row['object']}"
        # The ID can be the index of the row
        # We store the structured triple as separate fields for retrieval
        index_data.append({
            "id": f"triple_{_}",
            "text": text_representation,
            "tags": "type:triple",
            "subject": row['subject'],
            "relation": row['relation'],
            "object": row['object'],
            "doc_id": row['doc_id']
        })
        
    if not index_data:
        print("No data to index. Exiting.")
        sys.exit(0)

    # 5. Create and build the txtai index
    # Using "keyword: True" is the idiomatic way to create a sparse-only BM25 index.
    # This implicitly disables vector search components, preventing the download and
    # memory overhead of a default sentence-transformer model.
    print("\nInitializing txtai Embeddings for a sparse-only BM25 index...")
    embeddings = txtai.Embeddings({
        "keyword": True,   # Enable keyword-only index, implicitly disables vectors
        "content": True,   # Enable content storage to retrieve full data
        "bm25": {
            "stopwords": "english",
            "stemmer": "english",
            "tokenizer": "nltk.word_tokenize",
            "terms": True
        }
    })

    print(f"Indexing {len(index_data)} documents with BM25... This may take some time.")
    # The index method for BM25 is CPU-bound and can be memory intensive.
    embeddings.index(index_data)

    # 6. Save the index
    print(f"Saving index to: {args.index_path}")
    embeddings.save(args.index_path)

    print("\n--- Phase 2 Complete ---")
    print(f"txtai graph index saved to: {args.index_path}")

if __name__ == "__main__":
    main()
