import os
import json
import yaml
import argparse
from tqdm import tqdm
import sys
import re
import multiprocessing as mp
from more_itertools import ichunked
from itertools import cycle
import time
import pandas as pd
import pyarrow as pa
from collections import defaultdict

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

try:
    # from enrich_rag.graph import PersistentHyperGraph # No longer needed
    import torch
    import transformers
except ImportError:
    print("Error: Could not import EnrichRAG/torch/transformers.")
    sys.exit(1)

# --- Configuration ---
NUM_WORKERS = 16      
NLP_BATCH_SIZE = 64  
MP_CHUNK_SIZE = 256   

# ---
# HELPER FUNCTIONS
# ---

def stream_documents(corpus_file_path):
    """Reads the JSONL file line by line and yields (doc_text, doc_id)."""
    if not os.path.isfile(corpus_file_path) or not corpus_file_path.endswith('.jsonl'):
        print(f"Error: Corpus path '{corpus_file_path}' is not a valid .jsonl file.")
        return
    with open(corpus_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                doc_text = item.get('contents') or item.get('text', '')
                doc_id = item.get('id') or item.get('doc_id')
                if not doc_text or not doc_id:
                    continue
                yield (doc_text, str(doc_id))
            except json.JSONDecodeError:
                print("Warning: Skipping malformed line")
                
def count_lines(filepath):
    """Helper to get total lines for tqdm progress bar."""
    print("Counting total documents (this may take a minute for a 14GB file)...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, _ in enumerate(f):
                pass
        return i + 1
    except FileNotFoundError:
        return 0

# These globals will hold the model *per process*
oie_model = None
oie_tokenizer = None

def worker_init():
    """Simple initializer for each worker process."""
    global oie_model, oie_tokenizer
    oie_model = None
    oie_tokenizer = None
    print(f"[Worker {os.getpid()}]: Initialized.")

def worker_loop(worker_id, task_queue, result_queue):
    """
    This is the main function for each spawned worker process.
    It will be pinned to a specific GPU.
    """
    global oie_model, oie_tokenizer
    
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
    # Define the helper function inside the worker
    def parse_rebel_output(text):
        triples = []
        # Remove special tokens and split by triplet delimiter
        clean_text = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").strip()
        triplet_texts = clean_text.split("<triplet>")
        
        for triplet_text in triplet_texts:
            triplet_text = triplet_text.strip()
            if not triplet_text:
                continue
                
            try:
                # Find subj and obj markers
                subj_start = triplet_text.index("<subj>") + len("<subj>")
                obj_start = triplet_text.index("<obj>") + len("<obj>")
                
                # The relation is between the start of the string and the subj marker
                relation = triplet_text[:subj_start - len("<subj>")].strip()
                
                # The subject is between subj and obj markers
                subject = triplet_text[subj_start:obj_start - len("<obj>")].strip()
                
                # The object is from the obj marker to the end
                obj = triplet_text[obj_start:].strip()
                
                if subject and relation and obj:
                    triples.append({'subject': subject, 'relation': relation, 'object': obj})
            except ValueError:
                # This can happen if a triplet is malformed (e.g., missing <subj> or <obj>)
                # print(f"Warning: Malformed triplet, skipping: {triplet_text}")
                continue
                
        return triples

    try:
        # 1. Load the REBEL model
        num_visible_gpus = torch.cuda.device_count()
        device_id = worker_id % num_visible_gpus
        device = f"cuda:{device_id}"
        print(f"[Worker {worker_id}]: Loading REBEL-LARGE model on {device}...")

        model_name = 'Babelscape/rebel-large'
        oie_tokenizer = AutoTokenizer.from_pretrained(model_name)
        oie_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            load_in_8bit=True, 
            device_map=device
        )
        
        print(f"[Worker {worker_id}]: Model loaded on {device}.")
    except Exception as e:
        print(f"FATAL: [Worker {worker_id}] failed to load REBEL model. Error: {e}")
        result_queue.put(None) 
        return

    # 2. Start processing loop
    while True:
        try:
            doc_batch = task_queue.get() # This is a batch of DOCS
            if doc_batch is None:
                result_queue.put(None) 
                break
            
            if not doc_batch:
                result_queue.put([])
                continue

            # Simplified logic: process documents directly in batches.
            # The REBEL model was trained on sentences, so we truncate to 512 tokens.
            sub_batch_texts = [d[0] for d in doc_batch]
            sub_batch_doc_ids = [d[1] for d in doc_batch]

            inputs = oie_tokenizer(sub_batch_texts, 
                                      return_tensors="pt", 
                                      padding=True, 
                                      truncation=True, 
                                      max_length=512, # Increased max_length
                                     ).to(device)
            
            generated_ids = oie_model.generate(
                **inputs,
                max_length=512, # Increased max_length
                num_beams=1,
                num_return_sequences=1
            )
            
            decoded_outputs = oie_tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
            
            num_outputs = len(decoded_outputs)
            num_inputs = len(sub_batch_doc_ids)

            if num_outputs != num_inputs:
                print(f"Warning: [Worker {worker_id}] Mismatch! Got {num_outputs} outputs for {num_inputs} inputs. Truncating.")

            batch_results = []
            for i in range(min(num_outputs, num_inputs)):
                doc_id = sub_batch_doc_ids[i]
                output_text = decoded_outputs[i]
                
                triples = []
                entities = set()
                
                parsed_triples = parse_rebel_output(output_text)
                for triple in parsed_triples:
                    s = triple['subject']
                    r = triple['relation']
                    o = triple['object']
                    triples.append((s, r, o))
                    entities.add(s)
                    entities.add(o)
                
                batch_results.append((doc_id, "", list(entities), triples))
            
            result_queue.put(batch_results)
            
        except Exception as e:
            print(f"Error in worker {worker_id} loop: {e}")
            if doc_batch:
                # Send empty lists, not None
                failed_results = []
                for (doc_text, doc_id) in doc_batch:
                    failed_results.append((doc_id, "", [], [])) 
                result_queue.put(failed_results)
            
    print(f"[Worker {worker_id}]: Shutting down.")


def feed_queue(corpus_path, task_queue, num_workers):
    """
    NEW: This function now sends BATCHES of DOCUMENTS.
    """
    print("Main: Feeder process started.")
    corpus_file_path = corpus_path
    doc_generator = stream_documents(corpus_file_path)
    # MP_CHUNK_SIZE is now the number of *documents* per batch
    doc_batches = ichunked(doc_generator, MP_CHUNK_SIZE) 
    for batch in doc_batches:
        task_queue.put(list(batch)) # Put a list of (doc_text, doc_id)
    for _ in range(num_workers):
        task_queue.put(None)
    print("Main: All tasks sent to queue.")

def parallel_node_tokenize(node_text):
    # (This function is unchanged)
    stemmer = Stemmer.Stemmer("english")
    return stemmer.stemWords(str(node_text).split())

# --- Main Execution ---
def main():
    print("--- Starting Phase 0: Initial Graph Seeding (PARALLEL with REBEL-Large) ---")
    
    # 1. Load config
    base_config_path = os.path.join(project_root, "configs", "base.yaml")
    try:
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {base_config_path} not found.")
        sys.exit(1)
    
    # 2. Setup Argparser
    parser = argparse.ArgumentParser(description="Build Seed Graph (Parallel)")
    parser.add_argument("--corpus_path", type=str, default=base_config.get('corpus_path'))
    args = parser.parse_args()
    
    # --- Define output paths ---
    output_dir = os.path.join(project_root, "data", "graphs", "base")
    nodes_path = os.path.join(output_dir, "nodes.parquet")
    edges_path = os.path.join(output_dir, "edges.parquet")

    # 3. Initialize Graph and Lookup
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initializing in-memory lists for nodes and edges...")
    # We will aggregate results in main memory before saving to Parquet
    all_nodes = set()
    all_edges = []
    
    # 4. Get total doc count (This is now for *documents*)
    total_docs = count_lines(args.corpus_path)
    if total_docs == 0:
        print("Corpus file is empty. Exiting.")
        return
    
    # 5. Manual Process Pool Management (REBEL OIE)
    manager = mp.Manager()
    task_queue = manager.Queue(maxsize=NUM_WORKERS * 4) # Increased queue size
    result_queue = manager.Queue()
    
    print(f"Starting {NUM_WORKERS} parallel worker processes for REBEL OIE...")
    processes = []
    for i in range(NUM_WORKERS):
        p = mp.Process(target=worker_loop, args=(i, task_queue, result_queue))
        p.start()
        processes.append(p)
        
    feeder_process = mp.Process(target=feed_queue, args=(args.corpus_path, task_queue, NUM_WORKERS))
    feeder_process.start()

    # 6. Collect results
    print("Main: Waiting for REBEL OIE results...")
    workers_done = 0
    batch_counter = 0
    total_triples = 0
    
    with tqdm(total=total_docs, desc="Building Seed Graph (Docs)") as pbar:
        while workers_done < NUM_WORKERS:
            batch_result = result_queue.get() 
            
            if batch_result is None:
                workers_done += 1
                pbar.write(f"\nMain: Worker {workers_done}/{NUM_WORKERS} finished.")
                continue
            
            batch_counter += 1
            docs_in_batch = 0
            for result_tuple in batch_result:
                doc_id, _, entities, triples = result_tuple
                docs_in_batch += 1
                
                # Add chunk node
                all_nodes.add((doc_id, 'chunk'))
                
                # Add entity nodes
                for entity in entities:
                    all_nodes.add((entity, 'entity'))
                
                # Add edges
                for s, r, o in triples:
                    all_edges.append({
                        'subject': s,
                        'relation': r,
                        'object': o,
                        'doc_id': doc_id
                    })
                    total_triples += 1

            if pbar.n < total_docs:
                pbar.update(docs_in_batch) 

            if batch_counter % 100 == 0:
                pbar.write(f"\n--- Validation Check (Batch {batch_counter}) ---")
                pbar.write(f"Total Triples Extracted: {total_triples}")
                pbar.write(f"Total Unique Nodes: {len(all_nodes)}")
                pbar.write("-------------------------------------\n") 

    # 7. Cleanup workers
    feeder_process.join()
    for p in processes:
        p.join()
    
    print("\nMain: All OIE workers have finished.")
    
    # --- 8. Create and Save DataFrames ---
    print(f"Collected {len(all_nodes)} unique nodes and {len(all_edges)} edges.")
    print("Converting to Pandas DataFrames...")

    # Create Nodes DataFrame
    nodes_df = pd.DataFrame(list(all_nodes), columns=['node_id', 'type'])
    nodes_df.drop_duplicates(subset=['node_id'], inplace=True)
    
    # Create Edges DataFrame
    edges_df = pd.DataFrame(all_edges)

    print(f"Saving nodes to {nodes_path}...")
    nodes_df.to_parquet(nodes_path, engine='pyarrow')
    
    print(f"Saving edges to {edges_path}...")
    edges_df.to_parquet(edges_path, engine='pyarrow')

    print(f"\n--- Phase 0 Complete ---")
    print(f"Nodes saved to: {nodes_path}")
    print(f"Edges saved to: {edges_path}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()