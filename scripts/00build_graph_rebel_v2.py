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
import bm25s 
import Stemmer 
from collections import defaultdict

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

try:
    from enrich_rag.graph import PersistentHyperGraph
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
    from collections import defaultdict

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
            
            # --- Re-implement sentence splitting and batching ---
            worker_sentences = [] # List of (sentence, doc_id)
            for (doc_text, doc_id) in doc_batch:
                # Simpler sentence splitting regex
                for sent in re.split(r'(?<=[.?!])\s+', doc_text):
                    if len(sent.strip()) > 10: # Lowered threshold
                        worker_sentences.append( (sent.strip(), doc_id) )

            if not worker_sentences:
                result_queue.put([]) # Send empty result
                continue

            # This dictionary will aggregate results for the whole doc_batch
            aggregated_results = defaultdict(lambda: {"entities": set(), "triples": []})

            for sent_batch in ichunked(worker_sentences, NLP_BATCH_SIZE):
                sent_batch = list(sent_batch)
                if not sent_batch:
                    continue

                sub_batch_texts = [s[0] for s in sent_batch]
                sub_batch_doc_ids = [s[1] for s in sent_batch]

                if not sub_batch_texts:
                    continue

                inputs = oie_tokenizer(sub_batch_texts, 
                                          return_tensors="pt", 
                                          padding=True, 
                                          truncation=True, 
                                          max_length=256,
                                         ).to(device)
                
                generated_ids = oie_model.generate(
                    **inputs,
                    max_length=256, 
                    num_beams=5,
                    num_return_sequences=1
                )
                
                decoded_outputs = oie_tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
                
                num_outputs = len(decoded_outputs)
                num_inputs = len(sub_batch_doc_ids)

                if num_outputs != num_inputs:
                    print(f"Warning: [Worker {worker_id}] Mismatch! Got {num_outputs} outputs for {num_inputs} inputs. Truncating.")

                for i in range(min(num_outputs, num_inputs)):
                    doc_id = sub_batch_doc_ids[i]
                    output_text = decoded_outputs[i]
                    
                    parsed_triples = parse_rebel_output(output_text)
                    for triple in parsed_triples:
                        s = triple['subject']
                        r = triple['relation']
                        o = triple['object']
                        aggregated_results[doc_id]['triples'].append((s, r, o))
                        aggregated_results[doc_id]['entities'].add(s)
                        aggregated_results[doc_id]['entities'].add(o)

            # After processing all sentence batches, format the results
            final_results = []
            for doc_id, data in aggregated_results.items():
                final_results.append((doc_id, "", list(data['entities']), data['triples']))

            result_queue.put(final_results)
            
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
    
    # --- Arguments for unique filenames ---
    parser.add_argument("--graph_name", type=str, default="rebel_large_graph.json")
    parser.add_argument("--docs_name", type=str, default="rebel_large_docs.json")
    parser.add_argument("--node_index_name", type=str, default="rebel_large_bm25s_node_index")

    args = parser.parse_args()
    
    # --- Create full paths ---
    graph_path = os.path.join("data/graphs", args.graph_name)
    docs_path = os.path.join("data/graphs", args.docs_name)
    node_index_path = os.path.join("data/graphs", args.node_index_name)

    # 3. Initialize Graph and Lookup
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    os.makedirs(os.path.dirname(docs_path), exist_ok=True)
    
    print("Initializing new, empty graph...")
    graph = PersistentHyperGraph(graph_path, node_index_path=None, skip_index_load=True, edge_index_path=None)
    # We will aggregate results in main memory before adding to graph
    processed_docs = defaultdict(lambda: {"entities": set(), "triples": []})
    
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

    # 6. Collect results and build graph
    print("Main: Waiting for REBEL OIE results...")
    workers_done = 0
    
    # NEW: The tqdm total is now total_docs
    with tqdm(total=total_docs, desc="Building Seed Graph (Docs)") as pbar:
        while workers_done < NUM_WORKERS:
            # batch_result is a list of tuples from *one* doc_batch
            batch_result = result_queue.get() 
            
            if batch_result is None:
                workers_done += 1
                print(f"Main: Worker {workers_done}/{NUM_WORKERS} finished.")
                continue
            
            # Aggregate results by doc_id
            docs_in_batch = set()
            for result_tuple in batch_result:
                doc_id, chunk_text, entities, triples = result_tuple
                
                # This will no longer crash, as 'entities' is [] not None
                processed_docs[doc_id]["entities"].update(entities)
                processed_docs[doc_id]["triples"].extend(triples)
                docs_in_batch.add(doc_id)
            
            # Update bar by number of *documents* processed
            # This is an estimate, as one doc_batch from the queue
            # doesn't map 1:1 to N docs
            if pbar.n < total_docs:
                pbar.update(len(docs_in_batch))

    # 7. Cleanup workers
    feeder_process.join()
    for p in processes:
        p.join()
    
    print("\nMain: All OIE workers have finished.")
    
    # --- 8. Build the graph from the collected results (NEW STEP) ---
    print("Building final graph from collected triples...")
    for doc_id, data in tqdm(processed_docs.items(), desc="Adding to graph"):
        if data["entities"] or data["triples"]:
            # We don't have the full doc text, so we pass an empty string
            graph.add_chunk_and_facts(doc_id, 0, "", list(data["entities"]), data["triples"])

    # 9. Cache Graph and Lookup
    print("Saving graph to disk...")
    graph.save()
    with open(docs_path, 'w', encoding='utf-8') as f:
        # Convert sets to lists for JSON
        json_friendly_docs = {doc_id: {"entities": list(data["entities"]), "triples": data["triples"]} 
                              for doc_id, data in processed_docs.items()}
        json.dump(json_friendly_docs, f)
    print("Graph and docs lookup saved.")

    # --- 10. OPTIMIZED: Build and save the BM25 *ENTITY NODE* index ---
    print("\n--- Starting BM25 Node Index Build (Optimized) ---")
    
    all_nodes = list(graph.graph.nodes())
    entity_nodes = [
        n for n in all_nodes 
        if graph.graph.nodes[n].get('type') != 'chunk'
    ]
    
    print(f"Tokenizing and stemming {len(entity_nodes)} entity nodes (total nodes: {len(all_nodes)})...")
    num_cores = mp.cpu_count()
    print(f"Using {num_cores} cores...")
    
    with mp.Pool(processes=num_cores) as pool:
        tokenized_nodes = list(tqdm(
            pool.imap(parallel_node_tokenize, entity_nodes, chunksize=1000),
            total=len(entity_nodes),
            desc="Parallel Node Tokenizing"
        ))
    
    print("Node tokenization complete. Initializing bm2s index...")
    bm25_node_indexer = bm25s.BM25()
    
    print("Indexing nodes... This may take some time.")
    bm25_node_indexer.index(tokenized_nodes)
    
    print(f"Saving bm25s node index to: {node_index_path}...")
    os.makedirs(node_index_path, exist_ok=True)
    bm25_node_indexer.save(node_index_path)
    
    node_list_path = os.path.join(node_index_path, "node_list.json")
    with open(node_list_path, 'w') as f:
        json.dump(entity_nodes, f)
    print(f"Node list saved to: {node_list_path}")

    print(f"\n--- Phase 0 Complete ---")
    print(f"Seed Graph saved to: {graph_path}")
    print(f"Processed Docs lookup saved to: {docs_path}")
    print(f"BM25 Node Index saved to: {node_index_path}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
