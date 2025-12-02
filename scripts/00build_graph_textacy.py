import os
import json
import yaml
import argparse
from spacy.language import Language
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
from typing import List, Union
import spacy # Make sure spacy is imported at top level

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

try:
    from enrich_rag.graph import PersistentHyperGraph
    import textacy.extract 
except ImportError:
    print("Error: Could not import EnrichRAG modules or textacy/spacy.")
    print("Please run `pip install textacy`")
    sys.exit(1)

USEFUL_ENTITY_LABELS = {
    "PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART", "LOC", "FAC",
}
NUM_WORKERS = 8      
NLP_BATCH_SIZE = 4096 # spaCy is fast, so this can be high
MP_CHUNK_SIZE = 512  

# ---
# HELPER FUNCTIONS (MOVED TO GLOBAL SCOPE)
# ---

def get_span_text(span_or_list: Union[spacy.tokens.Span, List[spacy.tokens.Span]]) -> str:
    """ Converts a spaCy Span or a list of Spans into a single string. """
    if isinstance(span_or_list, list):
        return ", ".join(span.text.strip() for span in span_or_list)
    else:
        return span_or_list.text.strip()

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
                text = re.sub(r'\s+', ' ', doc_text).strip()
                yield (text, str(doc_id))
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed line: {line[:50]}...")
                
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

nlp_worker = None 

def worker_init():
    """Simple initializer for each worker process."""
    global nlp_worker
    nlp_worker = None
    print(f"[Worker {os.getpid()}]: Initialized.")

def worker_loop(worker_id, task_queue, result_queue):
    """
    This is the main function for each spawned worker process.
    It will be pinned to a specific GPU.
    """
    global nlp_worker
    
    import spacy
    import textacy.extract 
    import torch # Import torch to check device count
    
    try:
        # Use device index relative to what's visible
        num_visible_gpus = torch.cuda.device_count()
        if num_visible_gpus == 0:
            raise Exception("No GPUs visible to worker.")
        
        device_id = worker_id % num_visible_gpus
        spacy.require_gpu(device_id) 
        
        print(f"[Worker {worker_id}]: Loading spaCy model on GPU {device_id}...")
        nlp_worker = spacy.load("en_core_web_sm")
        print(f"[Worker {worker_id}]: Model loaded on GPU {device_id}.")
    except Exception as e:
        print(f"FATAL: [Worker {worker_id}] failed to load model. Error: {e}")
        result_queue.put(None) 
        return

    while True:
        try:
            doc_batch = task_queue.get()
            if doc_batch is None:
                result_queue.put(None) 
                break
            
            texts, doc_ids = zip(*doc_batch)
            batch_results = []
            
            # nlp.pipe processes all docs in parallel on the GPU
            for spacy_doc, doc_id in zip(nlp_worker.pipe(texts, batch_size=NLP_BATCH_SIZE), doc_ids):
                chunk_text = " ".join([sent.text for sent in spacy_doc.sents])
                
                entities = list(sorted(set([
                    ent.text.strip() for ent in spacy_doc.ents 
                    if ent.label_ in USEFUL_ENTITY_LABELS
                ])))
                
                triples_generator = textacy.extract.subject_verb_object_triples(spacy_doc)
                triples = []
                for s, v, o in triples_generator:
                    subject_text = get_span_text(s)
                    verb_text = " ".join(t.lemma_ for t in v) # Use lemma for relation
                    object_text = get_span_text(o)
                    triples.append((subject_text, verb_text, object_text))

                if entities or triples:
                    batch_results.append((doc_id, chunk_text, entities, triples))
                else:
                    batch_results.append((doc_id, None, None, None))
            
            result_queue.put(batch_results)
            
        except Exception as e:
            print(f"Error in worker {worker_id} loop: {e}")
            if doc_batch:
                failed_results = [(doc_id, None, None, None) for _, doc_id in doc_batch]
                result_queue.put(failed_results)
            
    print(f"[Worker {worker_id}]: Shutting down.")


def feed_queue(corpus_path, task_queue, num_workers):
    """
    This function now sends BATCHES of DOCUMENTS.
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
    """Top-level tokenizer for node indexing."""
    stemmer = Stemmer.Stemmer("english")
    return stemmer.stemWords(str(node_text).split())

def main():
    print("--- Starting Phase 0: Initial Graph Seeding (PARALLEL with Textacy) ---")
    
    base_config_path = os.path.join(project_root, "configs", "base.yaml")
    try:
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {base_config_path} not found.")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="Build Seed Graph (Parallel)")
    parser.add_argument("--corpus_path", type=str, default=base_config.get('corpus_path'))
    
    parser.add_argument("--graph_name", type=str, default="textacy_graph.json")
    parser.add_argument("--docs_name", type=str, default="textacy_docs.json")
    parser.add_argument("--node_index_name", type=str, default="textacy_bm25s_node_index")

    args = parser.parse_args()
    
    graph_path = os.path.join("data/graphs", args.graph_name)
    docs_path = os.path.join("data/graphs", args.docs_name)
    node_index_path = os.path.join("data/graphs", args.node_index_name)

    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    os.makedirs(os.path.dirname(docs_path), exist_ok=True)
    
    print("Initializing new, empty graph...")
    try:
        graph = PersistentHyperGraph(graph_path, node_index_path=None, skip_index_load=True)
    except TypeError:
        print("Error: graph.py __init__ might be out of date. Trying 4-arg init...")
        graph = PersistentHyperGraph(graph_path, node_index_path=None, edge_index_path=None, skip_index_load=True)
        
    processed_docs = defaultdict(lambda: {"entities": set(), "triples": []})
    
    total_docs = count_lines(args.corpus_path)
    if total_docs == 0:
        print("Corpus file is empty. Exiting.")
        return
    
    manager = mp.Manager()
    task_queue = manager.Queue(maxsize=NUM_WORKERS * 4) # Increased queue size
    result_queue = manager.Queue()
    
    print(f"Starting {NUM_WORKERS} parallel worker processes for NER/OIE...")
    processes = []
    for i in range(NUM_WORKERS):
        p = mp.Process(target=worker_loop, args=(i, task_queue, result_queue))
        p.start()
        processes.append(p)
        
    feeder_process = mp.Process(target=feed_queue, args=(args.corpus_path, task_queue, NUM_WORKERS))
    feeder_process.start()

    print("Main: Waiting for NER/OIE results...")
    workers_done = 0
    
    with tqdm(total=total_docs, desc="Building Seed Graph (Docs)") as pbar:
        while workers_done < NUM_WORKERS:
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
            
            if pbar.n < total_docs:
                pbar.update(len(docs_in_batch)) 

    feeder_process.join()
    for p in processes:
        p.join()
    
    print("\nMain: All OIE workers have finished.")
    
    print("Building final graph from collected triples...")
    for doc_id, data in tqdm(processed_docs.items(), desc="Adding to graph"):
        if data["entities"] or data["triples"]:
            graph.add_chunk_and_facts(doc_id, 0, "", list(data["entities"]), data["triples"])

    print("Saving graph to disk...")
    graph.save()
    with open(docs_path, 'w', encoding='utf-8') as f:
        # Convert sets to lists for JSON
        json_friendly_docs = {doc_id: {"entities": list(data["entities"]), "triples": data["triples"]} 
                              for doc_id, data in processed_docs.items()}
        json.dump(json_friendly_docs, f)
    print(f"Graph and docs lookup saved.")

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
    
    print("Node tokenization complete. Initializing bm25s index...")
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