import os
import json
import yaml
import argparse
from tqdm import tqdm
import sys
import multiprocessing as mp
from more_itertools import ichunked
import time
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

try:
    import torch
    import transformers
except ImportError:
    print("Error: Could not import EnrichRAG/torch/transformers.")
    sys.exit(1)

# Optimized Configuration
# Based on a b200 node with 8 GPUs and 224 CPU cores.
# We create a decoupled pipeline: Feeder -> GPU Workers -> CPU Workers -> Aggregator

# GPU Worker Pool: 2 workers per GPU to hide data transfer latency.
NUM_GPU_WORKERS = 16

# CPU Worker Pool: A large pool for the CPU-bound parsing task.
# We leave ample cores for the OS, GPU workers, and feeder process.
NUM_CPU_WORKERS = 128

# MP_CHUNK_SIZE: The number of documents each GPU worker processes in a single batch.
# Increased to better saturate the large VRAM on the GPUs.
MP_CHUNK_SIZE = 512

#
# HELPER FUNCTIONS
#

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

def parse_rebel_output(text):
    """Global helper function to parse raw model output into triples."""
    triples = []
    clean_text = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").strip()
    triplet_texts = clean_text.split("<triplet>")
    
    for triplet_text in triplet_texts:
        triplet_text = triplet_text.strip()
        if not triplet_text:
            continue
        try:
            subj_start = triplet_text.index("<subj>") + len("<subj>")
            obj_start = triplet_text.index("<obj>") + len("<obj>")
            relation = triplet_text[:subj_start - len("<subj>")].strip()
            subject = triplet_text[subj_start:obj_start - len("<obj>")].strip()
            obj = triplet_text[obj_start:].strip()
            if subject and relation and obj:
                triples.append({'subject': subject, 'relation': relation, 'object': obj})
        except ValueError:
            continue
    return triples

#
# WORKER LOOPS
#

def gpu_worker_loop(worker_id, task_queue, raw_output_queue):
    """
    This worker's ONLY job is to perform model inference on a GPU.
    It receives document batches and outputs raw model text.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    try:
        num_visible_gpus = torch.cuda.device_count()
        device_id = worker_id % num_visible_gpus
        device = f"cuda:{device_id}"
        print(f"[GPU Worker {worker_id}]: Loading REBEL-LARGE model on {device}...")

        model_name = 'Babelscape/rebel-large'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map=device
        )
        print(f"[GPU Worker {worker_id}]: Model loaded on {device}.")
    except Exception as e:
        print(f"FATAL: [GPU Worker {worker_id}] failed to load model. Error: {e}")
        return

    while True:
        doc_batch = task_queue.get()
        if doc_batch is None:
            break

        if not doc_batch:
            continue

        sub_batch_texts = [d[0] for d in doc_batch]
        sub_batch_doc_ids = [d[1] for d in doc_batch]

        try:
            inputs = tokenizer(
                sub_batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)

            generated_ids = model.generate(
                **inputs,
                max_length=512,
                num_beams=1,
                num_return_sequences=1
            )

            decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
            raw_output_queue.put((sub_batch_doc_ids, decoded_outputs))

        except Exception as e:
            print(f"Error in GPU worker {worker_id} loop: {e}")
            raw_output_queue.put((sub_batch_doc_ids, []))

    print(f"[GPU Worker {worker_id}]: Shutting down.")

def cpu_worker_loop(worker_id, raw_output_queue, result_queue):
    """
    This worker's ONLY job is to parse the raw text from the GPU workers.
    It is a CPU-bound task.
    """
    print(f"[CPU Worker {worker_id}]: Initialized.")
    while True:
        raw_batch = raw_output_queue.get()
        if raw_batch is None:
            break

        sub_batch_doc_ids, decoded_outputs = raw_batch
        
        num_outputs = len(decoded_outputs)
        num_inputs = len(sub_batch_doc_ids)

        if num_outputs != num_inputs:
            print(f"Warning: [CPU Worker {worker_id}] Mismatch! Got {num_outputs} outputs for {num_inputs} inputs. Truncating.")

        batch_results = []
        for i in range(min(num_outputs, num_inputs)):
            doc_id = sub_batch_doc_ids[i]
            output_text = decoded_outputs[i]
            
            entities = set()
            parsed_triples = parse_rebel_output(output_text)
            
            validated_triples = []
            for triple in parsed_triples:
                s = triple['subject']
                r = triple['relation']
                o = triple['object']
                if "<subj>" in s or "<obj>" in s or "<subj>" in o or "<obj>" in o:
                    continue
                validated_triples.append(triple)
                entities.add(s)
                entities.add(o)
            
            # Convert triples from dict to tuple for consistency
            triples_as_tuples = [(t['subject'], t['relation'], t['object']) for t in validated_triples]
            
            batch_results.append((doc_id, "", list(entities), triples_as_tuples))
        
        result_queue.put(batch_results)

    print(f"[CPU Worker {worker_id}]: Shutting down.")


def feed_queue(corpus_path, task_queue):
    """Feeds the initial task queue with batches of documents."""
    print("Main: Feeder process started.")
    doc_generator = stream_documents(corpus_path)
    doc_batches = ichunked(doc_generator, MP_CHUNK_SIZE)
    for batch in doc_batches:
        task_queue.put(list(batch))
    
    print("Main: Feeder finished. Sending shutdown signals to GPU workers.")
    for _ in range(NUM_GPU_WORKERS):
        task_queue.put(None)

def main():
    print("Starting Phase 0: Initial Graph Seeding (OPTIMIZED PIPELINE)")
    
    base_config_path = os.path.join(project_root, "configs", "base.yaml")
    try:
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {base_config_path} not found.")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="Build Seed Graph (Optimized Parallel)")
    parser.add_argument("--corpus_path", type=str, default=base_config.get('corpus_path'))
    args = parser.parse_args()
    
    output_dir = os.path.join(project_root, "data", "graphs", "base")
    nodes_path = os.path.join(output_dir, "nodes.parquet")
    edges_path = os.path.join(output_dir, "edges.parquet")
    os.makedirs(output_dir, exist_ok=True)
    
    total_docs = count_lines(args.corpus_path)
    if total_docs == 0:
        print("Corpus file is empty. Exiting.")
        return

    manager = mp.Manager()
    # Queue 1: Feeder -> GPU Workers
    task_queue = manager.Queue(maxsize=NUM_GPU_WORKERS * 4)
    # Queue 2: GPU Workers -> CPU Workers
    raw_output_queue = manager.Queue(maxsize=NUM_CPU_WORKERS * 4)
    # Queue 3: CPU Workers -> Main Aggregator
    result_queue = manager.Queue()

    # Start Feeder
    feeder_process = mp.Process(target=feed_queue, args=(args.corpus_path, task_queue))
    feeder_process.start()

    # Start GPU Workers
    gpu_processes = []
    print(f"Starting {NUM_GPU_WORKERS} GPU worker processes...")
    for i in range(NUM_GPU_WORKERS):
        p = mp.Process(target=gpu_worker_loop, args=(i, task_queue, raw_output_queue))
        p.start()
        gpu_processes.append(p)

    # Start CPU Workers
    cpu_processes = []
    print(f"Starting {NUM_CPU_WORKERS} CPU worker processes...")
    for i in range(NUM_CPU_WORKERS):
        p = mp.Process(target=cpu_worker_loop, args=(i, raw_output_queue, result_queue))
        p.start()
        cpu_processes.append(p)

    print("Main: Waiting for parsed results from CPU workers...")
    all_nodes = set()
    all_edges = []
    total_triples = 0
    
    with tqdm(total=total_docs, desc="Building Seed Graph (Docs)") as pbar:
        # We estimate the number of batches to expect.
        # This is for tqdm display and doesn't control the loop.
        expected_batches = (total_docs // MP_CHUNK_SIZE) + 1
        for batch_num in range(expected_batches):
            try:
                # We get results from the final queue
                batch_result = result_queue.get(timeout=600) # 10 min timeout
            except queue.Empty:
                print("\nWarning: Result queue was empty for 10 minutes. Assuming completion.")
                break

            docs_in_batch = 0
            for result_tuple in batch_result:
                doc_id, _, entities, triples = result_tuple
                docs_in_batch += 1
                
                all_nodes.add((doc_id, 'chunk'))
                for entity in entities:
                    all_nodes.add((entity, 'entity'))
                
                for s, r, o in triples:
                    all_edges.append({'subject': s, 'relation': r, 'object': o, 'doc_id': doc_id})
                    total_triples += 1
            
            pbar.update(docs_in_batch)
            
            if batch_num > 0 and batch_num % 100 == 0:
                pbar.write(f"\nValidation Check (Batch {batch_num})")
                pbar.write(f"Total Triples Extracted: {total_triples}")
                pbar.write(f"Total Unique Nodes: {len(all_nodes)}")
                pbar.write("-------------------------------------\n")

    print("\nMain: All batches processed. Cleaning up processes.")

    # Wait for feeder to finish (it will signal GPU workers)
    feeder_process.join()
    
    # Wait for all GPU workers to finish
    for p in gpu_processes:
        p.join()
    print("Main: All GPU workers have shut down.")

    # Now that GPU workers are done, signal CPU workers to shut down
    print("Main: Sending shutdown signals to CPU workers.")
    for _ in range(NUM_CPU_WORKERS):
        raw_output_queue.put(None)

    # Wait for all CPU workers to finish
    for p in cpu_processes:
        p.join()
    print("Main: All CPU workers have shut down.")

    print(f"Collected {len(all_nodes)} unique nodes and {len(all_edges)} edges.")
    print("Converting to Pandas DataFrames...")

    nodes_df = pd.DataFrame(list(all_nodes), columns=['node_id', 'type'])
    nodes_df.drop_duplicates(subset=['node_id'], inplace=True)
    
    edges_df = pd.DataFrame(all_edges)

    print(f"Saving nodes to {nodes_path}...")
    nodes_df.to_parquet(nodes_path, engine='pyarrow')
    
    print(f"Saving edges to {edges_path}...")
    edges_df.to_parquet(edges_path, engine='pyarrow')

    print(f"\nPhase 0 Complete")
    print(f"Nodes saved to: {nodes_path}")
    print(f"Edges saved to: {edges_path}")

if __name__ == "__main__":
    # 'spawn' is essential for CUDA multiprocessing
    mp.set_start_method("spawn", force=True)
    main()
