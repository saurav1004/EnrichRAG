import os
import json
import re
from tqdm import tqdm
import bm25s  
import Stemmer 
import time
import pickle

class Corpus:
    def __init__(self, corpus_path, bm25_index_path):
        """
        Initializes the Corpus by loading the pre-built bm25s index
        and the document lookup dictionary.
        """
        self.doc_lookup = {}
        self.doc_id_map = [] # Holds doc_ids in the exact order of the index
        
        print(f"[Corpus] Loading pre-built bm25s index from: {bm25_index_path}...")
        start_time = time.time()
        try:
            self.searcher = bm25s.BM25.load(bm25_index_path, mmap=True)
            
            # Initialize the stemmer for query tokenization
            self.stemmer = Stemmer.Stemmer("english")
            
            print(f"[Corpus] bm25s index loaded successfully. (Took {time.time() - start_time:.2f}s)")
        except FileNotFoundError:
            print(f"Error: bm25s index file not found at {bm25_index_path}")
            print("Please run 'scripts/01_build_bm25_index.py' first.")
            raise

        print(f"[Corpus] Loading doc texts & ID map from: {corpus_path}...")
        start_time = time.time()
        
        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Building corpus lookup"): 
                    try:
                        item = json.loads(line)
                        doc_text = item.get('contents') or item.get('text', '')
                        doc_id = str(item.get('id') or item.get('doc_id'))
                        if doc_text and doc_id:
                            self.doc_lookup[doc_id] = doc_text
                            self.doc_id_map.append(doc_id)
                    except json.JSONDecodeError:
                        continue 
        except FileNotFoundError:
            print(f"Error: Corpus file not found at {corpus_path}")
            raise
        
        print(f"[Corpus] Corpus lookups built. (Took {time.time() - start_time:.2f}s)")
        print(f"[Corpus] Total documents loaded in lookup: {len(self.doc_lookup)}")
        print("Corpus module ready.")

    def retrieve(self, query, k, exclude_doc_ids):
        tokenized_query = self.stemmer.stemWords(query.split())
        
        retrieve_k = k * 5 if k * 5 > 100 else 100 
        
        try:
            # We must wrap the single query in a list for the batch retrieval API
            doc_indices, doc_scores = self.searcher.retrieve([tokenized_query], k=retrieve_k)
            doc_indices = doc_indices[0]
        except Exception as e:
            self.log(f"Error during bm25s search: {e}")
            return [], []

        top_doc_ids = []
        for doc_index in doc_indices:
            if doc_index < len(self.doc_id_map):
                doc_id = self.doc_id_map[doc_index] # Map index back to string ID
                if doc_id not in exclude_doc_ids:
                    top_doc_ids.append(doc_id)
            if len(top_doc_ids) >= k:
                break
        
        return [self.doc_lookup[doc_id] for doc_id in top_doc_ids], top_doc_ids