import os
import json
import re
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
import txtai

from .llm import LLMAgent
from .graph import PersistentHyperGraph
from .corpus import Corpus

# --- Helper for Tool 4 ---
def parallel_node_tokenize(node_text):
    # This is now obsolete as we are not using bm25s here.
    # Kept for reference, can be removed later.
    from Stemmer import Stemmer
    stemmer = Stemmer("english")
    return stemmer.stemWords(str(node_text).split())

# Tool 1
class EnrichContextTool:
    def __init__(self, graph: PersistentHyperGraph, index_path: str, logger):
        self.graph = graph
        self.log = logger
        self.log(f"[EnrichContextTool] Loading txtai index from: {index_path}")
        try:
            self.index = txtai.Embeddings()
            self.index.load(index_path)
        except Exception as e:
            self.log(f"ERROR: Failed to load txtai index at {index_path}. Error: {e}")
            raise

    def run(self, query: str, k: int, exclude_nodes: set):
        """
        Uses txtai to find prize nodes/edges, then DuckDB to get 1-hop neighbors.
        """
        self.log(f"Running txtai search for query: '{query}' with k={k}")
        
        # 1. Find Prize Nodes and Edges with txtai
        # We ask for more results (k*2) to have a richer set to pull from.
        results = self.index.search(query, limit=k * 2)
        
        prize_nodes = set()
        context_summary = []

        for result in results:
            score = result['score']
            item = self.index.documents[result['id']] # Get full document from index
            
            if item['tags'] == 'type:entity':
                node_id = item['text']
                if node_id not in exclude_nodes:
                    prize_nodes.add(node_id)
                    context_summary.append(f"Entity: {node_id}")
            
            elif item['tags'] == 'type:triple':
                s, r, o = item['subject'], item['relation'], item['object']
                if s not in exclude_nodes:
                    prize_nodes.add(s)
                if o not in exclude_nodes:
                    prize_nodes.add(o)
                context_summary.append(f"Fact: ({s}, {r}, {o})")

        if not prize_nodes:
            self.log("txtai search returned no relevant nodes or edges.")
            return "", set()

        self.log(f"Found {len(prize_nodes)} initial prize nodes from txtai search.")

        # 2. Get 1-Hop Neighbors with DuckDB
        self.log(f"Fetching 1-hop neighbors for {len(prize_nodes)} nodes...")
        neighbor_edges_df = self.graph.get_neighbors(list(prize_nodes))

        if not neighbor_edges_df.empty:
            self.log(f"Found {len(neighbor_edges_df)} neighboring edges.")
            for _, row in neighbor_edges_df.iterrows():
                s, r, o = row['subject'], row['relation'], row['object']
                context_summary.append(f"Fact: ({s}, {r}, {o})")
                prize_nodes.add(s)
                prize_nodes.add(o)
        
        final_context_str = ". ".join(sorted(list(set(context_summary))))
        return final_context_str, prize_nodes

# Tool 2
class CheckSufficiencyTool:
    def __init__(self, llm: LLMAgent, epsilon, logger):
        self.llm = llm
        self.epsilon = epsilon
        self.log = logger

    def run(self, query, context, perplexity_prev):
        prompt = f"Context:\n{context}\n\nQuery: {query}\n\nAnswer:"
        perplexity_curr = self.llm.get_perplexity(prompt)
        info_gain = perplexity_prev - perplexity_curr
        if info_gain < self.epsilon:
            return "CONVERGED", perplexity_curr
        else:
            return "GAINED_INFO", perplexity_curr

# Tool 3
class AnalyzeDecideTool:
    def __init__(self, confidence_threshold, logger):
        self.threshold = confidence_threshold
        self.log = logger

    def run(self, perplexity_curr):
        if perplexity_curr < self.threshold:
            return "CONFIDENT"
        else:
            return "NEEDS_ENRICHMENT"

# Tool 4
class EnrichGraphTool:
    def __init__(self, llm: LLMAgent, corpus: Corpus, logger):
        self.llm = llm
        self.corpus = corpus
        self.log = logger

    def run(self, query: str, context: str, experiment_dir: Path, index_path: Path, k_docs: int):
        # 1. Ask LLM what to search for
        prompt = f"Context:\n{context}\n\nQuery: {query}\n\nBased on the context, what specific information is missing? Be concise."
        expansion_query = self.llm.generate(prompt, max_tokens=50, stop_tokens=["\n"])
        
        # 2. Retrieve new documents from the corpus
        # TODO: Need a way to exclude docs already processed in this run
        new_docs_text, new_doc_ids = self.corpus.retrieve(expansion_query, k=k_docs)
        
        if not new_docs_text:
            self.log("EnrichGraph found no new documents to process.")
            return False

        # 3. Use LLM to extract facts from new docs
        self.log(f"Enriching graph with {len(new_docs_text)} new documents...")
        new_nodes = set()
        new_edges = []
        
        for doc_id, doc_text in zip(new_doc_ids, new_docs_text):
            prompt = f"Extract all knowledge facts from the following text as a JSON list of triples. Each triple should be a dictionary with 'subject', 'relation', and 'object' keys. Example: [{{'subject': 'Paris', 'relation': 'is the capital of', 'object': 'France'}}]\n\nText: {doc_text}"
            facts_str = self.llm.generate(prompt, max_tokens=1024)
            triples = self._parse_llm_triples(facts_str) 
            
            for triple in triples:
                s, r, o = triple['subject'], triple['relation'], triple['object']
                new_nodes.add((s, 'entity'))
                new_nodes.add((o, 'entity'))
                new_edges.append({'subject': s, 'relation': r, 'object': o, 'doc_id': doc_id})

        if not new_edges:
            self.log("EnrichGraph extracted no new facts from the documents.")
            return True # Return true because we did process new docs, even if no facts were found

        # 4. Append new data to experiment's Parquet files
        self.log(f"Appending {len(new_nodes)} new nodes and {len(new_edges)} new edges to Parquet files.")
        
        nodes_df = pd.DataFrame(list(new_nodes), columns=['node_id', 'type'])
        edges_df = pd.DataFrame(new_edges)

        # Use a robust append method
        self._append_to_parquet(experiment_dir / "new_nodes.parquet", nodes_df)
        self._append_to_parquet(experiment_dir / "new_edges.parquet", edges_df)

        # 5. Upsert new data into the experiment's txtai index
        self.log("Upserting new data into txtai index...")
        index = txtai.Embeddings()
        index.load(str(index_path))

        index_data = []
        for _, row in nodes_df.iterrows():
            index_data.append({"id": row['node_id'], "text": row['node_id'], "tags": "type:entity"})
        
        for i, row in edges_df.iterrows():
            text_rep = f"{row['subject']} {row['relation']} {row['object']}"
            index_data.append({
                "id": f"new_triple_{i}_{doc_id}", # More unique ID
                "text": text_rep, "tags": "type:triple", **row.to_dict()
            })
            
        index.upsert(index_data)
        index.save(str(index_path))
        self.log("Index successfully updated and saved.")

        return True

    def _append_to_parquet(self, file_path: Path, df: pd.DataFrame):
        """Appends a DataFrame to a Parquet file."""
        if file_path.exists():
            existing_df = pd.read_parquet(file_path)
            combined_df = pd.concat([existing_df, df]).drop_duplicates(ignore_index=True)
            combined_df.to_parquet(file_path)
        else:
            df.to_parquet(file_path)

    def _parse_llm_triples(self, facts_str: str) -> list:
        """Safely parses a JSON list of triples from the LLM's output string."""
        try:
            # Find the first '[' and the last ']'
            start = facts_str.find('[')
            end = facts_str.rfind(']')
            if start != -1 and end != -1:
                json_str = facts_str[start:end+1]
                return json.loads(json_str)
            return []
        except json.JSONDecodeError:
            self.log(f"Warning: Could not parse triples from LLM output: {facts_str[:100]}...")
            return []