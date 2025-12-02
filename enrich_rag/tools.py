import os
import json
import re
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
import numpy as np
import txtai
import pcst_fast

from .llm import LLMAgent
from .graph import PersistentHyperGraph
from .corpus import Corpus

def parallel_node_tokenize(node_text):
    from Stemmer import Stemmer
    stemmer = Stemmer("english")
    return stemmer.stemWords(str(node_text).split())

class EnrichContextTool:
    def __init__(self, graph: PersistentHyperGraph, index_path: str, logger, prize_scale_factor: float):
        self.graph = graph
        self.log = logger
        self.prize_scale_factor = prize_scale_factor
        self.log(f"[EnrichContextTool] Loading txtai index from: {index_path}")
        try:
            self.index = txtai.Embeddings()
            self.index.load(index_path)
        except Exception as e:
            self.log(f"ERROR: Failed to load txtai index at {index_path}. Error: {e}")
            raise

    def run(self, query: str, k_nodes: int, k_hops: int, exclude_nodes: set):
        """
        Finds a contextually relevant subgraph using a "narrow and deep" search strategy,
        followed by a PCST algorithm and a robust fallback.
        """
        self.log(f"Running context enrichment for query: '{query}' with k_nodes={k_nodes}, k_hops={k_hops}")

        # Explicitly select all required fields.
        self.log("Tool 1: Performing broad search with txtai...")
        query_str = f"select id, text, tags, subject, object, score from txtai where similar('{query}')"
        results = self.index.search(query_str, limit=k_nodes * 5)
        
        if not results:
            self.log("Tool 1: Initial txtai search returned no results.")
            return "", set()
        
        self.log(f"Tool 1: Broad search returned {len(results)} results.")

        initial_prizes = {}
        
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        top_k_nodes = set()
        for result in sorted_results:
            if len(top_k_nodes) >= k_nodes:
                break

            # Check the 'tags' field to determine if it's a node or a triple
            if 'type:entity' in result.get('tags', ''):
                node_id = result.get('text')
                if node_id and node_id not in exclude_nodes:
                    top_k_nodes.add(node_id)
                    initial_prizes[node_id] = max(initial_prizes.get(node_id, 0), result['score'] * self.prize_scale_factor)
            
            elif 'type:triple' in result.get('tags', ''):
                # If it's a triple, extract the subject and object as potential nodes
                subject = result.get('subject')
                object_node = result.get('object')
                if subject and subject not in exclude_nodes:
                    top_k_nodes.add(subject)
                    initial_prizes[subject] = max(initial_prizes.get(subject, 0), result['score'] * self.prize_scale_factor)
                if object_node and object_node not in exclude_nodes:
                    top_k_nodes.add(object_node)
                    initial_prizes[object_node] = max(initial_prizes.get(object_node, 0), result['score'] * self.prize_scale_factor)

        if not top_k_nodes:
            self.log("Tool 1: Could not identify any valid top-k prize nodes from search results.")
            return "", set()

        self.log(f"Tool 1: Selected {len(top_k_nodes)} clean nodes for deep search: {list(top_k_nodes)}")

        self.log(f"Tool 1: Fetching candidate subgraph ({k_hops}-hop neighborhood)...")
        candidate_edges_df = self.graph.get_neighbors(node_ids=list(top_k_nodes), num_hops=k_hops)

        if candidate_edges_df.empty:
            self.log("Tool 1: No neighboring edges found for top-k prize nodes.")
            node_context = ". ".join([f"Entity: {n}" for n in top_k_nodes])
            return node_context, top_k_nodes

        try:
            final_context_str, final_nodes = self._run_pcst(initial_prizes, candidate_edges_df)
            if final_context_str:
                self.log("Tool 1: PCST strategy successful.")
                return final_context_str, final_nodes
        except Exception as e:
            self.log(f"Tool 1: PCST solver failed with an exception: {e}. Falling back to full subgraph.")

        self.log("Tool 1: PCST failed or returned empty tree. Falling back to full k-hop subgraph.")
        final_context_summary = []
        final_nodes = set(initial_prizes.keys())
        
        for _, row in candidate_edges_df.iterrows():
            s, r, o = row['subject'], row['relation'], row['object']
            final_context_summary.append(f"Fact: ({s}, {r}, {o})")
            final_nodes.add(s)
            final_nodes.add(o)
            
        final_context_str = ". ".join(sorted(list(set(final_context_summary))))
        return final_context_str, final_nodes

    def _run_pcst(self, initial_prizes, candidate_edges_df):
        """Helper function to run the PCST algorithm."""
        self.log("PCST: Preparing data for solver...")
        
        all_subgraph_nodes = pd.unique(candidate_edges_df[['subject', 'object']].values.ravel('K'))
        node_to_int = {node: i for i, node in enumerate(all_subgraph_nodes)}
        
        edges, costs = [], []
        for _, row in candidate_edges_df.iterrows():
            if row['subject'] in node_to_int and row['object'] in node_to_int:
                u, v = node_to_int[row['subject']], node_to_int[row['object']]
                edges.append((u, v))
                costs.append(1.0)

        prizes = np.zeros(len(all_subgraph_nodes))
        for node, prize in initial_prizes.items():
            if node in node_to_int:
                prizes[node_to_int[node]] = prize
        
        self.log(f"PCST: Running solver on a graph of {len(all_subgraph_nodes)} nodes and {len(edges)} edges...")
        root = -1
        valid_prize_nodes = {n: p for n, p in initial_prizes.items() if n in node_to_int}
        if valid_prize_nodes:
            root_node_str = max(valid_prize_nodes, key=valid_prize_nodes.get)
            root = node_to_int[root_node_str]

        if root == -1:
             self.log("PCST: Could not determine a valid root node. Aborting PCST.")
             return None, None

        vertex_indices, edge_indices = pcst_fast.pcst_fast(
            np.array(edges, dtype=np.int32), np.array(prizes, dtype=np.float64),
            np.array(costs, dtype=np.float64), root, 1, "gw", 0
        )

        if not vertex_indices.size > 0 or not edge_indices.size > 0:
            self.log("PCST: Solver returned an empty tree.")
            return None, None

        self.log(f"PCST: Solver returned a tree with {len(vertex_indices)} nodes and {len(edge_indices)} edges.")
        
        final_context_summary = []
        final_nodes = set()
        result_edges_df = candidate_edges_df.iloc[edge_indices]
        for _, row in result_edges_df.iterrows():
            s, r, o = row['subject'], row['relation'], row['object']
            final_context_summary.append(f"Fact: ({s}, {r}, {o})")
            final_nodes.add(s)
            final_nodes.add(o)

        final_context_str = ". ".join(sorted(list(set(final_context_summary))))
        return final_context_str, final_nodes

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

class AnalyzeDecideTool:
    def __init__(self, confidence_threshold, logger):
        self.threshold = confidence_threshold
        self.log = logger

    def run(self, perplexity_curr):
        if perplexity_curr < self.threshold:
            return "CONFIDENT"
        else:
            return "NEEDS_ENRICHMENT"

class EnrichGraphTool:
    def __init__(self, llm: LLMAgent, corpus: Corpus, logger):
        self.llm = llm
        self.corpus = corpus
        self.log = logger

    def run(self, query: str, context: str, experiment_dir: Path, index_path: Path, k_docs: int, exclude_doc_ids: set):
        prompt = f"Context:\n{context}\n\nQuery: {query}\n\nBased on the context, what specific information is missing? Be concise."
        expansion_query = self.llm.generate(prompt, max_tokens=50, stop_tokens=["\n"])
        
        new_docs_text, new_doc_ids = self.corpus.retrieve(
            expansion_query, 
            k=k_docs, 
            exclude_doc_ids=exclude_doc_ids
        )
        
        if not new_docs_text:
            self.log("EnrichGraph found no new documents to process.")
            return []

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
            return new_doc_ids

        self.log(f"Appending {len(new_nodes)} new nodes and {len(new_edges)} new edges to Parquet files.")
        
        nodes_df = pd.DataFrame(list(new_nodes), columns=['node_id', 'type'])
        edges_df = pd.DataFrame(new_edges)

        # Use a robust append method
        self._append_to_parquet(experiment_dir / "new_nodes.parquet", nodes_df)
        self._append_to_parquet(experiment_dir / "new_edges.parquet", edges_df)

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

        return new_doc_ids

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
            # Find the first '[' and the last ']' to extract the JSON array.
            start = facts_str.find('[')
            end = facts_str.rfind(']')
            if start != -1 and end != -1 and start < end:
                json_str = facts_str[start:end+1]
                return json.loads(json_str)
            return []
        except json.JSONDecodeError:
            self.log(f"Warning: Could not parse triples from LLM output: {facts_str[:100]}...")
            return []