import os
import json
import re
import multiprocessing as mp
from tqdm import tqdm
import bm25s
import Stemmer
from .llm import LLMAgent
from .graph import PersistentHyperGraph
from .corpus import Corpus

# --- NEW Helper for Tool 4 ---
def parallel_node_tokenize(node_text):
    stemmer = Stemmer.Stemmer("english")
    return stemmer.stemWords(str(node_text).split())

def parallel_edge_tokenize(edge_text):
    stemmer = Stemmer.Stemmer("english")
    return stemmer.stemWords(str(edge_text).split())

# Tool 1
class EnrichContextTool:
    def __init__(self, graph: PersistentHyperGraph, logger):
        self.graph = graph
        self.log = logger

    def run(self, query, k_nodes, k_edges, exclude_nodes):
        prizes_dict, prize_edges = self.graph.get_nodes_and_edges_by_bm25(query, k_nodes, k_edges, exclude_nodes)
        if not prizes_dict:
            return "", set() 
            
        subgraph, subgraph_hyperedges = self.graph.get_subgraph_by_pcst(prizes_dict, prize_edges)
        
        context_str = self.graph.to_text_summary(subgraph, subgraph_hyperedges)
        retrieved_nodes = set(subgraph.nodes())
        return context_str, retrieved_nodes

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

    def run(self, query, context, graph: PersistentHyperGraph, processed_docs, k_docs,
                node_index_path, edge_index_path, chunk_index_path):        
        prompt = f"Context:\n{context}\n\nQuery: {query}\n\nBased on the context, what specific information is missing? Be concise."
        expansion_query = self.llm.generate(prompt, max_tokens=50, stop_tokens=["\n"])
        
        exclude_doc_ids = set(processed_docs.keys())
        new_docs_text, new_doc_ids = self.corpus.retrieve(expansion_query, k=k_docs, exclude_doc_ids=exclude_doc_ids)
        
        if not new_docs_text:
            self.log("EnrichGraph found no new documents to process.")
            return False, [] 

        self.log(f"Enriching graph with {len(new_docs_text)} new documents...")
        newly_added_nodes = False
        new_edges = []
        
        for doc_id, doc_text in zip(new_doc_ids, new_docs_text):
            prompt = f"Extract all knowledge facts (triples and hyperedges) from the following text as a JSON list:\n\n{doc_text}"
            facts_str = self.llm.generate(prompt, max_tokens=1024)
            facts = self._parse_llm_facts(facts_str) 
            
            if facts:
                graph.add_deep_facts(doc_id, facts)
                newly_added_nodes = True
                
                for fact in facts:
                    if fact.get('type') == 'triple':
                        s, r, t = fact['nodes']
                        new_edges.append(f"{s} {r} {t}")

        if not newly_added_nodes:
            self.log("EnrichGraph extracted no new facts.")
            return True, new_doc_ids 

        self.log("[EnrichGraph] Re-building all BM25 indexes...")
        
        all_nodes = list(graph.graph.nodes())
        entity_nodes = [n for n in all_nodes if graph.graph.nodes[n].get('type') != 'chunk']
        
        all_edge_texts = []
        all_edge_list = []
        for u, v, data in graph.graph.edges(data=True):
            relation = data.get('relation', '')
            if relation and relation not in ['contains_entity'] and not relation.startswith('in_hyperedge'):
                all_edge_texts.append(f"{u} {relation} {v}")
                all_edge_list.append({"s": u, "r": relation, "t": v})
        
        num_cores = mp.cpu_count()
        
        self.log(f"Tokenizing {len(entity_nodes)} nodes and {len(all_edge_texts)} edges...")
        with mp.Pool(processes=num_cores) as pool:
            tokenized_entity_nodes = list(tqdm(pool.imap(parallel_node_tokenize, entity_nodes, chunksize=1000), total=len(entity_nodes), desc="Re-indexing Entities"))
            tokenized_edge_nodes = list(tqdm(pool.imap(parallel_edge_tokenize, all_edge_texts, chunksize=1000), total=len(all_edge_texts), desc="Re-indexing Edges"))

        self.log("Saving new node index...")
        bm25_node_indexer = bm25s.BM25()
        bm25_node_indexer.index(tokenized_entity_nodes)
        os.makedirs(node_index_path, exist_ok=True)
        bm25_node_indexer.save(node_index_path)
        node_list_path = os.path.join(node_index_path, "node_list.json")
        with open(node_list_path, 'w') as f:
            json.dump(entity_nodes, f)

        self.log("Saving new edge index...")
        bm25_edge_indexer = bm25s.BM25()
        bm25_edge_indexer.index(tokenized_edge_nodes)
        os.makedirs(edge_index_path, exist_ok=True)
        bm25_edge_indexer.save(edge_index_path)
        edge_list_path = os.path.join(edge_index_path, "edge_list.json")
        with open(edge_list_path, 'w') as f:
            json.dump(all_edge_list, f)

        self.log("New indexes saved.")

        graph.save()
        return True, new_doc_ids 

    def _parse_llm_facts(self, facts_str):
        try:
            match = re.search(r'\[.*\]', facts_str, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return []
        except json.JSONDecodeError:
            self.log(f"Warning: Could not parse facts: {facts_str[:100]}...")
            return []