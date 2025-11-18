import os
import json
import re
from .llm import LLMAgent
from .graph import PersistentHyperGraph
from .corpus import Corpus
from .tools import EnrichContextTool, CheckSufficiencyTool, AnalyzeDecideTool, EnrichGraphTool

class EnrichRAGPipeline:
    def __init__(self, cfg):
        self.cfg = cfg 
        
        # --- Logging Setup ---
        log_path = os.path.join(self.cfg['experiment_path'], 'run_log.txt')
        self.log_file = open(log_path, 'w')
        
        self._log("[Pipeline] Initializing LLM Agent...")
        self.llm = LLMAgent(cfg['llm_path'], max_input_len=cfg.get('llm_max_input_len', 4096))
        
        self._log("[Pipeline] Initializing Knowledge Graph (Fast Mode)...")
        self.graph = PersistentHyperGraph(
            cfg['graph_path'], 
            cfg['node_index_path'],
            cfg['edge_index_path']
        ) 
        self.graph_is_loaded = False
        
        self._log("[Pipeline] Initializing Corpus...")
        self.corpus = Corpus(cfg['corpus_path'], cfg['bm25_index_path']) 
        
        self._log("[Pipeline] Loading processed docs lookup...")
        with open(cfg['processed_docs_path'], 'r') as f:
            self.processed_docs = json.load(f)
            
        self._log("[Pipeline] Initializing Tools...")
        self.enrich_context = EnrichContextTool(self.graph, self._log)
        self.check_sufficiency = CheckSufficiencyTool(self.llm, cfg['info_gain_epsilon'], self._log)
        self.analyze_decide = AnalyzeDecideTool(cfg['confidence_threshold'], self._log)
        self.enrich_graph = EnrichGraphTool(self.llm, self.corpus, self._log)
        
        self.graph_path = cfg['graph_path'] 
        self.processed_docs_path = cfg['processed_docs_path']
        self.node_index_path = cfg['node_index_path'] 
        self.edge_index_path = cfg['edge_index_path']
        self.chunk_index_path = cfg['chunk_index_path']
        self._log("[Pipeline] EnrichRAG Pipeline is ready.")

    def _log(self, message):
        """Prints to console and writes to log file."""
        print(message)
        self.log_file.write(message + '\n')
        self.log_file.flush()

    def _load_full_graph_object(self):
        """Loads the 29GB graph object, only when needed by Tool 4."""
        self._log("[Pipeline] Tool 4 triggered: Loading full 29GB graph object...")
        self.graph = PersistentHyperGraph(self.graph_path, 
                                          node_index_path=self.node_index_path,
                                          edge_index_path=self.edge_index_path,
                                          skip_index_load=True)
        self.graph_is_loaded = True
        self._log("[Pipeline] Full graph object loaded into RAM.")

    def run_query(self, query):
        self._log(f"\n{'='*20} NEW QUERY {'='*20}")
        self._log(f"QUERY: {query}")
        self._log(f"{'='*50}\n")

        context = []
        retrieved_nodes = set()
        perplexity_prev = float('inf')
        full_context_str = ""
        
        for i in range(self.cfg['max_iterations']):
            self._log(f"--- [Query: {query[:50]}...] --- Iteration {i+1} ---")
            
            new_context_str, new_nodes = self.enrich_context.run(
                query, 
                self.cfg['pcst_k_prize_nodes'],
                self.cfg['pcst_k_prize_edges'],
                retrieved_nodes
            )
            
            if not new_context_str:
                self._log("Tool 1 (EnrichContext): Returned no new info.")
                status = "CONVERGED"
                perplexity_curr = perplexity_prev
            else:
                context.append(new_context_str)
                retrieved_nodes.update(new_nodes)
                full_context_str = "\n".join(context) 
                status, perplexity_curr = self.check_sufficiency.run(
                    query, 
                    full_context_str, 
                    perplexity_prev
                )
            
            self._log(f"Tool 2 (CheckSufficiency): Status={status}, Perplexity={perplexity_curr:.2f} (Prev: {perplexity_prev:.2f})")

            if status == "CONVERGED":
                decision = self.analyze_decide.run(perplexity_curr)
                self._log(f"Tool 3 (AnalyzeDecide): Decision={decision}")
                
                if decision == "CONFIDENT":
                    return self._run_final_generation(query, full_context_str)
                
                else: # NEEDS_ENRICHMENT
                    self._log("Tool 4 (EnrichGraph): Triggered.")
                    
                    if not self.graph_is_loaded:
                        self._load_full_graph_object()
                    
                    enrichment_happened, new_doc_ids = self.enrich_graph.run(
                        query,
                        full_context_str,
                        self.graph,
                        self.processed_docs,
                        self.cfg['enrich_graph_k_docs'],
                        self.node_index_path,
                        self.edge_index_path,
                        self.chunk_index_path
                    )
                    
                    with open(self.processed_docs_path, 'w') as f:
                        json.dump(self.processed_docs, f)

                    if not enrichment_happened:
                        self._log("EnrichGraph failed to find new docs, ending loop.")
                        return self._run_final_generation(query, full_context_str)

                    self._log("[Pipeline] Reloading graph and all indexes after enrichment...")
                    self.graph = PersistentHyperGraph(self.graph_path, self.node_index_path, self.edge_index_path)
                    self.enrich_context.graph = self.graph 
                    self.graph_is_loaded = False

                    context = []
                    retrieved_nodes = set()
                    perplexity_prev = float('inf')
                    full_context_str = ""
                    self._log("Retrying query with enriched graph...")
                    continue 

            perplexity_prev = perplexity_curr

        self._log("Max iterations reached. Generating with current context.")
        return self._run_final_generation(query, full_context_str)

    def _run_final_generation(self, query, context):
        prompt = f"Context:\n{context}\n\nQuery: {query}\n\nBased on the context, what is the final answer?"
        answer = self.llm.generate(prompt, max_tokens=256, stop_tokens=["\n", "Question:"])
        return answer.strip()