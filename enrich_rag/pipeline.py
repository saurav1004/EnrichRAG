import os
import json
import re
import uuid
import shutil
from pathlib import Path

from .llm import LLMAgent
from .graph import PersistentHyperGraph
from .corpus import Corpus
from .tools import EnrichContextTool, CheckSufficiencyTool, AnalyzeDecideTool, EnrichGraphTool

class EnrichRAGPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.experiment_id = cfg.get('experiment_name', str(uuid.uuid4()))
        
        # --- Directory and Path Setup ---
        project_root = Path(__file__).parent.parent
        self.base_dir = project_root / cfg['base_graph_path']
        self.experiment_dir = project_root / "data" / "graphs" / "experiments" / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        log_path = self.experiment_dir / 'run_log.txt'
        self.log_file = open(log_path, 'w')
        
        self._log(f"--- Initializing New EnrichRAG Run ---")
        self._log(f"Experiment ID: {self.experiment_id}")
        self._log(f"Experiment Directory: {self.experiment_dir}")
        self._log(f"\n--- RUN CONFIGURATION ---\n{json.dumps(self.cfg, indent=2)}\n-------------------------\n")

        base_index_path = self.base_dir / "graph_index.txtai"
        self.exp_index_path = self.experiment_dir / "graph_index.txtai"
        self._log(f"Copying base index from '{base_index_path}' to '{self.exp_index_path}'")
        try:
            shutil.copytree(base_index_path, self.exp_index_path)
        except FileExistsError:
            self._log("Experiment index already exists. Resuming run.")
        except FileNotFoundError:
            self._log(f"ERROR: Base index not found at '{base_index_path}'. Please run scripts/02_build_graph_index.py")
            raise
        except Exception as e:
            self._log(f"ERROR: Failed to copy base index: {e}")
            raise

        self._log("[Pipeline] Initializing LLM Agent...")
        self.llm = LLMAgent(cfg['llm_path'], max_input_len=cfg.get('llm_max_input_len', 4096))
        
        self._log("[Pipeline] Initializing Knowledge Graph with DuckDB...")
        self.graph = PersistentHyperGraph(
            base_dir=str(self.base_dir), 
            experiment_dir=str(self.experiment_dir)
        )
        
        self._log("[Pipeline] Initializing Corpus...")
        self.corpus = Corpus(cfg['corpus_path'], cfg['bm25_index_path']) 
            
        self._log("[Pipeline] Initializing Tools...")
        self.enrich_context = EnrichContextTool(
            self.graph, 
            str(self.exp_index_path), 
            self._log,
            prize_scale_factor=cfg.get('pcst_prize_scale_factor', 1000.0)
        )
        self.check_sufficiency = CheckSufficiencyTool(self.llm, cfg['info_gain_epsilon'], self._log)
        self.analyze_decide = AnalyzeDecideTool(cfg['confidence_threshold'], self._log)
        self.enrich_graph = EnrichGraphTool(self.llm, self.corpus, self._log)
        
        self._log("[Pipeline] EnrichRAG Pipeline is ready.")

    def _log(self, message):
        """Prints to console and writes to log file."""
        print(message)
        self.log_file.write(message + '\n')
        self.log_file.flush()

    def run_query(self, query):
        self._log(f"\n{'='*20} NEW QUERY {'='*20}")
        self._log(f"QUERY: {query}")
        self._log(f"{'='*50}\n")

        context = []
        retrieved_nodes = set()
        processed_doc_ids = set() 
        perplexity_prev = float('inf')
        full_context_str = ""
        
        for i in range(self.cfg['max_iterations']):
            self._log(f"--- [Query: {query[:50]}...] --- Iteration {i+1} ---")
            
            new_context_str, new_nodes = self.enrich_context.run(
                query,
                k_nodes=self.cfg.get('pcst_k_prize_nodes', 5),
                k_hops=self.cfg.get('graph_k_hops', 2),
                exclude_nodes=retrieved_nodes
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
                
                else: 
                    self._log("Tool 4 (EnrichGraph): Triggered.")
                    
                    newly_processed_ids = self.enrich_graph.run(
                        query=query,
                        context=full_context_str,
                        experiment_dir=self.experiment_dir,
                        index_path=self.exp_index_path,
                        k_docs=self.cfg['enrich_graph_k_docs'],
                        exclude_doc_ids=processed_doc_ids
                    )
                    
                    if not newly_processed_ids:
                        self._log("EnrichGraph failed to find new info, ending loop.")
                        return self._run_final_generation(query, full_context_str)

                    processed_doc_ids.update(newly_processed_ids)
                    self._log("Graph enriched. Continuing agent loop.")
                    
            perplexity_prev = perplexity_curr

        self._log("Max iterations reached. Generating with current context.")
        return self._run_final_generation(query, full_context_str)

    def _run_final_generation(self, query, context):
        prompt = f"Context:\n{context}\n\nQuery: {query}\n\nBased on the context, what is the final answer?"
        answer = self.llm.generate(prompt, max_tokens=256, stop_tokens=["\n", "Question:"])
        return answer.strip()