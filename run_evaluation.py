import argparse
import json
import os
from tqdm import tqdm

from enrich_rag import config
from enrich_rag.pipeline import EnrichRAGPipeline
from enrich_rag.benchmarks.loader import load_benchmark_data
from enrich_rag.benchmarks.metrics import evaluate

def main():
    # 1. Setup Argparse
    parser = argparse.ArgumentParser(description="Run the EnrichRAG Pipeline on a benchmark.")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the dataset-specific config file (e.g., configs/nq.yaml)"
    )
    args = parser.parse_args()

    # 2. Load Configuration
    # This loads and merges base.yaml + nq.yaml
    try:
        cfg = config.load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 3. Initialize Pipeline
    print("Initializing EnrichRAG Pipeline...")
    pipeline = EnrichRAGPipeline(cfg) # We'll pass the config dict

    # 4. Load Benchmark Data
    print(f"Loading benchmark data from: {cfg['dataset_path']}")
    benchmark_data = load_benchmark_data(cfg['dataset_path'], cfg['dataset_loader'])

    # 5. Run Pipeline
    predictions = []
    print(f"Running pipeline on {len(benchmark_data)} items...")
    
    for item in tqdm(benchmark_data, desc="Evaluating Benchmark"):
        query = item['question']
        generated_answer = pipeline.run_query(query)
        
        predictions.append({
            "id": item['id'],
            "question": query,
            "answer": generated_answer,
            "gold_answers": item['answers'] # Save gold for easier analysis
        })
    
    # 6. Save Predictions
    pred_path = os.path.join(cfg['experiment_path'], "predictions.json")
    with open(pred_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to: {pred_path}")

    # 7. Evaluate Results
    print("Calculating scores...")
    scores = evaluate(predictions, benchmark_data, cfg['metrics'])
    
    print(f"\n--- Evaluation Finished ---")
    print(json.dumps(scores, indent=2))
    
    # Save scores
    score_path = os.path.join(cfg['experiment_path'], "scores.json")
    with open(score_path, 'w') as f:
        json.dump(scores, f, indent=2)
    print(f"Scores saved to: {score_path}")

if __name__ == "__main__":
    main()