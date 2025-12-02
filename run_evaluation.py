import argparse
import json
from pathlib import Path
from tqdm import tqdm

from enrich_rag import config
from enrich_rag.pipeline import EnrichRAGPipeline
from enrich_rag.benchmarks.loader import load_benchmark_data
from enrich_rag.benchmarks.metrics import evaluate

def main():
    parser = argparse.ArgumentParser(description="Run the EnrichRAG Pipeline on a benchmark.")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the dataset-specific config file (e.g., configs/nq.yaml)"
    )
    args = parser.parse_args()

    try:
        cfg = config.load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("Initializing EnrichRAG Pipeline...")
    pipeline = EnrichRAGPipeline(cfg)

    print(f"Loading benchmark data from: {cfg['dataset_path']}")
    benchmark_data = load_benchmark_data(cfg['dataset_path'], cfg['dataset_loader'])

    pred_path = pipeline.experiment_dir / "predictions.jsonl"
    score_path = pipeline.experiment_dir / "scores.json"

    predictions = []
    print(f"Running pipeline on {len(benchmark_data)} items...")
    print(f"All outputs will be saved in: {pipeline.experiment_dir}")

    with open(pred_path, 'w') as f_out:
        for item in tqdm(benchmark_data, desc="Evaluating Benchmark"):
            query = item['question']
            generated_answer = pipeline.run_query(query)
            
            result_record = {
                "id": item['id'],
                "question": query,
                "answer": generated_answer,
                "gold_answers": item['answers']
            }
            predictions.append(result_record)
            f_out.write(json.dumps(result_record) + '\n')

    print("Calculating scores...")
    scores = evaluate(predictions, benchmark_data, cfg['metrics'])
    
    print(f"\n--- Evaluation Finished ---")
    print(f"Experiment Name: {cfg['experiment_name']}")
    print(json.dumps(scores, indent=2))
    
    with open(score_path, 'w') as f:
        json.dump(scores, f, indent=2)
    print(f"Scores saved to: {score_path}")
    print(f"Run log saved to: {pipeline.experiment_dir / 'run_log.txt'}")
    print(f"--- Run Complete. All artifacts saved in {pipeline.experiment_dir} ---")


if __name__ == "__main__":
    main()