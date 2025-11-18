#!/bin/bash
#
# run_experiments.sh
#
# This script automates running a hyperparameter grid search for the EnrichRAG project.
# It runs experiments in parallel across 8 GPUs, ensures all processes are cleaned up,
# and appends all results to a central markdown report.

# --- Dependency Check ---
if ! command -v jq &> /dev/null
then
    echo "ERROR: 'jq' is not installed. Please install it to parse JSON results."
    echo "e.g., 'sudo apt-get install jq' or 'brew install jq'"
    exit 1
fi

# --- Safety First: Cleanup on Exit ---
# This 'trap' command ensures that when the script exits (normally or via Ctrl+C),
# it kills all child processes in its process group. This is crucial for stopping
# the vllm servers started by the evaluation scripts.
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# --- Configuration ---
# Path to the validation/dev split for each benchmark.
# IMPORTANT: You should tune on a validation set, not the final test set.
# Please adjust these paths if they are different.
declare -A DEV_SETS
DEV_SETS["hotpotqa"]="data/benchmarks/hotpotqa/dev.jsonl"
DEV_SETS["2wikimultihopqa"]="data/benchmarks/2wikimultihopqa/dev.jsonl"
DEV_SETS["musique"]="data/benchmarks/musique/dev.jsonl"
DEV_SETS["popqa"]="data/benchmarks/popqa/dev.jsonl"
DEV_SETS["nq"]="data/benchmarks/nq/dev.jsonl"

# --- Reporting Setup ---
REPORT_FILE="results/ablation_report.md"
TEMP_CONFIG_DIR="configs/temp_experiments"
LOG_DIR="results/logs"
LOCK_FILE="/tmp/ablation_report.lock"

mkdir -p $TEMP_CONFIG_DIR
mkdir -p $LOG_DIR
mkdir -p results
cp configs/base.yaml $TEMP_CONFIG_DIR/base.yaml

# Initialize the report file with a header
echo "# Hyperparameter Ablation Report" > "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
printf "| %-30s | %-10s | %-8s | %-8s | %-8s | %-10s | %-10s | %-10s |\n" \
    "Benchmark" "Max Iter" "k_nodes" "k_edges" "k_docs" "Conf Thr" "EM" "F1" >> "$REPORT_FILE"
printf "|-%-30s-|-%-10s-|-%-8s-|-%-8s-|-%-8s-|-%-10s-|-%-10s-|-%-10s-|" \
    "------------------------------" "----------" "--------" "--------" "--------" "----------" "----------" "----------" >> "$REPORT_FILE"


# --- Helper Function to Run One Experiment ---
run_experiment() {
    # --- Parameters ---
    local gpu_id=$1
    local benchmark=$2
    local max_iter=$3
    local k_nodes=$4
    local k_edges=$5
    local k_docs=$6
    local conf_thresh=$7

    # --- Dynamic Config File Creation ---
    local dev_path=${DEV_SETS[$benchmark]}
    if [ -z "$dev_path" ]; then
        echo "ERROR: Dev set path for benchmark '$benchmark' not found. Skipping."
        return
    fi

    local exp_name="${benchmark}_it${max_iter}_kn${k_nodes}_ke${k_edges}_kd${k_docs}_ct${conf_thresh}"
    local temp_config_path="$TEMP_CONFIG_DIR/${exp_name}.yaml"

    # Generate the temporary config file
    # The base.yaml is already copied to the temp dir
    cp "$TEMP_CONFIG_DIR/base.yaml" "$temp_config_path"
    sed -i.bak "s|dataset_path:.*|dataset_path: \"$dev_path\"|" "$temp_config_path"
    sed -i.bak "s|dataset_loader:.*|dataset_loader: \"load_${benchmark}\"|" "$temp_config_path"
    sed -i.bak "s|max_iterations:.*|max_iterations: $max_iter|" "$temp_config_path"
    sed -i.bak "s|confidence_threshold:.*|confidence_threshold: $conf_thresh|" "$temp_config_path"
    sed -i.bak "s|pcst_k_prize_nodes:.*|pcst_k_prize_nodes: $k_nodes|" "$temp_config_path"
    sed -i.bak "s|pcst_k_prize_edges:.*|pcst_k_prize_edges: $k_edges|" "$temp_config_path"
    sed -i.bak "s|enrich_graph_k_docs:.*|enrich_graph_k_docs: $k_docs|" "$temp_config_path"
    rm "${temp_config_path}.bak"

    # --- Execute the Experiment in a Subshell ---
    (
        export CUDA_VISIBLE_DEVICES=$gpu_id
        echo "Starting $exp_name on GPU $gpu_id..."
        
        # Run the evaluation and redirect output to log files
        python run_evaluation.py --config "$temp_config_path" > "$LOG_DIR/${exp_name}.log" 2> "$LOG_DIR/${exp_name}.err"

        # Find the scores.json file in the latest experiment directory for this config
        local latest_exp_dir=$(find results -type d -name "${exp_name}_*" | sort | tail -n 1)
        local scores_file="$latest_exp_dir/scores.json"

        # --- Reporting ---
        if [ -f "$scores_file" ]; then
            local em_score=$(jq .em "$scores_file")
            local f1_score=$(jq .f1 "$scores_file")
            
            # Use a lock to prevent race conditions when writing to the report
            (
                flock 200
                printf "| %-30s | %-10s | %-8s | %-8s | %-8s | %-10s | %-10.4f | %-10.4f |\n" \
                    "$benchmark" "$max_iter" "$k_nodes" "$k_edges" "$k_docs" "$conf_thresh" "$em_score" "$f1_score" 
                    >> "$REPORT_FILE"
            ) 200>"$LOCK_FILE"
            
            echo "--- Finished experiment: $exp_name on GPU $gpu_id. Results logged. ---"
        else
            echo "--- ERROR: Scores file not found for experiment $exp_name! Looked in $latest_exp_dir ---"
            echo "--- Check logs in $LOG_DIR/${exp_name}.log and $LOG_DIR/${exp_name}.err for details. ---"
        fi
    ) &
}


# --- Main Execution Logic ---
declare -a experiments
# Format: "gpu_id;benchmark;max_iter;k_nodes;k_edges;k_docs;conf_thresh"

# --- Strategy 1: "Deep Dive" (for multi-hop)
experiments+=("0;hotpotqa;7;15;15;5;2.5")
experiments+=("1;2wikimultihopqa;7;15;15;5;2.5")
experiments+=("2;musique;7;15;15;5;2.5")

# --- Strategy 2: "Balanced" (strong baseline for all)
experiments+=("3;hotpotqa;5;10;10;3;3.0")
experiments+=("4;2wikimultihopqa;5;10;10;3;3.0")
experiments+=("5;musique;5;10;10;3;3.0")
experiments+=("6;popqa;5;10;10;3;3.0")
experiments+=("7;nq;5;10;10;3;3.0")

# --- Strategy 3: "Fast" (less computation)
experiments+=("0;hotpotqa;3;5;5;3;3.5") # Reuse GPUs
experiments+=("1;popqa;3;5;5;3;3.5")
experiments+=("2;nq;3;5;5;3;3.5")

# --- Process and Run Experiments in Parallel ---
MAX_PARALLEL_JOBS=8
job_count=0

for exp in "${experiments[@]}"; do
    IFS=';' read -r -a params <<< "$exp"
    run_experiment "${params[0]}" "${params[1]}" "${params[2]}" "${params[3]}" "${params[4]}" "${params[5]}" "${params[6]}"
    
    job_count=$((job_count + 1))
    if (( job_count >= MAX_PARALLEL_JOBS )); then
        echo "--- Reached max parallel jobs ($MAX_PARALLEL_JOBS). Waiting for a job to finish... ---"
        wait -n
        job_count=$((job_count - 1))
    fi
done

echo "--- All experiments launched. Waiting for remaining jobs to complete... ---"
wait
echo "--- All experiments finished. Report saved to $REPORT_FILE ---"
rm -f $LOCK_FILE
