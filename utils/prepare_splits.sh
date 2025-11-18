#!/bin/bash
#
# prepare_splits.sh
#
# This script checks for the existence of test.jsonl and dev.jsonl in each
# benchmark directory. If one is missing, it creates it by splitting the
# existing file into a 90% test / 10% dev ratio.
#
# WARNING: This is a destructive operation. For example, if you only have
# test.jsonl, this script will overwrite it with 90% of its original
# content and use the other 10% to create dev.jsonl.

set -e # Exit immediately if a command exits with a non-zero status.

BENCHMARK_DIR="data/benchmarks"

echo "--- Checking and Preparing Benchmark Data Splits ---"

if [ ! -d "$BENCHMARK_DIR" ]; then
    echo "ERROR: Benchmark directory '$BENCHMARK_DIR' not found."
    exit 1
fi

# Loop through each subdirectory in the benchmarks directory
for BENCHMARK_PATH in "$BENCHMARK_DIR"/*/; do
    # Ensure it's a directory before processing
    if [ ! -d "$BENCHMARK_PATH" ]; then
        continue
    fi

    BENCHMARK_NAME=$(basename "$BENCHMARK_PATH")
    echo "Checking benchmark: $BENCHMARK_NAME"

    TEST_FILE="${BENCHMARK_PATH}test.jsonl"
    DEV_FILE="${BENCHMARK_PATH}dev.jsonl"

    # --- Define the splitting function ---
    # $1: Source file to split
    # $2: New file to create (dev or test)
    # $3: Percentage for the NEW file (e.g., 10 for 10%)
    create_split() {
        local source_file=$1
        local new_file=$2
        local new_file_percent=$3
        local source_file_percent=$((100 - new_file_percent))

        echo "  > Splitting '$source_file'..."
        
        local total_lines=$(wc -l < "$source_file")
        if [ "$total_lines" -lt 10 ]; then
            echo "  > WARNING: Source file has fewer than 10 lines. Cannot split. Copying instead."
            cp "$source_file" "$new_file"
            return
        fi

        local new_file_lines=$((total_lines * new_file_percent / 100))
        local source_file_lines=$((total_lines - new_file_lines))

        # Temporary files for safe operation
        local shuffled_file="${source_file}.shuffled"
        local new_source_content="${source_file}.new_content"

        # Shuffle the original file to ensure a random split
        echo "  > Shuffling $total_lines lines..."
        shuf "$source_file" > "$shuffled_file"

        # Create the new split file (e.g., dev.jsonl)
        echo "  > Creating '$new_file' with $new_file_lines lines ($new_file_percent%)."
        head -n "$new_file_lines" "$shuffled_file" > "$new_file"

        # Create the new, smaller version of the source file
        echo "  > Resizing '$source_file' to $source_file_lines lines ($source_file_percent%)."
        tail -n "$source_file_lines" "$shuffled_file" > "$new_source_content"
        
        # Overwrite the original source file and clean up temporary files
        mv "$new_source_content" "$source_file"
        rm "$shuffled_file"
        
        echo "  > Done."
    }

    # --- Logic to check files and call the function ---
    if [ -f "$TEST_FILE" ] && [ -f "$DEV_FILE" ]; then
        echo "  > Found both test.jsonl and dev.jsonl. OK."
    elif [ -f "$TEST_FILE" ] && [ ! -f "$DEV_FILE" ]; then
        echo "  > Found test.jsonl, but dev.jsonl is missing."
        create_split "$TEST_FILE" "$DEV_FILE" 10 # Create dev.jsonl with 10% of test.jsonl
    elif [ ! -f "$TEST_FILE" ] && [ -f "$DEV_FILE" ]; then
        echo "  > Found dev.jsonl, but test.jsonl is missing."
        create_split "$DEV_FILE" "$TEST_FILE" 90 # Create test.jsonl with 90% of dev.jsonl
    else
        echo "  > ERROR: No test.jsonl or dev.jsonl found for this benchmark. Skipping."
    fi
    echo "" # Newline for readability
done

echo "--- Split check complete. ---"
