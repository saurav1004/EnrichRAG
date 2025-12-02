## Foundational Context & Prior Work

The `context/` directory contains full-repository snapshots of key academic and open-source projects that form the foundational basis for EnrichRAG. Understanding their contributions is crucial to understanding this project's design and goals.

*   **`context/lhrlab-graph-r1-8a5edab282632443.txt` (Graph-R1):** This is the primary baseline method that EnrichRAG aims to outperform. Graph-R1 uses a Reinforcement Learning (RL) agent to learn how to reason over a knowledge graph. EnrichRAG's core hypothesis is that a similar agentic loop can be achieved *without any training* by using a powerful LLM's intrinsic reasoning capabilities, making the system simpler and more adaptable.

*   **`context/ruc-nlpir-flashrag-8a5edab282632443.txt` (FlashRAG):** This project provides the evaluation framework and benchmark datasets. EnrichRAG will adopt the evaluation methodology from FlashRAG to ensure that its performance metrics (like EM and F1 scores) are directly comparable to the current state-of-the-art in the RAG field.

*   **`context/bupt-gamma-pathrag-8a5edab282632443.txt` (PathRAG):** This work is a key influence on the `EnrichContext` tool's design. While EnrichRAG does not use PathRAG directly, its concept of finding "relational paths" inspires EnrichRAG's more sophisticated subgraph retrieval strategy. Instead of simple k-hop lookups, EnrichRAG will use a Prize-Collecting Steiner Tree (PCST) algorithm to find the most relevant and compact subgraph connecting important entities, a technique influenced by the ideas in PathRAG.

*   **`context/neuml-txtai-8a5edab282632443.txt` (txtai):** This is the chosen search and indexing backend for the project. The most critical feature from `txtai` is its support for efficient, incremental index updates via its `upsert` method. This directly solves the core architectural challenge of the `EnrichGraph` tool, which needs to add new information to the knowledge graph's index in real-time without triggering a full, costly re-index.

# Project: EnrichRAG

## Project Overview

EnrichRAG is a Python-based, training-free, agentic Retrieval-Augmented Generation (RAG) pipeline. Its primary goal is to achieve state-of-the-art accuracy on open-domain question-answering benchmarks like NQ and TriviaQA. The project aims to build a persistent, evolving Knowledge Graph from a local document corpus, without the need for model training or external search APIs.

The core of EnrichRAG is an LLM-powered agent that intelligently decides when the knowledge graph is insufficient and what new information to extract from the corpus to "enrich" the graph in real-time.

## Architecture and Design

The project has undergone a significant architectural review to ensure the system is scalable, efficient, and maintainable, particularly concerning the critical requirement for incremental graph and index updates.

### Architectural Evolution

1.  **Initial Choice (`bm25s`):** The project initially used `bm25s` for all text indexing needs, chosen for its speed and low-overhead, Python-native implementation.
2.  **Discovery of Limitation:** We identified that the `EnrichGraph` tool, a core component of the agentic loop, requires the ability to *incrementally* update the search index as new facts are added to the graph. A full re-index on every update is not scalable.
3.  **Investigation:** We confirmed that `bm25s` is designed for static, one-time indexing and does not support an efficient incremental update workflow. This necessitated a change in the architecture.
4.  **Final Decision:** After a thorough discussion and correction of a faulty assumption about DuckDB's native FTS capabilities, the final, user-approved architecture was established.

### Final Architecture: `Parquet + DuckDB + txtai + DVC`

This stack provides a clear separation of concerns and leverages best-in-class tools for each task.

*   **`Parquet` (Data Storage):** All graph nodes and edges will be stored in a partitioned, columnar format using Parquet. This is highly efficient for storage and for analytical queries.
*   **`DuckDB` (Structured Queries):** Used as a high-speed query engine to perform structured SQL queries directly on the Parquet files. Its primary role is fast graph traversal (e.g., fetching 1-hop neighbors) and attribute-based filtering.
*   **`txtai` (Search & Indexing):** The dedicated search and retrieval engine. It is responsible for building a persistent, searchable index of the graph's text content. Crucially, it supports efficient incremental updates via its `upsert` method, solving the core limitation of the previous design.
*   **`DVC` (Data Versioning):** Manages version control for all large data artifacts. This includes the source Parquet files and the derivative `txtai` indexes, ensuring full reproducibility of experiments.

## Project Structure (Target)

This structure reflects the intended state of the repository after the architectural refactoring to `Parquet + DuckDB + txtai + DVC` is complete. It is designed to handle versioning and isolate experimental results.

```
EnrichRAG/
├── configs/
│   ├── base.yaml
│   └── ...
│
├── data/
│   ├── benchmarks/
│   │   └── nq_test.jsonl
│   │
│   ├── corpuses/
│   │   └── wiki18_100w.jsonl
│   │
│   └── graphs/
│       ├── base/
│       │   ├── nodes.parquet/
│       │   ├── edges.parquet/
│       │   └── graph_index.txtai/
│       │
│       └── experiments/
│           └── <experiment_id>/
│               ├── new_nodes.parquet/
│               ├── new_edges.parquet/
│               └── graph_index.txtai/
│
├── scripts/
│   ├── 00_build_graph_rebel.py    # Builds the 'base' graph to data/graphs/base/
│   ├── 01_build_corpus_index.py   # Builds the raw corpus index
│   └── 02_build_graph_index.py    # Builds the 'base' txtai index
│
├── enrich_rag/
│   ├── __init__.py
│   ├── config.py
│   ├── corpus.py
│   ├── graph.py
│   ├── llm.py
│   ├── pipeline.py
│   ├── tools.py
│   └── benchmarks/
│       ├── __init__.py
│       ├── loader.py
│       ├── metrics.py
│       └── utils.py
│
├── run_evaluation.py
└── requirements.txt
```

### Workflow and Structure

Here is a revised file structure and workflow that accounts for DVC and the EnrichGraph tool's behavior.

**1. The New Directory Structure**

I propose we create a base directory for the seed graph and an experiments directory to log the outputs of each run.

```
data/
└── graphs/
    ├── base/
    │   ├── nodes.parquet/
    │   ├── edges.parquet/
    │   └── graph_index.txtai/
    │
    └── experiments/
        ├── <experiment_id_1>/
        │   ├── new_nodes.parquet/
        │   ├── new_edges.parquet/
        │   └── graph_index.txtai/
        │
        └── <experiment_id_2>/
            ├── new_nodes.parquet/
            ├── new_edges.parquet/
            └── graph_index.txtai/
```

**2. How It Works in Practice**

Let's trace how this works during an experiment, for example, with an ID of run_abc.

*   **Initialization**:
    *   Before the experiment starts, the `EnrichRAGPipeline` creates the directory `data/graphs/experiments/run_abc/`.
    *   It then makes a copy of the base `txtai` index from `data/graphs/base/graph_index.txtai/` into `data/graphs/experiments/run_abc/graph_index.txtai/`. This gives the experiment its own private, modifiable index, leaving the base index untouched.

*   **Graph Representation (In `enrich_rag/graph.py`)**:
    *   The `PersistentHyperGraph` object for `run_abc` is initialized with knowledge of two data locations: the `base` directory and its own `run_abc` directory.
    *   When a query needs to be executed (e.g., fetching neighbors), DuckDB is instructed to read from both sets of Parquet files (e.g., `SELECT * FROM 'data/graphs/base/nodes.parquet' UNION ALL SELECT * FROM 'data/graphs/experiments/run_abc/new_nodes.parquet'`). This creates a unified, logical view of the entire graph for the agent.

*   **When `EnrichGraph` is Called**:
    *   The tool extracts new triples from the raw corpus.
    *   These new triples are appended as new rows to the `new_nodes.parquet` and `new_edges.parquet` files within the `data/graphs/experiments/run_abc/` directory.
    *   The `txtai` object, which points to the experiment-specific index, is updated by calling `txtai.upsert()` with the new data. This operation is fast and incremental.

**3. DVC Integration**

This structure makes DVC integration clean and powerful.

*   **Tracking the Base Graph**: The initial `data/graphs/base/` directory (containing the large Parquet files and the base `txtai` index) is tracked by DVC. We would run `dvc add data/graphs/base`, which creates a small `base.dvc` pointer file in Git.
*   **Tracking Experiments**: After `run_abc` is complete, the `data/graphs/experiments/run_abc/` directory contains all the artifacts of that run: the newly generated data and the final state of the `txtai` index. To save this result, you can simply run `dvc add data/graphs/experiments/run_abc`. This makes the entire experiment—and the exact data it generated—fully reproducible.
*   **Promoting a New Base**: If an experiment's enrichments are deemed valuable enough to become part of the new "standard" graph, you would run a separate, offline script to merge the `base` and `run_abc` Parquet files, creating a new `base_v2` directory, which would then be added to DVC.

## Current Project Status: Executing Offline Build

The architectural refactoring is complete. The offline data generation process has now begun on the experiments machine, starting with the graph creation script.

## Execution Workflow & TODO

This is the checklist for running the full pipeline on the **experiments machine**.

1.  **Generate Base Graph**
    *   **Command**: `python scripts/00build_graph_rebel.py`
    *   **Status**: `In Progress`
    *   **Output**: Creates `data/graphs/base/nodes.parquet` and `data/graphs/base/edges.parquet`.

2.  **Build Base Index**
    *   **Command**: `python scripts/02_build_graph_index.py`
    *   **Status**: `Next`
    *   **Output**: Creates the `data/graphs/base/graph_index.txtai/` directory.

3.  **Run Full Evaluation**
    *   **Command**: `python run_evaluation.py --config configs/nq.yaml`
    *   **Status**: `Pending`
    *   **Action**: This will trigger the full agentic pipeline, creating and using an experiment directory under `data/graphs/experiments/`.

4.  **Integrate DVC**
    *   **Status**: `Pending`
    *   **Action**: Once the base graph and index are successfully built, we will revisit setting up the DVC remote and run `dvc add data/graphs/base` to version the baseline data artifacts.

## Development Environment

The project utilizes a two-part development environment:

*   **Local Development:** ONLY Code editing tasks are performed on a MacBook.
*   **Experimentation & Script Execution:** All large-scale experiments and data processing scripts are run on a high-performance b200 node with the following specifications:
    *   **CPUs:** 224 CPU cores
    *   **GPUs:** 8 x NVIDIA GPUs
    *   **GPU Memory:** > 1TB total
    *   **RAM:** 2TB RAM

## Development Conventions

*   **Configuration:** Project settings are managed through `.yaml` files in the `configs/` directory.
*   **Data Storage:** All persistent graph data (nodes, edges) must be stored in partitioned **Parquet** files.
*   **Data Versioning:** All large data artifacts (Parquet files, `txtai` indexes, etc.) must be versioned using **DVC**.
*   **Data Querying:** Structured queries on the graph data (traversals, filtering) should be performed using **DuckDB** on the Parquet files.
*   **Indexing & Search:** Relevance-based text search and incremental indexing must be handled by **`txtai`**.
*   **Modular Design:** The project follows a modular design, with clear separation of concerns between the pipeline, graph, corpus, and tools.
*   **Benchmarking:** The project uses evaluation scripts from the FlashRAG project to ensure results are comparable to baselines.
*   **Open Information Extraction (OIE):** The project is currently evaluating `textacy` and `Babelscape/rebel-large` for building the initial seed graph.

## Gemini Added Memories
- Future experiment idea: Test asymmetric search patterns for PCST prizes. An 'Entity-Centric' strategy uses high k_nodes and low k_edges (e.g., 15/5), potentially good for comparison questions. A 'Relationship-Centric' strategy uses low k_nodes and high k_edges (e.g., 5/15), which could be powerful for multi-hop questions.
- This is a coding-only machine. Do not suggest running experiments, installations, or any commands that are not directly related to code editing. All experiments are run on a separate high-performance machine.

## Past Sessions Context
### DVC & Google Drive Setup Learnings
Attempting to configure DVC with a personal Google Drive account encountered several issues. The setup was postponed, but the key findings from the troubleshooting process are documented here for future reference.

*   **Google Authentication**: The standard OAuth browser-based login flow is blocked by Google for unverified applications like DVC. The correct, non-interactive method is to use a **Google Service Account**.
*   **Service Account Storage Quota**: Service Accounts are identities and do not have their own Google Drive storage quota. They cannot save files to a regular "My Drive" folder. To solve this, the Service Account must be added as a **Content manager** to a **Shared Drive**. The DVC remote must then point to the folder ID of this Shared Drive.
*   **DVC Configuration**: To force DVC to use the service account credentials, the `.dvc/config` file for the remote must be explicitly configured with `gdrive_use_service_account = true`. This is in addition to setting the `gdrive_service_account_json_file_path` to the location of the credentials file.
