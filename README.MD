# EnrichRAG: An Agentic, Training-Free RAG Pipeline

## 1.  Project Goal

**EnrichRAG** is a novel, training-free, agentic RAG pipeline. Its goal is to achieve state-of-the-art accuracy on complex, open-domain question-answering benchmarks (like NQ and TriviaQA) without the high cost of model training (like in Graph-R1) or reliance on external search APIs.

The core idea is to build a **persistent, evolving Knowledge Graph** *exclusively* from a local document corpus. An LLM-powered agent intelligently decides *when* the graph is insufficient and *what* new information to extract from the corpus to "enrich" the graph, improving its knowledge just-in-time.

## 2. Foundational Context & Prior Work

The design and goals of EnrichRAG were not created in a vacuum. They are a direct response to, and iteration upon, the current state-of-the-art in Graph-based Retrieval-Augmented Generation (GraphRAG). For any new engineer joining this project, understanding the limitations and design patterns of these foundational works is essential. The following files, located in the `/context` directory, were crucial in shaping our architecture and identifying our core research contribution.

## 3. Architecture: `Parquet + DuckDB + txtai + DVC`

This stack provides a clear separation of concerns and leverages best-in-class tools for each task.

*   **`Parquet` (Data Storage):** All graph nodes and edges are stored in a partitioned, columnar format. This is highly efficient for storage and for analytical queries.
*   **`DuckDB` (Structured Queries):** Used as a high-speed query engine to perform structured SQL queries directly on the Parquet files. Its primary role is fast graph traversal (e.g., fetching 1-hop neighbors) and attribute-based filtering.
*   **`txtai` (Search & Indexing):** The dedicated search and retrieval engine. It is responsible for building a persistent, searchable index of the graph's text content. Crucially, it supports efficient incremental updates via its `upsert` method, solving the core limitation of the previous design.
*   **`DVC` (Data Versioning):** Manages version control for all large data artifacts. This includes the source Parquet files and the derivative `txtai` indexes, ensuring full reproducibility of experiments.

## 4. The EnrichRAG Pipeline (The Goal)

The pipeline is an "online" (inference-time) agent loop that has access to four distinct tools.

*   **Phase 0: Offline Seed Graph:** A one-time, offline script builds a "shallow but wide" heterogeneous graph from the entire document corpus.
*   **Phase 1: Online Agent Loop:** For each query, the agent runs a loop to gather context using the tools below.
*   **Phase 2: Final Answer:** The agent generates an answer based on the context it gathered.

### The 4 Agent Tools (Theoretical Design)

1.  ** Tool 1: `EnrichContext` (The "Graph Read" Tool)**
    *   **Goal:** Retrieve a small, relevant subgraph from the *current* graph.
    *   **Algorithm (The "Hybrid Search PCST"):**
        1.  **Find Entry Points:** The agent runs a search query against the **`txtai` index** to find the most relevant nodes and edges.
        2.  **Get Neighbors:** The IDs of the top results are fed to **`DuckDB`**, which runs a high-speed SQL query over the **Parquet** files to retrieve the full 1-hop neighborhood of these initial results.
        3.  **Solve PCST:** A **Prize-Collecting Steiner Tree (PCST)** solver finds the most optimal, connected subgraph from the combined results.

2.  ** Tool 2: `CheckSufficiency` (The "Information-Theoretic Trigger")**
    *   **Goal:** Decide if the agent should continue retrieving or if the information has "converged."
    *   **Algorithm (Perplexity Convergence):** The agent calculates the **perplexity** (LLM "confidence") of a hypothetical answer. If adding more context doesn't lower the perplexity, the information has "converged."

3.  ** Tool 3: `Analyze & Decide` (The "Confidence Check")**
    *   **Goal:** Decide if the converged context is *good enough* to answer.
    *   **Algorithm (Confidence Threshold):** If the final perplexity is low enough (e.g., `< 3.0`), the agent proceeds to the final answer. If not, it triggers Tool 4.

4.  ** Tool 4: `EnrichGraph` (The "Graph Write" Tool)**
    *   **Goal:** Surgically expand the persistent knowledge graph with new information from the raw corpus.
    *   **Algorithm (Targeted Enrichment):**
        1.  **Localize Gap:** The LLM generates an `expansion_query`.
        2.  **Find New Docs:** The agent runs a search on the **corpus index** to find new documents.
        3.  **Deep Extraction:** The agent uses an LLM to run high-quality fact extraction on *only these new documents*.
        4.  **Update & Cache:** The new facts are added to the **Parquet** files (as a new partition). The **`txtai`** index is then incrementally updated using its `upsert` method. The new data versions are tracked with **`DVC`**.
        5.  **Retry:** The agent restarts the entire Phase 1 loop.

## 5. 🚦 Current Project State

**The project is currently in an architectural refactoring phase before completing the offline build (Phase 0).**

*   **COMPLETED:**
    *   The **Corpus Index** is **DONE**.
*   **CURRENT BOTTLENECK (Implementing New Architecture):**
    *   We have finalized the new, more robust data and indexing architecture: **Parquet + DuckDB + txtai + DVC**.
    *   The immediate next step is to implement this new architecture. This involves:
        1.  Refactoring the graph-building scripts (`00_build_...`) to output partitioned **Parquet** files.
        2.  Creating a new script (`02_build_graph_index.py`) to build the initial **`txtai`** index.
        3.  Modifying the agent tools (`enrich_rag/tools.py`) to use `DuckDB` for structured queries and `txtai` for relevance search and incremental updates.

## 6.  How to Run the Project

1.  **Install Dependencies:**
    *(Note: `txtai` and `duckdb` will be added to requirements.txt)*
    ```bash
    pip install -r requirements.txt
    ```

2.  **Build Corpus Index (Already Done):**
    ```bash
    # python scripts/01_build_corpus_index.py
    ```

3.  **Build Seed Graph (Future Step):**
    *This script will be refactored to output Parquet files.*
    ```bash
    # python scripts/00_build_graph_rebel.py
    ```

4.  **Build Graph Index (Future Step):**
    *This is a new script to be created for txtai.*
    ```bash
    # python scripts/02_build_graph_index.py
    ```

5.  **Run Evaluation (Future Step):**
    *Once all data and indexes are built using the new architecture.*
    ```bash
    python run_evaluation.py --config configs/nq.yaml
    ```