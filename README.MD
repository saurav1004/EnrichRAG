
![alt text](banner.png)

# EnrichRAG: An Agentic, Training-Free RAG Pipeline

**EnrichRAG** is a novel, training-free, agentic RAG pipeline. Its goal is to achieve state-of-the-art accuracy on complex, open-domain question-answering benchmarks (like NQ, 2wikimultihopqa, Musique, HotpotQA etc.) without the high cost of model training (like in Graph-R1) or reliance on external search APIs.

The core idea is to build a **persistent, evolving Knowledge Graph** *exclusively* from a local document corpus. An LLM-powered agent intelligently decides *when* the graph is insufficient and *what* new information to extract from the corpus to "enrich" the graph, improving its knowledge just-in-time.

## 1\. Motivation & Core Philosophy

EnrichRAG is designed to solve specific, persistent bottlenecks in standard RAG and Knowledge Graph (KG) systems:

### The Gaps We Are Solving

  * **The "Cold Start" Bottleneck:** Traditional KG creation is extremely time-consuming and computationally expensive, especially when processing massive corpora with LLMs. Creating a full graph upfront often results in redundancy and wasted compute on information that may never be queried.
  * **The Plausibility Problem (Context Flooding):** Most RAG methods attempt to flood the LLM's context window with as much retrieved information as possible. This leads to plausible-sounding but unfaithful generations. The model hallucinates because the context is noisy, or it fails to answer because of the **"Disconnected Facts"** problem—where the retrieval mechanism captures keywords but misses the logical relations binding them.

### Our Solutions

  * **Lazy, Demand-Driven Construction (`EnrichGraph`):**
    Instead of building a massive graph upfront, we employ "Lazy Construction." The KG grows organically based on the **query distribution**. This is an online enrichment method where the graph is updated only when necessary.
  * **Optimal Subgraph Retrieval (`PCST Solver`):**
    To solve the disconnected facts problem, we move beyond simple cosine similarity. We use a **Prize-Collecting Steiner Tree (PCST)** solver to identify the optimal connected subgraph. This ensures we retrieve a cohesive logical chain rather than isolated chunks, significantly reducing hallucinations.
  * **Uncertainty-Adaptive Compute:**
    EnrichRAG makes RAG adaptive. Harder queries trigger higher compute (via Graph Enrichment), while easier queries are answered quickly. Crucially, the compute spent on hard queries is "saved" into the persistent graph, benefiting all future similar queries.

-----

## 2\. Architecture

This stack provides a clear separation of concerns and leverages best-in-class tools for each task.

  * **`Parquet` (Data Storage):** All graph nodes and edges are stored in a partitioned, columnar format. This is highly efficient for storage and for analytical queries.
  * **`DuckDB` (Structured Queries):** Used as a high-speed query engine to perform structured SQL queries directly on the Parquet files. Its primary role is fast graph traversal (e.g., fetching 1-hop neighbors) and attribute-based filtering.
  * **`BM25/Dense Retrieval Index` (Vector Search):** An open-source vector search engine that provides efficient similarity search over embeddings. It is used to find relevant nodes and edges in the graph as well as relevant documents in the corpus.

-----

## 3\. The EnrichRAG Pipeline (The Goal)

The pipeline is an "online" (inference-time) agent loop that has access to four distinct tools.

  * **Phase 0: Offline Seed Graph:** A one-time, offline script builds a "shallow but wide" heterogeneous graph from the entire document corpus.
  * **Phase 1: Online Agent Loop:** For each query, the agent runs a loop to gather context using the tools below.
  * **Phase 2: Final Answer:** The agent generates an answer based on the context it gathered.

### The 4 Agent Tools (Theoretical Design)

![alt text](assets/image.png)

1.  **Tool 1: `EnrichContext` (The "Graph Read" Tool)**

      * **Goal:** Retrieve a small, relevant subgraph from the *current* graph to avoid context flooding.
      * **Algorithm (The "Hybrid Search PCST"):**
        1.  **Find Entry Points:** The agent runs a search query against the **`txtai` index** to find the most relevant nodes and edges.
        2.  **Get Neighbors:** The IDs of the top results are fed to **`DuckDB`**, which runs a high-speed SQL query over the **Parquet** files to retrieve the full 1-hop neighborhood of these initial results.
        3.  **Solve PCST:** A **Prize-Collecting Steiner Tree (PCST)** solver finds the most optimal, connected subgraph from the combined results. This mathematically ensures the retrieved context is logically connected, solving the "Disconnected Facts" problem.

2.  **Tool 2: `CheckSufficiency` (The "Information-Theoretic Trigger")**

      * **Goal:** Decide if the agent should continue retrieving or if the information has "converged."
      * **Algorithm (Perplexity Convergence):** The agent calculates the **perplexity** (LLM "confidence") of a hypothetical answer. If adding more context doesn't lower the perplexity, the information has "converged."

3.  **Tool 3: `Analyze & Decide` (The "Confidence Check")**

      * **Goal:** Decide if the converged context is *good enough* to answer.
      * **Algorithm (Confidence Threshold):** If the final perplexity is low enough (e.g., `< 3.0`), the agent proceeds to the final answer. If not, it triggers Tool 4.

4.  **Tool 4: `EnrichGraph` (The "Graph Write" Tool)**

      * **Goal:** Surgically expand the persistent knowledge graph with new information from the raw corpus (Lazy Construction).
      * **Algorithm (Targeted Enrichment):**
        1.  **Localize Gap:** The LLM generates an `expansion_query`.
        2.  **Find New Docs:** The agent runs a search on the **corpus index** to find new documents.
        3.  **Deep Extraction:** The agent uses an LLM to run high-quality fact extraction on *only these new documents*.
        4.  **Update & Cache:** The new facts are added to the **Parquet** files (as a new partition). The **`txtai`** index is then incrementally updated using its `upsert` method. The new data versions are tracked with **`DVC`**.
        5.  **Retry:** The agent restarts the entire Phase 1 loop.

-----

## 4\. How to Run the Project

1.  **Install Dependencies:**
    *(Note: `txtai` and `duckdb` will be added to requirements.txt)*

    ```bash
    pip install -r requirements.txt
    ```

2.  **Build Corpus Index:**

    ```bash
    # python scripts/01_build_corpus_index.py
    ```

3.  **Build Seed Graph:**
    *This script will be refactored to output Parquet files.*

    ```bash
    # python scripts/00_build_graph_rebel.py
    ```

4.  **Build Graph Index:**
    *This is a new script to be created for txtai.*

    ```bash
    # python scripts/02_build_graph_index.py
    ```

5.  **Run Evaluation:**
    *Once all data and indexes are built using the new architecture.*

    ```bash
    python run_evaluation.py --config configs/nq.yaml
    ```