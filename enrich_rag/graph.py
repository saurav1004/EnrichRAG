import os
import duckdb
import pandas as pd

class PersistentHyperGraph:
    def __init__(self, base_dir: str, experiment_dir: str = None):
        """
        Initializes the graph connection using DuckDB to query Parquet files.

        Args:
            base_dir (str): Path to the directory containing the base graph files 
                            (nodes.parquet, edges.parquet).
            experiment_dir (str, optional): Path to the directory for a specific 
                                            experiment, containing new nodes/edges. 
                                            Defaults to None.
        """
        print(f"[Graph] Initializing with base '{base_dir}' and experiment '{experiment_dir}'")
        self.base_nodes_path = os.path.join(base_dir, "nodes.parquet")
        self.base_edges_path = os.path.join(base_dir, "edges.parquet")
        
        self.exp_nodes_path = None
        self.exp_edges_path = None
        if experiment_dir:
            self.exp_nodes_path = os.path.join(experiment_dir, "new_nodes.parquet")
            self.exp_edges_path = os.path.join(experiment_dir, "new_edges.parquet")

        # Initialize a persistent DuckDB connection
        # For an in-memory database, use duckdb.connect(':memory:')
        self.con = duckdb.connect()
        print("[Graph] DuckDB connection established.")
        
        # Pre-build the SQL queries for fetching nodes and edges
        self._build_view_queries()

    def _build_view_queries(self):
        """Constructs the SQL queries needed to create composite views of the graph."""
        
        # --- Nodes Query ---
        self.nodes_query = f"SELECT * FROM read_parquet('{self.base_nodes_path}')"
        if self.exp_nodes_path and os.path.exists(self.exp_nodes_path):
            self.nodes_query += f"\nUNION ALL\nSELECT * FROM read_parquet('{self.exp_nodes_path}')"
            
        # --- Edges Query ---
        self.edges_query = f"SELECT * FROM read_parquet('{self.base_edges_path}')"
        if self.exp_edges_path and os.path.exists(self.exp_edges_path):
            self.edges_query += f"\nUNION ALL\nSELECT * FROM read_parquet('{self.exp_edges_path}')"

    def get_neighbors(self, node_ids: list) -> pd.DataFrame:
        """
        Finds all 1-hop neighbors for a given list of node IDs.

        Args:
            node_ids (list): A list of node_id strings to find neighbors for.

        Returns:
            pd.DataFrame: A DataFrame containing the edges connected to the input nodes.
        """
        if not node_ids:
            return pd.DataFrame()

        # Create a string representation of the list for the SQL IN clause
        node_list_str = ", ".join([f"'{node_id}'" for node_id in node_ids])

        query = f"""
            WITH all_edges AS ({self.edges_query})
            SELECT *
            FROM all_edges
            WHERE subject IN ({node_list_str}) OR object IN ({node_list_str})
        """
        
        try:
            return self.con.execute(query).fetchdf()
        except duckdb.Error as e:
            print(f"An error occurred during neighbor search: {e}")
            return pd.DataFrame()

    def get_all_nodes(self) -> pd.DataFrame:
        """Returns a DataFrame of all nodes in the composite graph."""
        try:
            return self.con.execute(self.nodes_query).fetchdf()
        except duckdb.Error as e:
            print(f"An error occurred while fetching all nodes: {e}")
            return pd.DataFrame()

    def get_all_edges(self) -> pd.DataFrame:
        """Returns a DataFrame of all edges in the composite graph."""
        try:
            return self.con.execute(self.edges_query).fetchdf()
        except duckdb.Error as e:
            print(f"An error occurred while fetching all edges: {e}")
            return pd.DataFrame()

    def close(self):
        """Closes the DuckDB connection."""
        if self.con:
            self.con.close()
            print("[Graph] DuckDB connection closed.")