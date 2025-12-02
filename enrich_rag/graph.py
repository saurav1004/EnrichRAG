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

        self.con = duckdb.connect()
        print("[Graph] DuckDB connection established.")
        
        self._build_view_queries()

    def _build_view_queries(self):
        """Constructs the SQL queries needed to create composite views of the graph."""
        
        self.nodes_query = f"SELECT * FROM read_parquet('{self.base_nodes_path}')"
        if self.exp_nodes_path and os.path.exists(self.exp_nodes_path):
            self.nodes_query += f"\nUNION ALL\nSELECT * FROM read_parquet('{self.exp_nodes_path}')"
            
        self.edges_query = f"SELECT * FROM read_parquet('{self.base_edges_path}')"
        if self.exp_edges_path and os.path.exists(self.exp_edges_path):
            self.edges_query += f"\nUNION ALL\nSELECT * FROM read_parquet('{self.exp_edges_path}')"

    def get_neighbors(self, node_ids: list, num_hops: int = 1) -> pd.DataFrame:
        """
        Finds all edges within a specified number of hops from a given list of node IDs.

        Args:
            node_ids (list): A list of node_id strings to start the traversal from.
            num_hops (int): The number of hops to traverse. Defaults to 1.

        Returns:
            pd.DataFrame: A DataFrame containing the edges within the k-hop neighborhood.
        """
        if not node_ids or num_hops < 1:
            return pd.DataFrame()

        # Properly escape single quotes in node IDs to prevent SQL injection/errors.
        escaped_node_ids = [node_id.replace("'", "''") for node_id in node_ids]
        node_list_str = ", ".join([f"'{node_id}'" for node_id in escaped_node_ids])

        full_query = f"""
            WITH RECURSIVE all_nodes_in_hops(node_id, depth) AS (
                -- Base case: the initial set of nodes at depth 0
                SELECT unnest(ARRAY[{node_list_str}]) AS node_id, 0 AS depth
                
                UNION ALL
                
                -- Recursive step: find neighbors of nodes from the previous hop.
                -- The two SELECT statements are wrapped in parentheses to form a single recursive member.
                (
                    SELECT DISTINCT e.object, prev.depth + 1
                    FROM ({self.edges_query}) e, all_nodes_in_hops prev
                    WHERE e.subject = prev.node_id AND prev.depth < {num_hops}
                    
                    UNION ALL
                    
                    SELECT DISTINCT e.subject, prev.depth + 1
                    FROM ({self.edges_query}) e, all_nodes_in_hops prev
                    WHERE e.object = prev.node_id AND prev.depth < {num_hops}
                )
            ),
            final_node_set AS (
                SELECT DISTINCT node_id FROM all_nodes_in_hops
            )
            -- Final query: select all edges where BOTH subject and object are in the final set of reachable nodes
            SELECT DISTINCT e.*
            FROM ({self.edges_query}) e
            WHERE e.subject IN (SELECT node_id FROM final_node_set)
              AND e.object IN (SELECT node_id FROM final_node_set)
        """
        
        try:
            return self.con.execute(full_query).fetchdf()
        except duckdb.Error as e:
            print(f"An error occurred during multi-hop neighbor search: {e}")
            # Also print the full query for debugging purposes
            print(f"Failing query:\n{full_query}")
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