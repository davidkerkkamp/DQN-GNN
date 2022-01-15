from typing import List, Tuple, Dict
import pandas as pd


class SewerGraphBuilder:
    def __init__(self):
        self.edge_indices: Tuple[List[int], List[int]] = ([], [])  # (rows, cols)
        self.sewer_ids: Dict[str, int] = {}
        self.sewer_nodes: Dict[str, List[int]] = {}
        self.isolated = set()
        self.num_edges = 0
        self.max_degree = 0

    # Scan the list of edge indices and compare with list of nodes to see which nodes don't have edges
    # and hence, are isolated
    def find_isolated_pipes(self):
        for k, v in self.sewer_ids.items():
            if v not in self.edge_indices[0]:
                self.isolated.add(k)

    # Assign an index to every sewer pipe ID, based on order of given feature matrix
    # to ensure feature matrix indices match edge adjacency matrix indices
    def create_feature_matrix_indices(self, sewer_ids):
        self.sewer_ids = {}  # Clear previous ID mapping
        for i, id in enumerate(sewer_ids):
            self.sewer_ids[id] = i

    # Build graph based on physical pipe lay-out, i.e. only add edge if pipes are connected physically
    # sewer_pipes: list of 3-tuples (id, start_node_id, end_node_id) or pd.DataFrame with these 3 columns
    def build_graph_node_connections(self, sewer_pipes):
        self.sewer_nodes = {}

        if isinstance(sewer_pipes, pd.DataFrame):
            for i, pipe in sewer_pipes.iterrows():
                self.add_node_connection(i, pipe['id'], pipe['start_node_id'], pipe['end_node_id'])
        else:
            for i, pipe in enumerate(sewer_pipes):
                self.add_node_connection(i, pipe[0], pipe[1], pipe[2])

        rows = []  # Build adjacency matrix in COO form
        cols = []  # rows contains row indices, cols contains column indices
        for node in self.sewer_nodes.values():
            if len(node) < 2:
                continue  # Only one pipe connected to node, so no edges to other pipes
            for from_pipe in node:
                for to_pipe in node:
                    if from_pipe != to_pipe:
                        rows.append(from_pipe)
                        cols.append(to_pipe)
        self.edge_indices = (rows, cols)
        self.num_edges = int(len(rows) / 2)
        self.max_degree = rows.count(max(set(rows), key=rows.count))

    def add_node_connection(self, i, pipe_id, start_node_id, end_node_id):
        # self.sewer_ids[pipe_id] = i  # Map pipe ID to index
        try:
            if not self.sewer_nodes.get(start_node_id):
                self.sewer_nodes[start_node_id] = []  # Init empty list
            self.sewer_nodes[start_node_id].append(self.sewer_ids[pipe_id])  # Map every node to connected pipe indices

            if not self.sewer_nodes.get(end_node_id):
                self.sewer_nodes[end_node_id] = []  # Init empty list
            self.sewer_nodes[end_node_id].append(self.sewer_ids[pipe_id])  # Map every node to connected pipe indices
        except KeyError as e:
            print(f"Unknown sewer ID, call create_feature_matrix_indices() first. Error message: {e}")

    # Build graph based on distance measure
    # Obtain input list 'sewer_pipe_relations' from Repository.get_sewers_with_neighbours()
    # sewer_pipe_relations: list of 10-tuples (s1_id, s1_x1, s1_y1, s1_x2, s1_y2, s2_id, s2_x1, s2_y1, s2_x2, s2_y2)
    def build_graph_pipe_proximity(self, sewer_pipe_relations: List[Tuple]):
        fresh_index = 0
        rows = []  # Build adjacency matrix in COO form
        cols = []  # rows contains row indices, cols contains column indices
        for i, pipe in enumerate(sewer_pipe_relations):
            s1_id, s2_id = pipe[0], pipe[5]

            try:
                rows.append(self.sewer_ids[s1_id])
                cols.append(self.sewer_ids[s2_id])
            except KeyError as e:
                print(f"Unknown sewer ID, call create_feature_matrix_indices() first. Error message: {e}")
        self.edge_indices = (rows, cols)
        self.num_edges = int(len(rows) / 2)
        self.max_degree = rows.count(max(set(rows), key=rows.count))
