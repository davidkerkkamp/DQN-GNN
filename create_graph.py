from NodeFeatureBuilder import NodeFeatureBuilder
from Repository import Repository
from SewerGraphBuilder import SewerGraphBuilder

repo = Repository('path/to/db/file')

# Specify constraints
min_construction_year = 1900  # Minimum pipe construction year
coords = [121222, 124893, 382361, 385282]  # Only retrieve pipes between these coordinates

# Retrieve all sewer pipes with given constraints.
# Returns list of tuples (id, start_node_id, end_node_id, length, systemtype, pipefunction,
# contentstype, material, construction_year, pipeshape, width, height, pipe_age, damage_class)
sewer_pipes = repo.get_sewers_with_damage_classes(min_constr_year=min_construction_year, area=coords)

# Retrieve pipes with their neighbours, using a distance of 20 meters
# Returns list of tuples (s1_id, s1_x1, s1_y1, s1_x2, s1_y2, s2_id, s2_x1, s2_y1, s2_x2, s2_y2)
# IMPORTANT to use same constraints as above, otherwise node feature matrix and edge indices will not match
sewer_pipes_neighbours = repo.get_sewers_with_neighbours(range=20,
                                                         min_constr_year=min_construction_year,
                                                         area=coords)

# Create pandas dataframe with node (pipe) features
feature_builder = NodeFeatureBuilder()
df = feature_builder.build_one_hot_matrix(sewer_pipes, type='damage_class')

# Build adjacency matrix (edge indices)
gb = SewerGraphBuilder()
gb.create_feature_matrix_indices(df['id'])
gb.build_graph_pipe_proximity(sewer_pipes_neighbours)

# Now gb.edge_indices contains a tuple with 2 lists where each element is the index of the node (pipe) in the
# 'df' created above (edge indices in COO form).
# Example: gb.edge_indices[0][0] = 0 and gb.edge_indices[1][0] = 1 means there is an edge between node 0 and 1