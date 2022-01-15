from typing import List, Tuple
import pandas as pd


# Build a feature matrix (pandas dataframe) from database data
class NodeFeatureBuilder:
    # sewer_pipes: list of (14 or 15) tuples: (id, start_node_id, end_node_id, length, systemtype, pipefunction,
    # contentstype, material, construction_year, pipeshape, width, height, pipe_age,
    # [damage_class | n_failures, failures_per_meter])
    def build_one_hot_matrix(self, sewer_pipes: List[Tuple], save=False, file_path='./data/pipes.csv',
                             type='damage_class'):
        if type == 'damage_class':
            df = pd.DataFrame(sewer_pipes, columns=['id', 'start_node_id', 'end_node_id', 'length', 'system_type',
                                                    'pipe_function', 'content_type', 'material', 'construction_year',
                                                    'pipe_shape', 'width', 'height', 'pipe_age', 'damage_class'])
        elif type == 'failure_rate':
            df = pd.DataFrame(sewer_pipes, columns=['id', 'start_node_id', 'end_node_id', 'length', 'system_type',
                                                    'pipe_function', 'content_type', 'material', 'construction_year',
                                                    'pipe_shape', 'width', 'height', 'pipe_age', 'n_failures',
                                                    'failures_per_meter'])
        else:
            raise Exception(f"Unknown type '{type}'")

        df.system_type = df.system_type.astype('category')
        df.pipe_function = df.pipe_function.astype('category')
        df.content_type = df.content_type.astype('category')
        df.material = df.material.astype('category')
        df.pipe_shape = df.pipe_shape.astype('category')

        system_type = pd.get_dummies(df.system_type, prefix='system_type')
        pipe_function = pd.get_dummies(df.pipe_function, prefix='pipe_function')
        content_type = pd.get_dummies(df.content_type, prefix='content_type')
        material = pd.get_dummies(df.material, prefix='material')
        pipe_shape = pd.get_dummies(df.pipe_shape, prefix='pipe_shape')

        df = pd.concat([df, system_type, pipe_function, content_type, material, pipe_shape], axis=1)

        if save:
            df.to_csv(file_path)
        return df
