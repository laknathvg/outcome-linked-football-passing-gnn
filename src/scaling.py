import copy
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from sklearn.preprocessing import StandardScaler

@dataclass
class GraphScalers:
    node_scaler: StandardScaler | None
    edge_scaler: StandardScaler | None
    edge_continuous_indices: list[int]


def make_feature_view(graph, node_indices: Sequence[int] | None, edge_indices: Sequence[int] | None):
    result = copy.deepcopy(graph)
    if node_indices is not None:
        result.x = result.x[:, list(node_indices)].clone()
    if edge_indices is not None:
        result.edge_attr = result.edge_attr[:, list(edge_indices)].clone()
    return result


def fit_graph_scalers(graphs: list, standardize_nodes: bool, edge_continuous_indices: list[int]) -> GraphScalers:
    node_scaler = StandardScaler().fit(np.vstack([g.x.cpu().numpy() for g in graphs])) if standardize_nodes else None
    edge_scaler = None
    if edge_continuous_indices:
        edge_matrix = np.vstack([g.edge_attr[:, edge_continuous_indices].cpu().numpy() for g in graphs if g.edge_attr is not None and g.edge_attr.numel() > 0])
        edge_scaler = StandardScaler().fit(edge_matrix)
    return GraphScalers(node_scaler, edge_scaler, edge_continuous_indices)


def transform_graphs(graphs: list, scalers: GraphScalers, history_features: np.ndarray | None = None) -> list:
    output = []
    for position, graph in enumerate(graphs):
        transformed = copy.deepcopy(graph)
        if scalers.node_scaler is not None:
            transformed.x = transformed.x.new_tensor(scalers.node_scaler.transform(transformed.x.cpu().numpy()))
        if scalers.edge_scaler is not None and transformed.edge_attr is not None and transformed.edge_attr.numel() > 0:
            values = transformed.edge_attr.cpu().numpy().copy()
            idx = scalers.edge_continuous_indices
            values[:, idx] = scalers.edge_scaler.transform(values[:, idx])
            transformed.edge_attr = transformed.edge_attr.new_tensor(values)
        if history_features is not None:
            transformed.global_features = transformed.x.new_tensor(history_features[position]).unsqueeze(0)
        output.append(transformed)
    return output


def aggregate_graph_features(graphs: list) -> np.ndarray:
    rows = []
    for graph in graphs:
        rows.append(np.concatenate([graph.x.cpu().numpy().sum(axis=0), graph.edge_attr.cpu().numpy().sum(axis=0)]))
    return np.vstack(rows)
