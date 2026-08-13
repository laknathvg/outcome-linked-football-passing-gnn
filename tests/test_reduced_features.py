import torch
from torch_geometric.data import Data
from src.scaling import make_feature_view


def test_reduced_feature_view_preserves_topology_and_expected_columns():
    graph = Data(
        x=torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]]),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.tensor([[10., 0.1, 1.], [20., -0.2, 0.]]),
        y=torch.tensor([1]),
    )
    reduced = make_feature_view(graph, node_indices=[0, 3], edge_indices=[0, 1])
    assert reduced.x.shape == (2, 2)
    assert reduced.edge_attr.shape == (2, 2)
    assert torch.equal(reduced.x, graph.x[:, [0, 3]])
    assert torch.equal(reduced.edge_attr, graph.edge_attr[:, [0, 1]])
    assert torch.equal(reduced.edge_index, graph.edge_index)
    assert torch.equal(reduced.y, graph.y)
