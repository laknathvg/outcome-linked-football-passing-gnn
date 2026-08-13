import torch
from torch_geometric.data import Data

from src.topology_permutation import permute_edge_destinations_degree_preserving


def test_destination_permutation_preserves_matched_quantities_and_changes_topology():
    src = torch.tensor([0,0,0,1,1,1,2,2,2,3,3,3], dtype=torch.long)
    dst = torch.tensor([1,2,3,0,2,3,0,1,3,0,1,2], dtype=torch.long)
    graph = Data(
        x=torch.arange(16, dtype=torch.float32).reshape(4,4),
        edge_index=torch.stack([src, dst]),
        edge_attr=torch.arange(36, dtype=torch.float32).reshape(12,3),
        role_ids=torch.tensor([0,1,2,3], dtype=torch.long),
        y=torch.tensor([1], dtype=torch.long),
    )
    permuted, audit = permute_edge_destinations_degree_preserving(
        graph, match_id=123, permutation_seed=1, swap_multiplier=3
    )
    assert torch.equal(permuted.edge_index[0], graph.edge_index[0])
    assert not torch.equal(permuted.edge_index[1], graph.edge_index[1])
    assert sorted(permuted.edge_index[1].tolist()) == sorted(graph.edge_index[1].tolist())
    assert torch.equal(permuted.x, graph.x)
    assert torch.equal(permuted.edge_attr, graph.edge_attr)
    assert torch.equal(permuted.role_ids, graph.role_ids)
    assert torch.equal(permuted.y, graph.y)
    assert int((permuted.edge_index[0] == permuted.edge_index[1]).sum()) == 0
    assert audit.changed_edges > 0
