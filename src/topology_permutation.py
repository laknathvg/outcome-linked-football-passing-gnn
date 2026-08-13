from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TopologyPermutationAudit:
    match_id: int
    permutation_seed: int
    edge_count: int
    requested_successful_swaps: int
    successful_swaps: int
    attempts: int
    changed_edges: int
    changed_fraction: float
    original_self_loops: int
    permuted_self_loops: int
    source_vector_preserved: bool
    destination_multiset_preserved: bool
    node_features_preserved: bool
    edge_attributes_preserved: bool
    roles_preserved: bool
    label_preserved: bool


def permute_edge_destinations_degree_preserving(
    graph,
    match_id: int,
    permutation_seed: int,
    swap_multiplier: int = 10,
    max_attempt_multiplier: int = 100,
):
    """Return a matched topology-disrupted copy of a PyG graph.

    Only edge destinations are rewired. The operation preserves:
      * every edge source and therefore source out-degree,
      * the multiset of edge destinations and therefore destination in-degree,
      * edge count,
      * edge-feature rows (features stay attached to the original pass-event row),
      * node features, role IDs, graph label and all other graph attributes.

    Rewiring is performed through random pairwise swaps of destination endpoints.
    A proposed swap is rejected if it would create an event-level self-loop. This
    avoids adding synthetic self-passes that are absent from the observed event
    graph. PyG/GATv2's separately configured internal self-loops remain unchanged.

    Randomness depends only on (permutation_seed, match_id), never on labels,
    folds, predictions, or model performance.
    """
    result = copy.deepcopy(graph)
    src = graph.edge_index[0].detach().cpu().numpy().astype(np.int64, copy=True)
    original_dst = graph.edge_index[1].detach().cpu().numpy().astype(np.int64, copy=True)
    dst = original_dst.copy()

    edge_count = int(len(dst))
    if edge_count < 2:
        raise RuntimeError(f"match_id={match_id} has fewer than two edges; destination permutation is undefined")

    original_self_loops = int(np.sum(src == original_dst))
    if original_self_loops != 0:
        raise RuntimeError(
            f"match_id={match_id} contains {original_self_loops} observed event-level self-loops; "
            "the matched no-new-self-loop permutation control expects none"
        )

    seed_sequence = np.random.SeedSequence([
        int(permutation_seed) & 0xFFFFFFFF,
        int(match_id) & 0xFFFFFFFF,
    ])
    rng = np.random.default_rng(seed_sequence)

    requested_swaps = max(edge_count, int(swap_multiplier) * edge_count)
    max_attempts = max(requested_swaps * int(max_attempt_multiplier), 1000)
    successful_swaps = 0
    attempts = 0

    while successful_swaps < requested_swaps and attempts < max_attempts:
        attempts += 1
        i = int(rng.integers(0, edge_count))
        j = int(rng.integers(0, edge_count - 1))
        if j >= i:
            j += 1

        di = int(dst[i])
        dj = int(dst[j])

        # No topological change if both rows already point to the same node.
        if di == dj:
            continue

        # Swapping destinations gives i -> dj and j -> di. Reject only if this
        # would create an event-level self-loop.
        if int(src[i]) == dj or int(src[j]) == di:
            continue

        dst[i], dst[j] = dj, di
        successful_swaps += 1

    if successful_swaps < requested_swaps:
        raise RuntimeError(
            f"match_id={match_id}, permutation_seed={permutation_seed}: only {successful_swaps} "
            f"valid destination swaps were achieved out of {requested_swaps} requested after {attempts} attempts"
        )

    changed_edges = int(np.sum(dst != original_dst))
    changed_fraction = float(changed_edges / edge_count)
    if changed_edges == 0:
        raise RuntimeError(
            f"match_id={match_id}, permutation_seed={permutation_seed}: topology permutation produced no changed edges"
        )

    permuted_self_loops = int(np.sum(src == dst))
    if permuted_self_loops != 0:
        raise RuntimeError(
            f"match_id={match_id}, permutation_seed={permutation_seed}: permutation introduced self-loops"
        )

    # A pure destination permutation must preserve the destination multiset.
    destination_multiset_preserved = bool(
        np.array_equal(np.sort(dst), np.sort(original_dst))
    )
    if not destination_multiset_preserved:
        raise RuntimeError("Destination multiset was not preserved")

    new_edge_index = graph.edge_index.clone()
    new_edge_index[1] = torch.as_tensor(
        dst,
        dtype=graph.edge_index.dtype,
        device=graph.edge_index.device,
    )
    result.edge_index = new_edge_index

    audit = TopologyPermutationAudit(
        match_id=int(match_id),
        permutation_seed=int(permutation_seed),
        edge_count=edge_count,
        requested_successful_swaps=int(requested_swaps),
        successful_swaps=int(successful_swaps),
        attempts=int(attempts),
        changed_edges=changed_edges,
        changed_fraction=changed_fraction,
        original_self_loops=original_self_loops,
        permuted_self_loops=permuted_self_loops,
        source_vector_preserved=bool(torch.equal(result.edge_index[0], graph.edge_index[0])),
        destination_multiset_preserved=destination_multiset_preserved,
        node_features_preserved=bool(torch.equal(result.x, graph.x)),
        edge_attributes_preserved=bool(torch.equal(result.edge_attr, graph.edge_attr)),
        roles_preserved=bool(torch.equal(result.role_ids, graph.role_ids)),
        label_preserved=bool(torch.equal(result.y, graph.y)),
    )

    if not all([
        audit.source_vector_preserved,
        audit.destination_multiset_preserved,
        audit.node_features_preserved,
        audit.edge_attributes_preserved,
        audit.roles_preserved,
        audit.label_preserved,
    ]):
        raise RuntimeError(f"Matched-control preservation check failed for match_id={match_id}")

    return result, audit
