
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool

from .models import HistoryMLP


class TopologyFreeEncoder(nn.Module):
    """
    Information-matched topology-free completed-match encoder.

    IMPORTANT:
    - Uses node/player features.
    - Uses the same 4-class role IDs and learned role embedding.
    - Uses every pass-event feature row: length, angle, key-pass.
    - DOES NOT use passer-recipient identities for neural representation.

    PyG's edge_index is touched ONLY to recover which graph in a mini-batch
    owns each edge_attr row. The actual source identity, destination identity,
    and source-destination pairing are never supplied to a learnable layer.

    Consequently, any within-graph rewiring of edge_index while keeping
    x, role_ids and edge_attr unchanged MUST leave this encoder output
    unchanged in evaluation mode.
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        config: dict,
    ):
        super().__init__()

        model_cfg = config["model"]

        hidden = int(model_cfg["hidden_dim"])              # 64
        role_dim = int(model_cfg["role_embedding_dim"])    # 8
        dropout = float(model_cfg["graph_dropout"])        # 0.30

        self.hidden_dim = hidden
        self.dropout = dropout

        # Same role embedding dimensionality as existing GNN.
        self.role_embedding = nn.Embedding(4, role_dim)

        # Shared player encoder.
        # Two transformations are intentionally used to provide a
        # non-graph analogue to the two-stage GATv2 representation path.
        self.player_mlp = nn.Sequential(
            nn.Linear(node_in_dim + role_dim, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # Shared pass-event encoder.
        # Each event is processed independently, then pooled as a set.
        self.pass_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # 4 role-level vectors + 1 pass-event vector.
        self.output_dim = hidden * 5

    def forward(self, batch):
        # -------------------------------------------------------------
        # PLAYER SET
        # -------------------------------------------------------------
        role_embedding = self.role_embedding(batch.role_ids)

        player_input = torch.cat(
            [batch.x, role_embedding],
            dim=1,
        )

        player_h = self.player_mlp(player_input)

        # Same broad role-aware ADD-pooling principle used by the
        # submitted positional GNN:
        # GK / DF / MF / FW are pooled separately.
        role_vectors = []

        for role_id in range(4):
            mask = batch.role_ids.eq(role_id).float().unsqueeze(-1)

            role_vectors.append(
                global_add_pool(
                    player_h * mask,
                    batch.batch,
                )
            )

        player_representation = torch.cat(
            role_vectors,
            dim=1,
        )

        # -------------------------------------------------------------
        # PASS-EVENT SET
        # -------------------------------------------------------------
        pass_h = self.pass_mlp(batch.edge_attr)

        # PyG concatenates the pass-event rows from all graphs in the
        # mini-batch. We need only the graph membership of each row.
        #
        # batch.batch maps NODE -> GRAPH.
        # Every edge lies entirely within one graph.
        #
        # This line therefore converts each edge row to its GRAPH ID.
        # The learned representation never receives source node ID,
        # destination node ID, or the source-destination pair.
        edge_graph_id = batch.batch[batch.edge_index[0]]

        pass_representation = global_add_pool(
            pass_h,
            edge_graph_id,
            size=player_representation.size(0),
        )

        return torch.cat(
            [player_representation, pass_representation],
            dim=1,
        )


class TopologyFreeNN(nn.Module):
    """
    Standalone topology-free completed-match neural comparator.

    Head width = 128 to mirror B-GNN's standalone classifier head.
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        config: dict,
    ):
        super().__init__()

        self.encoder = TopologyFreeEncoder(
            node_in_dim,
            edge_in_dim,
            config,
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 128),
            nn.ReLU(),
            nn.Dropout(float(config["model"]["classifier_dropout"])),
            nn.Linear(128, 2),
        )

    def forward(self, batch):
        z = self.encoder(batch)
        return self.classifier(z)


class TopologyFreeHybrid(nn.Module):
    """
    Topology-free completed-match neural branch +
    EXACT existing historical MLP +
    EXACT static sigmoid-constrained late fusion.
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        history_in_dim: int,
        config: dict,
    ):
        super().__init__()

        model_cfg = config["model"]

        self.encoder = TopologyFreeEncoder(
            node_in_dim,
            edge_in_dim,
            config,
        )

        # Same 64-unit completed-match head used in submitted C-Hybrid.
        self.completed_match_head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 64),
            nn.ReLU(),
            nn.Dropout(float(model_cfg["classifier_dropout"])),
            nn.Linear(64, 2),
        )

        # EXACT existing historical branch.
        self.history_head = HistoryMLP(
            history_in_dim,
            int(model_cfg["history_hidden_dim"]),
            float(model_cfg["history_dropout"]),
        )

        initial_lambda = min(
            max(
                float(model_cfg["fusion"]["initial_lambda"]),
                1e-6,
            ),
            1.0 - 1e-6,
        )

        self.raw_gate = nn.Parameter(
            torch.logit(torch.tensor(initial_lambda))
        )

    def forward(self, batch):
        completed_representation = self.encoder(batch)

        completed_logits = self.completed_match_head(
            completed_representation
        )

        history_logits = self.history_head(
            batch.global_features
        )

        gate = torch.sigmoid(self.raw_gate)

        return (
            gate * completed_logits
            + (1.0 - gate) * history_logits
        )

    def gate_value(self) -> float:
        return float(
            torch.sigmoid(self.raw_gate)
            .detach()
            .cpu()
        )
