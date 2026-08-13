import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool, global_mean_pool


class HistoryMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class GraphEncoder(nn.Module):
    def __init__(self, node_in_dim: int, edge_in_dim: int, config: dict, pooling: str | None = None):
        super().__init__()
        model = config["model"]
        self.pooling = pooling or model["pooling"]
        self.graph_dropout = float(model["graph_dropout"])
        role_dim = int(model["role_embedding_dim"])
        hidden = int(model["hidden_dim"])
        heads = int(model["attention_heads"])
        concat = bool(model["attention_concat"])
        self.role_embedding = nn.Embedding(4, role_dim)
        self.gat1 = GATv2Conv(node_in_dim + role_dim, hidden, heads=heads, concat=concat,
                              edge_dim=edge_in_dim, add_self_loops=bool(config["graph"]["add_self_loops_in_gatv2"]))
        first_out = hidden * heads if concat else hidden
        self.gat2 = GATv2Conv(first_out, hidden, heads=heads, concat=False,
                              edge_dim=edge_in_dim, add_self_loops=bool(config["graph"]["add_self_loops_in_gatv2"]))
        self.output_dim = hidden * 4 if self.pooling == "positional_add" else hidden

    def forward(self, x, edge_index, edge_attr, role_ids, batch):
        x = torch.cat([x, self.role_embedding(role_ids)], dim=1)
        x = F.dropout(F.elu(self.gat1(x, edge_index, edge_attr)), p=self.graph_dropout, training=self.training)
        x = F.dropout(F.elu(self.gat2(x, edge_index, edge_attr)), p=self.graph_dropout, training=self.training)
        if self.pooling == "global_mean":
            return global_mean_pool(x, batch)
        pooled = []
        for role_id in range(4):
            mask = role_ids.eq(role_id).float().unsqueeze(-1)
            pooled.append(global_add_pool(x * mask, batch))
        return torch.cat(pooled, dim=1)


class StructureGNN(nn.Module):
    def __init__(self, node_in_dim: int, edge_in_dim: int, config: dict, pooling: str | None = None):
        super().__init__()
        self.encoder = GraphEncoder(node_in_dim, edge_in_dim, config, pooling)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 128),
            nn.ReLU(),
            nn.Dropout(float(config["model"]["classifier_dropout"])),
            nn.Linear(128, 2),
        )

    def forward(self, batch):
        z = self.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.role_ids, batch.batch)
        return self.classifier(z)


class HybridGNN(nn.Module):
    def __init__(self, node_in_dim: int, edge_in_dim: int, history_in_dim: int, config: dict, pooling: str | None = None):
        super().__init__()
        model = config["model"]
        self.encoder = GraphEncoder(node_in_dim, edge_in_dim, config, pooling)
        self.graph_head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 64),
            nn.ReLU(),
            nn.Dropout(float(model["classifier_dropout"])),
            nn.Linear(64, 2),
        )
        self.history_head = HistoryMLP(history_in_dim, int(model["history_hidden_dim"]), float(model["history_dropout"]))
        initial_lambda = min(max(float(model["fusion"]["initial_lambda"]), 1e-6), 1 - 1e-6)
        self.raw_gate = nn.Parameter(torch.logit(torch.tensor(initial_lambda)))

    def forward(self, batch):
        z = self.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.role_ids, batch.batch)
        graph_logits = self.graph_head(z)
        history_logits = self.history_head(batch.global_features)
        gate = torch.sigmoid(self.raw_gate)
        return gate * graph_logits + (1.0 - gate) * history_logits

    def gate_value(self) -> float:
        return float(torch.sigmoid(self.raw_gate).detach().cpu())
