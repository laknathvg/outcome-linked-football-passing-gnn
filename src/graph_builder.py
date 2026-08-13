from dataclasses import dataclass
from math import atan2
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


@dataclass(frozen=True)
class GraphBuildResult:
    graph: Data | None
    exclusion_reason: str | None
    selected_player_count: int
    completed_edge_count: int


def _valid_location(value: Any) -> bool:
    return isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 2 and pd.notna(value[0]) and pd.notna(value[1])


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return default if pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return default


def _is_true(value: Any) -> bool:
    return bool(value is True or value == 1 or str(value).lower() == "true")


def resolve_player_columns(events: pd.DataFrame) -> tuple[str, str]:
    if all(c in events.columns for c in ["player_id", "pass_recipient_id"]) and events["player_id"].notna().any() and events["pass_recipient_id"].notna().any():
        return "player_id", "pass_recipient_id"
    if all(c in events.columns for c in ["player", "pass_recipient"]):
        return "player", "pass_recipient"
    raise ValueError("Missing passer/recipient columns.")


def completed_pass_mask(events: pd.DataFrame, team_name: str) -> pd.Series:
    mask = events["team"].eq(team_name) & events["type"].eq("Pass")
    recipient_col = "pass_recipient_id" if "pass_recipient_id" in events.columns else "pass_recipient"
    if recipient_col in events.columns:
        mask &= events[recipient_col].notna()
    if "pass_outcome" in events.columns:
        mask &= events["pass_outcome"].isna()
    if "pass_end_location" in events.columns:
        mask &= events["pass_end_location"].apply(_valid_location)
    return mask


def select_players(events: pd.DataFrame, team_name: str, node_count: int) -> tuple[dict[Any, int], str]:
    player_col, _ = resolve_player_columns(events)
    team_events = events[events["team"].eq(team_name) & events[player_col].notna() & events["location"].apply(_valid_location)].copy()
    if team_events.empty:
        return {}, player_col
    top_players = team_events[player_col].value_counts().head(node_count).index.tolist()
    if len(top_players) < node_count:
        return {}, player_col
    average_x = team_events[team_events[player_col].isin(top_players)].groupby(player_col)["location"].apply(lambda s: float(np.mean([loc[0] for loc in s])))
    available = [p for p in top_players if p in average_x.index]
    if len(available) < node_count:
        return {}, player_col
    ordered_players = average_x.loc[available].sort_values().index.tolist()
    return {player: idx for idx, player in enumerate(ordered_players[:node_count])}, player_col


def calculate_pass_angle(row: pd.Series) -> float:
    if "pass_angle" in row.index and pd.notna(row["pass_angle"]):
        return float(row["pass_angle"])
    start, end = row.get("location"), row.get("pass_end_location")
    if _valid_location(start) and _valid_location(end):
        return float(atan2(end[1] - start[1], end[0] - start[0]))
    return 0.0


def build_home_team_graph(events: pd.DataFrame, match_id: int, home_team: str, label: int, config: dict) -> GraphBuildResult:
    graph_config = config["graph"]
    node_count = int(graph_config["node_count"])
    try:
        node_map, player_col = select_players(events, home_team, node_count)
        _, recipient_col = resolve_player_columns(events)
    except ValueError as exc:
        return GraphBuildResult(None, str(exc), 0, 0)
    if len(node_map) != node_count:
        return GraphBuildResult(None, "fewer_than_required_eligible_players", len(node_map), 0)

    completed = events[completed_pass_mask(events, home_team)].copy()
    x = torch.zeros((node_count, 4), dtype=torch.float32)
    for player, node_index in node_map.items():
        player_events = events[events[player_col].eq(player) & events["team"].eq(home_team)]
        player_completed = completed[completed[player_col].eq(player)]
        x[node_index, 0] = float(len(player_completed))
        x[node_index, 1] = float(player_events["type"].eq("Shot").sum())
        if "shot_statsbomb_xg" in player_events.columns:
            x[node_index, 2] = float(player_events["shot_statsbomb_xg"].fillna(0.0).sum())
        x[node_index, 3] = float(player_events["type"].eq("Ball Recovery").sum())

    src, dst, attrs = [], [], []
    for _, row in completed.iterrows():
        passer, recipient = row.get(player_col), row.get(recipient_col)
        if passer not in node_map or recipient not in node_map:
            continue
        src.append(node_map[passer])
        dst.append(node_map[recipient])
        attrs.append([
            _as_float(row.get("pass_length"), 0.0),
            calculate_pass_angle(row),
            1.0 if _is_true(row.get("pass_shot_assist")) else 0.0,
        ])
    if not src:
        return GraphBuildResult(None, "no_completed_edges_between_selected_players", len(node_map), 0)

    role_template = graph_config["role_template"]
    if len(role_template) != node_count:
        raise ValueError("role_template length must equal node_count")

    graph = Data(
        x=x,
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.tensor(attrs, dtype=torch.float32),
        y=torch.tensor([int(label)], dtype=torch.long),
        role_ids=torch.tensor(role_template, dtype=torch.long),
        global_features=torch.zeros((1, 1), dtype=torch.float32),
        match_id=torch.tensor([int(match_id)], dtype=torch.long),
    )
    return GraphBuildResult(graph, None, node_count, len(src))
