from collections import deque
from typing import Any
import numpy as np
import pandas as pd

TEAM_STATS = [
    "goals_for", "goals_against", "shots_for", "shots_against",
    "shots_on_target_for", "shots_on_target_against", "corners_for", "corners_against",
]


def make_team_perspective_rows(match_stats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, match in match_stats.iterrows():
        common = {
            "match_id": int(match["match_id"]),
            "date": pd.to_datetime(match["date"]),
            "competition_id": match["competition_id"],
            "competition_name": match["competition_name"],
        }
        rows.append({**common, "side": "home", "team": match["home_team"], "opponent": match["away_team"],
                     "goals_for": match["home_score"], "goals_against": match["away_score"],
                     "shots_for": match["home_shots"], "shots_against": match["away_shots"],
                     "shots_on_target_for": match["home_shots_on_target"], "shots_on_target_against": match["away_shots_on_target"],
                     "corners_for": match["home_corners"], "corners_against": match["away_corners"]})
        rows.append({**common, "side": "away", "team": match["away_team"], "opponent": match["home_team"],
                     "goals_for": match["away_score"], "goals_against": match["home_score"],
                     "shots_for": match["away_shots"], "shots_against": match["home_shots"],
                     "shots_on_target_for": match["away_shots_on_target"], "shots_on_target_against": match["home_shots_on_target"],
                     "corners_for": match["away_corners"], "corners_against": match["home_corners"]})
    return pd.DataFrame(rows)


def compute_rolling_history(match_stats: pd.DataFrame, window: int, group_by_competition: bool) -> pd.DataFrame:
    long_df = make_team_perspective_rows(match_stats)
    grouping = ["team"] + (["competition_id"] if group_by_competition else [])
    output = []
    for _, group in long_df.groupby(grouping, dropna=False):
        previous = deque(maxlen=window)
        for _, row in group.sort_values(["date", "match_id"]).iterrows():
            result = row.to_dict()
            result["history_available"] = len(previous) == window
            result["history_source_match_ids"] = "|".join(str(int(item["match_id"])) for item in previous)
            for stat in TEAM_STATS:
                result[f"roll_{stat}"] = float(np.mean([float(item[stat]) for item in previous])) if len(previous) == window else np.nan
            output.append(result)
            previous.append(row.to_dict())
    history = pd.DataFrame(output)

    home = history[history["side"].eq("home")].copy()
    away = history[history["side"].eq("away")].copy()
    home_map = {f"roll_{s}": f"home_roll_{s}" for s in TEAM_STATS}
    away_map = {f"roll_{s}": f"away_roll_{s}" for s in TEAM_STATS}
    home = home.rename(columns=home_map)[["match_id", "history_available", "history_source_match_ids", *home_map.values()]]
    away = away.rename(columns=away_map)[["match_id", "history_available", "history_source_match_ids", *away_map.values()]]
    home = home.rename(columns={"history_available": "home_history_available", "history_source_match_ids": "home_history_source_match_ids"})
    away = away.rename(columns={"history_available": "away_history_available", "history_source_match_ids": "away_history_source_match_ids"})
    merged = match_stats.merge(home, on="match_id", how="left").merge(away, on="match_id", how="left")
    merged["history_available"] = merged["home_history_available"].fillna(False) & merged["away_history_available"].fillna(False)
    return merged


def align_samples(history_df: pd.DataFrame, graphs: dict[int, Any]):
    history_columns = [c for c in history_df.columns if c.startswith("home_roll_") or c.startswith("away_roll_")]
    samples = history_df[history_df["history_available"] & history_df["match_id"].isin(graphs)].copy()
    samples = samples.sort_values(["date", "match_id"]).reset_index(drop=True)
    graph_list = [graphs[int(mid)] for mid in samples["match_id"]]
    X_history = samples[history_columns].to_numpy(dtype=np.float32)
    y = samples["label"].to_numpy(dtype=np.int64)
    return samples, graph_list, X_history, y, history_columns
