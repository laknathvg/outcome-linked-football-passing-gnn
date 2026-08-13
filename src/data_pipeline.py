from typing import Any
import pandas as pd
from statsbombpy import sb
from tqdm import tqdm
from .graph_builder import build_home_team_graph


def _metadata_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ["home_team_name", "away_team_name", "team_name", "name"]:
            if key in value:
                return str(value[key])
        return None
    if pd.isna(value):
        return None
    return str(value)


def _integer(value: Any) -> int | None:
    try:
        return None if pd.isna(value) else int(value)
    except (TypeError, ValueError):
        return None


def count_team_stats(events: pd.DataFrame, team_name: str) -> dict[str, int]:
    team_events = events[events["team"].eq(team_name)]
    shots = team_events[team_events["type"].eq("Shot")]
    on_target_outcomes = {"Goal", "Saved", "Saved to Post", "Saved Off Target"}
    shots_on_target = int(shots["shot_outcome"].isin(on_target_outcomes).sum()) if "shot_outcome" in shots.columns else 0
    corners = int(team_events["pass_type"].eq("Corner").sum()) if "pass_type" in team_events.columns else 0
    return {"shots": int(len(shots)), "shots_on_target": shots_on_target, "corners": corners}


def select_competitions(competitions: pd.DataFrame, cohort_config: dict) -> pd.DataFrame:
    if cohort_config["mode"] == "all":
        return competitions.copy()
    if cohort_config["mode"] == "named":
        return competitions[competitions["competition_name"].isin(set(cohort_config["competition_names"]))].copy()
    raise ValueError(f"Unsupported cohort mode: {cohort_config['mode']}")


def harvest_dataset(config: dict) -> tuple[pd.DataFrame, dict[int, Any], pd.DataFrame]:
    competitions = sb.competitions()
    selected = select_competitions(competitions, config["cohort"])
    match_records = []
    for _, competition in selected.iterrows():
        try:
            matches = sb.matches(competition_id=int(competition["competition_id"]), season_id=int(competition["season_id"])).copy()
        except Exception:
            continue
        matches["_competition_id"] = int(competition["competition_id"])
        matches["_season_id"] = int(competition["season_id"])
        matches["_competition_name"] = competition["competition_name"]
        matches["_season_name"] = competition.get("season_name", "")
        matches["_competition_gender"] = competition.get("competition_gender", "")
        match_records.extend(matches.to_dict("records"))

    unique = {}
    for record in match_records:
        unique.setdefault(int(record["match_id"]), record)
    ordered = sorted(unique.values(), key=lambda row: (pd.to_datetime(row["match_date"]), int(row["match_id"])))
    maximum = config["cohort"].get("max_matches")
    if maximum:
        ordered = ordered[: int(maximum)]

    manifest_rows, stat_rows = [], []
    graphs = {}
    for match in tqdm(ordered, desc="Harvesting matches"):
        match_id = int(match["match_id"])
        manifest = {
            "match_id": match_id,
            "date": str(match.get("match_date", "")),
            "competition_id": match.get("_competition_id"),
            "competition_name": match.get("_competition_name"),
            "season_id": match.get("_season_id"),
            "season_name": match.get("_season_name"),
            "category": match.get("_competition_gender", ""),
            "home_team": None,
            "away_team": None,
            "label": None,
            "graph_included": False,
            "history_available": False,
            "final_included": False,
            "exclusion_reason": None,
            "selected_player_count": 0,
            "completed_edge_count": 0,
        }

        home_team = _metadata_text(match.get("home_team"))
        away_team = _metadata_text(match.get("away_team"))
        home_score = _integer(match.get("home_score"))
        away_score = _integer(match.get("away_score"))
        manifest["home_team"], manifest["away_team"] = home_team, away_team
        if home_team is None or away_team is None or home_score is None or away_score is None:
            manifest["exclusion_reason"] = "missing_authoritative_match_metadata"
            manifest_rows.append(manifest)
            continue

        label = int(home_score > away_score)
        manifest["label"] = label
        try:
            events = sb.events(match_id=match_id, split=False)
        except Exception:
            manifest["exclusion_reason"] = "event_load_failed"
            manifest_rows.append(manifest)
            continue

        event_teams = set(events["team"].dropna().astype(str).unique())
        if home_team not in event_teams or away_team not in event_teams:
            manifest["exclusion_reason"] = "metadata_team_missing_from_events"
            manifest_rows.append(manifest)
            continue

        home_stats, away_stats = count_team_stats(events, home_team), count_team_stats(events, away_team)
        stat_rows.append({
            "match_id": match_id,
            "date": pd.to_datetime(match["match_date"]),
            "competition_id": match.get("_competition_id"),
            "competition_name": match.get("_competition_name"),
            "season_id": match.get("_season_id"),
            "season_name": match.get("_season_name"),
            "category": match.get("_competition_gender", ""),
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "label": label,
            "home_shots": home_stats["shots"],
            "away_shots": away_stats["shots"],
            "home_shots_on_target": home_stats["shots_on_target"],
            "away_shots_on_target": away_stats["shots_on_target"],
            "home_corners": home_stats["corners"],
            "away_corners": away_stats["corners"],
        })

        built = build_home_team_graph(events, match_id, home_team, label, config)
        manifest["selected_player_count"] = built.selected_player_count
        manifest["completed_edge_count"] = built.completed_edge_count
        if built.graph is None:
            manifest["exclusion_reason"] = built.exclusion_reason
        else:
            manifest["graph_included"] = True
            graphs[match_id] = built.graph
        manifest_rows.append(manifest)

    return pd.DataFrame(stat_rows), graphs, pd.DataFrame(manifest_rows)
