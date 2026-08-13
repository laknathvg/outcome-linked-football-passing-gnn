from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from statsbombpy import sb
from tqdm import tqdm

from .data_pipeline import (
    _integer,
    _metadata_text,
    count_team_stats,
    select_competitions,
)
from .graph_builder import build_home_team_graph


@dataclass(frozen=True)
class RollingOriginSplit:
    fold: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    raw_train_count: int
    train_count_after_embargo: int
    test_start_date: pd.Timestamp
    test_end_date: pd.Timestamp


def _retry_call(fn: Callable[[], Any], attempts: int = 5, base_wait_seconds: float = 1.0):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return fn(), attempt, None
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(base_wait_seconds * (2 ** (attempt - 1)))
    return None, attempts, repr(last_error)


def _load_events_cached(match_id: int, cache_dir: Path, attempts: int = 5):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{match_id}.pkl"
    if cache_path.exists():
        try:
            return pd.read_pickle(cache_path), 0, "cache", None
        except Exception:
            cache_path.unlink(missing_ok=True)

    events, used_attempts, error = _retry_call(
        lambda: sb.events(match_id=match_id, split=False),
        attempts=attempts,
    )
    if events is not None:
        events.to_pickle(cache_path)
        return events, used_attempts, "statsbombpy", None
    return None, used_attempts, "statsbombpy", error


def harvest_dataset_with_retry_cache(
    config: dict,
    attempts: int = 5,
    fail_on_event_error: bool = True,
):
    competitions, comp_attempts, comp_error = _retry_call(sb.competitions, attempts=attempts)
    if competitions is None:
        raise RuntimeError(f"Could not load StatsBomb competitions after {comp_attempts} attempts: {comp_error}")

    selected = select_competitions(competitions, config["cohort"])
    match_records: list[dict[str, Any]] = []

    for _, competition in selected.iterrows():
        competition_id = int(competition["competition_id"])
        season_id = int(competition["season_id"])
        matches, used_attempts, error = _retry_call(
            lambda cid=competition_id, sid=season_id: sb.matches(competition_id=cid, season_id=sid).copy(),
            attempts=attempts,
        )
        if matches is None:
            raise RuntimeError(
                f"Could not load matches for competition_id={competition_id}, season_id={season_id} "
                f"after {used_attempts} attempts: {error}"
            )
        matches["_competition_id"] = competition_id
        matches["_season_id"] = season_id
        matches["_competition_name"] = competition["competition_name"]
        matches["_season_name"] = competition.get("season_name", "")
        matches["_competition_gender"] = competition.get("competition_gender", "")
        match_records.extend(matches.to_dict("records"))

    unique: dict[int, dict[str, Any]] = {}
    for record in match_records:
        unique.setdefault(int(record["match_id"]), record)

    ordered = sorted(
        unique.values(),
        key=lambda row: (pd.to_datetime(row["match_date"]), int(row["match_id"])),
    )

    maximum = config["cohort"].get("max_matches")
    if maximum:
        ordered = ordered[: int(maximum)]

    cache_dir = Path(config["project"].get("cache_dir", "data/cache")) / "events"
    manifest_rows: list[dict[str, Any]] = []
    stat_rows: list[dict[str, Any]] = []
    graphs: dict[int, Any] = {}

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
            "event_load_attempts": None,
            "event_source": None,
            "event_load_error": None,
        }

        home_team = _metadata_text(match.get("home_team"))
        away_team = _metadata_text(match.get("away_team"))
        home_score = _integer(match.get("home_score"))
        away_score = _integer(match.get("away_score"))
        manifest["home_team"] = home_team
        manifest["away_team"] = away_team

        if home_team is None or away_team is None or home_score is None or away_score is None:
            manifest["exclusion_reason"] = "missing_authoritative_match_metadata"
            manifest_rows.append(manifest)
            continue

        label = int(home_score > away_score)
        manifest["label"] = label

        events, used_attempts, source, error = _load_events_cached(
            match_id=match_id,
            cache_dir=cache_dir,
            attempts=attempts,
        )
        manifest["event_load_attempts"] = used_attempts
        manifest["event_source"] = source
        manifest["event_load_error"] = error

        if events is None:
            manifest["exclusion_reason"] = "event_load_failed"
            manifest_rows.append(manifest)
            if fail_on_event_error:
                raise RuntimeError(
                    f"Event loading failed for match_id={match_id} after {used_attempts} attempts. "
                    f"The run is stopped so transient network failures cannot silently change the cohort. Error: {error}"
                )
            continue

        event_teams = set(events["team"].dropna().astype(str).unique())
        if home_team not in event_teams or away_team not in event_teams:
            manifest["exclusion_reason"] = "metadata_team_missing_from_events"
            manifest_rows.append(manifest)
            continue

        home_stats = count_team_stats(events, home_team)
        away_stats = count_team_stats(events, away_team)

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

    candidate_manifest = pd.DataFrame([
        {
            "match_id": int(match["match_id"]),
            "date": str(match.get("match_date", "")),
            "competition_id": match.get("_competition_id"),
            "competition_name": match.get("_competition_name"),
            "season_id": match.get("_season_id"),
            "season_name": match.get("_season_name"),
            "home_team": _metadata_text(match.get("home_team")),
            "away_team": _metadata_text(match.get("away_team")),
        }
        for match in ordered
    ])

    return pd.DataFrame(stat_rows), graphs, pd.DataFrame(manifest_rows), candidate_manifest


def apply_team_embargo(samples: pd.DataFrame, train_indices: np.ndarray, embargo_matches: int) -> np.ndarray:
    train_indices = np.asarray(train_indices, dtype=int)
    if embargo_matches <= 0 or len(train_indices) == 0:
        return train_indices

    train = samples.iloc[train_indices]
    remove: set[int] = set()
    teams = pd.unique(pd.concat([train["home_team"], train["away_team"]], ignore_index=True))

    for team in teams:
        team_rows = train[
            train["home_team"].eq(team) | train["away_team"].eq(team)
        ].sort_values(["date", "match_id"])
        remove.update(team_rows.index.to_list()[-embargo_matches:])

    return np.asarray([idx for idx in train_indices if idx not in remove], dtype=int)


def rolling_origin_splits(
    samples: pd.DataFrame,
    initial_train_fraction: float = 0.40,
    n_splits: int = 5,
    embargo_matches: int = 3,
) -> list[RollingOriginSplit]:
    if not 0.0 < initial_train_fraction < 1.0:
        raise ValueError("initial_train_fraction must be between 0 and 1")

    working = samples.copy()
    working["_date"] = pd.to_datetime(working["date"]).dt.normalize()
    if not working.index.equals(pd.RangeIndex(len(working))):
        raise ValueError("samples must have a reset RangeIndex")

    target_initial_count = int(np.ceil(len(working) * initial_train_fraction))
    boundary_position = min(max(target_initial_count - 1, 0), len(working) - 1)
    initial_end_date = working.iloc[boundary_position]["_date"]

    future_dates = np.array(
        sorted(working.loc[working["_date"] > initial_end_date, "_date"].unique()),
        dtype="datetime64[ns]",
    )
    if len(future_dates) < n_splits:
        raise RuntimeError("Not enough distinct future dates to construct the requested rolling-origin folds")

    date_groups = [np.asarray(group) for group in np.array_split(future_dates, n_splits)]
    splits: list[RollingOriginSplit] = []

    for fold, test_dates_array in enumerate(date_groups, start=1):
        if len(test_dates_array) == 0:
            raise RuntimeError(f"Empty test date group in fold {fold}")
        test_dates = pd.to_datetime(test_dates_array).normalize()
        test_start = pd.Timestamp(test_dates.min())
        test_end = pd.Timestamp(test_dates.max())

        raw_train_indices = working.index[working["_date"] < test_start].to_numpy(dtype=int)
        test_indices = working.index[working["_date"].isin(test_dates)].to_numpy(dtype=int)
        train_indices = apply_team_embargo(working, raw_train_indices, embargo_matches)

        if len(train_indices) == 0 or len(test_indices) == 0:
            raise RuntimeError(f"Invalid empty train/test partition in fold {fold}")
        if pd.to_datetime(working.iloc[train_indices]["date"]).max() >= pd.to_datetime(working.iloc[test_indices]["date"]).min():
            raise RuntimeError(f"Chronology violation in rolling-origin fold {fold}")

        splits.append(RollingOriginSplit(
            fold=fold,
            train_indices=train_indices,
            test_indices=test_indices,
            raw_train_count=len(raw_train_indices),
            train_count_after_embargo=len(train_indices),
            test_start_date=test_start,
            test_end_date=test_end,
        ))

    return splits


def date_aware_inner_split(
    samples: pd.DataFrame,
    outer_train_indices: np.ndarray,
    validation_fraction: float = 0.20,
    embargo_matches: int = 3,
    min_train: int = 100,
    min_validation: int = 25,
):
    outer_train_indices = np.asarray(outer_train_indices, dtype=int)
    ordered = samples.iloc[outer_train_indices].sort_values(["date", "match_id"]).copy()
    ordered["_date"] = pd.to_datetime(ordered["date"]).dt.normalize()

    validation_target = max(min_validation, int(np.ceil(len(ordered) * validation_fraction)))
    cutoff_position = max(1, len(ordered) - validation_target)
    validation_start_date = pd.Timestamp(ordered.iloc[cutoff_position]["_date"])

    raw_inner_train = ordered.index[ordered["_date"] < validation_start_date].to_numpy(dtype=int)
    validation_indices = ordered.index[ordered["_date"] >= validation_start_date].to_numpy(dtype=int)
    inner_train_indices = apply_team_embargo(samples, raw_inner_train, embargo_matches)

    if len(inner_train_indices) < min_train:
        raise RuntimeError(
            f"Inner training set is too small after embargo: {len(inner_train_indices)} < {min_train}. "
            "The protocol is intentionally stopped rather than selecting a GNN epoch from an underpowered training set."
        )
    if len(validation_indices) < min_validation:
        raise RuntimeError(
            f"Inner validation set is too small: {len(validation_indices)} < {min_validation}."
        )
    if pd.to_datetime(samples.iloc[inner_train_indices]["date"]).max() >= pd.to_datetime(samples.iloc[validation_indices]["date"]).min():
        raise RuntimeError("Chronology violation in inner split")

    return inner_train_indices, validation_indices, {
        "raw_inner_train_count": int(len(raw_inner_train)),
        "inner_train_count_after_embargo": int(len(inner_train_indices)),
        "inner_validation_count": int(len(validation_indices)),
        "validation_start_date": str(validation_start_date.date()),
    }
