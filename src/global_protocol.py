from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GlobalRollingOriginSplit:
    fold: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    raw_train_count: int
    train_count_after_embargo: int
    test_start_date: pd.Timestamp
    test_end_date: pd.Timestamp


def _date_series(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["date"]).dt.normalize()


def apply_competition_team_embargo(
    samples: pd.DataFrame,
    train_indices: Iterable[int],
    embargo_matches: int,
) -> np.ndarray:
    train_indices = np.asarray(list(train_indices), dtype=int)
    if embargo_matches <= 0 or len(train_indices) == 0:
        return train_indices

    train = samples.iloc[train_indices].copy()
    train["_index"] = train.index
    remove: set[int] = set()

    for competition_id, comp in train.groupby("competition_id", dropna=False):
        teams = pd.unique(pd.concat([comp["home_team"], comp["away_team"]], ignore_index=True))
        for team in teams:
            team_rows = comp[
                comp["home_team"].eq(team) | comp["away_team"].eq(team)
            ].sort_values(["date", "match_id"])
            remove.update(team_rows["_index"].to_list()[-embargo_matches:])

    return np.asarray([idx for idx in train_indices if idx not in remove], dtype=int)


def competition_aware_rolling_origin_splits(
    samples: pd.DataFrame,
    initial_train_fraction: float = 0.40,
    n_splits: int = 5,
    embargo_matches: int = 3,
):
    """Construct pooled folds while preserving chronology separately in each competition.

    Large/evaluable competitions contribute their earliest ~40% of eligible observations to an
    initial training period and their remaining future *dates* to ``n_splits`` rolling-origin
    windows. A competition that cannot support the frozen five-window protocol is NOT forced into
    tiny or duplicated folds; instead it is retained in the coverage manifest with
    ``oof_eligible=False`` and a machine-readable exclusion reason. This is essential for the
    heterogeneous Global cohort, where some open-data tournaments contain too few distinct future
    match dates for five valid windows.

    Corresponding windows from OOF-eligible competitions are pooled. Thus, no competition is
    silently discarded and no under-sized competition is used to manufacture invalid test folds.
    """
    if not 0.0 < initial_train_fraction < 1.0:
        raise ValueError("initial_train_fraction must be between 0 and 1")
    if not samples.index.equals(pd.RangeIndex(len(samples))):
        raise ValueError("samples must use a reset RangeIndex")

    working = samples.copy()
    working["_date"] = _date_series(working)

    comp_specs: dict[object, dict] = {}
    coverage_rows: list[dict] = []

    for competition_id, group in working.groupby("competition_id", dropna=False):
        group = group.sort_values(["_date", "match_id"])
        n = len(group)
        target_initial_count = int(np.ceil(n * initial_train_fraction))
        boundary_position = min(max(target_initial_count - 1, 0), n - 1)
        initial_end_date = pd.Timestamp(group.iloc[boundary_position]["_date"])
        future_dates = np.array(
            sorted(group.loc[group["_date"] > initial_end_date, "_date"].unique()),
            dtype="datetime64[ns]",
        )

        base_row = {
            "competition_id": competition_id,
            "competition_name": group.iloc[0]["competition_name"],
            "eligible_n": int(n),
            "initial_end_date": str(initial_end_date.date()),
            "future_distinct_dates": int(len(future_dates)),
            "requested_folds": int(n_splits),
        }

        if len(future_dates) < n_splits:
            coverage_rows.append({
                **base_row,
                "folds_supported": int(len(future_dates)),
                "oof_eligible": False,
                "oof_exclusion_reason": "insufficient_future_dates_for_five_fold_protocol",
            })
            continue

        date_groups = [
            pd.to_datetime(np.asarray(x)).normalize()
            for x in np.array_split(future_dates, n_splits)
        ]

        # Pre-validate every competition-specific fold before admitting the competition to the
        # pooled OOF protocol. If one fold would have an empty/fully-embargoed training set, the
        # competition is transparently excluded from OOF instead of crashing the full study.
        exclusion_reason = None
        for fold_idx, test_dates in enumerate(date_groups, start=1):
            if len(test_dates) == 0:
                exclusion_reason = f"empty_test_date_group_fold_{fold_idx}"
                break
            test_start = pd.Timestamp(test_dates.min())
            raw_train = group.index[group["_date"] < test_start].to_numpy(dtype=int)
            test_idx = group.index[group["_date"].isin(test_dates)].to_numpy(dtype=int)
            train_after = apply_competition_team_embargo(working, raw_train, embargo_matches)
            if len(raw_train) == 0 or len(test_idx) == 0:
                exclusion_reason = f"empty_train_or_test_partition_fold_{fold_idx}"
                break
            if len(train_after) == 0:
                exclusion_reason = f"embargo_removed_all_training_rows_fold_{fold_idx}"
                break
            if pd.to_datetime(working.loc[train_after, "date"]).max() >= pd.to_datetime(working.loc[test_idx, "date"]).min():
                exclusion_reason = f"chronology_guard_failed_fold_{fold_idx}"
                break

        if exclusion_reason is not None:
            coverage_rows.append({
                **base_row,
                "folds_supported": 0,
                "oof_eligible": False,
                "oof_exclusion_reason": exclusion_reason,
            })
            continue

        comp_specs[competition_id] = {
            "group_indices": group.index.to_numpy(dtype=int),
            "initial_end_date": initial_end_date,
            "date_groups": date_groups,
        }
        coverage_rows.append({
            **base_row,
            "folds_supported": int(n_splits),
            "oof_eligible": True,
            "oof_exclusion_reason": None,
        })

    if not comp_specs:
        raise RuntimeError(
            "No competition can support the requested competition-aware five-fold rolling-origin protocol. "
            "Inspect global_competition_fold_coverage.csv / cohort composition."
        )

    splits: list[GlobalRollingOriginSplit] = []
    comp_fold_rows: list[dict] = []

    for fold in range(1, n_splits + 1):
        raw_train_all: list[int] = []
        test_all: list[int] = []
        test_dates_all: list[pd.Timestamp] = []

        for competition_id, spec in comp_specs.items():
            group = working.loc[spec["group_indices"]]
            test_dates = spec["date_groups"][fold - 1]
            test_start = pd.Timestamp(test_dates.min())
            test_end = pd.Timestamp(test_dates.max())
            raw_train = group.index[group["_date"] < test_start].to_numpy(dtype=int)
            test_idx = group.index[group["_date"].isin(test_dates)].to_numpy(dtype=int)
            train_after = apply_competition_team_embargo(working, raw_train, embargo_matches)

            # These conditions were pre-validated above; keep hard guards here so a future code
            # regression cannot silently change the protocol.
            if len(raw_train) == 0 or len(test_idx) == 0:
                raise RuntimeError(f"Unexpected empty competition train/test partition: competition_id={competition_id}, fold={fold}")
            if len(train_after) == 0:
                raise RuntimeError(f"Unexpected fully embargoed competition training partition: competition_id={competition_id}, fold={fold}")
            if pd.to_datetime(working.loc[train_after, "date"]).max() >= pd.to_datetime(working.loc[test_idx, "date"]).min():
                raise RuntimeError(f"Within-competition chronology violation: competition_id={competition_id}, fold={fold}")

            raw_train_all.extend(raw_train.tolist())
            test_all.extend(test_idx.tolist())
            test_dates_all.extend(pd.to_datetime(working.loc[test_idx, "date"]).tolist())
            comp_fold_rows.append({
                "competition_id": competition_id,
                "competition_name": group.iloc[0]["competition_name"],
                "fold": fold,
                "raw_train_n": int(len(raw_train)),
                "train_after_embargo_n": int(len(train_after)),
                "test_n": int(len(test_idx)),
                "test_start_date": str(test_start.date()),
                "test_end_date": str(test_end.date()),
            })

        raw_train_unique = np.asarray(sorted(set(raw_train_all)), dtype=int)
        test_unique = np.asarray(sorted(set(test_all)), dtype=int)
        train_after_unique = apply_competition_team_embargo(working, raw_train_unique, embargo_matches)

        if len(test_unique) == 0:
            raise RuntimeError(f"Global fold {fold} has no pooled test observations")
        if set(train_after_unique).intersection(set(test_unique)):
            raise RuntimeError(f"Train/test overlap detected in global fold {fold}")
        splits.append(GlobalRollingOriginSplit(
            fold=fold,
            train_indices=train_after_unique,
            test_indices=test_unique,
            raw_train_count=int(len(raw_train_unique)),
            train_count_after_embargo=int(len(train_after_unique)),
            test_start_date=pd.Timestamp(min(test_dates_all)),
            test_end_date=pd.Timestamp(max(test_dates_all)),
        ))

    return splits, pd.DataFrame(coverage_rows), pd.DataFrame(comp_fold_rows)

def competition_aware_inner_split(
    samples: pd.DataFrame,
    outer_train_indices: np.ndarray,
    validation_fraction: float = 0.20,
    embargo_matches: int = 3,
    min_total_train: int = 200,
    min_total_validation: int = 100,
):
    """Create pooled inner train/validation partitions with chronology preserved in each competition."""
    outer_train_indices = np.asarray(outer_train_indices, dtype=int)
    working = samples.copy()
    working["_date"] = _date_series(working)
    outer = working.loc[outer_train_indices]

    raw_inner_train_all: list[int] = []
    validation_all: list[int] = []
    audit_rows: list[dict] = []

    for competition_id, group in outer.groupby("competition_id", dropna=False):
        group = group.sort_values(["_date", "match_id"])
        distinct_dates = np.array(sorted(group["_date"].unique()), dtype="datetime64[ns]")

        # Tiny competition partitions remain training-only for epoch selection rather than
        # producing a meaningless 1-2 match validation subset. This is recorded explicitly.
        if len(group) < 12 or len(distinct_dates) < 3:
            raw_inner_train_all.extend(group.index.to_list())
            audit_rows.append({
                "competition_id": competition_id,
                "competition_name": group.iloc[0]["competition_name"],
                "outer_train_n": int(len(group)),
                "raw_inner_train_n": int(len(group)),
                "inner_train_after_embargo_n": int(len(group)),
                "inner_validation_n": 0,
                "validation_start_date": None,
                "status": "training_only_small_partition",
            })
            continue

        validation_target = max(1, int(np.ceil(len(group) * validation_fraction)))
        cutoff_position = max(1, len(group) - validation_target)
        validation_start_date = pd.Timestamp(group.iloc[cutoff_position]["_date"])
        raw_inner_train = group.index[group["_date"] < validation_start_date].to_numpy(dtype=int)
        validation = group.index[group["_date"] >= validation_start_date].to_numpy(dtype=int)
        train_after = apply_competition_team_embargo(working, raw_inner_train, embargo_matches)

        if len(train_after) == 0 or len(validation) == 0:
            raw_inner_train_all.extend(group.index.to_list())
            audit_rows.append({
                "competition_id": competition_id,
                "competition_name": group.iloc[0]["competition_name"],
                "outer_train_n": int(len(group)),
                "raw_inner_train_n": int(len(group)),
                "inner_train_after_embargo_n": int(len(group)),
                "inner_validation_n": 0,
                "validation_start_date": None,
                "status": "training_only_after_embargo_guard",
            })
            continue

        raw_inner_train_all.extend(raw_inner_train.tolist())
        validation_all.extend(validation.tolist())
        audit_rows.append({
            "competition_id": competition_id,
            "competition_name": group.iloc[0]["competition_name"],
            "outer_train_n": int(len(group)),
            "raw_inner_train_n": int(len(raw_inner_train)),
            "inner_train_after_embargo_n": int(len(train_after)),
            "inner_validation_n": int(len(validation)),
            "validation_start_date": str(validation_start_date.date()),
            "status": "chronological_validation",
        })

    raw_train_unique = np.asarray(sorted(set(raw_inner_train_all)), dtype=int)
    validation_unique = np.asarray(sorted(set(validation_all)), dtype=int)
    train_after_unique = apply_competition_team_embargo(working, raw_train_unique, embargo_matches)

    if len(train_after_unique) < min_total_train:
        raise RuntimeError(f"Global inner training set too small: {len(train_after_unique)} < {min_total_train}")
    if len(validation_unique) < min_total_validation:
        raise RuntimeError(f"Global inner validation set too small: {len(validation_unique)} < {min_total_validation}")
    if set(train_after_unique).intersection(set(validation_unique)):
        raise RuntimeError("Global inner train/validation overlap detected")

    # Check chronology separately in every competition that contributes validation rows.
    for competition_id in pd.unique(working.loc[validation_unique, "competition_id"]):
        comp_train = train_after_unique[working.loc[train_after_unique, "competition_id"].to_numpy() == competition_id]
        comp_val = validation_unique[working.loc[validation_unique, "competition_id"].to_numpy() == competition_id]
        if len(comp_train) and len(comp_val):
            if pd.to_datetime(working.loc[comp_train, "date"]).max() >= pd.to_datetime(working.loc[comp_val, "date"]).min():
                raise RuntimeError(f"Inner within-competition chronology violation for competition_id={competition_id}")

    return train_after_unique, validation_unique, pd.DataFrame(audit_rows)
