import numpy as np
import pandas as pd

from src.global_protocol import competition_aware_rolling_origin_splits


def _synthetic_samples():
    rows = []
    idx = 0
    for comp_id, comp_name in [(1, "League A"), (2, "Cup B")]:
        for day in range(1, 31):
            rows.append({
                "match_id": 1000 + idx,
                "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
                "competition_id": comp_id,
                "competition_name": comp_name,
                "home_team": f"H{day % 6}",
                "away_team": f"A{(day + 1) % 6}",
                "label": day % 2,
            })
            idx += 1
    return pd.DataFrame(rows).sort_values(["date", "match_id"]).reset_index(drop=True)


def test_competition_aware_splits_cover_future_without_overlap():
    samples = _synthetic_samples()
    splits, coverage, comp_folds = competition_aware_rolling_origin_splits(
        samples, initial_train_fraction=0.40, n_splits=5, embargo_matches=1
    )
    assert len(splits) == 5
    assert len(coverage) == 2
    seen = set()
    for split in splits:
        assert not set(split.train_indices).intersection(set(split.test_indices))
        for idx in split.test_indices:
            assert idx not in seen
            seen.add(int(idx))
    assert len(comp_folds) == 10


def test_small_competition_is_manifested_but_not_forced_into_oof():
    large = _synthetic_samples()
    small_rows = []
    base_id = 5000
    # 8 observations on only 4 dates. After the 40% initial window there cannot be 5 future dates.
    for i in range(8):
        small_rows.append({
            "match_id": base_id + i,
            "date": pd.Timestamp("2021-01-01") + pd.Timedelta(days=i // 2),
            "competition_id": 99,
            "competition_name": "Tiny Cup",
            "home_team": f"T{i % 4}",
            "away_team": f"U{(i + 1) % 4}",
            "label": i % 2,
        })
    samples = pd.concat([large, pd.DataFrame(small_rows)], ignore_index=True)
    samples = samples.sort_values(["date", "match_id"]).reset_index(drop=True)

    splits, coverage, comp_folds = competition_aware_rolling_origin_splits(
        samples, initial_train_fraction=0.40, n_splits=5, embargo_matches=1
    )

    tiny = coverage.loc[coverage.competition_id.eq(99)].iloc[0]
    assert bool(tiny.oof_eligible) is False
    assert tiny.oof_exclusion_reason == "insufficient_future_dates_for_five_fold_protocol"
    assert 99 not in set(comp_folds.competition_id)
    assert len(splits) == 5
