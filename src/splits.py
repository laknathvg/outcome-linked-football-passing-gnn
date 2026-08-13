from dataclasses import dataclass
from typing import Iterator
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

@dataclass(frozen=True)
class FoldSplit:
    fold: int
    train_indices: np.ndarray
    test_indices: np.ndarray


def _apply_team_embargo(samples: pd.DataFrame, train_indices: np.ndarray, embargo_matches: int) -> np.ndarray:
    if embargo_matches <= 0:
        return train_indices
    train = samples.iloc[train_indices]
    remove = set()
    teams = pd.unique(pd.concat([train["home_team"], train["away_team"]], ignore_index=True))
    for team in teams:
        positions = train[train["home_team"].eq(team) | train["away_team"].eq(team)].sort_values(["date", "match_id"]).index.to_list()
        remove.update(positions[-embargo_matches:])
    return np.asarray([i for i in train_indices if i not in remove], dtype=int)


def forward_chaining_splits(samples: pd.DataFrame, n_splits: int, embargo_matches: int) -> Iterator[FoldSplit]:
    blocks = np.array_split(np.arange(len(samples)), n_splits + 1)
    for fold_number in range(1, n_splits + 1):
        train_indices = np.concatenate(blocks[:fold_number])
        test_indices = blocks[fold_number]
        if not len(train_indices) or not len(test_indices):
            continue
        train_indices = _apply_team_embargo(samples, train_indices, embargo_matches)
        yield FoldSplit(fold_number, train_indices, test_indices)


def stratified_splits(y: np.ndarray, n_splits: int, seed: int) -> Iterator[FoldSplit]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_indices, test_indices) in enumerate(splitter.split(np.zeros((len(y), 1)), y), start=1):
        yield FoldSplit(fold, train_indices, test_indices)


def chronological_inner_split(samples: pd.DataFrame, outer_train_indices: np.ndarray, validation_fraction: float, embargo_matches: int):
    ordered = samples.iloc[outer_train_indices].sort_values(["date", "match_id"])
    validation_size = max(1, int(round(len(ordered) * validation_fraction)))
    validation_positions = ordered.index.to_numpy()[-validation_size:]
    train_positions = ordered.index.to_numpy()[:-validation_size]
    train_positions = _apply_team_embargo(samples, train_positions, embargo_matches)
    return train_positions, validation_positions
