import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def metric_row(truth, prediction, score):
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    return {
        "n": int(len(truth)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "win_precision": float(precision_score(truth, prediction, pos_label=1, zero_division=0)),
        "win_recall": float(recall_score(truth, prediction, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, score)) if len(np.unique(truth)) == 2 else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def build_results_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(["cohort", "protocol", "model", "fold", "seed"]):
        row = dict(zip(["cohort", "protocol", "model", "fold", "seed"], keys)); row.update(metric_row(group.true_label, group.predicted_class, group.predicted_score)); row["scope"] = "fold"; rows.append(row)
    for keys, group in predictions.groupby(["cohort", "protocol", "model"]):
        row = dict(zip(["cohort", "protocol", "model"], keys))
        row["fold"] = "pooled"
        row["seed"] = "multiple_fold_specific"
        row.update(metric_row(group.true_label, group.predicted_class, group.predicted_score))
        row["scope"] = "pooled"
        rows.append(row)
    return pd.DataFrame(rows)


def prediction_rows(samples, test_indices, truth, prediction, score, cohort, protocol, model, fold, seed, gate_value=None):
    rows = []
    for local, sample_index in enumerate(test_indices):
        sample = samples.iloc[int(sample_index)]
        rows.append({
            "match_id": int(sample["match_id"]), "date": sample["date"], "competition_id": sample["competition_id"],
            "competition_name": sample["competition_name"], "season_id": sample["season_id"], "category": sample.get("category", ""),
            "home_team": sample["home_team"], "away_team": sample["away_team"], "cohort": cohort, "protocol": protocol,
            "fold": fold, "seed": seed, "model": model, "true_label": int(truth[local]),
            "predicted_class": int(prediction[local]), "predicted_score": float(score[local]), "gate_value": gate_value,
        })
    return rows
