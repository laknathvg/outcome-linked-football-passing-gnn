from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import yaml
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src.evaluation import build_results_table, prediction_rows
from src.final_protocol import harvest_dataset_with_retry_cache
from src.global_protocol import (
    competition_aware_inner_split,
    competition_aware_rolling_origin_splits,
)
from src.history import align_samples, compute_rolling_history
from src.models import HistoryMLP, HybridGNN, StructureGNN
from src.reproducibility import save_environment, set_global_seed
from src.scaling import fit_graph_scalers, make_feature_view, transform_graphs
from src.training import (
    fit_graph_model,
    fit_history_model,
    predict_graph_model,
    predict_history_model,
    select_graph_epoch,
    select_history_epoch,
)

PROJECT_ROOT = Path.cwd()
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiment5_global_pooled_consistency"
FINAL_ZIP = Path("/content/football_gnn_experiment5_global_pooled_consistency_final.zip")
BASE_CONFIG_PATH = PROJECT_ROOT / "config_final.yaml"
BASE_SEED = 42
INITIAL_TRAIN_FRACTION = 0.40
N_OUTER_FOLDS = 5
TEAM_EMBARGO_MATCHES = 3
INNER_VALIDATION_FRACTION = 0.20
BOOTSTRAP_SAMPLES = 10000
SUBGROUP_BOOTSTRAP_SAMPLES = 5000


def fold_seed(fold: int) -> int:
    return BASE_SEED + 100 * fold


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def full_graph_views(graphs: list, indices: np.ndarray):
    return [
        make_feature_view(graphs[int(i)], node_indices=[0, 1, 2, 3], edge_indices=[0, 1, 2])
        for i in indices
    ]


def preprocess_graph_pair(train_raw, other_raw, train_history, other_history):
    scalers = fit_graph_scalers(
        graphs=train_raw,
        standardize_nodes=True,
        edge_continuous_indices=[0, 1],
    )
    return (
        transform_graphs(train_raw, scalers, train_history),
        transform_graphs(other_raw, scalers, other_history),
    )


def xgb_model(config: dict, seed: int):
    s = config["xgboost"]
    return xgb.XGBClassifier(
        n_estimators=int(s["n_estimators"]),
        max_depth=int(s["max_depth"]),
        learning_rate=float(s["learning_rate"]),
        subsample=float(s["subsample"]),
        colsample_bytree=float(s["colsample_bytree"]),
        eval_metric=s["eval_metric"],
        random_state=seed,
        n_jobs=int(s["n_jobs"]),
    )


def build_config():
    with BASE_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["project"]["seed"] = BASE_SEED
    config["project"]["output_dir"] = "outputs/experiment5_global_pooled_consistency"
    # External cache survives a Colab re-run even if the extracted package is refreshed.
    config["project"]["cache_dir"] = "/content/statsbomb_global_event_cache"
    config["cohort"] = {
        "name": "global",
        "mode": "all",
        "competition_names": [],
        "max_matches": None,
    }
    config["history"]["window_matches"] = 3
    config["history"]["group_by_competition"] = True
    config["graph"]["node_count"] = 11
    config["graph"]["node_features"] = [
        "completed_pass_count", "shot_count", "shot_xg_sum", "ball_recovery_count"
    ]
    config["graph"]["edge_features"] = ["pass_length", "pass_angle", "key_pass_indicator"]
    config["graph"]["standardize_node_features"] = True
    config["graph"]["standardize_edge_continuous_indices"] = [0, 1]
    config["model"]["hidden_dim"] = 64
    config["model"]["attention_heads"] = 4
    config["model"]["attention_concat"] = False
    config["model"]["role_embedding_dim"] = 8
    config["model"]["graph_dropout"] = 0.30
    config["model"]["classifier_dropout"] = 0.40
    config["model"]["history_hidden_dim"] = 32
    config["model"]["history_dropout"] = 0.20
    config["model"]["pooling"] = "positional_add"
    config["model"]["fusion"] = {"type": "static_sigmoid", "initial_lambda": 0.50}
    config["training"]["optimizer"] = "Adam"
    config["training"]["learning_rate"] = 0.002
    config["training"]["weight_decay"] = 0.0005
    config["training"]["batch_size"] = 32
    config["training"]["maximum_epochs"] = 50
    config["training"]["loss"] = "CrossEntropyLoss"
    config["training"]["early_stopping"] = {
        "metric": "macro_f1",
        "patience": 10,
        "minimum_delta": 0.0001,
        "restore_best_checkpoint": True,
    }
    config["validation"] = {
        "strategy": "competition_aware_pooled_rolling_origin_40pct_initial_train",
        "initial_train_fraction": INITIAL_TRAIN_FRACTION,
        "outer_folds": N_OUTER_FOLDS,
        "inner_validation_fraction": INNER_VALIDATION_FRACTION,
        "team_embargo_matches": TEAM_EMBARGO_MATCHES,
        "primary_metric": "macro_f1",
    }
    config["experiments"] = {
        "run_xgboost_history": True,
        "run_history_mlp": True,
        "run_structure_gnn": True,
        "run_hybrid_full": True,
    }
    return config


def metric_row_safe(truth, prediction, score):
    truth = np.asarray(truth, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    score = np.asarray(score, dtype=float)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    return {
        "n": int(len(truth)),
        "win_rate": float(np.mean(truth)) if len(truth) else np.nan,
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)) if len(np.unique(truth)) == 2 else np.nan,
        "macro_f1": float(f1_score(truth, prediction, labels=[0, 1], average="macro", zero_division=0)),
        "win_precision": float(precision_score(truth, prediction, pos_label=1, zero_division=0)),
        "win_recall": float(recall_score(truth, prediction, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, score)) if len(np.unique(truth)) == 2 else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def paired_bootstrap(truth, pred_a, pred_b, samples: int, seed: int):
    truth = np.asarray(truth, dtype=int)
    pred_a = np.asarray(pred_a, dtype=int)
    pred_b = np.asarray(pred_b, dtype=int)
    rng = np.random.default_rng(seed)
    n = len(truth)
    acc = np.empty(samples, dtype=float)
    f1 = np.empty(samples, dtype=float)
    for i in range(samples):
        idx = rng.integers(0, n, n)
        y = truth[idx]
        a = pred_a[idx]
        b = pred_b[idx]
        acc[i] = accuracy_score(y, a) - accuracy_score(y, b)
        f1[i] = (
            f1_score(y, a, labels=[0, 1], average="macro", zero_division=0)
            - f1_score(y, b, labels=[0, 1], average="macro", zero_division=0)
        )
    return {
        "accuracy_ci_lower": float(np.percentile(acc, 2.5)),
        "accuracy_ci_upper": float(np.percentile(acc, 97.5)),
        "macro_f1_ci_lower": float(np.percentile(f1, 2.5)),
        "macro_f1_ci_upper": float(np.percentile(f1, 97.5)),
    }


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def exact_mcnemar(truth, a, b):
    truth = np.asarray(truth, dtype=int)
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    ca = a == truth
    cb = b == truth
    a_correct_b_wrong = int(np.sum(ca & ~cb))
    b_correct_a_wrong = int(np.sum(cb & ~ca))
    discordant = a_correct_b_wrong + b_correct_a_wrong
    p = (
        float(binomtest(min(a_correct_b_wrong, b_correct_a_wrong), n=discordant, p=0.5).pvalue)
        if discordant else 1.0
    )
    return a_correct_b_wrong, b_correct_a_wrong, p


def paired_comparison(predictions, model_a, model_b, comparison, bootstrap_samples=BOOTSTRAP_SAMPLES, seed=2026):
    a = predictions[predictions["model"].eq(model_a)].sort_values("match_id").reset_index(drop=True)
    b = predictions[predictions["model"].eq(model_b)].sort_values("match_id").reset_index(drop=True)
    if len(a) != len(b) or not np.array_equal(a["match_id"].to_numpy(), b["match_id"].to_numpy()):
        raise RuntimeError(f"Predictions are not paired for {comparison}")
    truth = a["true_label"].to_numpy(dtype=int)
    pa = a["predicted_class"].to_numpy(dtype=int)
    pb = b["predicted_class"].to_numpy(dtype=int)
    ma = metric_row_safe(truth, pa, a["predicted_score"].to_numpy())
    mb = metric_row_safe(truth, pb, b["predicted_score"].to_numpy())
    boot = paired_bootstrap(truth, pa, pb, bootstrap_samples, seed)
    aw, bw, p = exact_mcnemar(truth, pa, pb)
    return {
        "comparison": comparison,
        "model_a": model_a,
        "model_b": model_b,
        "paired_n": int(len(truth)),
        "model_a_accuracy": ma["accuracy"],
        "model_b_accuracy": mb["accuracy"],
        "accuracy_difference_a_minus_b": ma["accuracy"] - mb["accuracy"],
        "accuracy_difference_pp": 100.0 * (ma["accuracy"] - mb["accuracy"]),
        **boot,
        "model_a_macro_f1": ma["macro_f1"],
        "model_b_macro_f1": mb["macro_f1"],
        "macro_f1_difference_a_minus_b": ma["macro_f1"] - mb["macro_f1"],
        "macro_f1_difference_pp": 100.0 * (ma["macro_f1"] - mb["macro_f1"]),
        "model_a_correct_b_wrong": aw,
        "model_b_correct_a_wrong": bw,
        "mcnemar_exact_p_raw": p,
        "bootstrap_samples": int(bootstrap_samples),
    }


def normalize_gender(value):
    text = str(value).strip().lower()
    if any(token in text for token in ["female", "women", "woman"]):
        return "women"
    if any(token in text for token in ["male", "men", "man"]):
        return "men"
    return "unknown"


def classify_format(name):
    text = str(name).strip().lower()
    tournament_tokens = [
        "world cup", "champions league", "europa league", "euro", "copa del rey",
        "copa america", "cup", "olympic", "uefa", "fifa", "africa cup", "gold cup",
    ]
    league_tokens = [
        "premier league", "super league", "major league soccer", "league", "liga",
        "bundesliga", "serie a", "ligue 1", "nwsl", "eredivisie", "division 1",
    ]
    if any(token in text for token in tournament_tokens):
        return "tournament", "keyword_tournament"
    if any(token in text for token in league_tokens):
        return "league", "keyword_league"
    return "unknown", "unclassified"


def subgroup_metrics(predictions, group_column, output_name):
    rows = []
    for (group_value, model), group in predictions.groupby([group_column, "model"], dropna=False):
        row = {group_column: group_value, "model": model}
        row.update(metric_row_safe(group.true_label, group.predicted_class, group.predicted_score))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / output_name, index=False)
    return df


def subgroup_pairwise(predictions, group_column, output_name, min_ci_n=50, bootstrap_samples=SUBGROUP_BOOTSTRAP_SAMPLES):
    rows = []
    for group_value, block in predictions.groupby(group_column, dropna=False):
        a = block[block.model.eq("C-Hybrid-full")].sort_values("match_id").reset_index(drop=True)
        b = block[block.model.eq("A-MLP")].sort_values("match_id").reset_index(drop=True)
        if len(a) == 0 or len(a) != len(b) or not np.array_equal(a.match_id.to_numpy(), b.match_id.to_numpy()):
            continue
        truth = a.true_label.to_numpy(dtype=int)
        pa = a.predicted_class.to_numpy(dtype=int)
        pb = b.predicted_class.to_numpy(dtype=int)
        ma = metric_row_safe(truth, pa, a.predicted_score.to_numpy())
        mb = metric_row_safe(truth, pb, b.predicted_score.to_numpy())
        aw, bw, p = exact_mcnemar(truth, pa, pb)
        can_ci = len(truth) >= min_ci_n and len(np.unique(truth)) == 2
        boot = (
            paired_bootstrap(truth, pa, pb, bootstrap_samples, seed=3100 + len(rows))
            if can_ci
            else {"accuracy_ci_lower": np.nan, "accuracy_ci_upper": np.nan, "macro_f1_ci_lower": np.nan, "macro_f1_ci_upper": np.nan}
        )
        rows.append({
            group_column: group_value,
            "paired_n": int(len(truth)),
            "hybrid_accuracy": ma["accuracy"],
            "mlp_accuracy": mb["accuracy"],
            "hybrid_minus_mlp_accuracy_pp": 100.0 * (ma["accuracy"] - mb["accuracy"]),
            "hybrid_macro_f1": ma["macro_f1"],
            "mlp_macro_f1": mb["macro_f1"],
            "hybrid_minus_mlp_macro_f1_pp": 100.0 * (ma["macro_f1"] - mb["macro_f1"]),
            **boot,
            "hybrid_correct_mlp_wrong": aw,
            "mlp_correct_hybrid_wrong": bw,
            "mcnemar_exact_p_raw": p,
            "ci_reported": bool(can_ci),
            "ci_rule": f"paired bootstrap reported when n>={min_ci_n} and both classes present",
            "bootstrap_samples_if_reported": int(bootstrap_samples),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["mcnemar_exact_p_holm"] = holm_adjust(out["mcnemar_exact_p_raw"].to_numpy())
    out.to_csv(OUTPUT_DIR / output_name, index=False)
    return out


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = build_config()
    with (PROJECT_ROOT / "config_experiment5_global.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    (OUTPUT_DIR / "experiment5_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    set_global_seed(BASE_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 100)
    print("EXPERIMENT 5 — GLOBAL POOLED CONSISTENCY")
    print("=" * 100)
    print("Device:", device)
    print("Cohort rule: all competition-season rows returned by StatsBomb Open Data")
    print("Evaluation: competition-aware pooled rolling origin; NOT held-out-competition transfer")

    save_environment(OUTPUT_DIR / "environment.json")
    with (OUTPUT_DIR / "environment_freeze.txt").open("w", encoding="utf-8") as handle:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=handle, check=True)

    print("\nSTEP 1 — HARVESTING GLOBAL STATSBOMB OPEN DATA")
    print("First pass: up to 5 attempts per event request. Failed events are recorded, not silently dropped.")
    match_stats, graph_map, manifest, candidate_manifest = harvest_dataset_with_retry_cache(
        config, attempts=5, fail_on_event_error=False
    )

    failed_events = manifest.loc[manifest.exclusion_reason.eq("event_load_failed")].copy()
    if len(failed_events):
        print(f"First pass left {len(failed_events)} unresolved event downloads.")
        print("Retry pass: cached successes are reused; unresolved matches receive up to 8 attempts.")
        time.sleep(3.0)
        match_stats, graph_map, manifest, candidate_manifest = harvest_dataset_with_retry_cache(
            config, attempts=8, fail_on_event_error=False
        )
        failed_events = manifest.loc[manifest.exclusion_reason.eq("event_load_failed")].copy()

    candidate_manifest.to_csv(OUTPUT_DIR / "candidate_match_manifest.csv", index=False)
    if len(failed_events):
        failed_events.to_csv(OUTPUT_DIR / "unresolved_event_downloads.csv", index=False)
        failed_ids = failed_events.match_id.astype(int).tolist()
        raise RuntimeError(
            "Global event harvesting remained incomplete after the retry pass. "
            f"Unresolved matches={len(failed_ids)}; first IDs={failed_ids[:20]}. "
            "The run is stopped rather than silently changing the cohort. "
            "Successful event files remain cached in /content/statsbomb_global_event_cache."
        )

    print("Candidate matches:", len(candidate_manifest))
    print("Matches with statistics:", len(match_stats))
    print("Graphs constructed:", len(graph_map))

    print("\nSTEP 2 — BUILDING COMPETITION-SPECIFIC THREE-MATCH PRE-MATCH HISTORY")
    history_df = compute_rolling_history(
        match_stats,
        window=int(config["history"]["window_matches"]),
        group_by_competition=True,
    )
    samples, graphs, X_history_raw, y, history_columns = align_samples(history_df, graph_map)
    samples["gender_group"] = samples["category"].map(normalize_gender)
    fmt = samples["competition_name"].map(classify_format)
    samples["competition_format"] = [x[0] for x in fmt]
    samples["competition_format_rule"] = [x[1] for x in fmt]

    history_ids = set(history_df.loc[history_df.history_available, "match_id"].astype(int))
    final_ids = set(samples.match_id.astype(int))
    manifest["history_available"] = manifest.match_id.isin(history_ids)
    manifest["final_included"] = manifest.match_id.isin(final_ids)
    manifest.loc[manifest.graph_included & ~manifest.history_available, "exclusion_reason"] = "insufficient_rolling_history"
    manifest.loc[manifest.final_included, "exclusion_reason"] = None
    if int((manifest.exclusion_reason == "event_load_failed").sum()) != 0:
        raise RuntimeError("Global run contains event_load_failed exclusions")

    manifest.to_csv(OUTPUT_DIR / "cohort_manifest.csv", index=False)
    samples.to_csv(OUTPUT_DIR / "included_samples.csv", index=False)
    (OUTPUT_DIR / "history_feature_order.json").write_text(json.dumps(history_columns, indent=2), encoding="utf-8")

    global_flow = pd.DataFrame([
        {"stage": "candidate_matches", "n": len(candidate_manifest)},
        {"stage": "event_rows_available", "n": len(match_stats)},
        {"stage": "graphs_constructed", "n": int(manifest.graph_included.sum())},
        {"stage": "history_available", "n": int(manifest.history_available.sum())},
        {"stage": "final_included", "n": int(manifest.final_included.sum())},
        {"stage": "final_win", "n": int(y.sum())},
        {"stage": "final_not_win", "n": int(len(y) - y.sum())},
    ])
    global_flow.to_csv(OUTPUT_DIR / "global_cohort_flow.csv", index=False)

    comp_flow = manifest.groupby(["competition_id", "competition_name"], dropna=False).agg(
        candidate_n=("match_id", "size"),
        graph_n=("graph_included", "sum"),
        history_n=("history_available", "sum"),
        final_included_n=("final_included", "sum"),
    ).reset_index()
    comp_flow.to_csv(OUTPUT_DIR / "global_cohort_flow_by_competition.csv", index=False)

    season_manifest = manifest.groupby(
        ["competition_id", "competition_name", "season_id", "season_name", "category"], dropna=False
    ).agg(
        candidate_n=("match_id", "size"),
        graph_n=("graph_included", "sum"),
        final_included_n=("final_included", "sum"),
    ).reset_index()
    season_manifest.to_csv(OUTPUT_DIR / "global_competition_season_manifest.csv", index=False)

    # Elite overlap is reported explicitly because Global is a pooled cohort, not external validation.
    elite_reference = pd.read_csv(PROJECT_ROOT / "reference_elite" / "included_samples.csv")
    elite_ids = set(elite_reference.match_id.astype(int))
    overlap_ids = set(samples.match_id.astype(int)).intersection(elite_ids)
    pd.DataFrame([{
        "global_eligible_n": int(len(samples)),
        "elite_reference_n": int(len(elite_reference)),
        "overlap_match_n": int(len(overlap_ids)),
        "global_non_elite_n": int(len(samples) - len(overlap_ids)),
        "interpretation": "Global includes Elite overlap and is not an external or unseen-competition test",
    }]).to_csv(OUTPUT_DIR / "global_elite_overlap.csv", index=False)

    print("Final eligible Global N:", len(samples))
    print("Competition IDs represented:", samples.competition_id.nunique())
    print("Competition names represented:", samples.competition_name.nunique())
    print("Win proportion:", round(float(y.mean()), 4))
    print("Elite-overlap matches:", len(overlap_ids))

    print("\nSTEP 3 — COMPETITION-AWARE POOLED ROLLING-ORIGIN FOLDS")
    splits, coverage_df, comp_fold_df = competition_aware_rolling_origin_splits(
        samples,
        initial_train_fraction=INITIAL_TRAIN_FRACTION,
        n_splits=N_OUTER_FOLDS,
        embargo_matches=TEAM_EMBARGO_MATCHES,
    )
    coverage_df.to_csv(OUTPUT_DIR / "global_competition_fold_coverage.csv", index=False)
    comp_fold_df.to_csv(OUTPUT_DIR / "global_competition_fold_details.csv", index=False)

    evaluable_comp_ids = set(coverage_df.loc[coverage_df.oof_eligible.eq(True), "competition_id"].tolist())
    oof_evaluable_n = int(samples.competition_id.isin(evaluable_comp_ids).sum())
    excluded_coverage = coverage_df.loc[~coverage_df.oof_eligible.eq(True)].copy()
    print("OOF-evaluable competition IDs:", len(evaluable_comp_ids))
    print("OOF-evaluable eligible observations:", oof_evaluable_n)
    print("Competitions retained in manifest but excluded from OOF:", len(excluded_coverage))
    if len(excluded_coverage):
        print(excluded_coverage[["competition_id", "competition_name", "eligible_n", "future_distinct_dates", "oof_exclusion_reason"]].to_string(index=False))

    prediction_rows_all = []
    fold_assignment_rows = []
    fold_summary_rows = []
    selected_epoch_rows = []
    inner_audit_rows = []
    runtime_rows = []
    parameter_rows = []

    for split in splits:
        fold = split.fold
        seed = fold_seed(fold)
        outer_train = split.train_indices
        outer_test = split.test_indices
        if len(np.unique(y[outer_train])) != 2 or len(np.unique(y[outer_test])) != 2:
            raise RuntimeError(f"Global fold {fold} lacks both classes")

        inner_train, inner_val, inner_audit = competition_aware_inner_split(
            samples,
            outer_train,
            validation_fraction=INNER_VALIDATION_FRACTION,
            embargo_matches=TEAM_EMBARGO_MATCHES,
            min_total_train=200,
            min_total_validation=100,
        )
        if len(np.unique(y[inner_train])) != 2 or len(np.unique(y[inner_val])) != 2:
            raise RuntimeError(f"Global fold {fold} inner partition lacks both classes")
        inner_audit["fold"] = fold
        inner_audit_rows.extend(inner_audit.to_dict("records"))

        print("\n" + "-" * 100)
        print(f"GLOBAL FOLD {fold}")
        print("Outer train after embargo:", len(outer_train))
        print("Outer test:", len(outer_test))
        print("Inner train:", len(inner_train))
        print("Inner validation:", len(inner_val))
        print("Fold seed:", seed)

        for level, role, indices in [
            ("outer", "train", outer_train), ("outer", "test", outer_test),
            ("inner", "train", inner_train), ("inner", "validation", inner_val),
        ]:
            for idx in indices:
                fold_assignment_rows.append({
                    "fold": fold, "level": level, "role": role,
                    "match_id": int(samples.iloc[int(idx)].match_id),
                    "strategy": "competition_aware_pooled_rolling_origin" if level == "outer" else "competition_aware_chronological_inner",
                })

        inner_scaler = StandardScaler().fit(X_history_raw[inner_train])
        X_inner_train = inner_scaler.transform(X_history_raw[inner_train])
        X_inner_val = inner_scaler.transform(X_history_raw[inner_val])

        history_factory = lambda: HistoryMLP(
            X_history_raw.shape[1], int(config["model"]["history_hidden_dim"]), float(config["model"]["history_dropout"])
        )
        structure_factory = lambda: StructureGNN(4, 3, config)
        hybrid_factory = lambda: HybridGNN(4, 3, X_history_raw.shape[1], config)
        if fold == 1:
            parameter_rows.extend([
                {"model": "A-MLP", "trainable_parameters": count_parameters(history_factory())},
                {"model": "B-GNN-full", "trainable_parameters": count_parameters(structure_factory())},
                {"model": "C-Hybrid-full", "trainable_parameters": count_parameters(hybrid_factory())},
            ])

        inner_graph_train_raw = full_graph_views(graphs, inner_train)
        inner_graph_val_raw = full_graph_views(graphs, inner_val)
        inner_graph_train, inner_graph_val = preprocess_graph_pair(
            inner_graph_train_raw, inner_graph_val_raw, X_inner_train, X_inner_val
        )

        history_sel = select_history_epoch(
            history_factory, X_inner_train, y[inner_train], X_inner_val, y[inner_val], config, seed, device
        )
        structure_sel = select_graph_epoch(
            structure_factory, inner_graph_train, inner_graph_val, y[inner_val], config, seed, device
        )
        hybrid_sel = select_graph_epoch(
            hybrid_factory, inner_graph_train, inner_graph_val, y[inner_val], config, seed, device
        )
        selected_epoch_rows.append({
            "fold": fold, "seed": seed,
            "A_MLP_best_epoch": history_sel.best_epoch,
            "A_MLP_validation_macro_f1": history_sel.best_metric,
            "B_GNN_best_epoch": structure_sel.best_epoch,
            "B_GNN_validation_macro_f1": structure_sel.best_metric,
            "C_Hybrid_best_epoch": hybrid_sel.best_epoch,
            "C_Hybrid_validation_macro_f1": hybrid_sel.best_metric,
            "inner_train_n": len(inner_train), "inner_validation_n": len(inner_val),
            "outer_train_n": len(outer_train), "outer_test_n": len(outer_test),
        })
        print("Selected epochs:", history_sel.best_epoch, structure_sel.best_epoch, hybrid_sel.best_epoch)

        outer_scaler = StandardScaler().fit(X_history_raw[outer_train])
        X_outer_train = outer_scaler.transform(X_history_raw[outer_train])
        X_outer_test = outer_scaler.transform(X_history_raw[outer_test])
        graph_train_raw = full_graph_views(graphs, outer_train)
        graph_test_raw = full_graph_views(graphs, outer_test)
        graph_train, graph_test = preprocess_graph_pair(
            graph_train_raw, graph_test_raw, X_outer_train, X_outer_test
        )

        def store(model_name, pred, score, gate=None):
            prediction_rows_all.extend(prediction_rows(
                samples, outer_test, y[outer_test], pred, score,
                cohort="global", protocol="global_competition_aware_pooled_rolling_origin",
                model=model_name, fold=fold, seed=seed, gate_value=gate,
            ))

        start = time.perf_counter()
        model = xgb_model(config, seed)
        model.fit(X_outer_train, y[outer_train])
        train_s = time.perf_counter() - start
        start = time.perf_counter(); pred = model.predict(X_outer_test); score = model.predict_proba(X_outer_test)[:, 1]; infer_s = time.perf_counter() - start
        store("A-XGB", pred, score)
        runtime_rows.append({"fold": fold, "model": "A-XGB", "train_seconds": train_s, "inference_seconds": infer_s, "test_n": len(outer_test)})

        start = time.perf_counter()
        model = fit_history_model(history_factory, X_outer_train, y[outer_train], history_sel.best_epoch, config, seed, device)
        train_s = time.perf_counter() - start
        start = time.perf_counter(); pred, score = predict_history_model(model, X_outer_test, device); infer_s = time.perf_counter() - start
        store("A-MLP", pred, score)
        runtime_rows.append({"fold": fold, "model": "A-MLP", "train_seconds": train_s, "inference_seconds": infer_s, "test_n": len(outer_test)})

        start = time.perf_counter()
        model = fit_graph_model(structure_factory, graph_train, structure_sel.best_epoch, config, seed, device)
        train_s = time.perf_counter() - start
        start = time.perf_counter(); pred, score = predict_graph_model(model, graph_test, config, device); infer_s = time.perf_counter() - start
        store("B-GNN-full", pred, score)
        runtime_rows.append({"fold": fold, "model": "B-GNN-full", "train_seconds": train_s, "inference_seconds": infer_s, "test_n": len(outer_test)})

        start = time.perf_counter()
        model = fit_graph_model(hybrid_factory, graph_train, hybrid_sel.best_epoch, config, seed, device)
        train_s = time.perf_counter() - start
        start = time.perf_counter(); pred, score = predict_graph_model(model, graph_test, config, device); infer_s = time.perf_counter() - start
        gate = model.gate_value()
        store("C-Hybrid-full", pred, score, gate)
        runtime_rows.append({"fold": fold, "model": "C-Hybrid-full", "train_seconds": train_s, "inference_seconds": infer_s, "test_n": len(outer_test)})

        fold_summary_rows.append({
            "fold": fold, "seed": seed,
            "outer_train_n": len(outer_train), "outer_test_n": len(outer_test),
            "inner_train_n": len(inner_train), "inner_validation_n": len(inner_val),
            "test_calendar_start": str(split.test_start_date.date()),
            "test_calendar_end": str(split.test_end_date.date()),
            "train_win_rate": float(y[outer_train].mean()), "test_win_rate": float(y[outer_test].mean()),
            "learned_lambda": gate,
        })
        print(f"Fold {fold} complete. lambda={gate:.6f}")

    predictions = pd.DataFrame(prediction_rows_all)
    assignments = pd.DataFrame(fold_assignment_rows)
    folds = pd.DataFrame(fold_summary_rows)
    epochs = pd.DataFrame(selected_epoch_rows)
    runtime = pd.DataFrame(runtime_rows)
    parameters = pd.DataFrame(parameter_rows)
    inner_audit = pd.DataFrame(inner_audit_rows)

    # Add normalized subgroup labels directly to OOF predictions.
    predictions["gender_group"] = predictions["category"].map(normalize_gender)
    fmt = predictions["competition_name"].map(classify_format)
    predictions["competition_format"] = [x[0] for x in fmt]
    predictions["competition_format_rule"] = [x[1] for x in fmt]

    expected = {"A-XGB", "A-MLP", "B-GNN-full", "C-Hybrid-full"}
    if set(predictions.model.unique()) != expected:
        raise RuntimeError("Global predictions do not contain the exact four frozen model families")
    if predictions.duplicated(["match_id", "model"]).any():
        raise RuntimeError("Duplicate global OOF match/model prediction detected")
    reference = None
    for model_name, group in predictions.groupby("model"):
        ids = set(group.match_id.astype(int))
        reference = ids if reference is None else reference
        if ids != reference:
            raise RuntimeError(f"Global paired prediction IDs differ for {model_name}")

    results = build_results_table(predictions)
    runtime["inference_ms_per_match"] = 1000.0 * runtime.inference_seconds / runtime.test_n
    predictions.to_csv(OUTPUT_DIR / "global_oof_predictions.csv", index=False)
    assignments.to_csv(OUTPUT_DIR / "global_fold_assignments.csv", index=False)
    folds.to_csv(OUTPUT_DIR / "global_fold_summary.csv", index=False)
    epochs.to_csv(OUTPUT_DIR / "global_selected_epochs.csv", index=False)
    runtime.to_csv(OUTPUT_DIR / "global_runtime_summary.csv", index=False)
    parameters.to_csv(OUTPUT_DIR / "global_parameter_counts.csv", index=False)
    inner_audit.to_csv(OUTPUT_DIR / "global_inner_split_audit.csv", index=False)
    results.to_csv(OUTPUT_DIR / "global_results.csv", index=False)
    results[results.scope.eq("pooled")].to_csv(OUTPUT_DIR / "global_pooled_primary_table.csv", index=False)

    # Main pooled comparisons.
    comparisons = pd.DataFrame([
        paired_comparison(predictions, "C-Hybrid-full", "A-MLP", "C-Hybrid-full vs A-MLP", seed=2026),
        paired_comparison(predictions, "B-GNN-full", "A-MLP", "B-GNN-full vs A-MLP", seed=2027),
        paired_comparison(predictions, "C-Hybrid-full", "B-GNN-full", "C-Hybrid-full vs B-GNN-full", seed=2028),
    ])
    comparisons["mcnemar_exact_p_holm"] = holm_adjust(comparisons.mcnemar_exact_p_raw.to_numpy())
    comparisons.to_csv(OUTPUT_DIR / "global_pooled_pairwise_inference.csv", index=False)

    # Per-competition metrics and Hybrid-vs-MLP paired deltas.
    per_comp = subgroup_metrics(predictions, "competition_name", "global_per_competition_metrics.csv")
    comp_pair = subgroup_pairwise(
        predictions, "competition_name", "global_per_competition_pairwise.csv", min_ci_n=50, bootstrap_samples=SUBGROUP_BOOTSTRAP_SAMPLES
    )

    # Competition-macro summary: every competition has equal weight.
    comp_macro_rows = []
    for model_name, block in per_comp.groupby("model"):
        comp_macro_rows.append({
            "model": model_name,
            "competition_count": int(block.competition_name.nunique()),
            "competition_macro_mean_accuracy": float(block.accuracy.mean()),
            "competition_macro_median_accuracy": float(block.accuracy.median()),
            "competition_macro_mean_macro_f1": float(block.macro_f1.mean()),
            "competition_macro_median_macro_f1": float(block.macro_f1.median()),
            "macro_f1_q25": float(block.macro_f1.quantile(0.25)),
            "macro_f1_q75": float(block.macro_f1.quantile(0.75)),
        })
    pd.DataFrame(comp_macro_rows).to_csv(OUTPUT_DIR / "global_competition_macro_summary.csv", index=False)

    if len(comp_pair):
        def delta_distribution(block, label):
            d = block.hybrid_minus_mlp_macro_f1_pp
            return {
                "subset": label,
                "competition_count": int(len(block)),
                "mean_delta_pp": float(d.mean()),
                "median_delta_pp": float(d.median()),
                "q25_delta_pp": float(d.quantile(0.25)),
                "q75_delta_pp": float(d.quantile(0.75)),
                "positive_delta_competitions": int((d > 0).sum()),
                "negative_delta_competitions": int((d < 0).sum()),
                "zero_delta_competitions": int((d == 0).sum()),
            }
        dist_rows = [delta_distribution(comp_pair, "all_oof_competitions")]
        dist_rows.append(delta_distribution(comp_pair[comp_pair.paired_n >= 20], "competitions_with_oof_n_ge_20"))
        pd.DataFrame(dist_rows).to_csv(OUTPUT_DIR / "global_competition_delta_distribution.csv", index=False)

    # Gender and competition-format results, as explicitly requested by supervisor feedback.
    subgroup_metrics(predictions, "gender_group", "global_gender_metrics.csv")
    subgroup_pairwise(predictions, "gender_group", "global_gender_pairwise.csv", min_ci_n=50, bootstrap_samples=BOOTSTRAP_SAMPLES)
    subgroup_metrics(predictions, "competition_format", "global_format_metrics.csv")
    subgroup_pairwise(predictions, "competition_format", "global_format_pairwise.csv", min_ci_n=50, bootstrap_samples=BOOTSTRAP_SAMPLES)

    format_map = predictions[["competition_name", "competition_format", "competition_format_rule"]].drop_duplicates().sort_values("competition_name")
    format_map.to_csv(OUTPUT_DIR / "global_competition_format_mapping.csv", index=False)

    oof_counts = predictions[predictions.model.eq("C-Hybrid-full")].groupby(
        ["competition_id", "competition_name", "gender_group", "competition_format"], dropna=False
    ).size().rename("oof_n").reset_index()
    coverage_full = comp_flow.merge(oof_counts, on=["competition_id", "competition_name"], how="left")
    coverage_full["oof_n"] = coverage_full.oof_n.fillna(0).astype(int)
    coverage_full.to_csv(OUTPUT_DIR / "global_competition_coverage_summary.csv", index=False)

    metadata = {
        "artifact": "Experiment 5 — Global pooled consistency",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_definition": "all StatsBomb Open Data competition-season rows returned at run time; exact manifest saved",
        "eligible_n_all_competitions": int(len(samples)),
        "oof_evaluable_eligible_n": int(oof_evaluable_n),
        "paired_outer_test_n": int(len(reference)),
        "competition_id_count_eligible": int(samples.competition_id.nunique()),
        "competition_id_count_oof_evaluable": int(len(evaluable_comp_ids)),
        "competition_id_count_excluded_from_oof": int(len(excluded_coverage)),
        "competition_name_count_eligible": int(samples.competition_name.nunique()),
        "competition_name_count_oof": int(predictions[predictions.model.eq("C-Hybrid-full")].competition_name.nunique()),
        "elite_overlap_n": int(len(overlap_ids)),
        "validation": "competition-aware pooled rolling-origin; 40% initial within each OOF-evaluable competition; five future date windows; undersized competitions retained in manifest and excluded from OOF with explicit reason",
        "not_a_transfer_test": True,
        "history_grouping": "team within competition_id",
        "team_embargo_matches": TEAM_EMBARGO_MATCHES,
        "primary_model_comparison": "C-Hybrid-full vs exact A-MLP",
        "main_bootstrap_samples": BOOTSTRAP_SAMPLES,
        "subgroup_bootstrap_samples": SUBGROUP_BOOTSTRAP_SAMPLES,
        "subgroup_ci_rule": "per-competition paired bootstrap only when paired n>=50 and both classes present",
        "seed_policy": "42 + 100*fold, shared across model families within fold",
        "claim_boundary": "performance within a heterogeneous pooled cohort; not unseen-competition generalization",
    }
    (OUTPUT_DIR / "reproducibility_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    readme = f"""EXPERIMENT 5 — GLOBAL POOLED CONSISTENCY
=========================================

Purpose
-------
Evaluate whether the frozen full Hybrid's performance pattern remains within the heterogeneous pooled
StatsBomb Open Data cohort. This is NOT leave-one-competition-out transfer and must not be described
as external generalization.

Protocol
--------
- All StatsBomb Open Data competition-season rows returned at execution time; exact manifests saved.
- One home-team graph per completed match.
- Same full graph features and architecture frozen from Elite Experiment 1.
- Pre-match three-match histories remain competition-specific.
- Competition-aware pooled rolling-origin evaluation: for every competition that can support the
  frozen five-window protocol, the first ~40% is initial training and remaining future dates are
  divided into five windows; corresponding windows are pooled across competitions.
- Competitions with too few distinct future dates or an invalid embargoed partition are retained in
  the coverage manifest and explicitly excluded from OOF evaluation; they are never forced into tiny
  or duplicated test folds.
- Three-match team embargo is applied within competition.
- Inner validation also preserves chronology within competition.
- Training-only preprocessing, fixed seeds, max 50 epochs, patience 10.
- Primary metric: macro F1.

Required heterogeneity reporting
--------------------------------
The artifact includes per-competition sample sizes and metrics, equal-weight competition-macro summaries,
Hybrid-minus-MLP deltas by competition, men's/women's subgroup results, league/tournament subgroup results,
and paired confidence intervals where subgroup sample size permits.

Claim boundary
--------------
Use wording such as "performance within a heterogeneous pooled cohort". Do not claim unseen-competition,
unseen-team, or external transfer. Global overlaps the Elite cohort; the exact overlap is saved in
global_elite_overlap.csv.
"""
    (OUTPUT_DIR / "README_EXPERIMENT5.txt").write_text(readme, encoding="utf-8")

    # Hash and package. Raw event cache is intentionally excluded; exact source manifests are included.
    excluded_parts = {"__pycache__", ".pytest_cache", ".git", "cache"}
    checksum_rows = []
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if not path.is_file() or path.suffix == ".zip" or path.name == "SHA256SUMS.csv":
            continue
        if any(part in excluded_parts for part in path.parts):
            continue
        checksum_rows.append({
            "path": str(path.relative_to(PROJECT_ROOT)),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        })
    pd.DataFrame(checksum_rows).to_csv(OUTPUT_DIR / "SHA256SUMS.csv", index=False)

    if FINAL_ZIP.exists():
        FINAL_ZIP.unlink()
    with zipfile.ZipFile(FINAL_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(PROJECT_ROOT.rglob("*")):
            if not path.is_file() or path.suffix == ".zip":
                continue
            if any(part in excluded_parts for part in path.parts):
                continue
            archive.write(path, Path("football_gnn_experiment5_global") / path.relative_to(PROJECT_ROOT))

    pooled = results[results.scope.eq("pooled")].sort_values("model")
    primary = comparisons[comparisons.comparison.eq("C-Hybrid-full vs A-MLP")].iloc[0]
    print("\n" + "=" * 100)
    print("EXPERIMENT 5 COMPLETED")
    print("=" * 100)
    print("Eligible Global N (all competitions):", len(samples))
    print("OOF-evaluable eligible N:", oof_evaluable_n)
    print("Paired outer-test N:", len(reference))
    print("Competition names in eligible cohort:", samples.competition_name.nunique())
    print("Competition IDs excluded from OOF:", len(excluded_coverage))
    print("Competition names represented in OOF:", predictions[predictions.model.eq("C-Hybrid-full")].competition_name.nunique())
    print("Hybrid macro F1:", f"{primary.model_a_macro_f1:.4f}")
    print("A-MLP macro F1:", f"{primary.model_b_macro_f1:.4f}")
    print("Hybrid - A-MLP macro-F1 difference:", f"{primary.macro_f1_difference_pp:+.3f} pp")
    print("95% CI:", f"[{100*primary.macro_f1_ci_lower:+.3f}, {100*primary.macro_f1_ci_upper:+.3f}] pp")
    print("McNemar raw p:", f"{primary.mcnemar_exact_p_raw:.6f}")
    print("McNemar Holm p:", f"{primary.mcnemar_exact_p_holm:.6f}")
    print("Final archive:", FINAL_ZIP)
    print("\nPOOLED RESULTS")
    print(pooled[["model", "n", "accuracy", "balanced_accuracy", "macro_f1", "win_precision", "win_recall", "roc_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
