from __future__ import annotations

import hashlib
import json
import os
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from src.evaluation import build_results_table, prediction_rows
from src.final_protocol import (
    date_aware_inner_split,
    harvest_dataset_with_retry_cache,
    rolling_origin_splits,
)
from src.graph_builder import calculate_pass_angle, completed_pass_mask
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
BASE_CONFIG_PATH = PROJECT_ROOT / "config_final.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "elite_rolling_origin_final"
FINAL_ZIP = Path("/content/football_gnn_elite_rolling_origin_authoritative_final.zip")
BASE_SEED = 42
INITIAL_TRAIN_FRACTION = 0.40
N_OUTER_FOLDS = 5
TEAM_EMBARGO_MATCHES = 3
INNER_VALIDATION_FRACTION = 0.20
MIN_OUTER_TRAIN = 200
MIN_INNER_TRAIN = 100
MIN_INNER_VALIDATION = 25
BOOTSTRAP_SAMPLES = 10000


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def fold_seed(fold: int) -> int:
    return BASE_SEED + fold * 100


def xgb_model(config: dict, seed: int):
    settings = config["xgboost"]
    return xgb.XGBClassifier(
        n_estimators=int(settings["n_estimators"]),
        max_depth=int(settings["max_depth"]),
        learning_rate=float(settings["learning_rate"]),
        subsample=float(settings["subsample"]),
        colsample_bytree=float(settings["colsample_bytree"]),
        eval_metric=settings["eval_metric"],
        random_state=seed,
        n_jobs=int(settings["n_jobs"]),
    )


def full_graph_views(graphs: list, indices: np.ndarray):
    return [
        make_feature_view(
            graphs[int(index)],
            node_indices=[0, 1, 2, 3],
            edge_indices=[0, 1, 2],
        )
        for index in indices
    ]


def preprocess_graph_pair(train_raw, other_raw, train_history, other_history):
    scalers = fit_graph_scalers(
        graphs=train_raw,
        standardize_nodes=True,
        edge_continuous_indices=[0, 1],
    )
    train = transform_graphs(train_raw, scalers, train_history)
    other = transform_graphs(other_raw, scalers, other_history)
    return train, other


def save_config(config: dict):
    with BASE_CONFIG_PATH.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    with (PROJECT_ROOT / "config_final_rolling_origin.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def build_frozen_config():
    with BASE_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config["project"]["seed"] = BASE_SEED
    config["project"]["output_dir"] = "outputs/elite_rolling_origin_final"
    config["project"]["cache_dir"] = "data/cache"

    config["cohort"]["name"] = "elite"
    config["cohort"]["mode"] = "named"
    config["cohort"]["competition_names"] = ["La Liga", "Champions League"]
    config["cohort"]["max_matches"] = None

    config["graph"]["node_count"] = 11
    config["graph"]["node_features"] = [
        "completed_pass_count",
        "shot_count",
        "shot_xg_sum",
        "ball_recovery_count",
    ]
    config["graph"]["edge_features"] = [
        "pass_length",
        "pass_angle",
        "key_pass_indicator",
    ]
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
        "strategy": "rolling_origin_40pct_initial_train",
        "initial_train_fraction": INITIAL_TRAIN_FRACTION,
        "outer_folds": N_OUTER_FOLDS,
        "inner_validation_fraction": INNER_VALIDATION_FRACTION,
        "team_embargo_matches": TEAM_EMBARGO_MATCHES,
        "minimum_outer_train_after_embargo": MIN_OUTER_TRAIN,
        "minimum_inner_train_after_embargo": MIN_INNER_TRAIN,
        "minimum_inner_validation": MIN_INNER_VALIDATION,
        "primary_metric": "macro_f1",
    }

    config["experiments"] = {
        "run_xgboost_history": True,
        "run_history_mlp": True,
        "run_structure_gnn": True,
        "run_hybrid_full": True,
    }
    return config


def manual_integrity_checks(config):
    events = pd.DataFrame({
        "team": ["Home", "Home", "Home", "Away"],
        "type": ["Pass", "Pass", "Pass", "Pass"],
        "pass_recipient": ["P2", "P3", None, "P9"],
        "pass_outcome": [None, "Incomplete", None, None],
        "pass_end_location": [[20.0, 20.0], [30.0, 30.0], [40.0, 40.0], [50.0, 50.0]],
    })
    assert completed_pass_mask(events, "Home").tolist() == [True, False, False, False]

    row = pd.Series({
        "location": [10.0, 20.0],
        "pass_end_location": [30.0, 20.0],
        "pass_angle": None,
    })
    assert abs(calculate_pass_angle(row)) < 1e-9

    set_global_seed(BASE_SEED)
    a_np = np.random.rand(5)
    a_torch = torch.rand(5)
    set_global_seed(BASE_SEED)
    assert np.allclose(a_np, np.random.rand(5))
    assert torch.allclose(a_torch, torch.rand(5))

    forbidden = ["np.random.randint", "random.randint", "torch.randint"]
    found = []
    for source in (PROJECT_ROOT / "src").rglob("*.py"):
        if source.name == "final_protocol.py":
            continue
        text = source.read_text(encoding="utf-8")
        for pattern in forbidden:
            if pattern in text:
                found.append((str(source), pattern))
    assert not found, f"Forbidden random-edge construction found: {found}"

    test_hybrid = HybridGNN(4, 3, 16, config)
    assert abs(test_hybrid.gate_value() - 0.5) < 1e-6


def paired_bootstrap(truth, pred_a, pred_b, samples=BOOTSTRAP_SAMPLES, seed=2026):
    rng = np.random.default_rng(seed)
    n = len(truth)
    acc_diffs = np.empty(samples, dtype=float)
    f1_diffs = np.empty(samples, dtype=float)
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        yb = truth[idx]
        a = pred_a[idx]
        b = pred_b[idx]
        acc_diffs[i] = accuracy_score(yb, a) - accuracy_score(yb, b)
        f1_diffs[i] = (
            f1_score(yb, a, average="macro", zero_division=0)
            - f1_score(yb, b, average="macro", zero_division=0)
        )
    return {
        "accuracy_ci_lower": float(np.percentile(acc_diffs, 2.5)),
        "accuracy_ci_upper": float(np.percentile(acc_diffs, 97.5)),
        "macro_f1_ci_lower": float(np.percentile(f1_diffs, 2.5)),
        "macro_f1_ci_upper": float(np.percentile(f1_diffs, 97.5)),
    }


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main():
    # Remove older diagnostic outputs/configs to keep one authoritative artifact.
    shutil.rmtree(PROJECT_ROOT / "outputs", ignore_errors=True)
    for old_name in [
        "config_smoke.yaml",
        "config_elite_primary_final.yaml",
        "README_FINAL_ELITE_EXPERIMENT.txt",
        "ZENODO_README.txt",
        "SHA256SUMS.csv",
        "run_pipeline.py",
        "AUDIT_FINDINGS.md",
        "COLAB_STEP_BY_STEP.md",
        "cohort_manifest.csv",
        "fold_predictions_final.csv",
        "results_final.csv",
    ]:
        (PROJECT_ROOT / old_name).unlink(missing_ok=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = build_frozen_config()
    save_config(config)
    manual_integrity_checks(config)

    set_global_seed(BASE_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 100)
    print("FINAL ROLLING-ORIGIN ELITE PRIMARY EXPERIMENT")
    print("=" * 100)
    print("Device:", device)
    print("Initial training fraction:", INITIAL_TRAIN_FRACTION)
    print("Outer folds:", N_OUTER_FOLDS)
    print("Team embargo:", TEAM_EMBARGO_MATCHES)
    print("Minimum outer training size after embargo:", MIN_OUTER_TRAIN)
    print("Minimum inner training size after embargo:", MIN_INNER_TRAIN)

    save_environment(OUTPUT_DIR / "environment.json")
    with (PROJECT_ROOT / "environment_freeze.txt").open("w", encoding="utf-8") as handle:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=handle, check=True)
    (OUTPUT_DIR / "config_resolved.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\nSTEP 1 — HARVESTING ELITE DATA WITH RETRIES AND LOCAL CACHE")
    match_stats, graph_map, manifest, candidate_manifest = harvest_dataset_with_retry_cache(
        config,
        attempts=5,
        fail_on_event_error=True,
    )
    candidate_manifest.to_csv(OUTPUT_DIR / "candidate_match_manifest.csv", index=False)
    print("Candidate matches:", len(candidate_manifest))
    print("Matches with extracted statistics:", len(match_stats))
    print("Passing graphs constructed:", len(graph_map))

    print("\nSTEP 2 — BUILDING PRE-MATCH THREE-MATCH HISTORY")
    history_df = compute_rolling_history(
        match_stats=match_stats,
        window=int(config["history"]["window_matches"]),
        group_by_competition=bool(config["history"]["group_by_competition"]),
    )
    samples, graphs, X_history_raw, y, history_columns = align_samples(history_df, graph_map)
    print("Final eligible cohort N:", len(samples))
    print("Win proportion:", round(float(y.mean()), 4))
    print("History dimension:", X_history_raw.shape[1])

    history_ids = set(history_df.loc[history_df["history_available"], "match_id"].astype(int))
    final_ids = set(samples["match_id"].astype(int))
    manifest["history_available"] = manifest["match_id"].isin(history_ids)
    manifest["final_included"] = manifest["match_id"].isin(final_ids)
    manifest.loc[
        manifest["graph_included"] & ~manifest["history_available"],
        "exclusion_reason",
    ] = "insufficient_rolling_history"
    manifest.loc[manifest["final_included"], "exclusion_reason"] = None

    if int((manifest["exclusion_reason"] == "event_load_failed").sum()) != 0:
        raise RuntimeError("Final run contains event_load_failed exclusions; this is not allowed")

    manifest.to_csv(OUTPUT_DIR / "cohort_manifest.csv", index=False)
    samples.to_csv(OUTPUT_DIR / "included_samples.csv", index=False)
    (OUTPUT_DIR / "history_feature_order.json").write_text(json.dumps(history_columns, indent=2), encoding="utf-8")

    cohort_flow = pd.DataFrame([
        {"stage": "candidate_matches", "n": len(candidate_manifest)},
        {"stage": "event_rows_available", "n": len(match_stats)},
        {"stage": "graphs_constructed", "n": int(manifest["graph_included"].sum())},
        {"stage": "history_available", "n": int(manifest["history_available"].sum())},
        {"stage": "final_included", "n": int(manifest["final_included"].sum())},
        {"stage": "final_win", "n": int(y.sum())},
        {"stage": "final_not_win", "n": int(len(y) - y.sum())},
    ])
    cohort_flow.to_csv(OUTPUT_DIR / "cohort_flow.csv", index=False)

    print("\nSTEP 3 — CREATING 40% INITIAL-TRAIN ROLLING-ORIGIN FOLDS")
    splits = rolling_origin_splits(
        samples,
        initial_train_fraction=INITIAL_TRAIN_FRACTION,
        n_splits=N_OUTER_FOLDS,
        embargo_matches=TEAM_EMBARGO_MATCHES,
    )
    if len(splits) != N_OUTER_FOLDS:
        raise RuntimeError(f"Expected {N_OUTER_FOLDS} folds, got {len(splits)}")

    prediction_output = []
    fold_assignment_rows = []
    fold_summary_rows = []
    selected_epoch_rows = []
    parameter_count_rows = []
    runtime_rows = []

    for split in splits:
        fold = split.fold
        outer_train = split.train_indices
        outer_test = split.test_indices
        seed = fold_seed(fold)

        if len(outer_train) < MIN_OUTER_TRAIN:
            raise RuntimeError(
                f"Fold {fold} outer training set is too small after embargo: {len(outer_train)} < {MIN_OUTER_TRAIN}"
            )
        if len(np.unique(y[outer_train])) != 2 or len(np.unique(y[outer_test])) != 2:
            raise RuntimeError(f"Fold {fold} lacks both classes in train or test")

        inner_train, inner_val, inner_meta = date_aware_inner_split(
            samples,
            outer_train,
            validation_fraction=INNER_VALIDATION_FRACTION,
            embargo_matches=TEAM_EMBARGO_MATCHES,
            min_train=MIN_INNER_TRAIN,
            min_validation=MIN_INNER_VALIDATION,
        )
        if len(np.unique(y[inner_train])) != 2 or len(np.unique(y[inner_val])) != 2:
            raise RuntimeError(f"Fold {fold} lacks both classes in inner train or validation")

        print("\n" + "-" * 100)
        print(f"FOLD {fold}")
        print("Raw outer train before embargo:", split.raw_train_count)
        print("Outer train after embargo:", len(outer_train))
        print("Outer test:", len(outer_test), f"({split.test_start_date.date()} to {split.test_end_date.date()})")
        print("Inner train after embargo:", len(inner_train))
        print("Inner validation:", len(inner_val))
        print("Fold seed:", seed)

        for level, role, indices in [
            ("outer", "train", outer_train),
            ("outer", "test", outer_test),
            ("inner", "train", inner_train),
            ("inner", "validation", inner_val),
        ]:
            for idx in indices:
                fold_assignment_rows.append({
                    "fold": fold,
                    "level": level,
                    "role": role,
                    "match_id": int(samples.iloc[int(idx)]["match_id"]),
                    "strategy": "rolling_origin_40pct_initial_train" if level == "outer" else "date_aware_chronological_inner",
                })

        inner_history_scaler = StandardScaler().fit(X_history_raw[inner_train])
        X_inner_train = inner_history_scaler.transform(X_history_raw[inner_train])
        X_inner_val = inner_history_scaler.transform(X_history_raw[inner_val])

        history_factory = lambda: HistoryMLP(
            input_dim=X_history_raw.shape[1],
            hidden_dim=int(config["model"]["history_hidden_dim"]),
            dropout=float(config["model"]["history_dropout"]),
        )
        structure_factory = lambda: StructureGNN(4, 3, config)
        hybrid_factory = lambda: HybridGNN(4, 3, X_history_raw.shape[1], config)

        if fold == 1:
            parameter_count_rows.extend([
                {"model": "A-MLP", "trainable_parameters": count_parameters(history_factory())},
                {"model": "B-GNN-full", "trainable_parameters": count_parameters(structure_factory())},
                {"model": "C-Hybrid-full", "trainable_parameters": count_parameters(hybrid_factory())},
            ])

        inner_graph_train_raw = full_graph_views(graphs, inner_train)
        inner_graph_val_raw = full_graph_views(graphs, inner_val)
        inner_graph_train, inner_graph_val = preprocess_graph_pair(
            inner_graph_train_raw,
            inner_graph_val_raw,
            X_inner_train,
            X_inner_val,
        )

        history_selection = select_history_epoch(
            history_factory,
            X_inner_train,
            y[inner_train],
            X_inner_val,
            y[inner_val],
            config,
            seed,
            device,
        )
        structure_selection = select_graph_epoch(
            structure_factory,
            inner_graph_train,
            inner_graph_val,
            y[inner_val],
            config,
            seed,
            device,
        )
        hybrid_selection = select_graph_epoch(
            hybrid_factory,
            inner_graph_train,
            inner_graph_val,
            y[inner_val],
            config,
            seed,
            device,
        )

        selected_epoch_rows.append({
            "fold": fold,
            "seed": seed,
            "outer_raw_train_n": split.raw_train_count,
            "outer_train_after_embargo_n": len(outer_train),
            "outer_test_n": len(outer_test),
            **inner_meta,
            "A_MLP_best_epoch": history_selection.best_epoch,
            "A_MLP_validation_macro_f1": history_selection.best_metric,
            "B_GNN_best_epoch": structure_selection.best_epoch,
            "B_GNN_validation_macro_f1": structure_selection.best_metric,
            "C_Hybrid_best_epoch": hybrid_selection.best_epoch,
            "C_Hybrid_validation_macro_f1": hybrid_selection.best_metric,
        })
        print(
            "Selected epochs:",
            f"A-MLP={history_selection.best_epoch}",
            f"B-GNN={structure_selection.best_epoch}",
            f"C-Hybrid={hybrid_selection.best_epoch}",
        )

        outer_history_scaler = StandardScaler().fit(X_history_raw[outer_train])
        X_outer_train = outer_history_scaler.transform(X_history_raw[outer_train])
        X_outer_test = outer_history_scaler.transform(X_history_raw[outer_test])
        graph_train_raw = full_graph_views(graphs, outer_train)
        graph_test_raw = full_graph_views(graphs, outer_test)
        graph_train, graph_test = preprocess_graph_pair(
            graph_train_raw,
            graph_test_raw,
            X_outer_train,
            X_outer_test,
        )

        def store(model_name, prediction, score, gate_value=None):
            prediction_output.extend(prediction_rows(
                samples=samples,
                test_indices=outer_test,
                truth=y[outer_test],
                prediction=prediction,
                score=score,
                cohort="elite",
                protocol="authoritative_rolling_origin_primary",
                model=model_name,
                fold=fold,
                seed=seed,
                gate_value=gate_value,
            ))

        start = time.perf_counter()
        model = xgb_model(config, seed)
        model.fit(X_outer_train, y[outer_train])
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        pred = model.predict(X_outer_test)
        score = model.predict_proba(X_outer_test)[:, 1]
        inference_seconds = time.perf_counter() - start
        store("A-XGB", pred, score)
        runtime_rows.append({"fold": fold, "model": "A-XGB", "train_seconds": train_seconds, "inference_seconds": inference_seconds, "test_n": len(outer_test)})

        start = time.perf_counter()
        model = fit_history_model(history_factory, X_outer_train, y[outer_train], history_selection.best_epoch, config, seed, device)
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        pred, score = predict_history_model(model, X_outer_test, device)
        inference_seconds = time.perf_counter() - start
        store("A-MLP", pred, score)
        runtime_rows.append({"fold": fold, "model": "A-MLP", "train_seconds": train_seconds, "inference_seconds": inference_seconds, "test_n": len(outer_test)})

        start = time.perf_counter()
        model = fit_graph_model(structure_factory, graph_train, structure_selection.best_epoch, config, seed, device)
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        pred, score = predict_graph_model(model, graph_test, config, device)
        inference_seconds = time.perf_counter() - start
        store("B-GNN-full", pred, score)
        runtime_rows.append({"fold": fold, "model": "B-GNN-full", "train_seconds": train_seconds, "inference_seconds": inference_seconds, "test_n": len(outer_test)})

        start = time.perf_counter()
        model = fit_graph_model(hybrid_factory, graph_train, hybrid_selection.best_epoch, config, seed, device)
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        pred, score = predict_graph_model(model, graph_test, config, device)
        inference_seconds = time.perf_counter() - start
        learned_lambda = model.gate_value()
        store("C-Hybrid-full", pred, score, gate_value=learned_lambda)
        runtime_rows.append({"fold": fold, "model": "C-Hybrid-full", "train_seconds": train_seconds, "inference_seconds": inference_seconds, "test_n": len(outer_test)})

        fold_summary_rows.append({
            "fold": fold,
            "seed": seed,
            "test_start_date": str(split.test_start_date.date()),
            "test_end_date": str(split.test_end_date.date()),
            "outer_raw_train_n": split.raw_train_count,
            "outer_train_after_embargo_n": len(outer_train),
            "outer_test_n": len(outer_test),
            "inner_train_after_embargo_n": len(inner_train),
            "inner_validation_n": len(inner_val),
            "train_win_rate": float(y[outer_train].mean()),
            "test_win_rate": float(y[outer_test].mean()),
            "learned_lambda": learned_lambda,
        })
        print(f"Fold {fold} completed. Learned lambda={learned_lambda:.6f}")

    predictions_df = pd.DataFrame(prediction_output)
    fold_assignments_df = pd.DataFrame(fold_assignment_rows)
    selected_epochs_df = pd.DataFrame(selected_epoch_rows)
    fold_summary_df = pd.DataFrame(fold_summary_rows)
    parameter_counts_df = pd.DataFrame(parameter_count_rows)
    runtime_df = pd.DataFrame(runtime_rows)

    expected_models = {"A-XGB", "A-MLP", "B-GNN-full", "C-Hybrid-full"}
    if set(predictions_df["model"].unique()) != expected_models:
        raise RuntimeError("The final prediction file does not contain exactly the four primary models")
    if predictions_df.duplicated(["match_id", "model"]).any():
        raise RuntimeError("A match/model pair appears more than once in outer-test predictions")

    reference_ids = None
    for model_name, group in predictions_df.groupby("model"):
        ids = set(group["match_id"].astype(int))
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise RuntimeError(f"Paired prediction IDs differ for {model_name}")

    results_df = build_results_table(predictions_df)
    predictions_df.to_csv(OUTPUT_DIR / "fold_predictions_final.csv", index=False)
    fold_assignments_df.to_csv(OUTPUT_DIR / "fold_assignments.csv", index=False)
    selected_epochs_df.to_csv(OUTPUT_DIR / "selected_epochs.csv", index=False)
    fold_summary_df.to_csv(OUTPUT_DIR / "fold_summary.csv", index=False)
    parameter_counts_df.to_csv(OUTPUT_DIR / "parameter_counts.csv", index=False)
    runtime_df["inference_ms_per_match"] = 1000.0 * runtime_df["inference_seconds"] / runtime_df["test_n"]
    runtime_df.to_csv(OUTPUT_DIR / "runtime_summary.csv", index=False)
    results_df.to_csv(OUTPUT_DIR / "results_final.csv", index=False)

    pooled = results_df[results_df["scope"] == "pooled"].copy()
    pooled.to_csv(OUTPUT_DIR / "paper_primary_table.csv", index=False)

    hybrid = predictions_df[predictions_df["model"] == "C-Hybrid-full"].sort_values("match_id").reset_index(drop=True)
    mlp = predictions_df[predictions_df["model"] == "A-MLP"].sort_values("match_id").reset_index(drop=True)
    if not np.array_equal(hybrid["match_id"].to_numpy(), mlp["match_id"].to_numpy()):
        raise RuntimeError("Hybrid and MLP predictions are not paired")

    truth = hybrid["true_label"].to_numpy()
    hybrid_pred = hybrid["predicted_class"].to_numpy()
    mlp_pred = mlp["predicted_class"].to_numpy()
    hybrid_acc = accuracy_score(truth, hybrid_pred)
    mlp_acc = accuracy_score(truth, mlp_pred)
    hybrid_f1 = f1_score(truth, hybrid_pred, average="macro", zero_division=0)
    mlp_f1 = f1_score(truth, mlp_pred, average="macro", zero_division=0)
    bootstrap = paired_bootstrap(truth, hybrid_pred, mlp_pred)

    hybrid_correct = hybrid_pred == truth
    mlp_correct = mlp_pred == truth
    h_correct_m_wrong = int(np.sum(hybrid_correct & ~mlp_correct))
    m_correct_h_wrong = int(np.sum(mlp_correct & ~hybrid_correct))
    discordant = h_correct_m_wrong + m_correct_h_wrong
    p_value = (
        float(binomtest(min(h_correct_m_wrong, m_correct_h_wrong), n=discordant, p=0.5, alternative="two-sided").pvalue)
        if discordant > 0 else 1.0
    )

    pairwise_df = pd.DataFrame([{
        "comparison": "C-Hybrid-full_minus_A-MLP",
        "paired_n": len(truth),
        "hybrid_accuracy": hybrid_acc,
        "mlp_accuracy": mlp_acc,
        "accuracy_difference": hybrid_acc - mlp_acc,
        "accuracy_difference_pp": 100.0 * (hybrid_acc - mlp_acc),
        "accuracy_ci_lower": bootstrap["accuracy_ci_lower"],
        "accuracy_ci_upper": bootstrap["accuracy_ci_upper"],
        "hybrid_macro_f1": hybrid_f1,
        "mlp_macro_f1": mlp_f1,
        "macro_f1_difference": hybrid_f1 - mlp_f1,
        "macro_f1_difference_pp": 100.0 * (hybrid_f1 - mlp_f1),
        "macro_f1_ci_lower": bootstrap["macro_f1_ci_lower"],
        "macro_f1_ci_upper": bootstrap["macro_f1_ci_upper"],
        "hybrid_correct_mlp_wrong": h_correct_m_wrong,
        "mlp_correct_hybrid_wrong": m_correct_h_wrong,
        "mcnemar_exact_p": p_value,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }])
    pairwise_df.to_csv(OUTPUT_DIR / "primary_pairwise_inference.csv", index=False)

    gate_df = hybrid[["fold", "gate_value"]].drop_duplicates().sort_values("fold").reset_index(drop=True)
    gate_df.to_csv(OUTPUT_DIR / "static_gate_by_fold.csv", index=False)

    metadata = {
        "artifact": "Authoritative Elite Rolling-Origin Primary Experiment",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "Elite",
        "competitions": ["La Liga", "Champions League"],
        "candidate_matches": int(len(candidate_manifest)),
        "final_included_observations": int(len(samples)),
        "paired_outer_test_observations": int(len(reference_ids)),
        "initial_train_fraction": INITIAL_TRAIN_FRACTION,
        "outer_folds": N_OUTER_FOLDS,
        "team_embargo_matches": TEAM_EMBARGO_MATCHES,
        "inner_validation_fraction": INNER_VALIDATION_FRACTION,
        "minimum_outer_train_after_embargo": MIN_OUTER_TRAIN,
        "minimum_inner_train_after_embargo": MIN_INNER_TRAIN,
        "seed_policy": "42 + 100*fold; same fold seed for all four models",
        "primary_metric": "macro_f1",
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "event_download_attempts": 5,
        "event_load_failures_allowed": False,
    }
    (OUTPUT_DIR / "reproducibility_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Root convenience copies requested for the final artifact.
    shutil.copy2(OUTPUT_DIR / "cohort_manifest.csv", PROJECT_ROOT / "cohort_manifest.csv")
    shutil.copy2(OUTPUT_DIR / "fold_predictions_final.csv", PROJECT_ROOT / "fold_predictions_final.csv")
    shutil.copy2(OUTPUT_DIR / "results_final.csv", PROJECT_ROOT / "results_final.csv")

    readme = f"""AUTHORITATIVE FOOTBALL GNN — FINAL ROLLING-ORIGIN PRIMARY EXPERIMENT
===================================================================

This artifact supersedes the earlier smoke and six-block forward-chaining diagnostic runs.

Why the protocol changed
------------------------
The earlier first outer fold retained only 67 training observations and only 15 inner-training
observations after the three-match team embargo. That was too small for defensible epoch selection
for the GNN/Hybrid models. The final protocol therefore fixes a 40% initial chronological training
window before five rolling-origin test windows. No result was used to choose this correction.

Final protocol
--------------
- Elite cohort: La Liga + Champions League.
- One home-team sample per completed match.
- Official match metadata provides home/away identity and final outcome.
- Actual completed StatsBomb pass recipients only; no random recipients.
- Full graph node features: completed passes, shots, shot xG sum, ball recoveries.
- Full graph edge features: pass length, pass angle, key-pass indicator.
- Three-match pre-match history from each team's perspective, grouped by competition.
- Initial chronological training period: 40% of eligible observations, adjusted to avoid splitting a date.
- Five non-overlapping chronological test windows on the remaining future period.
- Expanding/rolling-origin training before each test window.
- Three-match per-team embargo at outer and inner boundaries.
- Date-aware inner chronological validation for epoch selection.
- Minimum outer training size after embargo: {MIN_OUTER_TRAIN}.
- Minimum inner training size after embargo: {MIN_INNER_TRAIN}.
- Training-only scaling.
- Adam, lr=0.002, weight_decay=5e-4, batch=32, max_epochs=50, patience=10.
- Primary metric: macro F1.
- Fixed fold seed: 42 + 100*fold, shared by all four models in the same fold.
- Event downloads retry up to five times and are locally cached for the run. The final experiment aborts
  rather than silently dropping a match because of a transient event-download failure.

Primary models
--------------
A-XGB, A-MLP, B-GNN-full, C-Hybrid-full.

Primary inference
-----------------
C-Hybrid-full versus A-MLP using paired outer-test predictions, 10,000 paired match-level bootstrap
resamples, and an exact McNemar test.

Authoritative files
-------------------
config_final.yaml
cohort_manifest.csv
fold_predictions_final.csv
results_final.csv
outputs/elite_rolling_origin_final/candidate_match_manifest.csv
outputs/elite_rolling_origin_final/cohort_flow.csv
outputs/elite_rolling_origin_final/fold_assignments.csv
outputs/elite_rolling_origin_final/fold_summary.csv
outputs/elite_rolling_origin_final/selected_epochs.csv
outputs/elite_rolling_origin_final/primary_pairwise_inference.csv
outputs/elite_rolling_origin_final/static_gate_by_fold.csv
outputs/elite_rolling_origin_final/runtime_summary.csv
outputs/elite_rolling_origin_final/parameter_counts.csv
outputs/elite_rolling_origin_final/environment.json
environment_freeze.txt
src/final_protocol.py
run_final_rolling_origin.py

Publication rule
----------------
All manuscript values must be regenerated from results_final.csv and the paired prediction file.
Legacy notebook percentages must not be mixed with this implementation.
"""
    (OUTPUT_DIR / "README_EXPERIMENT1.txt").write_text(
    readme,
    encoding="utf-8"
)

    # Create checksums for files that will be packaged, excluding cache and the checksum file itself.
    excluded_parts = {"__pycache__", ".pytest_cache", ".git", "cache"}
    checksum_rows = []
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS.csv" or path.suffix == ".zip":
            continue
        if any(part in excluded_parts for part in path.parts):
            continue
        checksum_rows.append({
            "path": str(path.relative_to(PROJECT_ROOT)),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        })
    pd.DataFrame(checksum_rows).to_csv(PROJECT_ROOT / "SHA256SUMS.csv", index=False)

    if FINAL_ZIP.exists():
        FINAL_ZIP.unlink()
    with zipfile.ZipFile(FINAL_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(PROJECT_ROOT.rglob("*")):
            if not path.is_file() or path.suffix == ".zip":
                continue
            if any(part in excluded_parts for part in path.parts):
                continue
            archive.write(path, Path("football_gnn_authoritative") / path.relative_to(PROJECT_ROOT))

    print("\n" + "=" * 100)
    print("FINAL ROLLING-ORIGIN RUN COMPLETED")
    print("=" * 100)
    print("Final eligible cohort N:", len(samples))
    print("Paired outer-test N:", len(reference_ids))
    print("Minimum outer training N:", int(fold_summary_df["outer_train_after_embargo_n"].min()))
    print("Minimum inner training N:", int(fold_summary_df["inner_train_after_embargo_n"].min()))
    print("Hybrid accuracy:", f"{hybrid_acc:.4f}")
    print("A-MLP accuracy:", f"{mlp_acc:.4f}")
    print("Accuracy difference:", f"{100*(hybrid_acc-mlp_acc):+.3f} pp")
    print("Accuracy 95% CI:", f"[{100*bootstrap['accuracy_ci_lower']:+.3f}, {100*bootstrap['accuracy_ci_upper']:+.3f}] pp")
    print("Hybrid macro F1:", f"{hybrid_f1:.4f}")
    print("A-MLP macro F1:", f"{mlp_f1:.4f}")
    print("Macro-F1 difference:", f"{100*(hybrid_f1-mlp_f1):+.3f} pp")
    print("Macro-F1 95% CI:", f"[{100*bootstrap['macro_f1_ci_lower']:+.3f}, {100*bootstrap['macro_f1_ci_upper']:+.3f}] pp")
    print("McNemar exact p:", f"{p_value:.6f}")
    print("Final archive:", FINAL_ZIP)


if __name__ == "__main__":
    main()
