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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from src.evaluation import build_results_table, prediction_rows
from src.final_protocol import harvest_dataset_with_retry_cache
from src.models import StructureGNN, HybridGNN
from src.reproducibility import save_environment, set_global_seed
from src.scaling import (
    aggregate_graph_features,
    fit_graph_scalers,
    make_feature_view,
    transform_graphs,
)
from src.training import fit_graph_model, predict_graph_model, select_graph_epoch

PROJECT_ROOT = Path.cwd()
REFERENCE_DIR = PROJECT_ROOT / "reference_experiment1"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiment4_reduced_feature_robustness"
FINAL_ZIP = Path("/content/football_gnn_experiment4_reduced_feature_robustness_final.zip")

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 2026
EXPECTED_ELIGIBLE_N = 757
EXPECTED_OUTER_TEST_N = 454

FULL_NODE_INDICES = [0, 1, 2, 3]
FULL_EDGE_INDICES = [0, 1, 2]
REDUCED_NODE_INDICES = [0, 3]       # completed passes, ball recoveries
REDUCED_EDGE_INDICES = [0, 1]       # pass length, pass angle

REMOVED_FEATURES = [
    "node: shot_count",
    "node: shot_xg_sum",
    "edge: key_pass_indicator",
]
RETAINED_FEATURES = [
    "node: completed_pass_count",
    "node: ball_recovery_count",
    "edge: pass_length",
    "edge: pass_angle",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def graph_views(graphs: list, indices: np.ndarray, node_indices: list[int], edge_indices: list[int]):
    return [
        make_feature_view(
            graphs[int(index)],
            node_indices=node_indices,
            edge_indices=edge_indices,
        )
        for index in indices
    ]


def preprocess_graph_pair(train_raw, other_raw, train_history=None, other_history=None):
    # Every retained reduced edge feature is continuous (length, angle).
    edge_continuous = list(range(train_raw[0].edge_attr.shape[1]))
    scalers = fit_graph_scalers(
        graphs=train_raw,
        standardize_nodes=True,
        edge_continuous_indices=edge_continuous,
    )
    train = transform_graphs(train_raw, scalers, history_features=train_history)
    other = transform_graphs(other_raw, scalers, history_features=other_history)
    return train, other


def paired_bootstrap(truth, pred_a, pred_b, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    n = len(truth)
    acc_diff = np.empty(samples, dtype=float)
    f1_diff = np.empty(samples, dtype=float)
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        yb = truth[idx]
        a = pred_a[idx]
        b = pred_b[idx]
        acc_diff[i] = accuracy_score(yb, a) - accuracy_score(yb, b)
        f1_diff[i] = (
            f1_score(yb, a, average="macro", zero_division=0)
            - f1_score(yb, b, average="macro", zero_division=0)
        )
    return {
        "accuracy_ci_lower": float(np.percentile(acc_diff, 2.5)),
        "accuracy_ci_upper": float(np.percentile(acc_diff, 97.5)),
        "macro_f1_ci_lower": float(np.percentile(f1_diff, 2.5)),
        "macro_f1_ci_upper": float(np.percentile(f1_diff, 97.5)),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted_sorted = np.zeros(m, dtype=float)
    running = 0.0
    for rank, original_index in enumerate(order):
        candidate = (m - rank) * p[original_index]
        running = max(running, candidate)
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.zeros(m, dtype=float)
    for rank, original_index in enumerate(order):
        adjusted[original_index] = adjusted_sorted[rank]
    return adjusted.tolist()


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def make_xgb(config: dict, seed: int):
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


def require_reference_files():
    required = [
        "candidate_match_manifest.csv",
        "cohort_manifest.csv",
        "fold_assignments.csv",
        "fold_predictions_final.csv",
        "included_samples.csv",
        "history_feature_order.json",
        "selected_epochs.csv",
        "results_final.csv",
        "parameter_counts.csv",
    ]
    missing = [name for name in required if not (REFERENCE_DIR / name).exists()]
    if missing:
        raise RuntimeError(f"Missing frozen Experiment 1 reference files: {missing}")


def main():
    require_reference_files()
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with (PROJECT_ROOT / "config_final.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    # Hard freeze of the authoritative Experiment 1 protocol.
    if config["validation"]["strategy"] != "rolling_origin_40pct_initial_train":
        raise RuntimeError("Experiment 4 requires the frozen Experiment 1 rolling-origin protocol")
    if config["model"]["pooling"] != "positional_add":
        raise RuntimeError("Experiment 4 must retain frozen positional pooling")
    if config["graph"].get("reduced_node_indices") != REDUCED_NODE_INDICES:
        raise RuntimeError("Reduced node-feature indices differ from the preregistered [0, 3]")
    if config["graph"].get("reduced_edge_indices") != REDUCED_EDGE_INDICES:
        raise RuntimeError("Reduced edge-feature indices differ from the preregistered [0, 1]")

    exp_config = {
        "experiment": "Experiment 4 - reduced-feature robustness",
        "scientific_question": (
            "Does the Hybrid retain useful performance after removing outcome-adjacent shot count, "
            "shot xG sum, and key-pass indicator features?"
        ),
        "football_question": (
            "If the model is no longer told how many shots the team took, how much xG it created, "
            "or which passes directly created shots, does the passing/history representation still help?"
        ),
        "removed_features": REMOVED_FEATURES,
        "retained_features": RETAINED_FEATURES,
        "frozen_protocol": {
            "eligible_n": EXPECTED_ELIGIBLE_N,
            "paired_outer_test_n": EXPECTED_OUTER_TEST_N,
            "outer_protocol": "40% initial train rolling origin with 5 future test folds",
            "team_embargo_matches": 3,
            "same_outer_folds": True,
            "same_inner_folds": True,
            "same_training_seeds": True,
            "same_graph_topology": True,
            "same_positional_pooling": True,
            "same_optimizer_and_training": True,
        },
        "models": [
            "A-MLP (frozen Experiment 1 reference)",
            "B-GNN-full (frozen Experiment 1 reference)",
            "C-Hybrid-full (frozen Experiment 1 reference)",
            "B-GNN-reduced",
            "C-Hybrid-reduced",
            "D-Aggregate-full",
            "D-Aggregate-reduced",
        ],
        "predeclared_comparisons": [
            "C-Hybrid-reduced vs A-MLP",
            "C-Hybrid-reduced vs D-Aggregate-reduced",
            "C-Hybrid-full vs C-Hybrid-reduced",
            "B-GNN-full vs B-GNN-reduced",
            "D-Aggregate-full vs D-Aggregate-reduced",
        ],
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    (OUTPUT_DIR / "experiment4_config.json").write_text(json.dumps(exp_config, indent=2), encoding="utf-8")
    save_environment(OUTPUT_DIR / "environment.json")
    with (OUTPUT_DIR / "environment_freeze.txt").open("w", encoding="utf-8") as handle:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=handle, check=True)

    samples = pd.read_csv(REFERENCE_DIR / "included_samples.csv").reset_index(drop=True)
    assignments = pd.read_csv(REFERENCE_DIR / "fold_assignments.csv")
    frozen_predictions = pd.read_csv(REFERENCE_DIR / "fold_predictions_final.csv")
    selected_epochs = pd.read_csv(REFERENCE_DIR / "selected_epochs.csv")
    candidate_manifest = pd.read_csv(REFERENCE_DIR / "candidate_match_manifest.csv")
    with (REFERENCE_DIR / "history_feature_order.json").open("r", encoding="utf-8") as handle:
        history_columns = json.load(handle)

    if len(samples) != EXPECTED_ELIGIBLE_N:
        raise RuntimeError(f"Eligible N changed: expected {EXPECTED_ELIGIBLE_N}, got {len(samples)}")
    if len(history_columns) != 16 or any(col not in samples.columns for col in history_columns):
        raise RuntimeError("Frozen 16-dimensional history feature order is missing or inconsistent")

    y = samples["label"].astype(int).to_numpy()
    X_history_raw = samples[history_columns].astype(float).to_numpy()
    id_to_index = {int(mid): idx for idx, mid in enumerate(samples["match_id"].astype(int))}

    # Reuse exact frozen Experiment 1 out-of-fold predictions where nothing changes.
    frozen_frames = []
    rename_map = {
        "A-MLP": "A-MLP",
        "B-GNN-full": "B-GNN-full",
        "C-Hybrid-full": "C-Hybrid-full",
    }
    for old_name, new_name in rename_map.items():
        frame = frozen_predictions[frozen_predictions["model"].eq(old_name)].copy()
        if len(frame) != EXPECTED_OUTER_TEST_N:
            raise RuntimeError(f"Frozen {old_name} prediction count is not {EXPECTED_OUTER_TEST_N}")
        frame["protocol"] = "experiment4_reduced_feature_robustness"
        frame["model"] = new_name
        frozen_frames.append(frame)

    print("=" * 100)
    print("EXPERIMENT 4 — REDUCED-FEATURE ROBUSTNESS")
    print("=" * 100)
    print("Frozen eligible cohort N:", len(samples))
    print("Frozen paired outer-test N:", EXPECTED_OUTER_TEST_N)
    print("Removed: shot count, shot xG sum, key-pass indicator")
    print("Retained: completed-pass count, ball recoveries, pass length, pass angle")
    print("Same real passing topology, positional pooling, historical branch, folds, embargo, and seeds.")

    print("\nSTEP 1 — RECONSTRUCTING THE EXACT FROZEN STATSBOMB GRAPHS")
    _, graph_map, reharvest_manifest, new_candidate_manifest = harvest_dataset_with_retry_cache(
        config, attempts=5, fail_on_event_error=True
    )

    if set(candidate_manifest["match_id"].astype(int)) != set(new_candidate_manifest["match_id"].astype(int)):
        raise RuntimeError("StatsBomb candidate-match set changed from frozen Experiment 1")

    missing_graphs = [mid for mid in samples["match_id"].astype(int) if mid not in graph_map]
    if missing_graphs:
        raise RuntimeError(f"Could not reconstruct {len(missing_graphs)} frozen graphs: {missing_graphs[:10]}")

    manifest_lookup = reharvest_manifest.set_index("match_id")
    for row in samples.itertuples(index=False):
        current = manifest_lookup.loc[int(row.match_id)]
        if str(current["home_team"]) != str(row.home_team) or str(current["away_team"]) != str(row.away_team):
            raise RuntimeError(f"Team metadata changed for match_id={row.match_id}")
        label_now = int(graph_map[int(row.match_id)].y.view(-1)[0].item())
        if label_now != int(row.label):
            raise RuntimeError(f"Label changed for match_id={row.match_id}")

    graphs = [graph_map[int(mid)] for mid in samples["match_id"].astype(int)]
    samples.to_csv(OUTPUT_DIR / "frozen_included_samples.csv", index=False)
    assignments.to_csv(OUTPUT_DIR / "frozen_fold_assignments.csv", index=False)
    selected_epochs.to_csv(OUTPUT_DIR / "frozen_experiment1_selected_epochs.csv", index=False)
    reharvest_manifest.to_csv(OUTPUT_DIR / "reharvest_manifest.csv", index=False)

    # Explicit feature audit before training.
    audit_rows = []
    for sample_idx, graph in enumerate(graphs):
        reduced = make_feature_view(graph, REDUCED_NODE_INDICES, REDUCED_EDGE_INDICES)
        if reduced.x.shape[1] != 2 or reduced.edge_attr.shape[1] != 2:
            raise RuntimeError(f"Reduced feature dimensionality failed for match {samples.iloc[sample_idx]['match_id']}")
        # The retained reduced features must be exact slices of the full graph.
        if not torch.equal(reduced.x, graph.x[:, REDUCED_NODE_INDICES]):
            raise RuntimeError("Reduced node-feature slice changed values")
        if not torch.equal(reduced.edge_attr, graph.edge_attr[:, REDUCED_EDGE_INDICES]):
            raise RuntimeError("Reduced edge-feature slice changed values")
        audit_rows.append({
            "match_id": int(samples.iloc[sample_idx]["match_id"]),
            "full_node_dim": int(graph.x.shape[1]),
            "reduced_node_dim": int(reduced.x.shape[1]),
            "full_edge_dim": int(graph.edge_attr.shape[1]),
            "reduced_edge_dim": int(reduced.edge_attr.shape[1]),
            "edge_count_preserved": int(reduced.edge_index.shape[1]) == int(graph.edge_index.shape[1]),
            "topology_preserved": bool(torch.equal(reduced.edge_index, graph.edge_index)),
            "label_preserved": int(reduced.y.view(-1)[0].item()) == int(graph.y.view(-1)[0].item()),
        })
    pd.DataFrame(audit_rows).to_csv(OUTPUT_DIR / "reduced_feature_integrity_audit.csv", index=False)

    def indices_for(fold: int, level: str, role: str) -> np.ndarray:
        ids = assignments.loc[
            assignments["fold"].eq(fold)
            & assignments["level"].eq(level)
            & assignments["role"].eq(role),
            "match_id",
        ].astype(int).tolist()
        if not ids:
            raise RuntimeError(f"Missing frozen assignment fold={fold}, level={level}, role={role}")
        return np.asarray([id_to_index[mid] for mid in ids], dtype=int)

    frozen_test_ids = set(assignments.loc[
        assignments["level"].eq("outer") & assignments["role"].eq("test"), "match_id"
    ].astype(int))
    for frame in frozen_frames:
        if set(frame["match_id"].astype(int)) != frozen_test_ids:
            raise RuntimeError("Frozen reference predictions do not match exact outer-test fold assignments")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Graph reconstruction, feature audit, and frozen-fold verification passed.")
    print("Device:", device)

    prediction_frames = list(frozen_frames)
    selected_rows = []
    runtime_rows = []
    gate_rows = []

    # Parameter/input audit. Full models are included as references; reduced models are newly trained.
    param_rows = [
        {
            "model": "B-GNN-full",
            "node_input_dim": 4,
            "edge_input_dim": 3,
            "trainable_parameters": count_parameters(StructureGNN(4, 3, config, pooling="positional_add")),
        },
        {
            "model": "B-GNN-reduced",
            "node_input_dim": 2,
            "edge_input_dim": 2,
            "trainable_parameters": count_parameters(StructureGNN(2, 2, config, pooling="positional_add")),
        },
        {
            "model": "C-Hybrid-full",
            "node_input_dim": 4,
            "edge_input_dim": 3,
            "trainable_parameters": count_parameters(HybridGNN(4, 3, 16, config, pooling="positional_add")),
        },
        {
            "model": "C-Hybrid-reduced",
            "node_input_dim": 2,
            "edge_input_dim": 2,
            "trainable_parameters": count_parameters(HybridGNN(2, 2, 16, config, pooling="positional_add")),
        },
        {
            "model": "D-Aggregate-full",
            "node_input_dim": 4,
            "edge_input_dim": 3,
            "aggregate_input_dim": 7,
            "trainable_parameters": None,
        },
        {
            "model": "D-Aggregate-reduced",
            "node_input_dim": 2,
            "edge_input_dim": 2,
            "aggregate_input_dim": 4,
            "trainable_parameters": None,
        },
    ]
    pd.DataFrame(param_rows).to_csv(OUTPUT_DIR / "reduced_parameter_counts.csv", index=False)

    print("\nSTEP 2 — TRAINING REDUCED B-GNN / HYBRID AND DIRECT AGGREGATE CONTROLS")
    for fold in range(1, 6):
        outer_train = indices_for(fold, "outer", "train")
        outer_test = indices_for(fold, "outer", "test")
        inner_train = indices_for(fold, "inner", "train")
        inner_val = indices_for(fold, "inner", "validation")

        seed_row = selected_epochs.loc[selected_epochs["fold"].eq(fold)]
        if len(seed_row) != 1:
            raise RuntimeError(f"Could not uniquely resolve frozen seed for fold {fold}")
        training_seed = int(seed_row.iloc[0]["seed"])

        expected_fold_ids = set(samples.iloc[outer_test]["match_id"].astype(int))
        if expected_fold_ids != set(frozen_frames[1].loc[frozen_frames[1]["fold"].eq(fold), "match_id"].astype(int)):
            raise RuntimeError(f"Fold {fold} test IDs changed from Experiment 1")

        print("\n" + "-" * 100)
        print(
            f"FOLD {fold}: outer train={len(outer_train)}, test={len(outer_test)}, "
            f"inner train={len(inner_train)}, inner val={len(inner_val)}, seed={training_seed}"
        )

        # Inner preprocessing for epoch selection.
        inner_history_scaler = StandardScaler().fit(X_history_raw[inner_train])
        X_inner_train = inner_history_scaler.transform(X_history_raw[inner_train])
        X_inner_val = inner_history_scaler.transform(X_history_raw[inner_val])

        inner_train_raw = graph_views(graphs, inner_train, REDUCED_NODE_INDICES, REDUCED_EDGE_INDICES)
        inner_val_raw = graph_views(graphs, inner_val, REDUCED_NODE_INDICES, REDUCED_EDGE_INDICES)
        inner_train_scaled, inner_val_scaled = preprocess_graph_pair(
            inner_train_raw,
            inner_val_raw,
            train_history=X_inner_train,
            other_history=X_inner_val,
        )

        b_factory = lambda: StructureGNN(2, 2, config, pooling="positional_add")
        c_factory = lambda: HybridGNN(2, 2, 16, config, pooling="positional_add")

        start = time.perf_counter()
        b_selection = select_graph_epoch(
            b_factory, inner_train_scaled, inner_val_scaled, y[inner_val], config, training_seed, device
        )
        b_select_seconds = time.perf_counter() - start

        start = time.perf_counter()
        c_selection = select_graph_epoch(
            c_factory, inner_train_scaled, inner_val_scaled, y[inner_val], config, training_seed, device
        )
        c_select_seconds = time.perf_counter() - start

        # Outer preprocessing refitted on the complete outer training partition only.
        outer_history_scaler = StandardScaler().fit(X_history_raw[outer_train])
        X_outer_train = outer_history_scaler.transform(X_history_raw[outer_train])
        X_outer_test = outer_history_scaler.transform(X_history_raw[outer_test])

        red_outer_train_raw = graph_views(graphs, outer_train, REDUCED_NODE_INDICES, REDUCED_EDGE_INDICES)
        red_outer_test_raw = graph_views(graphs, outer_test, REDUCED_NODE_INDICES, REDUCED_EDGE_INDICES)
        red_outer_train, red_outer_test = preprocess_graph_pair(
            red_outer_train_raw,
            red_outer_test_raw,
            train_history=X_outer_train,
            other_history=X_outer_test,
        )

        set_global_seed(training_seed)
        start = time.perf_counter()
        b_model = fit_graph_model(
            b_factory,
            red_outer_train,
            epochs=b_selection.best_epoch,
            config=config,
            seed=training_seed,
            device=device,
        )
        b_fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        b_pred, b_score = predict_graph_model(b_model, red_outer_test, config, device)
        b_predict_seconds = time.perf_counter() - start

        set_global_seed(training_seed)
        start = time.perf_counter()
        c_model = fit_graph_model(
            c_factory,
            red_outer_train,
            epochs=c_selection.best_epoch,
            config=config,
            seed=training_seed,
            device=device,
        )
        c_fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        c_pred, c_score = predict_graph_model(c_model, red_outer_test, config, device)
        c_predict_seconds = time.perf_counter() - start
        c_gate = c_model.gate_value()

        prediction_frames.append(pd.DataFrame(prediction_rows(
            samples=samples,
            test_indices=outer_test,
            truth=y[outer_test],
            prediction=b_pred,
            score=b_score,
            cohort="elite",
            protocol="experiment4_reduced_feature_robustness",
            model="B-GNN-reduced",
            fold=fold,
            seed=training_seed,
            gate_value=None,
        )))
        prediction_frames.append(pd.DataFrame(prediction_rows(
            samples=samples,
            test_indices=outer_test,
            truth=y[outer_test],
            prediction=c_pred,
            score=c_score,
            cohort="elite",
            protocol="experiment4_reduced_feature_robustness",
            model="C-Hybrid-reduced",
            fold=fold,
            seed=training_seed,
            gate_value=c_gate,
        )))

        # Direct in-match aggregate controls: same match-derived features but no graph message passing.
        # They are fitted only on outer-training data, with training-only scaling.
        full_train_raw = graph_views(graphs, outer_train, FULL_NODE_INDICES, FULL_EDGE_INDICES)
        full_test_raw = graph_views(graphs, outer_test, FULL_NODE_INDICES, FULL_EDGE_INDICES)
        D_full_train_raw = aggregate_graph_features(full_train_raw)
        D_full_test_raw = aggregate_graph_features(full_test_raw)
        D_full_scaler = StandardScaler().fit(D_full_train_raw)
        D_full_train = D_full_scaler.transform(D_full_train_raw)
        D_full_test = D_full_scaler.transform(D_full_test_raw)

        D_red_train_raw = aggregate_graph_features(red_outer_train_raw)
        D_red_test_raw = aggregate_graph_features(red_outer_test_raw)
        D_red_scaler = StandardScaler().fit(D_red_train_raw)
        D_red_train = D_red_scaler.transform(D_red_train_raw)
        D_red_test = D_red_scaler.transform(D_red_test_raw)

        start = time.perf_counter()
        d_full = make_xgb(config, training_seed)
        d_full.fit(D_full_train, y[outer_train])
        d_full_fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        d_full_pred = d_full.predict(D_full_test)
        d_full_score = d_full.predict_proba(D_full_test)[:, 1]
        d_full_predict_seconds = time.perf_counter() - start

        start = time.perf_counter()
        d_red = make_xgb(config, training_seed)
        d_red.fit(D_red_train, y[outer_train])
        d_red_fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        d_red_pred = d_red.predict(D_red_test)
        d_red_score = d_red.predict_proba(D_red_test)[:, 1]
        d_red_predict_seconds = time.perf_counter() - start

        prediction_frames.append(pd.DataFrame(prediction_rows(
            samples=samples,
            test_indices=outer_test,
            truth=y[outer_test],
            prediction=d_full_pred,
            score=d_full_score,
            cohort="elite",
            protocol="experiment4_reduced_feature_robustness",
            model="D-Aggregate-full",
            fold=fold,
            seed=training_seed,
            gate_value=None,
        )))
        prediction_frames.append(pd.DataFrame(prediction_rows(
            samples=samples,
            test_indices=outer_test,
            truth=y[outer_test],
            prediction=d_red_pred,
            score=d_red_score,
            cohort="elite",
            protocol="experiment4_reduced_feature_robustness",
            model="D-Aggregate-reduced",
            fold=fold,
            seed=training_seed,
            gate_value=None,
        )))

        selected_rows.extend([
            {
                "fold": fold,
                "model": "B-GNN-reduced",
                "seed": training_seed,
                "best_epoch": int(b_selection.best_epoch),
                "inner_validation_macro_f1": float(b_selection.best_metric),
                "inner_train_n": len(inner_train),
                "inner_validation_n": len(inner_val),
                "outer_train_n": len(outer_train),
                "outer_test_n": len(outer_test),
            },
            {
                "fold": fold,
                "model": "C-Hybrid-reduced",
                "seed": training_seed,
                "best_epoch": int(c_selection.best_epoch),
                "inner_validation_macro_f1": float(c_selection.best_metric),
                "inner_train_n": len(inner_train),
                "inner_validation_n": len(inner_val),
                "outer_train_n": len(outer_train),
                "outer_test_n": len(outer_test),
            },
        ])
        runtime_rows.extend([
            {
                "fold": fold,
                "model": "B-GNN-reduced",
                "selection_seconds": b_select_seconds,
                "fit_seconds": b_fit_seconds,
                "predict_seconds": b_predict_seconds,
                "total_model_seconds": b_select_seconds + b_fit_seconds + b_predict_seconds,
            },
            {
                "fold": fold,
                "model": "C-Hybrid-reduced",
                "selection_seconds": c_select_seconds,
                "fit_seconds": c_fit_seconds,
                "predict_seconds": c_predict_seconds,
                "total_model_seconds": c_select_seconds + c_fit_seconds + c_predict_seconds,
            },
            {
                "fold": fold,
                "model": "D-Aggregate-full",
                "selection_seconds": 0.0,
                "fit_seconds": d_full_fit_seconds,
                "predict_seconds": d_full_predict_seconds,
                "total_model_seconds": d_full_fit_seconds + d_full_predict_seconds,
            },
            {
                "fold": fold,
                "model": "D-Aggregate-reduced",
                "selection_seconds": 0.0,
                "fit_seconds": d_red_fit_seconds,
                "predict_seconds": d_red_predict_seconds,
                "total_model_seconds": d_red_fit_seconds + d_red_predict_seconds,
            },
        ])
        gate_rows.append({"fold": fold, "seed": training_seed, "reduced_gate_value": c_gate})
        print(
            f"Selected epochs: B-reduced={b_selection.best_epoch}, C-reduced={c_selection.best_epoch}; "
            f"C-reduced lambda={c_gate:.6f}"
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    model_names = {
        "A-MLP",
        "B-GNN-full",
        "C-Hybrid-full",
        "B-GNN-reduced",
        "C-Hybrid-reduced",
        "D-Aggregate-full",
        "D-Aggregate-reduced",
    }
    if set(predictions["model"].unique()) != model_names:
        raise RuntimeError(f"Unexpected model set: {set(predictions['model'].unique())}")
    for model_name, group in predictions.groupby("model"):
        if len(group) != EXPECTED_OUTER_TEST_N:
            raise RuntimeError(f"{model_name} prediction N={len(group)}, expected {EXPECTED_OUTER_TEST_N}")
        if set(group["match_id"].astype(int)) != frozen_test_ids:
            raise RuntimeError(f"{model_name} does not use the exact frozen outer-test observations")

    predictions.to_csv(OUTPUT_DIR / "reduced_predictions.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(OUTPUT_DIR / "reduced_selected_epochs.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(OUTPUT_DIR / "runtime_summary.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(OUTPUT_DIR / "reduced_gate_by_fold.csv", index=False)

    results = build_results_table(predictions)
    results.to_csv(OUTPUT_DIR / "reduced_results.csv", index=False)

    print("\n" + "=" * 100)
    print("POOLED REDUCED-FEATURE RESULTS")
    print("=" * 100)
    pooled = results[results["scope"].eq("pooled")].copy()
    print(pooled[[
        "model", "n", "accuracy", "balanced_accuracy", "macro_f1", "win_precision", "win_recall",
        "roc_auc", "tn", "fp", "fn", "tp"
    ]].to_string(index=False))

    comparisons = [
        # label, model A, model B, bootstrap seed, positive interpretation
        (
            "Reduced hybrid vs historical branch",
            "C-Hybrid-reduced",
            "A-MLP",
            BOOTSTRAP_SEED + 1,
            "Positive difference means the reduced Hybrid still outperforms the pre-match historical MLP.",
        ),
        (
            "Reduced hybrid vs reduced direct aggregate",
            "C-Hybrid-reduced",
            "D-Aggregate-reduced",
            BOOTSTRAP_SEED + 2,
            "Positive difference means graph/history modeling outperforms a direct aggregate control using the same reduced in-match variables.",
        ),
        (
            "Hybrid full vs reduced",
            "C-Hybrid-full",
            "C-Hybrid-reduced",
            BOOTSTRAP_SEED + 3,
            "Positive difference means the removed outcome-adjacent variables improved the full Hybrid; a small difference supports robustness.",
        ),
        (
            "GNN full vs reduced",
            "B-GNN-full",
            "B-GNN-reduced",
            BOOTSTRAP_SEED + 4,
            "Positive difference means the removed outcome-adjacent variables improved the graph-only model.",
        ),
        (
            "Aggregate full vs reduced",
            "D-Aggregate-full",
            "D-Aggregate-reduced",
            BOOTSTRAP_SEED + 5,
            "Positive difference measures the contribution of the removed attacking-output variables to the direct aggregate baseline.",
        ),
    ]

    pairwise_rows = []
    raw_p = []
    for label, a_name, b_name, bootstrap_seed, interpretation in comparisons:
        a = predictions[predictions["model"].eq(a_name)].sort_values(["fold", "match_id"]).reset_index(drop=True)
        b = predictions[predictions["model"].eq(b_name)].sort_values(["fold", "match_id"]).reset_index(drop=True)
        if not np.array_equal(a["match_id"].astype(int).to_numpy(), b["match_id"].astype(int).to_numpy()):
            raise RuntimeError(f"Pairing failed for {label}")
        truth = a["true_label"].astype(int).to_numpy()
        a_pred = a["predicted_class"].astype(int).to_numpy()
        b_pred = b["predicted_class"].astype(int).to_numpy()
        a_acc = accuracy_score(truth, a_pred)
        b_acc = accuracy_score(truth, b_pred)
        a_f1 = f1_score(truth, a_pred, average="macro", zero_division=0)
        b_f1 = f1_score(truth, b_pred, average="macro", zero_division=0)
        ci = paired_bootstrap(truth, a_pred, b_pred, samples=BOOTSTRAP_SAMPLES, seed=bootstrap_seed)

        a_correct = a_pred == truth
        b_correct = b_pred == truth
        a_correct_b_wrong = int(np.sum(a_correct & ~b_correct))
        b_correct_a_wrong = int(np.sum(b_correct & ~a_correct))
        discordant = a_correct_b_wrong + b_correct_a_wrong
        if discordant:
            p_value = float(binomtest(
                min(a_correct_b_wrong, b_correct_a_wrong),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue)
        else:
            p_value = 1.0
        raw_p.append(p_value)

        pairwise_rows.append({
            "comparison": label,
            "model_a": a_name,
            "model_b": b_name,
            "paired_n": len(truth),
            "model_a_accuracy": a_acc,
            "model_b_accuracy": b_acc,
            "accuracy_difference_a_minus_b": a_acc - b_acc,
            "accuracy_difference_pp": (a_acc - b_acc) * 100,
            "accuracy_ci_lower": ci["accuracy_ci_lower"],
            "accuracy_ci_upper": ci["accuracy_ci_upper"],
            "model_a_macro_f1": a_f1,
            "model_b_macro_f1": b_f1,
            "macro_f1_difference_a_minus_b": a_f1 - b_f1,
            "macro_f1_difference_pp": (a_f1 - b_f1) * 100,
            "macro_f1_ci_lower": ci["macro_f1_ci_lower"],
            "macro_f1_ci_upper": ci["macro_f1_ci_upper"],
            "model_a_correct_b_wrong": a_correct_b_wrong,
            "model_b_correct_a_wrong": b_correct_a_wrong,
            "mcnemar_exact_p_raw": p_value,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "interpretation": interpretation,
        })

    adjusted = holm_adjust(raw_p)
    for row, adjusted_p in zip(pairwise_rows, adjusted):
        row["mcnemar_exact_p_holm"] = adjusted_p

    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_df.to_csv(OUTPUT_DIR / "reduced_pairwise_inference.csv", index=False)

    primary = pairwise_df[pairwise_df["comparison"].isin([
        "Reduced hybrid vs historical branch",
        "Reduced hybrid vs reduced direct aggregate",
        "Hybrid full vs reduced",
    ])].copy()
    primary["reporting_rule"] = (
        "Report point estimate and paired 95% bootstrap CI. Do not claim statistical superiority when the CI contains zero "
        "or Holm-adjusted McNemar p >= 0.05. Robustness does not require identical full and reduced scores; it asks whether "
        "useful performance/complementarity remains after removal of outcome-adjacent variables."
    )
    primary.to_csv(OUTPUT_DIR / "reduced_primary_summary.csv", index=False)

    print("\n" + "=" * 100)
    print("PAIRED REDUCED-FEATURE INFERENCE")
    print("=" * 100)
    print(pairwise_df.to_string(index=False))

    metadata = {
        "artifact": "Experiment 4 - Reduced-Feature Robustness",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "eligible_cohort_n": EXPECTED_ELIGIBLE_N,
        "paired_outer_test_n": EXPECTED_OUTER_TEST_N,
        "removed_features": REMOVED_FEATURES,
        "retained_features": RETAINED_FEATURES,
        "same_graph_topology": True,
        "same_positional_pooling": True,
        "same_history_features": True,
        "frozen_folds": True,
        "frozen_training_seeds": True,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "primary_comparisons": [
            "C-Hybrid-reduced vs A-MLP",
            "C-Hybrid-reduced vs D-Aggregate-reduced",
            "C-Hybrid-full vs C-Hybrid-reduced",
        ],
    }
    (OUTPUT_DIR / "reproducibility_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    readme = f"""
EXPERIMENT 4 — REDUCED-FEATURE ROBUSTNESS
==========================================

QUESTION
--------
Does the Hybrid retain useful outcome-linked classification performance after removing the most outcome-adjacent attacking variables?

FOOTBALL INTERPRETATION
-----------------------
The full graph tells the model not only how the home team circulated the ball, but also how many shots it produced,
how much StatsBomb shot xG it accumulated, and which passes directly created shots. Those variables are naturally close
to the final result of the match. A team that creates 20 shots and 2.5 xG has already revealed a lot about how the match went.

The reduced experiment deliberately removes:
- shot count
- shot xG sum
- key-pass indicator

It retains:
- completed-pass count per player
- ball-recovery count per player
- pass length
- pass angle
- the exact observed passer-recipient network
- GK/DF/MF/FW positional pooling
- all 16 pre-match historical variables

FROZEN PROTOCOL
---------------
Eligible Elite cohort: {EXPECTED_ELIGIBLE_N}
Paired outer-test observations: {EXPECTED_OUTER_TEST_N}
Same rolling-origin folds: yes
Same inner chronological folds: yes
Same three-match team embargo: yes
Same training seeds: yes
Same topology: yes
Same positional pooling: yes
Same optimizer/training settings: yes

MODELS
------
A-MLP                : frozen exact historical branch from Experiment 1
B-GNN-full           : frozen full-feature graph-only result from Experiment 1
C-Hybrid-full        : frozen full-feature Hybrid result from Experiment 1
B-GNN-reduced        : new graph-only model with reduced graph features
C-Hybrid-reduced     : new Hybrid with reduced graph features + unchanged 16-D history
D-Aggregate-full     : direct in-match aggregate control, full graph features, no message passing
D-Aggregate-reduced  : direct in-match aggregate control, reduced graph features, no message passing

PREDECLARED COMPARISONS
-----------------------
1. C-Hybrid-reduced vs A-MLP
   Does complementary graph/history performance remain after outcome-adjacent graph features are removed?

2. C-Hybrid-reduced vs D-Aggregate-reduced
   Does the reduced Hybrid outperform a direct aggregate in-match baseline using the same reduced match-derived variables?

3. C-Hybrid-full vs C-Hybrid-reduced
   How much does removing shots/xG/key-pass information change Hybrid performance?

4. B-GNN-full vs B-GNN-reduced
   How much does removing those variables change graph-only performance?

5. D-Aggregate-full vs D-Aggregate-reduced
   How strongly does the simple direct baseline depend on those attacking-output variables?

STATISTICS
----------
Paired match-level bootstrap: {BOOTSTRAP_SAMPLES} resamples
Exact McNemar tests: yes
Holm correction across the five predeclared comparisons: yes

REPORTING RULE
--------------
A reduced model does not need to equal the full model to demonstrate robustness. The relevant question is whether useful
performance and/or complementarity remains once obvious attacking-output clues are removed. Confidence intervals and corrected
p-values must be reported; negative or null findings are retained.
"""
    (OUTPUT_DIR / "README_EXPERIMENT4.txt").write_text(readme.strip() + "\n", encoding="utf-8")

    # Package checksum manifest.
    excluded_parts = {"__pycache__", ".pytest_cache", ".git"}
    checksum_rows = []
    for artifact_file in sorted(PROJECT_ROOT.rglob("*")):
        if not artifact_file.is_file() or artifact_file.suffix == ".zip":
            continue
        if any(part in excluded_parts for part in artifact_file.parts):
            continue
        checksum_rows.append({
            "path": str(artifact_file.relative_to(PROJECT_ROOT)),
            "bytes": int(artifact_file.stat().st_size),
            "sha256": sha256_file(artifact_file),
        })
    pd.DataFrame(checksum_rows).to_csv(OUTPUT_DIR / "SHA256SUMS.csv", index=False)

    if FINAL_ZIP.exists():
        FINAL_ZIP.unlink()
    with zipfile.ZipFile(FINAL_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for artifact_file in sorted(PROJECT_ROOT.rglob("*")):
            if not artifact_file.is_file() or artifact_file.suffix == ".zip":
                continue
            if any(part in excluded_parts for part in artifact_file.parts):
                continue
            archive.write(
                artifact_file,
                Path("football_gnn_experiment4_reduced") / artifact_file.relative_to(PROJECT_ROOT),
            )

    print("\n" + "=" * 100)
    print("EXPERIMENT 4 COMPLETED SUCCESSFULLY")
    print("=" * 100)
    print("Final archive:", FINAL_ZIP)
    print("Archive size (MB):", round(FINAL_ZIP.stat().st_size / (1024 * 1024), 2))


if __name__ == "__main__":
    main()
