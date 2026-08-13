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
import yaml
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from src.evaluation import build_results_table, prediction_rows
from src.final_protocol import harvest_dataset_with_retry_cache
from src.models import StructureGNN, HybridGNN
from src.reproducibility import save_environment, set_global_seed
from src.scaling import fit_graph_scalers, make_feature_view, transform_graphs
from src.training import fit_graph_model, predict_graph_model, select_graph_epoch

PROJECT_ROOT = Path.cwd()
REFERENCE_DIR = PROJECT_ROOT / "reference_experiment1"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiment3_pooling_ablation"
FINAL_ZIP = Path("/content/football_gnn_experiment3_pooling_ablation_final.zip")

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 2026
EXPECTED_ELIGIBLE_N = 757
EXPECTED_OUTER_TEST_N = 454


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def full_graph_views(graphs: list, indices: np.ndarray):
    return [
        make_feature_view(
            graphs[int(index)],
            node_indices=[0, 1, 2, 3],
            edge_indices=[0, 1, 2],
        )
        for index in indices
    ]


def preprocess_graph_pair(train_raw, other_raw, train_history=None, other_history=None):
    scalers = fit_graph_scalers(
        graphs=train_raw,
        standardize_nodes=True,
        edge_continuous_indices=[0, 1],
    )
    train = transform_graphs(train_raw, scalers, history_features=train_history)
    other = transform_graphs(other_raw, scalers, history_features=other_history)
    return train, other


def paired_bootstrap(truth, positional_pred, global_pred, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    n = len(truth)
    acc_diff = np.empty(samples, dtype=float)
    f1_diff = np.empty(samples, dtype=float)
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        yb = truth[idx]
        pos = positional_pred[idx]
        glob = global_pred[idx]
        acc_diff[i] = accuracy_score(yb, pos) - accuracy_score(yb, glob)
        f1_diff[i] = (
            f1_score(yb, pos, average="macro", zero_division=0)
            - f1_score(yb, glob, average="macro", zero_division=0)
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

    if config["validation"]["strategy"] != "rolling_origin_40pct_initial_train":
        raise RuntimeError("Experiment 3 requires the frozen Experiment 1 rolling-origin protocol")
    if config["model"]["pooling"] != "positional_add":
        raise RuntimeError("Frozen Experiment 1 reference must use positional_add pooling")

    exp_config = {
        "experiment": "Experiment 3 - positional pooling vs global mean pooling",
        "scientific_question": "Does role-aware positional pooling outperform global mean pooling under the frozen Elite protocol?",
        "primary_comparison": "C-Hybrid-positional vs C-Hybrid-globalmean",
        "supporting_comparison": "B-GNN-positional vs B-GNN-globalmean",
        "frozen_protocol": {
            "eligible_n": EXPECTED_ELIGIBLE_N,
            "paired_outer_test_n": EXPECTED_OUTER_TEST_N,
            "outer_protocol": "40% initial train rolling origin with 5 future test folds",
            "team_embargo_matches": 3,
            "same_folds": True,
            "same_training_seeds": True,
            "same_features": True,
            "same_graph_topology": True,
            "same_optimizer_and_training": True,
        },
        "manipulated_component": "graph-level pooling only",
        "positional_pooling": "role-specific global-add pooling for FW/MF/DF/GK followed by concatenation",
        "control_pooling": "global mean pooling across all 11 player nodes",
        "capacity_note": "This is a pooling ablation, not a parameter-matched capacity control; parameter counts are reported explicitly.",
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    (OUTPUT_DIR / "experiment3_config.json").write_text(json.dumps(exp_config, indent=2), encoding="utf-8")
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

    # Reuse the exact frozen positional predictions from Experiment 1.
    b_pos = frozen_predictions[frozen_predictions["model"].eq("B-GNN-full")].copy()
    c_pos = frozen_predictions[frozen_predictions["model"].eq("C-Hybrid-full")].copy()
    if len(b_pos) != EXPECTED_OUTER_TEST_N or len(c_pos) != EXPECTED_OUTER_TEST_N:
        raise RuntimeError("Frozen positional prediction counts do not match Experiment 1")
    if set(b_pos["match_id"].astype(int)) != set(c_pos["match_id"].astype(int)):
        raise RuntimeError("Frozen B-GNN and C-Hybrid test observations are not paired")

    for frame, model_name in [(b_pos, "B-GNN-positional"), (c_pos, "C-Hybrid-positional")]:
        frame["protocol"] = "experiment3_pooling_ablation"
        frame["model"] = model_name

    print("=" * 100)
    print("EXPERIMENT 3 — POSITIONAL POOLING VS GLOBAL MEAN POOLING")
    print("=" * 100)
    print("Frozen eligible cohort N:", len(samples))
    print("Frozen paired outer-test N:", EXPECTED_OUTER_TEST_N)
    print("Primary comparison: C-Hybrid-positional vs C-Hybrid-globalmean")
    print("Supporting comparison: B-GNN-positional vs B-GNN-globalmean")
    print("Only the graph-level pooling operator changes.")

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
    if frozen_test_ids != set(b_pos["match_id"].astype(int)):
        raise RuntimeError("Frozen positional predictions do not match outer-test fold assignments")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Graph reconstruction and frozen-fold verification passed.")
    print("Device:", device)

    prediction_frames = [b_pos, c_pos]
    selected_rows = []
    runtime_rows = []
    gate_rows = []

    # Parameter-count audit.
    param_rows = [
        {
            "model": "B-GNN-positional",
            "trainable_parameters": count_parameters(StructureGNN(4, 3, config, pooling="positional_add")),
        },
        {
            "model": "B-GNN-globalmean",
            "trainable_parameters": count_parameters(StructureGNN(4, 3, config, pooling="global_mean")),
        },
        {
            "model": "C-Hybrid-positional",
            "trainable_parameters": count_parameters(HybridGNN(4, 3, 16, config, pooling="positional_add")),
        },
        {
            "model": "C-Hybrid-globalmean",
            "trainable_parameters": count_parameters(HybridGNN(4, 3, 16, config, pooling="global_mean")),
        },
    ]
    pd.DataFrame(param_rows).to_csv(OUTPUT_DIR / "pooling_parameter_counts.csv", index=False)

    print("\nSTEP 2 — TRAINING GLOBAL-MEAN POOLING CONTROLS ON THE EXACT FROZEN FOLDS")
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
        if expected_fold_ids != set(b_pos.loc[b_pos["fold"].eq(fold), "match_id"].astype(int)):
            raise RuntimeError(f"Fold {fold} test IDs changed from Experiment 1")

        print("\n" + "-" * 100)
        print(
            f"FOLD {fold}: outer train={len(outer_train)}, test={len(outer_test)}, "
            f"inner train={len(inner_train)}, inner val={len(inner_val)}, seed={training_seed}"
        )

        # Inner preprocessing: fit history scaler and graph scaler on inner train only.
        inner_history_scaler = StandardScaler().fit(X_history_raw[inner_train])
        X_inner_train = inner_history_scaler.transform(X_history_raw[inner_train])
        X_inner_val = inner_history_scaler.transform(X_history_raw[inner_val])

        inner_train_raw = full_graph_views(graphs, inner_train)
        inner_val_raw = full_graph_views(graphs, inner_val)
        inner_train_scaled, inner_val_scaled = preprocess_graph_pair(
            inner_train_raw,
            inner_val_raw,
            train_history=X_inner_train,
            other_history=X_inner_val,
        )

        b_factory = lambda: StructureGNN(4, 3, config, pooling="global_mean")
        c_factory = lambda: HybridGNN(4, 3, 16, config, pooling="global_mean")

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

        # Outer preprocessing: refit scalers on complete outer training data only.
        outer_history_scaler = StandardScaler().fit(X_history_raw[outer_train])
        X_outer_train = outer_history_scaler.transform(X_history_raw[outer_train])
        X_outer_test = outer_history_scaler.transform(X_history_raw[outer_test])

        outer_train_raw = full_graph_views(graphs, outer_train)
        outer_test_raw = full_graph_views(graphs, outer_test)
        outer_train_scaled, outer_test_scaled = preprocess_graph_pair(
            outer_train_raw,
            outer_test_raw,
            train_history=X_outer_train,
            other_history=X_outer_test,
        )

        set_global_seed(training_seed)
        start = time.perf_counter()
        b_model = fit_graph_model(
            b_factory,
            outer_train_scaled,
            epochs=b_selection.best_epoch,
            config=config,
            seed=training_seed,
            device=device,
        )
        b_fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        b_pred, b_score = predict_graph_model(b_model, outer_test_scaled, config, device)
        b_predict_seconds = time.perf_counter() - start

        set_global_seed(training_seed)
        start = time.perf_counter()
        c_model = fit_graph_model(
            c_factory,
            outer_train_scaled,
            epochs=c_selection.best_epoch,
            config=config,
            seed=training_seed,
            device=device,
        )
        c_fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        c_pred, c_score = predict_graph_model(c_model, outer_test_scaled, config, device)
        c_predict_seconds = time.perf_counter() - start
        c_gate = c_model.gate_value()

        b_rows = pd.DataFrame(prediction_rows(
            samples=samples,
            test_indices=outer_test,
            truth=y[outer_test],
            prediction=b_pred,
            score=b_score,
            cohort="elite",
            protocol="experiment3_pooling_ablation",
            model="B-GNN-globalmean",
            fold=fold,
            seed=training_seed,
            gate_value=None,
        ))
        c_rows = pd.DataFrame(prediction_rows(
            samples=samples,
            test_indices=outer_test,
            truth=y[outer_test],
            prediction=c_pred,
            score=c_score,
            cohort="elite",
            protocol="experiment3_pooling_ablation",
            model="C-Hybrid-globalmean",
            fold=fold,
            seed=training_seed,
            gate_value=c_gate,
        ))
        prediction_frames.extend([b_rows, c_rows])

        selected_rows.extend([
            {
                "fold": fold,
                "model": "B-GNN-globalmean",
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
                "model": "C-Hybrid-globalmean",
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
                "model": "B-GNN-globalmean",
                "selection_seconds": b_select_seconds,
                "fit_seconds": b_fit_seconds,
                "predict_seconds": b_predict_seconds,
                "total_model_seconds": b_select_seconds + b_fit_seconds + b_predict_seconds,
            },
            {
                "fold": fold,
                "model": "C-Hybrid-globalmean",
                "selection_seconds": c_select_seconds,
                "fit_seconds": c_fit_seconds,
                "predict_seconds": c_predict_seconds,
                "total_model_seconds": c_select_seconds + c_fit_seconds + c_predict_seconds,
            },
        ])
        gate_rows.append({"fold": fold, "seed": training_seed, "globalmean_gate_value": c_gate})
        print(
            f"Selected epochs: B-global={b_selection.best_epoch}, C-global={c_selection.best_epoch}; "
            f"C-global lambda={c_gate:.6f}"
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    model_names = {
        "B-GNN-positional",
        "B-GNN-globalmean",
        "C-Hybrid-positional",
        "C-Hybrid-globalmean",
    }
    if set(predictions["model"].unique()) != model_names:
        raise RuntimeError(f"Unexpected model set: {set(predictions['model'].unique())}")
    for model_name, group in predictions.groupby("model"):
        if len(group) != EXPECTED_OUTER_TEST_N:
            raise RuntimeError(f"{model_name} prediction N={len(group)}, expected {EXPECTED_OUTER_TEST_N}")
        if set(group["match_id"].astype(int)) != frozen_test_ids:
            raise RuntimeError(f"{model_name} does not use the exact frozen outer-test observations")

    predictions.to_csv(OUTPUT_DIR / "pooling_predictions.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(OUTPUT_DIR / "pooling_selected_epochs.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(OUTPUT_DIR / "runtime_summary.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(OUTPUT_DIR / "globalmean_gate_by_fold.csv", index=False)

    results = build_results_table(predictions)
    results.to_csv(OUTPUT_DIR / "pooling_results.csv", index=False)

    print("\n" + "=" * 100)
    print("POOLED POOLING-ABLATION RESULTS")
    print("=" * 100)
    pooled = results[results["scope"].eq("pooled")].copy()
    print(pooled[[
        "model", "n", "accuracy", "balanced_accuracy", "macro_f1", "win_precision", "win_recall",
        "roc_auc", "tn", "fp", "fn", "tp"
    ]].to_string(index=False))

    # Paired inference for the two predeclared comparisons.
    comparisons = [
        ("B-GNN", "B-GNN-positional", "B-GNN-globalmean", BOOTSTRAP_SEED + 1),
        ("C-Hybrid", "C-Hybrid-positional", "C-Hybrid-globalmean", BOOTSTRAP_SEED + 2),
    ]
    pairwise_rows = []
    raw_p = []

    for family, pos_name, global_name, bootstrap_seed in comparisons:
        pos = predictions[predictions["model"].eq(pos_name)].sort_values(["fold", "match_id"]).reset_index(drop=True)
        glob = predictions[predictions["model"].eq(global_name)].sort_values(["fold", "match_id"]).reset_index(drop=True)
        if not np.array_equal(pos["match_id"].astype(int).to_numpy(), glob["match_id"].astype(int).to_numpy()):
            raise RuntimeError(f"Pairing failed for {family}")
        truth = pos["true_label"].astype(int).to_numpy()
        pos_pred = pos["predicted_class"].astype(int).to_numpy()
        glob_pred = glob["predicted_class"].astype(int).to_numpy()
        pos_acc = accuracy_score(truth, pos_pred)
        glob_acc = accuracy_score(truth, glob_pred)
        pos_f1 = f1_score(truth, pos_pred, average="macro", zero_division=0)
        glob_f1 = f1_score(truth, glob_pred, average="macro", zero_division=0)
        ci = paired_bootstrap(truth, pos_pred, glob_pred, samples=BOOTSTRAP_SAMPLES, seed=bootstrap_seed)

        pos_correct = pos_pred == truth
        glob_correct = glob_pred == truth
        pos_correct_global_wrong = int(np.sum(pos_correct & ~glob_correct))
        global_correct_pos_wrong = int(np.sum(glob_correct & ~pos_correct))
        discordant = pos_correct_global_wrong + global_correct_pos_wrong
        if discordant:
            p_value = float(binomtest(
                min(pos_correct_global_wrong, global_correct_pos_wrong),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue)
        else:
            p_value = 1.0
        raw_p.append(p_value)

        pairwise_rows.append({
            "comparison_family": family,
            "comparison": f"{pos_name}_minus_{global_name}",
            "paired_n": len(truth),
            "positional_accuracy": pos_acc,
            "globalmean_accuracy": glob_acc,
            "positional_minus_globalmean_accuracy": pos_acc - glob_acc,
            "positional_minus_globalmean_accuracy_pp": (pos_acc - glob_acc) * 100,
            "accuracy_ci_lower": ci["accuracy_ci_lower"],
            "accuracy_ci_upper": ci["accuracy_ci_upper"],
            "positional_macro_f1": pos_f1,
            "globalmean_macro_f1": glob_f1,
            "positional_minus_globalmean_macro_f1": pos_f1 - glob_f1,
            "positional_minus_globalmean_macro_f1_pp": (pos_f1 - glob_f1) * 100,
            "macro_f1_ci_lower": ci["macro_f1_ci_lower"],
            "macro_f1_ci_upper": ci["macro_f1_ci_upper"],
            "positional_correct_global_wrong": pos_correct_global_wrong,
            "global_correct_positional_wrong": global_correct_pos_wrong,
            "mcnemar_exact_p_raw": p_value,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        })

    adjusted = holm_adjust(raw_p)
    for row, adjusted_p in zip(pairwise_rows, adjusted):
        row["mcnemar_exact_p_holm"] = adjusted_p
    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_df.to_csv(OUTPUT_DIR / "pooling_pairwise_inference.csv", index=False)

    primary = pairwise_df[pairwise_df["comparison_family"].eq("C-Hybrid")].copy()
    primary["interpretation_rule"] = (
        "Positive difference favors positional pooling; negative difference favors global mean pooling. "
        "Do not claim statistical superiority if the 95% CI includes zero or Holm-adjusted McNemar p >= 0.05."
    )
    primary.to_csv(OUTPUT_DIR / "pooling_primary_summary.csv", index=False)

    print("\n" + "=" * 100)
    print("PAIRWISE POOLING INFERENCE")
    print("=" * 100)
    print(pairwise_df.to_string(index=False))

    metadata = {
        "artifact": "Experiment 3 - Positional Pooling vs Global Mean Pooling",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "eligible_cohort_n": EXPECTED_ELIGIBLE_N,
        "paired_outer_test_n": EXPECTED_OUTER_TEST_N,
        "models": sorted(model_names),
        "primary_comparison": "C-Hybrid-positional vs C-Hybrid-globalmean",
        "supporting_comparison": "B-GNN-positional vs B-GNN-globalmean",
        "frozen_folds": True,
        "frozen_training_seeds": True,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "capacity_matched": False,
        "capacity_note": "Pooling changes representation dimensionality; parameter counts are reported rather than concealed.",
    }
    (OUTPUT_DIR / "reproducibility_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    readme = f"""
EXPERIMENT 3 — POSITIONAL POOLING VS GLOBAL MEAN POOLING
========================================================

QUESTION
--------
Does role-aware positional pooling improve outcome-linked classification relative to global mean pooling?

FOOTBALL INTERPRETATION
-----------------------
Positional pooling keeps separate summaries for goalkeeper, defenders, midfielders, and forwards before classification.
Global mean pooling mixes all eleven player embeddings into one team-average representation.

The ablation therefore tests whether preserving broad tactical-unit identity (GK/DF/MF/FW) is useful beyond simply averaging the team.

FROZEN FROM EXPERIMENT 1
------------------------
Eligible Elite cohort: {EXPECTED_ELIGIBLE_N}
Paired outer-test observations: {EXPECTED_OUTER_TEST_N}
Same rolling-origin folds: yes
Same three-match team embargo: yes
Same inner folds: yes
Same training seeds: yes
Same graph topology: yes
Same node and edge features: yes
Same optimizer/training settings: yes

CONDITIONS
----------
B-GNN-positional       : frozen Experiment 1 B-GNN predictions
B-GNN-globalmean       : identical B-GNN except global mean pooling
C-Hybrid-positional    : frozen Experiment 1 Hybrid predictions
C-Hybrid-globalmean    : identical Hybrid except global mean pooling

PRIMARY COMPARISON
------------------
C-Hybrid-positional vs C-Hybrid-globalmean

SUPPORTING COMPARISON
---------------------
B-GNN-positional vs B-GNN-globalmean

STATISTICS
----------
Paired outer-test evaluation
10,000-sample paired match-level bootstrap
Exact McNemar test
Holm correction across the two predeclared pooling comparisons

IMPORTANT CAPACITY NOTE
-----------------------
This is a pooling ablation, not a parameter-matched capacity control. Positional concatenation produces a larger pooled vector than global mean pooling. The exact trainable parameter counts are saved in pooling_parameter_counts.csv and must be reported transparently if this result is discussed.

PUBLICATION RULE
----------------
If positional pooling does not outperform global mean pooling, report the negative ablation and narrow the architectural claim. Do not retune the primary experiment after seeing this result.
""".strip() + "\n"
    (OUTPUT_DIR / "README_EXPERIMENT3.txt").write_text(readme, encoding="utf-8")

    # SHA-256 manifest over the experiment package (excluding generated zip/cache folders).
    excluded_parts = {"__pycache__", ".pytest_cache", ".git", "data"}
    checksum_rows = []
    for artifact in sorted(PROJECT_ROOT.rglob("*")):
        if not artifact.is_file() or artifact.suffix == ".zip":
            continue
        if any(part in excluded_parts for part in artifact.parts):
            continue
        checksum_rows.append({
            "path": str(artifact.relative_to(PROJECT_ROOT)),
            "bytes": int(artifact.stat().st_size),
            "sha256": sha256_file(artifact),
        })
    pd.DataFrame(checksum_rows).to_csv(OUTPUT_DIR / "SHA256SUMS.csv", index=False)

    if FINAL_ZIP.exists():
        FINAL_ZIP.unlink()
    with zipfile.ZipFile(FINAL_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for artifact in sorted(PROJECT_ROOT.rglob("*")):
            if not artifact.is_file() or artifact.suffix == ".zip":
                continue
            if any(part in excluded_parts for part in artifact.parts):
                continue
            archive.write(
                artifact,
                Path("football_gnn_experiment3_pooling") / artifact.relative_to(PROJECT_ROOT),
            )

    print("\n" + "=" * 100)
    print("EXPERIMENT 3 COMPLETED SUCCESSFULLY")
    print("=" * 100)
    print("Final archive:", FINAL_ZIP)
    print("Archive size (MB):", round(FINAL_ZIP.stat().st_size / (1024 * 1024), 2))


if __name__ == "__main__":
    main()
