from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

from src.evaluation import build_results_table, prediction_rows
from src.final_protocol import harvest_dataset_with_retry_cache
from src.models import StructureGNN
from src.reproducibility import save_environment, set_global_seed
from src.scaling import fit_graph_scalers, make_feature_view, transform_graphs
from src.topology_permutation import permute_edge_destinations_degree_preserving
from src.training import fit_graph_model, predict_graph_model, select_graph_epoch

PROJECT_ROOT = Path.cwd()
REFERENCE_DIR = PROJECT_ROOT / "reference_experiment1"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiment2_topology_permutation"
FINAL_ZIP = Path("/content/football_gnn_experiment2_topology_permutation_final.zip")

PERMUTATION_SEEDS = [1, 2, 3]
SWAP_MULTIPLIER = 10
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


def preprocess_graph_pair(train_raw, other_raw):
    scalers = fit_graph_scalers(
        graphs=train_raw,
        standardize_nodes=True,
        edge_continuous_indices=[0, 1],
    )
    train = transform_graphs(train_raw, scalers, history_features=None)
    other = transform_graphs(other_raw, scalers, history_features=None)
    return train, other


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


def paired_bootstrap_two_models(
    truth: np.ndarray,
    observed_pred: np.ndarray,
    permuted_pred: np.ndarray,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
):
    rng = np.random.default_rng(seed)
    n = len(truth)
    accuracy_differences = np.empty(samples, dtype=float)
    macro_f1_differences = np.empty(samples, dtype=float)
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        yb = truth[idx]
        obs = observed_pred[idx]
        perm = permuted_pred[idx]
        accuracy_differences[i] = accuracy_score(yb, obs) - accuracy_score(yb, perm)
        macro_f1_differences[i] = (
            f1_score(yb, obs, average="macro", zero_division=0)
            - f1_score(yb, perm, average="macro", zero_division=0)
        )
    return {
        "accuracy_ci_lower": float(np.percentile(accuracy_differences, 2.5)),
        "accuracy_ci_upper": float(np.percentile(accuracy_differences, 97.5)),
        "macro_f1_ci_lower": float(np.percentile(macro_f1_differences, 2.5)),
        "macro_f1_ci_upper": float(np.percentile(macro_f1_differences, 97.5)),
    }


def paired_bootstrap_observed_vs_mean_permuted(
    truth: np.ndarray,
    observed_pred: np.ndarray,
    permuted_predictions: list[np.ndarray],
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED + 17,
):
    rng = np.random.default_rng(seed)
    n = len(truth)
    accuracy_differences = np.empty(samples, dtype=float)
    macro_f1_differences = np.empty(samples, dtype=float)
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        yb = truth[idx]
        obs = observed_pred[idx]
        obs_acc = accuracy_score(yb, obs)
        obs_f1 = f1_score(yb, obs, average="macro", zero_division=0)
        perm_acc = np.mean([accuracy_score(yb, pred[idx]) for pred in permuted_predictions])
        perm_f1 = np.mean([
            f1_score(yb, pred[idx], average="macro", zero_division=0)
            for pred in permuted_predictions
        ])
        accuracy_differences[i] = obs_acc - perm_acc
        macro_f1_differences[i] = obs_f1 - perm_f1
    return {
        "accuracy_ci_lower": float(np.percentile(accuracy_differences, 2.5)),
        "accuracy_ci_upper": float(np.percentile(accuracy_differences, 97.5)),
        "macro_f1_ci_lower": float(np.percentile(macro_f1_differences, 2.5)),
        "macro_f1_ci_upper": float(np.percentile(macro_f1_differences, 97.5)),
    }


def require_reference_files():
    required = [
        "included_samples.csv",
        "fold_assignments.csv",
        "fold_predictions_final.csv",
        "selected_epochs.csv",
        "results_final.csv",
        "candidate_match_manifest.csv",
        "cohort_manifest.csv",
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

    # Freeze exactly the Experiment 1 graph/model/training settings.
    if config["validation"]["strategy"] != "rolling_origin_40pct_initial_train":
        raise RuntimeError("The supplied config is not the frozen Experiment 1 rolling-origin protocol")
    if int(config["validation"]["outer_folds"]) != 5:
        raise RuntimeError("Experiment 2 requires the five frozen Experiment 1 folds")

    experiment_config = {
        "experiment": "Experiment 2 - observed topology vs destination-permuted topology",
        "scientific_question": "Does observed passer-recipient topology outperform a matched topology-disrupted graph representation?",
        "observed_reference": "Frozen B-GNN-full predictions from Experiment 1",
        "permutation_seeds": PERMUTATION_SEEDS,
        "swap_multiplier": SWAP_MULTIPLIER,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "training_seed_policy": "Reuse exact fold-specific Experiment 1 seed",
        "topology_control": {
            "changes": ["edge destinations only"],
            "preserves": [
                "node features",
                "edge count",
                "edge source vector",
                "source out-degree",
                "destination multiset",
                "destination in-degree",
                "edge feature rows",
                "role IDs",
                "graph labels",
                "no synthetic event-level self-passes",
            ],
            "label_independent": True,
            "fold_independent": True,
        },
    }
    (OUTPUT_DIR / "experiment2_config.json").write_text(
        json.dumps(experiment_config, indent=2), encoding="utf-8"
    )
    save_environment(OUTPUT_DIR / "environment.json")
    with (OUTPUT_DIR / "environment_freeze.txt").open("w", encoding="utf-8") as handle:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=handle, check=True)

    frozen_samples = pd.read_csv(REFERENCE_DIR / "included_samples.csv").reset_index(drop=True)
    frozen_assignments = pd.read_csv(REFERENCE_DIR / "fold_assignments.csv")
    frozen_predictions = pd.read_csv(REFERENCE_DIR / "fold_predictions_final.csv")
    frozen_selected_epochs = pd.read_csv(REFERENCE_DIR / "selected_epochs.csv")
    frozen_candidate_manifest = pd.read_csv(REFERENCE_DIR / "candidate_match_manifest.csv")

    if len(frozen_samples) != EXPECTED_ELIGIBLE_N:
        raise RuntimeError(
            f"Frozen Experiment 1 eligible N changed: expected {EXPECTED_ELIGIBLE_N}, got {len(frozen_samples)}"
        )

    observed = frozen_predictions[frozen_predictions["model"].eq("B-GNN-full")].copy()
    if len(observed) != EXPECTED_OUTER_TEST_N:
        raise RuntimeError(
            f"Frozen observed B-GNN outer-test N changed: expected {EXPECTED_OUTER_TEST_N}, got {len(observed)}"
        )
    if observed["match_id"].duplicated().any():
        raise RuntimeError("Frozen observed B-GNN predictions contain duplicate match IDs")

    observed["protocol"] = "experiment2_topology_permutation"
    observed["model"] = "B-GNN-observed"
    observed["permutation_seed"] = np.nan

    print("=" * 100)
    print("EXPERIMENT 2 — OBSERVED TOPOLOGY VS DESTINATION-PERMUTED TOPOLOGY")
    print("=" * 100)
    print("Frozen eligible cohort N:", len(frozen_samples))
    print("Frozen paired outer-test N:", len(observed))
    print("Permutation seeds:", PERMUTATION_SEEDS)
    print("Only edge destinations are changed; node/edge features and degree marginals are preserved.")

    print("\nSTEP 1 — RE-HARVESTING STATSBOMB EVENTS TO RECONSTRUCT THE FROZEN GRAPHS")
    _, graph_map, new_manifest, new_candidate_manifest = harvest_dataset_with_retry_cache(
        config,
        attempts=5,
        fail_on_event_error=True,
    )

    frozen_candidate_ids = set(frozen_candidate_manifest["match_id"].astype(int))
    new_candidate_ids = set(new_candidate_manifest["match_id"].astype(int))
    if frozen_candidate_ids != new_candidate_ids:
        raise RuntimeError(
            "StatsBomb candidate-match set differs from frozen Experiment 1. Stop rather than silently changing the experiment."
        )

    frozen_ids = frozen_samples["match_id"].astype(int).tolist()
    missing_graphs = [match_id for match_id in frozen_ids if match_id not in graph_map]
    if missing_graphs:
        raise RuntimeError(
            f"Could not reconstruct graphs for {len(missing_graphs)} frozen Experiment 1 matches: {missing_graphs[:10]}"
        )

    # Verify current authoritative metadata remains aligned with frozen Experiment 1.
    manifest_lookup = new_manifest.set_index("match_id")
    for row in frozen_samples.itertuples(index=False):
        current = manifest_lookup.loc[int(row.match_id)]
        if str(current["home_team"]) != str(row.home_team) or str(current["away_team"]) != str(row.away_team):
            raise RuntimeError(f"Team metadata changed for match_id={row.match_id}")
        graph_label = int(graph_map[int(row.match_id)].y.view(-1)[0].item())
        if graph_label != int(row.label):
            raise RuntimeError(f"Label changed for match_id={row.match_id}")

    samples = frozen_samples.copy().reset_index(drop=True)
    y = samples["label"].astype(int).to_numpy()
    graphs = [graph_map[int(match_id)] for match_id in samples["match_id"]]
    id_to_index = {int(match_id): idx for idx, match_id in enumerate(samples["match_id"].astype(int))}

    # Copy frozen protocol files directly into Experiment 2 outputs.
    samples.to_csv(OUTPUT_DIR / "frozen_included_samples.csv", index=False)
    frozen_assignments.to_csv(OUTPUT_DIR / "frozen_fold_assignments.csv", index=False)
    frozen_selected_epochs.to_csv(OUTPUT_DIR / "frozen_experiment1_selected_epochs.csv", index=False)
    new_manifest.to_csv(OUTPUT_DIR / "reharvest_manifest.csv", index=False)

    def indices_for(fold: int, level: str, role: str) -> np.ndarray:
        ids = frozen_assignments.loc[
            frozen_assignments["fold"].eq(fold)
            & frozen_assignments["level"].eq(level)
            & frozen_assignments["role"].eq(role),
            "match_id",
        ].astype(int).tolist()
        if not ids:
            raise RuntimeError(f"No frozen assignment rows for fold={fold}, level={level}, role={role}")
        missing = [match_id for match_id in ids if match_id not in id_to_index]
        if missing:
            raise RuntimeError(f"Frozen fold contains match IDs missing from frozen cohort: {missing[:10]}")
        return np.asarray([id_to_index[match_id] for match_id in ids], dtype=int)

    # Verify the exact frozen outer-test coverage.
    union_test_ids = set(
        frozen_assignments.loc[
            frozen_assignments["level"].eq("outer") & frozen_assignments["role"].eq("test"),
            "match_id",
        ].astype(int)
    )
    if union_test_ids != set(observed["match_id"].astype(int)):
        raise RuntimeError("Frozen B-GNN predictions and frozen outer-test assignments do not match exactly")

    print("Graph reconstruction and frozen-fold verification passed.")

    all_prediction_frames = [observed]
    audit_rows = []
    selected_epoch_rows = []
    runtime_rows = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    for permutation_seed in PERMUTATION_SEEDS:
        print("\n" + "=" * 100)
        print(f"TOPOLOGY PERMUTATION SEED {permutation_seed}")
        print("=" * 100)

        permuted_graphs = []
        seed_audits = []
        for idx, graph in enumerate(graphs):
            match_id = int(samples.iloc[idx]["match_id"])
            permuted, audit = permute_edge_destinations_degree_preserving(
                graph,
                match_id=match_id,
                permutation_seed=permutation_seed,
                swap_multiplier=SWAP_MULTIPLIER,
            )
            permuted_graphs.append(permuted)
            seed_audits.append(asdict(audit))
        audit_rows.extend(seed_audits)

        seed_audit_df = pd.DataFrame(seed_audits)
        print(
            "Topology changed fraction: mean=",
            f"{seed_audit_df['changed_fraction'].mean():.4f}",
            "min=",
            f"{seed_audit_df['changed_fraction'].min():.4f}",
            "max=",
            f"{seed_audit_df['changed_fraction'].max():.4f}",
        )
        if int(seed_audit_df["permuted_self_loops"].sum()) != 0:
            raise RuntimeError("Permutation audit found synthetic event-level self-loops")
        if not seed_audit_df[
            [
                "source_vector_preserved",
                "destination_multiset_preserved",
                "node_features_preserved",
                "edge_attributes_preserved",
                "roles_preserved",
                "label_preserved",
            ]
        ].all().all():
            raise RuntimeError("Matched topology-control preservation audit failed")

        seed_prediction_rows = []

        for fold in range(1, 6):
            outer_train = indices_for(fold, "outer", "train")
            outer_test = indices_for(fold, "outer", "test")
            inner_train = indices_for(fold, "inner", "train")
            inner_val = indices_for(fold, "inner", "validation")

            seed_row = frozen_selected_epochs.loc[frozen_selected_epochs["fold"].eq(fold)]
            if len(seed_row) != 1:
                raise RuntimeError(f"Could not uniquely resolve frozen training seed for fold {fold}")
            training_seed = int(seed_row.iloc[0]["seed"])

            # Assert exact observed test IDs for this fold.
            observed_fold_ids = set(
                observed.loc[observed["fold"].eq(fold), "match_id"].astype(int)
            )
            expected_fold_ids = set(samples.iloc[outer_test]["match_id"].astype(int))
            if observed_fold_ids != expected_fold_ids:
                raise RuntimeError(f"Fold {fold} test IDs differ from frozen observed B-GNN predictions")

            print(
                f"Fold {fold}: outer train={len(outer_train)}, test={len(outer_test)}, "
                f"inner train={len(inner_train)}, inner val={len(inner_val)}, training seed={training_seed}"
            )

            structure_factory = lambda: StructureGNN(4, 3, config)

            # Inner epoch selection on permuted topology using the exact frozen split.
            inner_train_raw = full_graph_views(permuted_graphs, inner_train)
            inner_val_raw = full_graph_views(permuted_graphs, inner_val)
            inner_train_scaled, inner_val_scaled = preprocess_graph_pair(inner_train_raw, inner_val_raw)

            start = time.perf_counter()
            selection = select_graph_epoch(
                structure_factory,
                inner_train_scaled,
                inner_val_scaled,
                y[inner_val],
                config,
                training_seed,
                device,
            )
            selection_seconds = time.perf_counter() - start

            # Refit preprocessing on the full frozen outer-training set.
            outer_train_raw = full_graph_views(permuted_graphs, outer_train)
            outer_test_raw = full_graph_views(permuted_graphs, outer_test)
            outer_train_scaled, outer_test_scaled = preprocess_graph_pair(outer_train_raw, outer_test_raw)

            start = time.perf_counter()
            model = fit_graph_model(
                structure_factory,
                outer_train_scaled,
                epochs=selection.best_epoch,
                config=config,
                seed=training_seed,
                device=device,
            )
            fit_seconds = time.perf_counter() - start

            start = time.perf_counter()
            pred, score = predict_graph_model(model, outer_test_scaled, config, device)
            predict_seconds = time.perf_counter() - start

            seed_prediction_rows.extend(
                prediction_rows(
                    samples=samples,
                    test_indices=outer_test,
                    truth=y[outer_test],
                    prediction=pred,
                    score=score,
                    cohort="elite",
                    protocol="experiment2_topology_permutation",
                    model=f"B-GNN-permuted-p{permutation_seed}",
                    fold=fold,
                    seed=training_seed,
                    gate_value=None,
                )
            )

            selected_epoch_rows.append({
                "permutation_seed": permutation_seed,
                "fold": fold,
                "training_seed": training_seed,
                "best_epoch": int(selection.best_epoch),
                "inner_validation_macro_f1": float(selection.best_metric),
                "inner_train_n": int(len(inner_train)),
                "inner_validation_n": int(len(inner_val)),
                "outer_train_n": int(len(outer_train)),
                "outer_test_n": int(len(outer_test)),
            })
            runtime_rows.append({
                "permutation_seed": permutation_seed,
                "fold": fold,
                "selection_seconds": selection_seconds,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "total_model_seconds": selection_seconds + fit_seconds + predict_seconds,
            })
            print(
                f"  selected epoch={selection.best_epoch}, inner macro-F1={selection.best_metric:.4f}"
            )

        seed_predictions = pd.DataFrame(seed_prediction_rows)
        if len(seed_predictions) != EXPECTED_OUTER_TEST_N:
            raise RuntimeError(
                f"Permutation seed {permutation_seed} produced {len(seed_predictions)} predictions; expected {EXPECTED_OUTER_TEST_N}"
            )
        if set(seed_predictions["match_id"].astype(int)) != set(observed["match_id"].astype(int)):
            raise RuntimeError(f"Permutation seed {permutation_seed} is not paired to the exact observed test matches")
        seed_predictions["permutation_seed"] = permutation_seed
        all_prediction_frames.append(seed_predictions)

    predictions = pd.concat(all_prediction_frames, ignore_index=True)
    predictions.to_csv(OUTPUT_DIR / "topology_predictions.csv", index=False)

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(OUTPUT_DIR / "topology_permutation_audit.csv", index=False)
    audit_summary = audit_df.groupby("permutation_seed").agg(
        graphs=("match_id", "count"),
        mean_edges=("edge_count", "mean"),
        mean_changed_fraction=("changed_fraction", "mean"),
        min_changed_fraction=("changed_fraction", "min"),
        max_changed_fraction=("changed_fraction", "max"),
        total_original_self_loops=("original_self_loops", "sum"),
        total_permuted_self_loops=("permuted_self_loops", "sum"),
    ).reset_index()
    audit_summary.to_csv(OUTPUT_DIR / "topology_permutation_summary.csv", index=False)

    selected_epochs_df = pd.DataFrame(selected_epoch_rows)
    selected_epochs_df.to_csv(OUTPUT_DIR / "topology_selected_epochs.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(OUTPUT_DIR / "runtime_summary.csv", index=False)

    results = build_results_table(predictions)
    results.to_csv(OUTPUT_DIR / "topology_results.csv", index=False)

    print("\n" + "=" * 100)
    print("POOLED TOPOLOGY RESULTS")
    print("=" * 100)
    pooled = results[results["scope"].eq("pooled")].copy()
    print(
        pooled[
            [
                "model", "n", "accuracy", "balanced_accuracy", "macro_f1",
                "win_precision", "win_recall", "roc_auc", "tn", "fp", "fn", "tp"
            ]
        ].to_string(index=False)
    )

    observed_aligned = observed.sort_values(["fold", "match_id"]).reset_index(drop=True)
    truth = observed_aligned["true_label"].astype(int).to_numpy()
    observed_pred = observed_aligned["predicted_class"].astype(int).to_numpy()
    observed_accuracy = accuracy_score(truth, observed_pred)
    observed_f1 = f1_score(truth, observed_pred, average="macro", zero_division=0)

    pairwise_rows = []
    permutation_prediction_arrays = []
    raw_p_values = []

    for permutation_seed in PERMUTATION_SEEDS:
        perm = predictions[predictions["model"].eq(f"B-GNN-permuted-p{permutation_seed}")].copy()
        perm = perm.sort_values(["fold", "match_id"]).reset_index(drop=True)
        if not np.array_equal(
            observed_aligned["match_id"].astype(int).to_numpy(),
            perm["match_id"].astype(int).to_numpy(),
        ):
            raise RuntimeError(f"Prediction alignment failed for permutation seed {permutation_seed}")
        perm_pred = perm["predicted_class"].astype(int).to_numpy()
        permutation_prediction_arrays.append(perm_pred)
        perm_accuracy = accuracy_score(truth, perm_pred)
        perm_f1 = f1_score(truth, perm_pred, average="macro", zero_division=0)
        ci = paired_bootstrap_two_models(
            truth,
            observed_pred,
            perm_pred,
            samples=BOOTSTRAP_SAMPLES,
            seed=BOOTSTRAP_SEED + permutation_seed,
        )
        observed_correct = observed_pred == truth
        perm_correct = perm_pred == truth
        observed_correct_perm_wrong = int(np.sum(observed_correct & ~perm_correct))
        perm_correct_observed_wrong = int(np.sum(perm_correct & ~observed_correct))
        discordant = observed_correct_perm_wrong + perm_correct_observed_wrong
        if discordant:
            p_value = float(
                binomtest(
                    min(observed_correct_perm_wrong, perm_correct_observed_wrong),
                    n=discordant,
                    p=0.5,
                    alternative="two-sided",
                ).pvalue
            )
        else:
            p_value = 1.0
        raw_p_values.append(p_value)
        pairwise_rows.append({
            "permutation_seed": permutation_seed,
            "paired_n": len(truth),
            "observed_accuracy": observed_accuracy,
            "permuted_accuracy": perm_accuracy,
            "observed_minus_permuted_accuracy": observed_accuracy - perm_accuracy,
            "observed_minus_permuted_accuracy_pp": (observed_accuracy - perm_accuracy) * 100,
            "accuracy_ci_lower": ci["accuracy_ci_lower"],
            "accuracy_ci_upper": ci["accuracy_ci_upper"],
            "observed_macro_f1": observed_f1,
            "permuted_macro_f1": perm_f1,
            "observed_minus_permuted_macro_f1": observed_f1 - perm_f1,
            "observed_minus_permuted_macro_f1_pp": (observed_f1 - perm_f1) * 100,
            "macro_f1_ci_lower": ci["macro_f1_ci_lower"],
            "macro_f1_ci_upper": ci["macro_f1_ci_upper"],
            "observed_correct_permuted_wrong": observed_correct_perm_wrong,
            "permuted_correct_observed_wrong": perm_correct_observed_wrong,
            "mcnemar_exact_p_raw": p_value,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        })

    adjusted = holm_adjust(raw_p_values)
    for row, adjusted_p in zip(pairwise_rows, adjusted):
        row["mcnemar_exact_p_holm"] = adjusted_p

    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_df.to_csv(OUTPUT_DIR / "topology_pairwise_by_permutation_seed.csv", index=False)

    # Primary summary: observed versus the mean performance of the three fixed
    # topology-permutation replicates. This avoids selecting the most favorable
    # randomization after inspecting results.
    permuted_accuracies = [
        accuracy_score(truth, pred) for pred in permutation_prediction_arrays
    ]
    permuted_f1s = [
        f1_score(truth, pred, average="macro", zero_division=0)
        for pred in permutation_prediction_arrays
    ]
    aggregate_ci = paired_bootstrap_observed_vs_mean_permuted(
        truth,
        observed_pred,
        permutation_prediction_arrays,
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED + 99,
    )
    primary_summary = pd.DataFrame([{
        "paired_outer_test_n": len(truth),
        "permutation_seeds": "|".join(map(str, PERMUTATION_SEEDS)),
        "observed_accuracy": observed_accuracy,
        "mean_permuted_accuracy": float(np.mean(permuted_accuracies)),
        "sd_permuted_accuracy": float(np.std(permuted_accuracies, ddof=1)),
        "observed_minus_mean_permuted_accuracy": observed_accuracy - float(np.mean(permuted_accuracies)),
        "observed_minus_mean_permuted_accuracy_pp": (observed_accuracy - float(np.mean(permuted_accuracies))) * 100,
        "accuracy_difference_ci_lower": aggregate_ci["accuracy_ci_lower"],
        "accuracy_difference_ci_upper": aggregate_ci["accuracy_ci_upper"],
        "observed_macro_f1": observed_f1,
        "mean_permuted_macro_f1": float(np.mean(permuted_f1s)),
        "sd_permuted_macro_f1": float(np.std(permuted_f1s, ddof=1)),
        "observed_minus_mean_permuted_macro_f1": observed_f1 - float(np.mean(permuted_f1s)),
        "observed_minus_mean_permuted_macro_f1_pp": (observed_f1 - float(np.mean(permuted_f1s))) * 100,
        "macro_f1_difference_ci_lower": aggregate_ci["macro_f1_ci_lower"],
        "macro_f1_difference_ci_upper": aggregate_ci["macro_f1_ci_upper"],
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "inference_note": "McNemar tests are reported separately for each of the three fixed permutation seeds with Holm correction; no post-hoc permutation seed was selected.",
    }])
    primary_summary.to_csv(OUTPUT_DIR / "topology_primary_summary.csv", index=False)

    print("\n" + "=" * 100)
    print("PRIMARY TOPOLOGY SUMMARY")
    print("=" * 100)
    print(primary_summary.to_string(index=False))

    print("\nPAIRWISE OBSERVED VS EACH FIXED PERMUTATION SEED")
    print(
        pairwise_df[
            [
                "permutation_seed",
                "observed_minus_permuted_accuracy_pp",
                "observed_minus_permuted_macro_f1_pp",
                "mcnemar_exact_p_raw",
                "mcnemar_exact_p_holm",
            ]
        ].to_string(index=False)
    )

    readme = f"""EXPERIMENT 2 — OBSERVED TOPOLOGY VS DESTINATION-PERMUTED TOPOLOGY
================================================================================

Scientific question
-------------------
Does the observed passer-recipient topology outperform a matched topology-disrupted graph representation?

Frozen reference
----------------
Experiment 1 eligible cohort N: {len(samples)}
Frozen paired outer-test N: {len(observed)}
Outer/inner fold assignments: reused exactly from Experiment 1
Observed comparator: frozen B-GNN-full out-of-fold predictions from Experiment 1

Permutation design
------------------
Three fixed topology-permutation seeds: {PERMUTATION_SEEDS}
Only edge destinations are rewired.
Randomization depends only on permutation seed and match ID.
No label, fold, prediction, or performance information enters the permutation.

Preserved exactly
-----------------
- node features
- edge count
- edge source vector / source out-degree
- destination multiset / destination in-degree
- edge-feature rows: pass length, pass angle, key-pass indicator
- role IDs
- graph labels
- no synthetic event-level self-passes

Training/evaluation
-------------------
Each permuted condition uses the exact frozen Experiment 1 inner/outer fold assignments,
the exact fold-specific training seed, the same GATv2 architecture, the same optimizer,
training-only scaling, and the same chronological model-selection procedure.

Primary statistical reporting
-----------------------------
- observed B-GNN pooled performance
- each of three fixed permutation-seed pooled performances
- mean and SD of permuted performance across the three fixed seeds
- observed minus mean-permuted effect sizes
- paired match-level bootstrap CIs with {BOOTSTRAP_SAMPLES} resamples
- exact McNemar test per permutation seed
- Holm adjustment across the three McNemar tests

Important interpretation rule
-----------------------------
If observed topology clearly exceeds all/most permuted controls, this supports a topology-specific contribution.
If observed and permuted performance are similar, topology-specific claims must be weakened or removed.
If permuted topology performs better, investigate representation or leakage problems before submission.
"""
    (OUTPUT_DIR / "README_EXPERIMENT2.txt").write_text(readme, encoding="utf-8")

    metadata = {
        "artifact": "Experiment 2 matched destination-permutation topology control",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "eligible_cohort_n": len(samples),
        "paired_outer_test_n": len(observed),
        "permutation_seeds": PERMUTATION_SEEDS,
        "swap_multiplier": SWAP_MULTIPLIER,
        "training_seed_source": "frozen Experiment 1 fold assignments/selected epochs",
        "observed_predictions_source": "frozen Experiment 1 B-GNN-full OOF predictions",
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }
    (OUTPUT_DIR / "reproducibility_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    # Create a checksum manifest for all experiment code/reference/output files,
    # excluding transient StatsBomb event cache and archive files.
    excluded_parts = {"__pycache__", ".pytest_cache", ".git", "data"}
    checksum_rows = []
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if not path.is_file() or path.suffix == ".zip":
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
    include_roots = [
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "tests",
        REFERENCE_DIR,
        OUTPUT_DIR,
    ]
    top_level_files = [
        PROJECT_ROOT / "config_final.yaml",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "run_experiment2_topology.py",
    ]
    with zipfile.ZipFile(FINAL_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for root in include_roots:
            for path in sorted(root.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and ".pytest_cache" not in path.parts:
                    archive.write(path, Path("football_gnn_experiment2_topology") / path.relative_to(PROJECT_ROOT))
        for path in top_level_files:
            if path.exists():
                archive.write(path, Path("football_gnn_experiment2_topology") / path.name)

    print("\n" + "=" * 100)
    print("EXPERIMENT 2 COMPLETED SUCCESSFULLY")
    print("=" * 100)
    print("Output directory:", OUTPUT_DIR)
    print("Final archive:", FINAL_ZIP)
    print("Archive size (MB):", round(FINAL_ZIP.stat().st_size / (1024 * 1024), 2))


if __name__ == "__main__":
    main()
