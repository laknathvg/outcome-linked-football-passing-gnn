
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from torch_geometric.data import Batch

from src.final_protocol import harvest_dataset_with_retry_cache
from src.history import align_samples, compute_rolling_history
from src.models import HistoryMLP, StructureGNN, HybridGNN
from src.reproducibility import save_environment, set_global_seed
from src.scaling import (
    fit_graph_scalers,
    make_feature_view,
    transform_graphs,
)
from src.training import (
    fit_graph_model,
    predict_graph_model,
    select_graph_epoch,
)
from src.topology_free import (
    TopologyFreeNN,
    TopologyFreeHybrid,
)


ROOT = Path.cwd()

BASE_SEED = 42
BOOTSTRAP_SAMPLES = 10_000

MODEL_TF = "TF-NN"
MODEL_TF_HYBRID = "TF-Hybrid"

FROZEN_COMPARISONS = [
    ("B-GNN-full", MODEL_TF, "B-GNN-full vs TF-NN"),
    (
        "C-Hybrid-full",
        MODEL_TF_HYBRID,
        "C-Hybrid-full vs TF-Hybrid",
    ),
    (
        MODEL_TF_HYBRID,
        "A-MLP",
        "TF-Hybrid vs A-MLP",
    ),
]


def count_parameters(model):
    return int(
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )


def fold_seed(fold: int) -> int:
    # EXACT submitted seed policy.
    return BASE_SEED + 100 * int(fold)


def load_config(cohort: str) -> dict:
    if cohort == "elite":
        path = ROOT / "config_final.yaml"
    else:
        path = ROOT / "config_experiment5_global.yaml"

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    # Persist event caches outside the cloned repository.
    # This does NOT change scientific methodology.
    if cohort == "elite":
        config["project"]["cache_dir"] = (
            "/content/statsbomb_elite_r1_cache"
        )
        config["cohort"]["name"] = "elite"
        config["cohort"]["mode"] = "named"
        config["cohort"]["competition_names"] = [
            "La Liga",
            "Champions League",
        ]
        config["cohort"]["max_matches"] = None

    else:
        config["project"]["cache_dir"] = (
            "/content/statsbomb_global_event_cache"
        )
        config["cohort"] = {
            "name": "global",
            "mode": "all",
            "competition_names": [],
            "max_matches": None,
        }

    # Assert the frozen submitted model/training configuration.
    assert int(config["graph"]["node_count"]) == 11
    assert int(config["model"]["hidden_dim"]) == 64
    assert int(config["model"]["role_embedding_dim"]) == 8
    assert float(config["model"]["graph_dropout"]) == 0.30
    assert float(config["model"]["classifier_dropout"]) == 0.40
    assert int(config["model"]["history_hidden_dim"]) == 32
    assert float(config["model"]["history_dropout"]) == 0.20
    assert float(config["training"]["learning_rate"]) == 0.002
    assert float(config["training"]["weight_decay"]) == 0.0005
    assert int(config["training"]["batch_size"]) == 32
    assert int(config["training"]["maximum_epochs"]) == 50

    return config


def paths_for(cohort: str):
    if cohort == "elite":
        return {
            "assignments": (
                ROOT
                / "reference_experiment1"
                / "fold_assignments.csv"
            ),
            "predictions": (
                ROOT
                / "reference_experiment1"
                / "fold_predictions_final.csv"
            ),
            "output": (
                ROOT
                / "outputs"
                / "experiment6_topology_free_elite"
            ),
        }

    return {
        "assignments": (
            ROOT
            / "outputs"
            / "experiment5_global_pooled_consistency"
            / "global_fold_assignments.csv"
        ),
        "predictions": (
            ROOT
            / "outputs"
            / "experiment5_global_pooled_consistency"
            / "global_oof_predictions.csv"
        ),
        "output": (
            ROOT
            / "outputs"
            / "experiment6_topology_free_global"
        ),
    }


def full_graph_views(graphs, indices):
    return [
        make_feature_view(
            graphs[int(i)],
            node_indices=[0, 1, 2, 3],
            edge_indices=[0, 1, 2],
        )
        for i in indices
    ]


def preprocess_graph_pair(
    train_raw,
    other_raw,
    train_history,
    other_history,
):
    # EXACT same fold-local preprocessing logic as submitted study.
    scalers = fit_graph_scalers(
        graphs=train_raw,
        standardize_nodes=True,
        edge_continuous_indices=[0, 1],
    )

    return (
        transform_graphs(
            train_raw,
            scalers,
            train_history,
        ),
        transform_graphs(
            other_raw,
            scalers,
            other_history,
        ),
    )


def metric_row(truth, pred, score):
    truth = np.asarray(truth, dtype=int)
    pred = np.asarray(pred, dtype=int)
    score = np.asarray(score, dtype=float)

    row = {
        "n": int(len(truth)),
        "accuracy": float(
            accuracy_score(truth, pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(truth, pred)
        ),
        "macro_f1": float(
            f1_score(
                truth,
                pred,
                average="macro",
                labels=[0, 1],
                zero_division=0,
            )
        ),
        "win_precision": float(
            precision_score(
                truth,
                pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "win_recall": float(
            recall_score(
                truth,
                pred,
                pos_label=1,
                zero_division=0,
            )
        ),
    }

    if len(np.unique(truth)) == 2:
        row["roc_auc"] = float(
            roc_auc_score(truth, score)
        )
    else:
        row["roc_auc"] = np.nan

    return row


def paired_bootstrap(
    truth,
    pred_a,
    pred_b,
    samples=BOOTSTRAP_SAMPLES,
    seed=2026,
):
    truth = np.asarray(truth, dtype=int)
    pred_a = np.asarray(pred_a, dtype=int)
    pred_b = np.asarray(pred_b, dtype=int)

    rng = np.random.default_rng(seed)

    n = len(truth)

    accuracy_differences = np.empty(
        samples,
        dtype=float,
    )

    f1_differences = np.empty(
        samples,
        dtype=float,
    )

    for i in range(samples):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        y = truth[idx]
        a = pred_a[idx]
        b = pred_b[idx]

        accuracy_differences[i] = (
            accuracy_score(y, a)
            - accuracy_score(y, b)
        )

        f1_differences[i] = (
            f1_score(
                y,
                a,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
            - f1_score(
                y,
                b,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        )

    return {
        "accuracy_ci_lower_pp":
            100.0
            * float(
                np.percentile(
                    accuracy_differences,
                    2.5,
                )
            ),

        "accuracy_ci_upper_pp":
            100.0
            * float(
                np.percentile(
                    accuracy_differences,
                    97.5,
                )
            ),

        "macro_f1_ci_lower_pp":
            100.0
            * float(
                np.percentile(
                    f1_differences,
                    2.5,
                )
            ),

        "macro_f1_ci_upper_pp":
            100.0
            * float(
                np.percentile(
                    f1_differences,
                    97.5,
                )
            ),
    }


def exact_mcnemar(
    truth,
    pred_a,
    pred_b,
):
    truth = np.asarray(truth, dtype=int)
    pred_a = np.asarray(pred_a, dtype=int)
    pred_b = np.asarray(pred_b, dtype=int)

    a_correct = pred_a == truth
    b_correct = pred_b == truth

    a_correct_b_wrong = int(
        np.sum(
            a_correct
            & ~b_correct
        )
    )

    b_correct_a_wrong = int(
        np.sum(
            b_correct
            & ~a_correct
        )
    )

    discordant = (
        a_correct_b_wrong
        + b_correct_a_wrong
    )

    if discordant == 0:
        p = 1.0
    else:
        p = float(
            binomtest(
                min(
                    a_correct_b_wrong,
                    b_correct_a_wrong,
                ),
                n=discordant,
                p=0.5,
            ).pvalue
        )

    return (
        a_correct_b_wrong,
        b_correct_a_wrong,
        p,
    )


def holm_adjust(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    m = len(values)
    order = np.argsort(values)

    adjusted = np.empty(
        m,
        dtype=float,
    )

    running = 0.0

    for rank, index in enumerate(order):
        current = min(
            1.0,
            (m - rank) * values[index],
        )

        running = max(
            running,
            current,
        )

        adjusted[index] = running

    return adjusted


def normalize_frozen_predictions(df):
    """
    Current repository prediction outputs already contain these fields,
    but this validates rather than silently guessing.
    """
    required = {
        "match_id",
        "model",
        "true_label",
        "predicted_class",
        "predicted_score",
    }

    missing = required.difference(df.columns)

    if missing:
        raise RuntimeError(
            "Frozen prediction file is missing expected columns: "
            + str(sorted(missing))
        )

    result = df.copy()

    result["match_id"] = (
        result["match_id"]
        .astype(int)
    )

    result["true_label"] = (
        result["true_label"]
        .astype(int)
    )

    result["predicted_class"] = (
        result["predicted_class"]
        .astype(int)
    )

    return result


def paired_comparison(
    predictions,
    model_a,
    model_b,
    name,
    seed,
):
    a = (
        predictions[
            predictions["model"].eq(model_a)
        ]
        .sort_values("match_id")
        .reset_index(drop=True)
    )

    b = (
        predictions[
            predictions["model"].eq(model_b)
        ]
        .sort_values("match_id")
        .reset_index(drop=True)
    )

    if len(a) != len(b):
        raise RuntimeError(
            f"{name}: paired N differs."
        )

    if not np.array_equal(
        a["match_id"].to_numpy(),
        b["match_id"].to_numpy(),
    ):
        raise RuntimeError(
            f"{name}: match IDs are not paired."
        )

    if not np.array_equal(
        a["true_label"].to_numpy(),
        b["true_label"].to_numpy(),
    ):
        raise RuntimeError(
            f"{name}: truth labels differ."
        )

    truth = a["true_label"].to_numpy()

    pred_a = (
        a["predicted_class"]
        .to_numpy()
    )

    pred_b = (
        b["predicted_class"]
        .to_numpy()
    )

    metric_a = metric_row(
        truth,
        pred_a,
        a["predicted_score"].to_numpy(),
    )

    metric_b = metric_row(
        truth,
        pred_b,
        b["predicted_score"].to_numpy(),
    )

    boot = paired_bootstrap(
        truth,
        pred_a,
        pred_b,
        samples=BOOTSTRAP_SAMPLES,
        seed=seed,
    )

    aw, bw, p = exact_mcnemar(
        truth,
        pred_a,
        pred_b,
    )

    return {
        "comparison": name,
        "model_a": model_a,
        "model_b": model_b,
        "paired_n": int(len(truth)),

        "model_a_accuracy":
            metric_a["accuracy"],

        "model_b_accuracy":
            metric_b["accuracy"],

        "accuracy_difference_pp":
            100.0
            * (
                metric_a["accuracy"]
                - metric_b["accuracy"]
            ),

        "model_a_macro_f1":
            metric_a["macro_f1"],

        "model_b_macro_f1":
            metric_b["macro_f1"],

        "macro_f1_difference_pp":
            100.0
            * (
                metric_a["macro_f1"]
                - metric_b["macro_f1"]
            ),

        **boot,

        "model_a_correct_b_wrong":
            aw,

        "model_b_correct_a_wrong":
            bw,

        "mcnemar_exact_p_raw":
            p,

        "bootstrap_samples":
            BOOTSTRAP_SAMPLES,
    }


def get_indices(
    assignments,
    id_to_index,
    fold,
    level,
    role,
):
    block = assignments[
        assignments["fold"].eq(fold)
        & assignments["level"].eq(level)
        & assignments["role"].eq(role)
    ]

    ids = (
        block["match_id"]
        .astype(int)
        .tolist()
    )

    missing = [
        mid
        for mid in ids
        if mid not in id_to_index
    ]

    if missing:
        raise RuntimeError(
            f"Frozen assignment contains "
            f"{len(missing)} match IDs "
            f"not reconstructed in cohort. "
            f"First IDs={missing[:10]}"
        )

    return np.asarray(
        [
            id_to_index[mid]
            for mid in ids
        ],
        dtype=int,
    )


def topology_invariance_test(
    graph,
    config,
    history_dim,
    device,
):
    """
    Scientific integrity test:
    TF models MUST be invariant to within-graph adjacency rewiring
    when node features, roles and edge_attr rows are unchanged.
    """

    from copy import deepcopy

    set_global_seed(777)

    model = TopologyFreeHybrid(
        4,
        3,
        history_dim,
        config,
    ).to(device)

    model.eval()

    original = deepcopy(graph)

    destination_permuted = deepcopy(graph)

    # Rotate destination assignments.
    destination_permuted.edge_index = (
        destination_permuted.edge_index.clone()
    )

    destination_permuted.edge_index[1] = torch.roll(
        destination_permuted.edge_index[1],
        shifts=1,
    )

    both_permuted = deepcopy(graph)

    both_permuted.edge_index = (
        both_permuted.edge_index.clone()
    )

    # Also alter source identities inside the same graph.
    both_permuted.edge_index[0] = torch.roll(
        both_permuted.edge_index[0],
        shifts=1,
    )

    both_permuted.edge_index[1] = torch.roll(
        both_permuted.edge_index[1],
        shifts=2,
    )

    b1 = Batch.from_data_list(
        [original]
    ).to(device)

    b2 = Batch.from_data_list(
        [destination_permuted]
    ).to(device)

    b3 = Batch.from_data_list(
        [both_permuted]
    ).to(device)

    with torch.no_grad():
        z1 = model(b1)
        z2 = model(b2)
        z3 = model(b3)

    d12 = float(
        torch.max(
            torch.abs(z1 - z2)
        )
        .cpu()
    )

    d13 = float(
        torch.max(
            torch.abs(z1 - z3)
        )
        .cpu()
    )

    passed = (
        d12 < 1e-7
        and d13 < 1e-7
    )

    if not passed:
        raise RuntimeError(
            "TOPOLOGY INVARIANCE TEST FAILED. "
            f"destination diff={d12}, "
            f"source+destination diff={d13}"
        )

    return {
        "passed": True,
        "max_abs_logit_diff_destination_permutation":
            d12,
        "max_abs_logit_diff_source_and_destination_permutation":
            d13,
        "interpretation":
            (
                "Topology-free model output is invariant "
                "to within-graph rewiring when node "
                "features, roles, pass-event attributes "
                "and history are fixed."
            ),
    }


def harvest(cohort, config):
    print(
        "\nHarvesting StatsBomb Open Data..."
    )

    if cohort == "elite":
        return harvest_dataset_with_retry_cache(
            config,
            attempts=5,
            fail_on_event_error=True,
        )

    # Same defensive retry logic as original Global experiment.
    result = harvest_dataset_with_retry_cache(
        config,
        attempts=5,
        fail_on_event_error=False,
    )

    match_stats, graph_map, manifest, candidate = result

    failed = manifest[
        manifest["exclusion_reason"]
        .eq("event_load_failed")
    ]

    if len(failed):
        print(
            f"Global first pass left "
            f"{len(failed)} failed event downloads; retrying."
        )

        time.sleep(3)

        result = harvest_dataset_with_retry_cache(
            config,
            attempts=8,
            fail_on_event_error=False,
        )

        match_stats, graph_map, manifest, candidate = result

        failed = manifest[
            manifest["exclusion_reason"]
            .eq("event_load_failed")
        ]

    if len(failed):
        raise RuntimeError(
            "Global harvesting remains incomplete. "
            "Stop rather than changing the cohort."
        )

    return (
        match_stats,
        graph_map,
        manifest,
        candidate,
    )


def main(cohort):
    paths = paths_for(cohort)
    output_dir = paths["output"]

    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = load_config(cohort)

    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\n" + "=" * 100)
    print(
        f"EXPERIMENT 6 — TOPOLOGY-FREE NEURAL BASELINE — "
        f"{cohort.upper()}"
    )
    print("=" * 100)

    print("Device:", device)
    print("Source commit:", source_commit)

    # -------------------------------------------------------------
    # WRITE PROTOCOL BEFORE ANY TRAINING OCCURS
    # -------------------------------------------------------------

    frozen_protocol = {
        "experiment":
            "Experiment 6 - Information-matched topology-free neural baseline",

        "cohort":
            cohort,

        "protocol_frozen_before_training":
            True,

        "reviewer_target":
            "Reviewer 1 Comment 2",

        "primary_scientific_question":
            (
                "Does adjacency-aware graph learning outperform "
                "an information-matched topology-free neural "
                "representation?"
            ),

        "topology_free_information_retained": [
            "11 individual player feature vectors",
            "completed-pass count",
            "shot count",
            "summed shot xG",
            "ball recoveries",
            "same broad role IDs",
            "same 8-D learnable role embeddings",
            "every pass-event length",
            "every pass-event angle",
            "every pass-event key-pass indicator",
        ],

        "information_deliberately_removed": [
            "passer identity for each pass event",
            "recipient identity for each pass event",
            "source-destination pairing",
            "adjacency/message passing",
        ],

        "player_encoder":
            "shared MLP: (4 + 8 role embedding) -> 64 -> 64",

        "player_pooling":
            "same GK/DF/MF/FW role-aware additive pooling",

        "pass_encoder":
            "shared MLP: 3 -> 64 -> 64",

        "pass_pooling":
            "permutation-invariant additive set pooling",

        "topology_free_representation_dim":
            320,

        "standalone_head":
            "320 -> 128 -> 2",

        "hybrid_completed_match_head":
            "320 -> 64 -> 2",

        "historical_branch":
            "exact existing HistoryMLP, 16 -> 32 -> 2",

        "fusion":
            "exact static sigmoid-constrained scalar logit fusion",

        "primary_comparison":
            "C-Hybrid-full vs TF-Hybrid",

        "secondary_comparisons": [
            "B-GNN-full vs TF-NN",
            "TF-Hybrid vs A-MLP",
        ],

        "multiplicity":
            "Holm adjustment across the three Experiment 6 McNemar tests",

        "bootstrap_samples":
            BOOTSTRAP_SAMPLES,

        "interpretation_rules": {
            "graph_supported":
                (
                    "If adjacency-aware model advantage has "
                    "paired CI above zero, report evidence "
                    "for graph-specific benefit."
                ),

            "graph_inconclusive":
                (
                    "If point estimate favors graph model but "
                    "CI crosses zero, report suggestive but "
                    "unestablished graph-specific benefit."
                ),

            "no_graph_requirement":
                (
                    "If topology-free and graph models are "
                    "similar, conclude rich completed-match "
                    "information is useful but graph learning "
                    "is not shown to be necessary."
                ),

            "topology_free_better":
                (
                    "If topology-free model performs better, "
                    "report it without rerunning to obtain a "
                    "preferred result."
                ),
        },
    }

    (
        output_dir
        / "protocol_frozen_before_training.json"
    ).write_text(
        json.dumps(
            frozen_protocol,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "source_commit.txt"
    ).write_text(
        source_commit + "\n",
        encoding="utf-8",
    )

    save_environment(
        output_dir
        / "environment.json"
    )

    # -------------------------------------------------------------
    # HARVEST THE SAME COHORT
    # -------------------------------------------------------------

    set_global_seed(BASE_SEED)

    (
        match_stats,
        graph_map,
        manifest,
        candidate_manifest,
    ) = harvest(
        cohort,
        config,
    )

    print(
        "Candidate matches:",
        len(candidate_manifest),
    )

    print(
        "Graphs reconstructed:",
        len(graph_map),
    )

    history_df = compute_rolling_history(
        match_stats=match_stats,
        window=int(
            config["history"]["window_matches"]
        ),
        group_by_competition=True,
    )

    (
        samples,
        graphs,
        X_history_raw,
        y,
        history_columns,
    ) = align_samples(
        history_df,
        graph_map,
    )

    samples["match_id"] = (
        samples["match_id"]
        .astype(int)
    )

    print(
        "Eligible observations reconstructed:",
        len(samples),
    )

    print(
        "History dimensions:",
        X_history_raw.shape[1],
    )

    if X_history_raw.shape[1] != 16:
        raise RuntimeError(
            f"Expected 16 history features, "
            f"got {X_history_raw.shape[1]}"
        )

    # Submitted Elite should reconstruct exactly 757 eligible samples.
    if cohort == "elite" and len(samples) != 757:
        raise RuntimeError(
            "Elite cohort mismatch: expected 757 eligible "
            f"observations, reconstructed {len(samples)}."
        )

    # -------------------------------------------------------------
    # LOAD EXACT FROZEN ASSIGNMENTS AND COMPARATOR PREDICTIONS
    # -------------------------------------------------------------

    assignments = pd.read_csv(
        paths["assignments"]
    )

    frozen_predictions = (
        normalize_frozen_predictions(
            pd.read_csv(
                paths["predictions"]
            )
        )
    )

    print(
        "Frozen assignment rows:",
        len(assignments),
    )

    print(
        "Frozen comparator models:",
        sorted(
            frozen_predictions["model"]
            .unique()
            .tolist()
        ),
    )

    required_frozen_models = {
        "A-MLP",
        "B-GNN-full",
        "C-Hybrid-full",
    }

    absent = (
        required_frozen_models
        - set(
            frozen_predictions["model"]
            .unique()
        )
    )

    if absent:
        raise RuntimeError(
            "Frozen predictions do not contain: "
            + str(sorted(absent))
        )

    id_to_index = {
        int(mid): int(index)
        for index, mid
        in enumerate(
            samples["match_id"]
            .tolist()
        )
    }

    folds = sorted(
        assignments["fold"]
        .astype(int)
        .unique()
        .tolist()
    )

    if folds != [1, 2, 3, 4, 5]:
        raise RuntimeError(
            f"Expected frozen folds [1,2,3,4,5], got {folds}"
        )

    # -------------------------------------------------------------
    # MODEL INFORMATION / PARAMETER COUNTS
    # -------------------------------------------------------------

    history_dim = int(
        X_history_raw.shape[1]
    )

    existing_gnn = StructureGNN(
        4,
        3,
        config,
    )

    existing_hybrid = HybridGNN(
        4,
        3,
        history_dim,
        config,
    )

    tf_nn = TopologyFreeNN(
        4,
        3,
        config,
    )

    tf_hybrid = TopologyFreeHybrid(
        4,
        3,
        history_dim,
        config,
    )

    parameter_counts = pd.DataFrame([
        {
            "model": "B-GNN-full",
            "trainable_parameters":
                count_parameters(existing_gnn),
        },
        {
            "model": "TF-NN",
            "trainable_parameters":
                count_parameters(tf_nn),
        },
        {
            "model": "C-Hybrid-full",
            "trainable_parameters":
                count_parameters(existing_hybrid),
        },
        {
            "model": "TF-Hybrid",
            "trainable_parameters":
                count_parameters(tf_hybrid),
        },
    ])

    parameter_counts.to_csv(
        output_dir
        / "parameter_counts.csv",
        index=False,
    )

    information_table = pd.DataFrame([
        {
            "model": "D-Aggregate-full",
            "individual_player_features": False,
            "role_information": False,
            "individual_pass_event_attributes": False,
            "exact_passer_recipient_adjacency": False,
            "historical_context": False,
            "note": "Seven summed completed-match features",
        },
        {
            "model": "TF-NN",
            "individual_player_features": True,
            "role_information": True,
            "individual_pass_event_attributes": True,
            "exact_passer_recipient_adjacency": False,
            "historical_context": False,
            "note": "Information-matched topology-free neural control",
        },
        {
            "model": "B-GNN-full",
            "individual_player_features": True,
            "role_information": True,
            "individual_pass_event_attributes": True,
            "exact_passer_recipient_adjacency": True,
            "historical_context": False,
            "note": "Submitted graph-only comparator",
        },
        {
            "model": "TF-Hybrid",
            "individual_player_features": True,
            "role_information": True,
            "individual_pass_event_attributes": True,
            "exact_passer_recipient_adjacency": False,
            "historical_context": True,
            "note": "Topology-free completed-match NN plus exact A-MLP",
        },
        {
            "model": "C-Hybrid-full",
            "individual_player_features": True,
            "role_information": True,
            "individual_pass_event_attributes": True,
            "exact_passer_recipient_adjacency": True,
            "historical_context": True,
            "note": "Submitted late-fusion GNN",
        },
    ])

    information_table.to_csv(
        output_dir
        / "model_information_access.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # TOPOLOGY-INVARIANCE TEST BEFORE TRAINING
    # -------------------------------------------------------------

    # Use one full, unscaled graph and attach a valid history row.
    test_graph = make_feature_view(
        graphs[0],
        node_indices=[0, 1, 2, 3],
        edge_indices=[0, 1, 2],
    )

    test_graph.global_features = torch.tensor(
        X_history_raw[0],
        dtype=torch.float32,
    ).unsqueeze(0)

    invariant_result = topology_invariance_test(
        test_graph,
        config,
        history_dim,
        device,
    )

    (
        output_dir
        / "topology_invariance_test.json"
    ).write_text(
        json.dumps(
            invariant_result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nTopology-invariance test:",
        invariant_result,
    )

    # -------------------------------------------------------------
    # TRAIN NEW MODELS ON EXACT FROZEN FOLDS
    # -------------------------------------------------------------

    prediction_rows = []
    selected_epoch_rows = []
    runtime_rows = []

    for fold in folds:

        seed = fold_seed(fold)

        outer_train = get_indices(
            assignments,
            id_to_index,
            fold,
            "outer",
            "train",
        )

        outer_test = get_indices(
            assignments,
            id_to_index,
            fold,
            "outer",
            "test",
        )

        inner_train = get_indices(
            assignments,
            id_to_index,
            fold,
            "inner",
            "train",
        )

        inner_val = get_indices(
            assignments,
            id_to_index,
            fold,
            "inner",
            "validation",
        )

        print("\n" + "-" * 100)
        print(
            f"{cohort.upper()} FOLD {fold}"
        )
        print("Seed:", seed)
        print(
            "Outer train:",
            len(outer_train),
        )
        print(
            "Outer test:",
            len(outer_test),
        )
        print(
            "Inner train:",
            len(inner_train),
        )
        print(
            "Inner validation:",
            len(inner_val),
        )

        # ---------------------------------------------------------
        # INNER HISTORY SCALER
        # ---------------------------------------------------------

        inner_history_scaler = (
            StandardScaler()
            .fit(
                X_history_raw[
                    inner_train
                ]
            )
        )

        X_inner_train = (
            inner_history_scaler
            .transform(
                X_history_raw[
                    inner_train
                ]
            )
        )

        X_inner_val = (
            inner_history_scaler
            .transform(
                X_history_raw[
                    inner_val
                ]
            )
        )

        # ---------------------------------------------------------
        # EXACT SAME GRAPH FEATURE SCALING
        # ---------------------------------------------------------

        inner_graph_train_raw = (
            full_graph_views(
                graphs,
                inner_train,
            )
        )

        inner_graph_val_raw = (
            full_graph_views(
                graphs,
                inner_val,
            )
        )

        (
            inner_graph_train,
            inner_graph_val,
        ) = preprocess_graph_pair(
            inner_graph_train_raw,
            inner_graph_val_raw,
            X_inner_train,
            X_inner_val,
        )

        # ---------------------------------------------------------
        # FACTORIES
        # ---------------------------------------------------------

        tf_factory = lambda: TopologyFreeNN(
            4,
            3,
            config,
        )

        tf_hybrid_factory = (
            lambda: TopologyFreeHybrid(
                4,
                3,
                history_dim,
                config,
            )
        )

        # ---------------------------------------------------------
        # INNER CHRONOLOGICAL EPOCH SELECTION
        # ---------------------------------------------------------

        tf_selection = select_graph_epoch(
            tf_factory,
            inner_graph_train,
            inner_graph_val,
            y[inner_val],
            config,
            seed,
            device,
        )

        tf_hybrid_selection = (
            select_graph_epoch(
                tf_hybrid_factory,
                inner_graph_train,
                inner_graph_val,
                y[inner_val],
                config,
                seed,
                device,
            )
        )

        selected_epoch_rows.append({
            "fold": fold,
            "seed": seed,

            "TF_NN_best_epoch":
                tf_selection.best_epoch,

            "TF_NN_validation_macro_f1":
                tf_selection.best_metric,

            "TF_Hybrid_best_epoch":
                tf_hybrid_selection.best_epoch,

            "TF_Hybrid_validation_macro_f1":
                tf_hybrid_selection.best_metric,
        })

        print(
            "Selected epochs:",
            f"TF-NN={tf_selection.best_epoch}, "
            f"TF-Hybrid={tf_hybrid_selection.best_epoch}",
        )

        # ---------------------------------------------------------
        # OUTER TRAINING PREPROCESSING
        # ---------------------------------------------------------

        outer_history_scaler = (
            StandardScaler()
            .fit(
                X_history_raw[
                    outer_train
                ]
            )
        )

        X_outer_train = (
            outer_history_scaler
            .transform(
                X_history_raw[
                    outer_train
                ]
            )
        )

        X_outer_test = (
            outer_history_scaler
            .transform(
                X_history_raw[
                    outer_test
                ]
            )
        )

        graph_train_raw = (
            full_graph_views(
                graphs,
                outer_train,
            )
        )

        graph_test_raw = (
            full_graph_views(
                graphs,
                outer_test,
            )
        )

        (
            graph_train,
            graph_test,
        ) = preprocess_graph_pair(
            graph_train_raw,
            graph_test_raw,
            X_outer_train,
            X_outer_test,
        )

        # ---------------------------------------------------------
        # TRAIN TF-NN
        # ---------------------------------------------------------

        start = time.perf_counter()

        model = fit_graph_model(
            tf_factory,
            graph_train,
            tf_selection.best_epoch,
            config,
            seed,
            device,
        )

        train_seconds = (
            time.perf_counter()
            - start
        )

        start = time.perf_counter()

        pred, score = predict_graph_model(
            model,
            graph_test,
            config,
            device,
        )

        inference_seconds = (
            time.perf_counter()
            - start
        )

        for local_position, sample_index in enumerate(
            outer_test
        ):
            prediction_rows.append({
                "match_id":
                    int(
                        samples.iloc[
                            int(sample_index)
                        ]["match_id"]
                    ),

                "true_label":
                    int(
                        y[
                            int(sample_index)
                        ]
                    ),

                "predicted_class":
                    int(
                        pred[
                            local_position
                        ]
                    ),

                "predicted_score":
                    float(
                        score[
                            local_position
                        ]
                    ),

                "model":
                    MODEL_TF,

                "fold":
                    int(fold),

                "seed":
                    int(seed),

                "gate_value":
                    np.nan,

                "cohort":
                    cohort,
            })

        runtime_rows.append({
            "fold": fold,
            "model": MODEL_TF,
            "train_seconds":
                train_seconds,
            "inference_seconds":
                inference_seconds,
            "test_n":
                len(outer_test),
        })

        # ---------------------------------------------------------
        # TRAIN TF-HYBRID
        # ---------------------------------------------------------

        start = time.perf_counter()

        model = fit_graph_model(
            tf_hybrid_factory,
            graph_train,
            tf_hybrid_selection.best_epoch,
            config,
            seed,
            device,
        )

        train_seconds = (
            time.perf_counter()
            - start
        )

        start = time.perf_counter()

        pred, score = predict_graph_model(
            model,
            graph_test,
            config,
            device,
        )

        inference_seconds = (
            time.perf_counter()
            - start
        )

        gate = model.gate_value()

        for local_position, sample_index in enumerate(
            outer_test
        ):
            prediction_rows.append({
                "match_id":
                    int(
                        samples.iloc[
                            int(sample_index)
                        ]["match_id"]
                    ),

                "true_label":
                    int(
                        y[
                            int(sample_index)
                        ]
                    ),

                "predicted_class":
                    int(
                        pred[
                            local_position
                        ]
                    ),

                "predicted_score":
                    float(
                        score[
                            local_position
                        ]
                    ),

                "model":
                    MODEL_TF_HYBRID,

                "fold":
                    int(fold),

                "seed":
                    int(seed),

                "gate_value":
                    float(gate),

                "cohort":
                    cohort,
            })

        runtime_rows.append({
            "fold":
                fold,
            "model":
                MODEL_TF_HYBRID,
            "train_seconds":
                train_seconds,
            "inference_seconds":
                inference_seconds,
            "test_n":
                len(outer_test),
        })

        print(
            f"Fold {fold} complete. "
            f"TF-Hybrid lambda={gate:.6f}"
        )

    # -------------------------------------------------------------
    # VALIDATE NEW OOF OUTPUT
    # -------------------------------------------------------------

    new_predictions = pd.DataFrame(
        prediction_rows
    )

    if new_predictions.duplicated(
        ["match_id", "model"]
    ).any():
        raise RuntimeError(
            "Duplicate new match/model OOF predictions."
        )

    tf_ids = set(
        new_predictions[
            new_predictions["model"]
            .eq(MODEL_TF)
        ]["match_id"]
        .astype(int)
    )

    tfh_ids = set(
        new_predictions[
            new_predictions["model"]
            .eq(MODEL_TF_HYBRID)
        ]["match_id"]
        .astype(int)
    )

    original_hybrid_ids = set(
        frozen_predictions[
            frozen_predictions["model"]
            .eq("C-Hybrid-full")
        ]["match_id"]
        .astype(int)
    )

    if not (
        tf_ids
        == tfh_ids
        == original_hybrid_ids
    ):
        raise RuntimeError(
            "New topology-free predictions are not "
            "paired to the exact frozen OOF match set."
        )

    if cohort == "elite":
        if len(tf_ids) != 454:
            raise RuntimeError(
                f"Elite expected paired N=454, got {len(tf_ids)}"
            )

    print(
        "\nExact frozen OOF pairing verified."
    )

    print(
        "Paired test N:",
        len(tf_ids),
    )

    # -------------------------------------------------------------
    # COMBINE ONLY NECESSARY FROZEN COMPARATORS
    # -------------------------------------------------------------

    frozen_keep = frozen_predictions[
        frozen_predictions["model"].isin(
            [
                "A-MLP",
                "B-GNN-full",
                "C-Hybrid-full",
            ]
        )
    ].copy()

    combined = pd.concat(
        [
            frozen_keep,
            new_predictions,
        ],
        ignore_index=True,
        sort=False,
    )

    combined.to_csv(
        output_dir
        / "combined_oof_predictions.csv",
        index=False,
    )

    new_predictions.to_csv(
        output_dir
        / "topology_free_oof_predictions.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # POOLED MODEL METRICS
    # -------------------------------------------------------------

    metric_rows = []

    for model_name in [
        "A-MLP",
        "B-GNN-full",
        MODEL_TF,
        "C-Hybrid-full",
        MODEL_TF_HYBRID,
    ]:

        block = (
            combined[
                combined["model"]
                .eq(model_name)
            ]
            .sort_values("match_id")
        )

        row = {
            "model":
                model_name,
        }

        row.update(
            metric_row(
                block[
                    "true_label"
                ].to_numpy(),

                block[
                    "predicted_class"
                ].to_numpy(),

                block[
                    "predicted_score"
                ].to_numpy(),
            )
        )

        metric_rows.append(
            row
        )

    pooled_metrics = pd.DataFrame(
        metric_rows
    )

    pooled_metrics.to_csv(
        output_dir
        / "pooled_metrics.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # FROZEN THREE-COMPARISON EXPERIMENT-6 FAMILY
    # -------------------------------------------------------------

    comparison_rows = []

    for i, (
        model_a,
        model_b,
        label,
    ) in enumerate(
        FROZEN_COMPARISONS
    ):

        comparison_rows.append(
            paired_comparison(
                combined,
                model_a,
                model_b,
                label,
                seed=2600 + i,
            )
        )

    comparisons = pd.DataFrame(
        comparison_rows
    )

    comparisons[
        "mcnemar_exact_p_holm"
    ] = holm_adjust(
        comparisons[
            "mcnemar_exact_p_raw"
        ].to_numpy()
    )

    comparisons.to_csv(
        output_dir
        / "pairwise_inference.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # SAVE SELECTED EPOCHS, RUNTIMES, SUMMARY
    # -------------------------------------------------------------

    selected_epochs = pd.DataFrame(
        selected_epoch_rows
    )

    selected_epochs.to_csv(
        output_dir
        / "selected_epochs.csv",
        index=False,
    )

    runtime = pd.DataFrame(
        runtime_rows
    )

    runtime[
        "inference_ms_per_match"
    ] = (
        1000.0
        * runtime["inference_seconds"]
        / runtime["test_n"]
    )

    runtime.to_csv(
        output_dir
        / "runtime_summary.csv",
        index=False,
    )

    summary = {
        "experiment":
            "Experiment 6 - topology-free neural baseline",

        "cohort":
            cohort,

        "eligible_n":
            int(len(samples)),

        "paired_oof_n":
            int(len(tf_ids)),

        "source_commit":
            source_commit,

        "device":
            str(device),

        "bootstrap_samples":
            BOOTSTRAP_SAMPLES,

        "fold_seed_policy":
            "42 + 100*fold",

        "topology_invariance_test":
            invariant_result,

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "claim_boundary":
            (
                "This experiment tests whether adjacency-aware "
                "graph learning adds predictive information "
                "beyond an information-matched topology-free "
                "neural representation. Interpretation is "
                "determined by the frozen comparison rules, "
                "not by a desired direction of result."
            ),
    }

    (
        output_dir
        / "experiment_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # PRINT THE RESULTS WE NEED TO INSPECT
    # -------------------------------------------------------------

    print("\n" + "=" * 100)
    print("POOLED METRICS")
    print("=" * 100)

    print(
        pooled_metrics[
            [
                "model",
                "n",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "roc_auc",
            ]
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 100)
    print("PAIRWISE INFERENCE")
    print("=" * 100)

    print(
        comparisons[
            [
                "comparison",
                "paired_n",
                "macro_f1_difference_pp",
                "macro_f1_ci_lower_pp",
                "macro_f1_ci_upper_pp",
                "mcnemar_exact_p_raw",
                "mcnemar_exact_p_holm",
            ]
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 100)
    print("PARAMETER COUNTS")
    print("=" * 100)

    print(
        parameter_counts.to_string(
            index=False
        )
    )

    print("\nExperiment completed successfully.")
    print("Output directory:", output_dir)

    return output_dir


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cohort",
        choices=[
            "elite",
            "global",
        ],
        default="elite",
    )

    args = parser.parse_args()

    main(args.cohort)
