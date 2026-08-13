# Outcome-Linked Post-Match Classification of Football Passing Networks Using a Hybrid Late-Fusion Graph Neural Network

This repository contains the implementation and reproducibility materials for the study
**“Outcome-Linked Post-Match Classification of Football Passing Networks Using a Hybrid Late-Fusion Graph Neural Network.”**

## Authors

- Laknath Gunawardhana
- Achala Aponso
- Maninda Edirisooriya
- Lakshika Chandradeva
- Mahmoud Aldraimli
- Naomi Krishnarajah

## Scope

The study evaluates retrospective Win/Not-Win classification of completed home-team football performances by combining a completed-match passing-network representation with competition-specific pre-match historical context.

The framework is not a pre-match forecasting system, a causal measure of tactical quality, or an evaluation of generalization to unseen competitions.

## Data

Raw event data are obtained from StatsBomb Open Data and are not redistributed in this repository. The repository includes code, cohort/sample manifests, fold assignments, out-of-fold predictions, statistical outputs, and reproducibility metadata.

## Final experiments

1. Elite complementarity evaluation — `run_final_rolling_origin.py`
2. Matched destination-permutation topology control — `run_experiment2_topology.py`
3. Positional-pooling vs global-mean pooling ablation — `run_experiment3_pooling.py`
4. Reduced-feature robustness analysis — `run_experiment4_reduced.py`
5. Global pooled-consistency evaluation — `run_experiment5_global.py`

## Key cohort counts

### Elite
- Eligible observations: N = 757
- Paired outer-test observations: N = 454

### Global
- Eligible observations: N = 3,098
- OOF-evaluable eligible observations: N = 2,929
- Paired outer-test observations: N = 1,745
- Fully evaluable competitions: 11
- Elite/Global overlap: 757 observations

## Repository layout

- `src/` — final common implementation used across Experiments 1–5
- `tests/` — reproducibility and experiment-specific tests
- `reference_experiment1/` — frozen N=454 Experiment 1 reference required by Experiments 2–4
- `reference_elite/` — Elite sample reference used by Experiment 5 overlap checks
- `outputs/` — final derived outputs for Experiments 2–5
- `figures/` — source data for the paper's evidence-summary figure
- `environment/` — environment freeze
- `config_final.yaml` — final shared model/evaluation configuration

## Important implementation note

The experiment runners were originally executed in Google Colab and contain packaging paths under `/content/` for creating final ZIP archives. Those packaging paths do not affect the scientific model/evaluation logic. If running outside Colab, adjust only the final archive-output path as needed.

## Reproducibility artifact

A version-specific Zenodo DOI will be added here after the immutable submission artifact is deposited.

## Development repository

This repository is intended to match the methodology and results reported in the IEEE Access manuscript.
