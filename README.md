# Completed-Match Representations and Historical Context for Retrospective Football Outcome Classification: A Controlled Graph and Topology-Free Evaluation


This repository contains the implementation and reproducibility materials for the revised IEEE Access study:

**“Completed-Match Representations and Historical Context for Retrospective Football Outcome Classification: A Controlled Graph and Topology-Free Evaluation.”**

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

## Experiments

1. Elite completed-match/history complementarity.
2. Matched destination-permutation topology control.
3. Positional versus global-mean pooling sensitivity.
4. Outcome-adjacent feature ablation.
5. Global pooled complementarity and heterogeneity.
6. Topology-free neural controls in Elite and Global.
7. Dependency-aware and repeated-training-seed robustness analyses.

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

The revised reproducibility release (v2.0.0) is archived at:

**https://doi.org/10.5281/zenodo.23262333**

The original submitted-manuscript release (v1.0.0) remains available at:

**https://doi.org/10.5281/zenodo.21918636**
## Development repository

This repository is intended to match the methodology and results reported in the IEEE Access manuscript.

## Revision

Version 2.0.0 corresponds to the revised IEEE Access manuscript
Access-2026-42462.

The revision adds topology-free neural controls, dependency-aware
block-bootstrap sensitivity analyses, repeated-training-seed analyses,
and revised interpretation of graph topology and positional pooling.
