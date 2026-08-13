EXPERIMENT 2 — OBSERVED TOPOLOGY VS DESTINATION-PERMUTED TOPOLOGY
================================================================================

Scientific question
-------------------
Does the observed passer-recipient topology outperform a matched topology-disrupted graph representation?

Frozen reference
----------------
Experiment 1 eligible cohort N: 757
Frozen paired outer-test N: 454
Outer/inner fold assignments: reused exactly from Experiment 1
Observed comparator: frozen B-GNN-full out-of-fold predictions from Experiment 1

Permutation design
------------------
Three fixed topology-permutation seeds: [1, 2, 3]
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
- paired match-level bootstrap CIs with 10000 resamples
- exact McNemar test per permutation seed
- Holm adjustment across the three McNemar tests

Important interpretation rule
-----------------------------
If observed topology clearly exceeds all/most permuted controls, this supports a topology-specific contribution.
If observed and permuted performance are similar, topology-specific claims must be weakened or removed.
If permuted topology performs better, investigate representation or leakage problems before submission.
