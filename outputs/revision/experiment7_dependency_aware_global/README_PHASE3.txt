IEEE ACCESS R1 — PHASE 3
DEPENDENCY-AWARE STATISTICAL INFERENCE
======================================

Purpose
-------
Assess whether the Global paired performance conclusions remain stable
when dependence between matches is acknowledged through block resampling.

Frozen data
-----------
Paired Global OOF observations: 1745

No neural model was retrained and no prediction was modified.

Primary sensitivity analysis
----------------------------
Competition-season cluster bootstrap.

Block definition:
competition_id x season_id

Number of blocks:
20

Bootstrap replicates:
10000

Whole competition-season blocks are sampled with replacement.
All matches belonging to every sampled block are retained.
The same sampled blocks are used for both models in each paired comparison.

Secondary coarser sensitivity
-----------------------------
Competition-level cluster bootstrap.

Number of competition blocks:
11

Reference analysis
------------------
Individual-match paired bootstrap, retained only to show how the
uncertainty interval changes when dependency assumptions are relaxed.

Primary comparison
------------------
C-Hybrid-full vs A-MLP.

Additional comparisons
----------------------
C-Hybrid-full vs TF-Hybrid
TF-Hybrid vs A-MLP
B-GNN-full vs TF-NN

Interpretation
--------------
If the block-bootstrap interval becomes wider or crosses zero while
the individual-match interval does not, the paper must explicitly state
that inferential support is sensitive to the assumed dependence structure.

The block bootstrap is a sensitivity analysis rather than proof that all
dependency has been eliminated. Cross-season and other forms of residual
dependence may remain.