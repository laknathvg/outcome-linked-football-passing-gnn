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
Eligible Elite cohort: 757
Paired outer-test observations: 454
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
