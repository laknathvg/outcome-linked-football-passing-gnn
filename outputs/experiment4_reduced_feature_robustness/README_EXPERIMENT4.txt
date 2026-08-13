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
Eligible Elite cohort: 757
Paired outer-test observations: 454
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
Paired match-level bootstrap: 10000 resamples
Exact McNemar tests: yes
Holm correction across the five predeclared comparisons: yes

REPORTING RULE
--------------
A reduced model does not need to equal the full model to demonstrate robustness. The relevant question is whether useful
performance and/or complementarity remains once obvious attacking-output clues are removed. Confidence intervals and corrected
p-values must be reported; negative or null findings are retained.
