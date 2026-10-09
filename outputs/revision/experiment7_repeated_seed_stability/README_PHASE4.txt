IEEE ACCESS R1 — PHASE 4
REPEATED-TRAINING-SEED VARIABILITY
==================================

Purpose
-------
Quantify neural-training stochasticity that is not captured by
resampling fixed test predictions.

Seed schedules
--------------
Five complete seed schedules are reported.

S0:
142, 242, 342, 442, 542
This is the exact frozen original experiment.

S1:
1142, 1242, 1342, 1442, 1542

S2:
2142, 2242, 2342, 2442, 2542

S3:
3142, 3242, 3342, 3442, 3542

S4:
4142, 4242, 4342, 4442, 4542

S1-S4 are independent additional training repetitions.

For every new seed/fold/model:
1. model parameters are reinitialized;
2. training batches are reshuffled according to the seed;
3. dropout stochasticity follows the new seed;
4. the inner-validation early-stopping epoch is reselected;
5. the model is retrained on the full outer training partition;
6. predictions are generated on the exact same frozen OOF test fold.

Elite analysis
--------------
A-MLP
C-Hybrid-positional
C-Hybrid-globalmean

Questions:
- Is C-Hybrid vs A-MLP stable to training seed?
- Is the positional-vs-global-mean pooling effect stable to seed?

Global analysis
---------------
A-MLP
C-Hybrid-positional

Question:
- Is the main Global Hybrid-vs-MLP advantage stable to seed?

Reporting
---------
For each model:
- macro-F1 mean
- sample SD
- min/max
- range

For each comparison:
- difference for every seed schedule
- mean difference
- sample SD
- min/max difference
- number of schedules favoring each model

Important interpretation
------------------------
This is a limited repeated-seed stability analysis, not a search for the
best-performing seed and not a replacement for observation-level or
dependency-aware uncertainty analyses.

All five schedules must be reported, regardless of direction.