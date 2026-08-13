EXPERIMENT 5 — GLOBAL POOLED CONSISTENCY
=========================================

Purpose
-------
Evaluate whether the frozen full Hybrid's performance pattern remains within the heterogeneous pooled
StatsBomb Open Data cohort. This is NOT leave-one-competition-out transfer and must not be described
as external generalization.

Protocol
--------
- All StatsBomb Open Data competition-season rows returned at execution time; exact manifests saved.
- One home-team graph per completed match.
- Same full graph features and architecture frozen from Elite Experiment 1.
- Pre-match three-match histories remain competition-specific.
- Competition-aware pooled rolling-origin evaluation: for every competition that can support the
  frozen five-window protocol, the first ~40% is initial training and remaining future dates are
  divided into five windows; corresponding windows are pooled across competitions.
- Competitions with too few distinct future dates or an invalid embargoed partition are retained in
  the coverage manifest and explicitly excluded from OOF evaluation; they are never forced into tiny
  or duplicated test folds.
- Three-match team embargo is applied within competition.
- Inner validation also preserves chronology within competition.
- Training-only preprocessing, fixed seeds, max 50 epochs, patience 10.
- Primary metric: macro F1.

Required heterogeneity reporting
--------------------------------
The artifact includes per-competition sample sizes and metrics, equal-weight competition-macro summaries,
Hybrid-minus-MLP deltas by competition, men's/women's subgroup results, league/tournament subgroup results,
and paired confidence intervals where subgroup sample size permits.

Claim boundary
--------------
Use wording such as "performance within a heterogeneous pooled cohort". Do not claim unseen-competition,
unseen-team, or external transfer. Global overlaps the Elite cohort; the exact overlap is saved in
global_elite_overlap.csv.
