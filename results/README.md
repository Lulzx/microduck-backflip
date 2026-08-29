# Curated backflip evidence

The repository intentionally does not version bulk training logs, sweeps, or
every exploratory evaluation. Selected promoted checkpoints and evidence are
force-added beside this manifest so the documented milestone can be replayed:

- `checkpoints/v88-assisted-cube-to-mat-model650/`: launch checkpoint and
  frozen run configuration for the first complete assisted cube-to-mat result.
- `checkpoints/v44-unified-model510/`: apex-to-mat recovery checkpoint used by
  the v88 policy handoff.
- `research_cube_to_mat_harness_v88_g0p16_m0p40_64_seed45_warp1.json`: complete
  deterministic 64-world evaluation for the v88 assisted configuration.
- `videos/v88-cube-mat-harness-success-seed45-env28-warp1/`: exact 4 s replay,
  evaluator output, and per-step trace for successful environment 28.

- `threaded_v15_model1175_strict_seed42.json`: best pre-research strict
  standing-start baseline; reliable takeoff but incomplete rotation.
- `phase_residual_authority5_model0_assisted_seed42.json`: assisted teacher
  distribution with 5% policy authority before impact. This is an assisted
  diagnostic, not an autonomous result.
- `research_efgcl_v25_model50_recovery_seed42.json`: recovery-slice baseline
  before stage-wise landing rewards; no strict holds.
- `research_efgcl_v25_model50_recovery_trace_seed42.json`: deterministic trace
  used to identify immediate post-contact angular instability.
- `research_efgcl_v26_model50_recovery_seed42.json`: first nonzero strict
  recovery hold after introducing stage-wise landing objectives.
- `research_efgcl_v26_model50_recovery_trace_seed42.json`: corresponding
  per-control-step diagnostic trace.
- `research_efgcl_v26_model50_assisted_seed42.json`: full standing-start
  diagnostic for the same checkpoint with virtual-spotter scale 1.0.
- `videos/v26-model50-best-recovery/*.mp4`: environment 29, which achieved the
  checkpoint's sole strict 0.5 s recovery hold before later tipping over.
- `videos/v26-model50-assisted-standing/*.mp4`: environment 29 from an assisted
  standing start; it visibly takes off and rotates but ends in body contact.

Recovery-start evaluations seed the state machine at a completed revolution
and feet-down contact. They isolate the landing controller and **do not prove a
standing-start backflip**. The project acceptance gates require the independent
standing-start batteries documented in `docs/backflip.md`.

Older JSONs retain their original local checkpoint path for provenance. Only
the two checkpoints required to replay promoted v44/v88 evidence are committed.
A final autonomous policy is still withheld until it passes the documented
multi-seed nominal and backlash batteries.
