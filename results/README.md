# Curated backflip evidence

The repository intentionally does not version training logs, PyTorch
checkpoints, bulk sweeps, or every exploratory evaluation. The JSON files
force-added beside this manifest are the minimum evidence needed to audit the
current claims in the documentation:

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

Each JSON records its checkpoint path for provenance, but checkpoints are not
committed because they are generated artifacts. A promoted final policy will
be released separately only after it passes the documented multi-seed nominal
and backlash batteries.
