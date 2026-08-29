# v88 assisted cube-to-mat launch checkpoint

This directory freezes the launch actor used in the first reproducible
standing-on-cube to raised-mat backflip with a strict stable landing. The
recovery actor is the already published
[`v44-unified-model510`](../v44-unified-model510/) checkpoint.

This is an **assisted simulation milestone**, not an autonomous policy and not
physical-robot evidence. The evaluator supplies a 16 N upward / 1.50 Nm
backward launch pulse from 0.30 to 0.40 s and, only after the measured airborne
rotation reaches 360 degrees, a bounded pitch-damping harness with gain
0.16 Nms/rad and maximum torque 0.40 Nm.

- task: `Mjlab-BackflipReferenceMatDistill-Pedestal-MicroDuck`
- deterministic acceptance: `WARP_NUM_THREADS=1`, seed 45, 64 worlds
- result: 64/64 takeoffs, 6/64 full revolutions, 5/64 full-revolution
  feet-first landings, 1/64 strict stable landings
- best: environment 28, 361.215 degrees, 0.92 s strict hold, 1.40 s posture hold
- video/evaluation/trace: [`../../videos/v88-cube-mat-harness-success-seed45-env28-warp1/`](../../videos/v88-cube-mat-harness-success-seed45-env28-warp1/)

Reproduce from the repository root:

```bash
WARP_NUM_THREADS=1 PYTHONPATH=src uv run --python .venv/bin/python --no-sync \
  scripts/eval_backflip.py \
  results/checkpoints/v88-assisted-cube-to-mat-model650/model_650.pt \
  --task-id Mjlab-BackflipReferenceMatDistill-Pedestal-MicroDuck \
  --num-envs 64 --seed 45 --start-mode standing --assist-scale 1 \
  --assist-force-n 16 --assist-torque-nm 1.50 \
  --landing-damping-gain 0.16 --landing-damping-max-nm 0.40 \
  --recovery-checkpoint results/checkpoints/v44-unified-model510/model_510.pt \
  --recovery-switch-mode approach --recovery-phase-mode local \
  --recovery-approach-angle-deg 180 --recovery-approach-tilt-deg 180 \
  --recovery-approach-height-m 1.0 \
  --render-env-index 28 \
  --video-dir results/videos/v88-cube-mat-harness-success-seed45-env28-warp1
```

SHA-256:

```text
86cc4ec03839d2d1407be63c9b17a22bd96c5368bd0c2d05fc06b08014c8d598  model_650.pt
63599b8501a57314db55e1b549f690411f898d4488da21fcf02d9bd480b4f21d  agent.yaml
606be779610cbf59b7e84b5eea082cab8de24c0b700e89f6ca3bfb6ae70ecd50  env.yaml
```
