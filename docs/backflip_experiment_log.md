# Backflip experiment log

This is the append-oriented laboratory record for the MicroDuck backflip
project. It records failures as well as successes. Numerical claims must point
to a JSON result, checkpoint, trace, benchmark, or terminal validation. The
task definition and safety gates live in [backflip.md](backflip.md); paper
analysis and first-principles decomposition live in
[backflip_research.md](backflip_research.md).

## Status snapshot — 2026-08-29 17:16 IST

- Goal: standing start, one uninterrupted airborne backward revolution,
  feet-first contact, then a strict 0.5 s stable hold in simulation.
- Hardware claim: **none**. No physical backflip has been attempted or proven.
- Best strict standing-start checkpoint: v15 `model_1175`; 128/128 takeoffs,
  mean 145.17 degrees and maximum 274 degrees, no 300-degree flight and no
  landing. Evidence: `results/threaded_v15_model1175_strict_seed42.json`.
- Best assisted teacher distribution: untrained 5%-authority phase residual;
  64/64 takeoffs, 64/64 >=300 degrees, 46/64 >=340 degrees, 46/64 first ground
  contacts on feet, 33/64 landing latches, and 0/64 strict stable holds.
  Evidence: `results/phase_residual_authority5_model0_assisted_seed42.json`.
- Current bottleneck: impact recovery. v25 recovery-start checkpoints 25, 50,
  and 75 all produced 0/32 strict stable holds.
- Active run: v26 stage-wise landing, 256 worlds, CPU, seed 42, 600 iterations.
  Run directory:
  `logs/rsl_rl/microduck_backflip/2026-08-29_17-14-12_research-efgcl-v26-stagewise-landing-256`.

## Reproducibility envelope

- Source checkout: `/Users/lulzx/work/microduck-backflip/microduck_rl`
- Source revision at v26 start: `d424a0c` plus the run-captured dirty diff.
- Python environment: repository `.venv`
- MuJoCo 3.10.0, Warp 1.12.0, PyTorch 2.9.1
- Simulator device: Warp CPU on Apple Silicon; CUDA is not available in this
  build. CuMetal is deliberately out of the active training path.
- Patched Warp checkout: `/Users/lulzx/work/warp-cpu-threadpool`
- Training must use `.venv/bin/microduck-train` directly. Running plain
  `uv run` may synchronize and replace the patched Warp package.
- Focused validation command:

  ```bash
  PYTHONPATH=src uv run --python .venv/bin/python --no-sync --with pytest \
    pytest tests/test_backflip_mdp.py tests/test_backflip_cfg.py -q
  ```

- v26 validation before launch: 30 passed in 3.97 s; `compileall` and
  `git diff --check` passed.

## Acceptance gates

One episode succeeds only when it starts standing, takes off above the state
machine's clearance threshold, accumulates at least 340 degrees of genuine
backward pitch before first recontact, contacts on the feet, and then remains
for 25 consecutive 50 Hz control steps with feet contact, trunk height at
least 0.095 m, tilt under 20 degrees, and angular speed under 2 rad/s.

Final simulation acceptance requires three 128-episode nominal batteries
(seeds 42, 123, and 2026) with >=99% takeoff, >=95% reaching 340 degrees,
>=90% strict stable landings, no nonfinite state, and visual rejection of
head/trunk/shoulder landings. The backlash task must then retain >=80% stable
landings. These thresholds have **not** been met.

## Simulator performance investigations

Measured physical-step throughput on this machine:

| Backend | Configuration | Physical steps/s | Decision |
|---|---:|---:|---|
| MJWarp | Warp CPU, 2048 worlds | 29,722 | Baseline was simulator-bound |
| MuJoCo C | One thread | 76,825 | Faster per world, not an mjlab drop-in |
| `mujoco.rollout` | 12 threads, RL-style one-control-step calls | 647,512 at 2048 worlds | Real speed, but exposes state plus sensors rather than mjlab's roughly 45 batched fields |
| MJWarp | Independent-process proxy, 12 processes | about 166,000 | Approximate 5.5x CPU ceiling |

The Python fan-out prototype cost about 130 microseconds per Warp launch;
MJWarp issues 137 launches per control step. A correct upstreamable native
thread pool must live in Warp's native CPU launcher and must first repair CPU
atomics (`atomic_add` is not atomic in the inspected path). v15/v26 use the
available patched CPU path, but the backflip result remains the primary goal;
backend work is not counted as behavior success.

## Chronological experiment ledger

Runs v2-v15 are reconstructed from immutable run names, saved checkpoints,
and result JSON files. v16 onward includes the decision rationale captured
during the research pass.

| Version | Hypothesis/change | Measured outcome | Decision |
|---|---|---|---|
| v2 | First explicit airborne backflip task and evaluator | Produced saved discovery checkpoints but no accepted standing backflip | Continue reward/state-machine development |
| v3-v4 | Clean discovery and revised airborne shaping | No accepted full rotation in standing evaluation | Add reverse curriculum and launch-specific signals |
| v5-v9 | Crouch and mid-flight curricula; successive launch/rotation shaping | Generated partial-motion artifacts, but no strict stable landing | Keep strict standing evaluation separate from curriculum states |
| v10 | Supported-push reward | Standing launch remained insufficient | Add ballistic feasibility target |
| v11/v11b | Ballistic feasible-push objective | Improved launch mechanics but did not close rotation/landing | Permit asymmetric strategy and specialize standing launch |
| v12 | Asymmetric/corkscrew allowance | Best evaluated policies still incomplete | Preserve mild asymmetry but retain airborne-pitch accounting |
| v13 | Standing-launch specialist | Improved takeoff, insufficient rotation | Add compact flight pose/tuck |
| v14 | Tuck discovery and native-backend investigation | Native MuJoCo throughput was promising, but backend incompatibility was too large for a drop-in | Continue MJWarp behavior work |
| v15 | Tuck task on patched threaded CPU path | `model_1175`: takeoff 100%, mean 145.17 degrees, max 274 degrees, stable 0/128 | Strict baseline retained; exploration needs successful trajectories |
| v16 | First EFGCL virtual spotter implementation | Wrench was not applied correctly; strict takeoff 0/128 | Invalid experiment; repair event semantics |
| v17 | Validated pulse configuration | A zero-weight reward hook was skipped, so the intended event path was inactive | Invalid experiment; move wrench to a step event |
| v18 | Live step-event wrench | Assisted policy learned to counteract the spotter because assisted launch rewards were free; model 25 mean rotation 93.16 degrees | Gate launch credit by autonomy |
| v19 | Multiply launch shaping by `1-assist_scale` | model 25 assisted: >=300 26/64, >=340 8/64, first contact feet 62/64; model 50: >=340 31/32 but mean 558.35 degrees, stable 0 | Exploration solved; overspin/recovery now dominant |
| v20 | Nominal-action prior plus late pitch braking | model 25 assisted: >=340 32/32 and feet-first latch 5/32; model 50 became bimodal/regressed | Bound assisted action authority directly |
| v21 | 20% residual authority while fully assisted | model 25: >=340 53/64, landing latch 25/64, stable 0; model 75 mean rotation 454.47 degrees and landing latch 1/32 | Make recovery lexicographically more important |
| v22 | Recovery-dominant reward | model 25: >=340 27/32, landing latch 5/32, stable 0 | Curriculum success did not match strict evaluator; align definitions |
| v23 | Stable latch for curriculum, full authority in unassisted reverse states | model 25: >=340 29/32 but first contact feet 4/32 and stable 0 | Assisted residual still corrupts approach and restricts recovery |
| v24 | 5% assisted residual before impact, 100% after impact | Untrained assisted diagnostic was strong: >=340 46/64, landing latch 33/64; stable 0 | Preserve teacher; isolate landing recovery |
| v25 | Add 20% recovery starts seeded feet-down, near-upright, full authority | models 25/50/75: stable 0/32. model 50 trace reached 8.6 rad/s yaw, lost foot-only support near 0.38 s, settled on body at 0.060 m | Product landing reward has a collapsed capture basin |
| v26 | Stage-wise independent upright/height/stillness/foot-support rewards plus one-shot stable-streak frontier | Running; iteration 0 already showed nonzero stability-frontier reward, but strict success remained zero | Evaluate recovery checkpoint 25 before continuing |

## v26 exact run record

Hypothesis: v25 did not lack post-impact samples; it lacked a usable objective
when one part of the landing state was poor. Replace most of the narrow product
reward with independent landing-stage objectives, and pay each newly extended
step of the 25-step stable streak once. Do not loosen strict success.

Code changes:

- `backflip_landing_upright`: broad nonnegative upright score after landing.
- `backflip_landing_height`: linear capture basin from 0.060 m to standing.
- `backflip_landing_stillness`: Gaussian angular-speed score with 4 rad/s scale.
- `backflip_landing_foot_support`: retains foot contact after touchdown.
- `backflip_stability_progress`: one-shot longest-streak frontier; rebuilding an
  already-seen short streak pays zero.
- Narrow landing composite reduced from weight 400 to 100; independent landing
  weights are 100/75/75/50 and frontier weight is 200.
- Strict success, assistance decay, standing evaluator, and action authority
  are unchanged.

Launch command:

```bash
WARP_NUM_THREADS=12 .venv/bin/microduck-train \
  Mjlab-Backflip-Flat-MicroDuck --gpu-ids None \
  --env.scene.num-envs 256 --env.seed 42 --agent.seed 42 \
  --agent.max-iterations 600 --agent.save-interval 25 \
  --agent.run-name research-efgcl-v26-stagewise-landing-256 \
  --agent.logger tensorboard
```

Initial evidence: 1,177-1,344 environment steps/s during iterations 1-7.
At iteration 0, mean episode stability-progress return was 0.625 while strict
success return was 0. This proves the frontier signal is reachable, not that a
landing has been learned. Next gate: recovery-only evaluation of model 25 over
at least 32 randomized starts, followed by a per-step trace if stable rate is
zero.

### v26 checkpoint evaluations

**2026-08-29 17:17 IST — model 25, recovery starts, seed 42, 64 episodes.**

- Stable landing: 0/64.
- Body-only ground contact: 42/64 (65.625%), improved from 64/64 in v25 model
  50 over its 32-start diagnostic.
- Mean time to body-only contact among failures: 0.568 s, versus 0.413 s for
  v25 model 50.
- Decision: stage-wise terms improve survival, but checkpoint 25 is too early
  to establish a strict hold. Continue to model 50 and add condition-specific
  hold telemetry.
- Evidence: `results/research_efgcl_v26_model25_recovery_seed42.json` and
  `results/research_efgcl_v26_model25_recovery_trace_seed42.json`.

**2026-08-29 17:19 IST — model 50, recovery starts, seed 42, 64 episodes.**

- Strict stable landing: **1/64 (1.5625%)**, the first nonzero strict recovery
  result in v16-v26.
- Body-only ground contact: 37/64 (57.8125%).
- Every trial reached each individual condition at least once: feet contact,
  upright under 20 degrees, height >=0.095 m, and angular speed <2 rad/s. The
  remaining problem is their simultaneity and duration.
- Longest feet/upright/height posture hold: mean 0.146 s, median 0.10 s,
  p90 0.26 s, maximum 1.04 s.
- Longest fully strict hold: mean 0.0425 s, median 0.02 s, p90 0.06 s,
  maximum 0.54 s.
- Decision: hypothesis supported but far below acceptance. Continue v26 and
  evaluate model 75/100. The angular-speed gate is currently the largest gap
  between posture-only and strict duration; do not modify the threshold.
- Evidence: `results/research_efgcl_v26_model50_recovery_seed42.json` and
  `results/research_efgcl_v26_model50_recovery_trace_seed42.json`.

Evaluator change made after model 25: result JSON now includes longest
posture-only and strict streak distributions, per-condition ever-hit rates,
and trace fields for tilt, angular speed, each strict predicate, and current
stable streak length. This changes diagnostics only, not physics, policy
observations, reward, or success semantics.

**2026-08-29 17:23 IST — model 100, recovery starts, seed 42, 64 episodes.**

- Strict stable landing: 0/64; body-only contact: 64/64.
- Longest posture-only hold: mean 0.085 s, maximum 0.20 s.
- Longest strict hold: mean 0.023 s, maximum 0.04 s.
- Decision: regression; model 50 remains the promoted recovery checkpoint.
- Evidence: `results/research_efgcl_v26_model100_recovery_seed42.json` (kept
  locally; not part of the curated public evidence set).

**2026-08-29 17:26 IST — model 50 visual/provenance pass.**

- Deterministic replay identified environment 29 as the strict recovery
  success. The selected-environment recorder produced a 1280x720, 50 fps,
  4.0 s H.264 MP4. Visual inspection confirms an initial upright recovery,
  followed by a later side fall; it is not a durable stand.
- An assisted standing-start battery over 64 worlds produced 64/64 takeoffs,
  63/64 >=300 degrees, 46/64 >=340 degrees, 27/64 landing latches, 45/64 first
  ground contacts on feet, and 0/64 strict holds. Environment 29 was rendered:
  it visibly takes off and rotates, then lands on the side/body.
- Recovery video:
  `results/videos/v26-model50-best-recovery/backflip-model_50-recovery-step-0.mp4`.
- Assisted full-attempt video:
  `results/videos/v26-model50-assisted-standing/backflip-model_50-standing-step-0.mp4`.
- These videos are diagnostic evidence only. Neither proves an autonomous
  standing-start backflip.

**2026-08-29 17:30 IST — model 175, recovery starts, seed 42, 64 episodes.**

- Strict stable landing: 0/64; body-only contact: 60/64 (93.75%).
- Longest posture-only hold: mean 0.090 s, maximum 0.22 s.
- Longest strict hold: mean 0.0266 s, maximum 0.08 s.
- Decision: model 50 remains the best verified v26 checkpoint and the published
  videos remain current. Continue the run through the planned spawn-mixture
  transition rather than promoting the numerically newest checkpoint.
- Evidence: `results/research_efgcl_v26_model175_recovery_seed42.json` (local,
  not in the curated public evidence set).

## v27 elevated cube curriculum

**2026-08-29 17:38-17:47 IST — task construction and pre-training baseline.**

Hypothesis: a 0.25 m elevated launch cube supplies additional gravitational
flight time without asking the XL330 actuators for more vertical impulse. A
small asymmetric backward translation is permitted. Success still requires a
genuine airborne backward revolution, first lower-floor recontact on the feet,
clearance beyond the cube edge, and the unchanged continuous 0.5 s strict hold.
This is a curriculum milestone; it does not replace the eventual flat-ground
acceptance gate.

Implementation and falsification details:

- Added a real MuJoCo collision floor plus a literal 0.25 x 0.25 x 0.25 m
  cube in `backflip_pedestal_terrain.py` and registered
  `Mjlab-Backflip-Pedestal-MicroDuck`.
- The first 0.18 m prototype was too narrow: visual replay showed the nominal
  feet on its edges, making an accidental fall dominate before the launch.
  It was rejected before training.
- Standing/crouch slices now reset near the cube's backward edge. Mid-flight
  and recovery slices reset 0.375 m from its center on the lower floor, so no
  landing lesson begins intersecting the cube.
- The EFGCL spotter gains a temporary 5 N body-relative backward component in
  this task only, alongside the existing 16 N lift and 1.40 Nm pitch torque.
  Assistance remains success-annealed; strict evaluation still uses zero.
- Landing latching requires horizontal root distance greater than the 0.125 m
  cube half-width plus 0.04 m margin. Touching down on the cube never counts.
- Focused configuration/state-machine tests: 34 passed. A 16-world,
  one-iteration live simulator smoke run completed without NaN at
  `logs/rsl_rl/microduck_backflip_pedestal/2026-08-29_17-38-22_pedestal-smoke-16`.

Warm-start baseline using v26 model 50, 64 standing starts, seed 42, full
assistance:

- Takeoff 64/64; >=300 degrees 5/64; >=340 degrees 5/64; >=360 degrees 4/64.
- Valid upright feet-first lower-floor landing latch: **3/64 (4.6875%)**.
- Strict 0.5 s stable landing: 0/64; body-only contact: 63/64.
- Maximum rotation 438.91 degrees; maximum peak height 0.5211 m.
- This is the first cube-start result and supplies a nonzero landing signal,
  but it is not success.
- Evidence: `results/research_pedestal_v3_model50_assisted_seed42.json`.

Training command (the checkpoint was copied byte-identically into the target
experiment directory because the runner does not resolve cross-experiment
resume paths; SHA-256 `b74c9ec557cb60a60e494102b7a69b20d7de10bc0f378065442dd2a3e3febac7`):

```bash
WARP_NUM_THREADS=12 .venv/bin/microduck-train \
  Mjlab-Backflip-Pedestal-MicroDuck --gpu-ids None \
  --env.scene.num-envs 256 --env.seed 42 --agent.seed 42 \
  --agent.max-iterations 400 --agent.save-interval 50 \
  --agent.run-name pedestal-v1-model50-warmstart-256 \
  --agent.logger tensorboard --agent.resume True \
  --agent.load-run warmstart-v26-model50 \
  --agent.load-checkpoint model_50.pt
```

Run directory:
`logs/rsl_rl/microduck_backflip_pedestal/2026-08-29_17-46-31_pedestal-v1-model50-warmstart-256`.
The resumed counter begins at iteration 51 and is scheduled through 450.
Checkpoint promotion will use the strict standing-only evaluator, not mean PPO
reward.

**2026-08-29 17:52 IST — pedestal model 100, seed 42, 64 assisted standing starts.**

- Takeoff 64/64; >=300 degrees 6/64; >=340 degrees 6/64; >=360 degrees 5/64.
- Valid upright feet-first lower-floor landing latch: 3/64 (4.6875%); strict
  0.5 s stable landing: 0/64; body-only contact: 64/64.
- Maximum rotation 413.04 degrees. This preserves the landing signal but is not
  an improvement over the warm-start baseline and is not promoted.
- Environment 31 was rendered because it latches a valid lower-floor landing.
  Visual inspection confirms that it stands on the cube, translates clear,
  rotates backward, reaches the feet first, then collapses onto its body. The
  clip is diagnostic evidence of the remaining recovery failure, not success.
- Evidence: `results/research_pedestal_v27_model100_assisted_seed42.json` and
  `results/videos/v27-pedestal-model100-assisted-contact/backflip-model_100-standing-step-0.mp4`.
- Decision: continue through model 150. The independent landing rewards are
  increasing episode length, but the strict reward remains zero, so no
  assistance decay or checkpoint promotion is justified yet.

**2026-08-29 17:56-18:05 IST — pedestal models 150/200/250 and plateau decision.**

| checkpoint | >=340 | landing latch | strict 0.5 s | body-only contact |
|---|---:|---:|---:|---:|
| model 150 | 4/64 | 2/64 | 0/64 | 64/64 |
| model 200 | 5/64 | **5/64** | 0/64 | 64/64 |
| model 250 | 6/64 | 4/64 | 0/64 | 62/64 |

Model 200 is the best cube landing-contact checkpoint, but none improves the
strict hold. At iteration 200 the curriculum changed from 35% to 55% standing
starts and halved preload reward; model 250 still had no stable streak. The run
was stopped at iteration 258 rather than spending another ~15 minutes on a
falsified recovery hypothesis. Evidence is in the corresponding
`results/research_pedestal_v27_model{150,200,250}_assisted_seed42.json` files
(local unless explicitly curated).

## v28 impact-recovery specialist and policy composition

Hypothesis: the launch/flight controller and impact-recovery controller are
distinct skills. Retain pedestal model 200 through the first valid feet contact,
then switch to a specialist trained only to absorb and stabilize the landing.
This is a deterministic state-machine composition, not a relaxed success gate.

The first composition probe used v26 model 50 as the recovery actor. It
activated in 3/64 valid-contact worlds but produced 0 strict holds. Code review
then exposed a distribution error: recovery resets randomized horizontal and
angular velocity but always set vertical velocity to zero, while pedestal
touchdowns are descending impacts.

v28 changes:

- `reset_backflip_state` now accepts a recovery-only vertical-velocity range.
- Specialist stages progress from -0.40 m/s, 5 degree, 0.75 rad/s disturbances
  to -3.25 m/s, 45 degree, 10 rad/s disturbances.
- `eval_backflip.py --recovery-checkpoint` loads a second actor and switches
  each world only after the existing state machine validates a rotated,
  upright, feet-first contact clear of the cube.
- The strict 0.5 s definition, pedestal geometry, actuator limits, and launch
  actor are unchanged.

Training command:

```bash
WARP_NUM_THREADS=12 .venv/bin/microduck-train \
  Mjlab-BackflipRecovery-Flat-MicroDuck --gpu-ids None \
  --env.scene.num-envs 256 --env.seed 42 --agent.seed 42 \
  --agent.max-iterations 400 --agent.save-interval 50 \
  --agent.run-name impact-recovery-v1-model50-warmstart-256 \
  --agent.logger tensorboard --agent.resume True \
  --agent.load-run warmstart-v26-model50 \
  --agent.load-checkpoint model_50.pt
```

Run directory:
`logs/rsl_rl/microduck_backflip_recovery/2026-08-29_18-07-19_impact-recovery-v1-model50-warmstart-256`.

**2026-08-29 18:13 IST — recovery model 100 and composed visual probe.**

- Recovery-only, initial impact stage, seed 42: 59/64 (92.1875%) strict
  0.5-second holds. Median longest strict hold 0.63 s, maximum 1.10 s, no
  nonfinite state. This validates the easy specialist stage only.
- Composed pedestal launch model 200 + recovery model 100: recovery activated
  in 3/64 valid-contact worlds, but strict full-attempt holds remained 0/64.
  The actual touchdown distribution is still outside the specialist's first
  stage; training continues toward the higher-impact stages.
- Environment 31 was rendered as the current composed diagnostic. It stands on
  the cube, clears it, rotates backward and contacts the lower floor, then
  collapses. It is explicitly a failure video, not a successful backflip.
- Evidence:
  `results/research_v28_impact_recovery_model100_seed42.json`,
  `results/research_pedestal_v28_hierarchical_launch200_recovery100_seed42.json`,
  and
  `results/videos/v28-pedestal-hierarchical-launch200-recovery100-env31/backflip-model_200-standing-step-0.mp4`.

**2026-08-29 18:17-18:24 IST — measured impact gap and high-spin restart.**

- Recovery model 150 on the explicit medium profile achieved **63/64 (98.4%)**
  strict holds under downward speeds through -1.25 m/s, tilts through 15
  degrees, and angular speeds through 3 rad/s. Median strict hold was 1.58 s.
- Composing model 150 behind pedestal launch model 200 still produced 0/64
  full strict holds, both with contact-triggered and final-approach switching.
- A selected-world trace of environment 31 measured the actual first contact:
  359.93 degrees rotation, 15.06 degrees tilt, -1.67 m/s vertical speed, and
  **17.58 rad/s angular speed**. The recovery actor's first action arrived on
  the following 20 ms control step. The previous medium maximum of 3 rad/s,
  and even the planned 10 rad/s final stage, did not cover this state.
- Evaluator recovery profiles now make checkpoint difficulty explicit instead
  of silently evaluating every checkpoint on the easy reset distribution.
  Selected-world tracing follows `--render-env-index` rather than always world
  zero. An optional `approach` switch activates at >=330 degrees, <=40 degrees
  tilt, <=0.36 m height while descending; it did not solve the distribution
  gap by itself.
- The original v28 run was stopped at model 200. Its checkpoint was resumed
  with hard/extreme angular-speed limits raised to 12/20 rad/s, bracketing the
  measured touchdown.

High-spin resume command:

```bash
WARP_NUM_THREADS=12 .venv/bin/microduck-train \
  Mjlab-BackflipRecovery-Flat-MicroDuck --gpu-ids None \
  --env.scene.num-envs 256 --env.seed 42 --agent.seed 42 \
  --agent.max-iterations 250 --agent.save-interval 50 \
  --agent.run-name impact-recovery-v2-high-spin-256 \
  --agent.logger tensorboard --agent.resume True \
  --agent.load-run 2026-08-29_18-07-19_impact-recovery-v1-model50-warmstart-256 \
  --agent.load-checkpoint model_200.pt
```

Run directory:
`logs/rsl_rl/microduck_backflip_recovery/2026-08-29_18-23-32_impact-recovery-v2-high-spin-256`.

**2026-08-29 18:29-18:36 IST — post-contact hypothesis rejected; v29 late-flight specialist.**

- High-spin recovery model 250 achieved 63/64 strict holds on the synthetic
  hard profile (up to -2.5 m/s and 12 rad/s), yet its composed pedestal result
  remained 0/64 even with the approach switch.
- The synthetic reset begins at 0.10–0.13 m trunk height in a nominal joint
  pose. The measured real first contact is at 0.247 m with the legs extended.
  Randomizing velocity after placing the robot on the ground therefore does
  not reproduce pre-impact foot placement or contact geometry. The
  post-contact-only specialist hypothesis is rejected despite its high
  in-distribution score.
- Added `Mjlab-BackflipTouchdown-Flat-MicroDuck`: 100% late-airborne RSI,
  ballistic angle/height/spin coupling, body-only-contact termination, zero
  assistance, and unchanged strict landing semantics.
- Initial curriculum: 330–350 degrees, 0.22–0.30 m, 12–18 rad/s. It expands at
  iterations 100 and 250 to 280–355 degrees, 0.20–0.38 m, 10–22 rad/s.
- Focused tests: 35 passed. A 16-world one-iteration live smoke test completed
  without NaNs. The task was warm-started from pedestal launch model 200,
  SHA-256 `1923cca2d3fca768b6d39ffa627fb6a3b1e731b20ef737f21cf463d3d85a1c42`.

Training command:

```bash
WARP_NUM_THREADS=12 .venv/bin/microduck-train \
  Mjlab-BackflipTouchdown-Flat-MicroDuck --gpu-ids None \
  --env.scene.num-envs 256 --env.seed 42 --agent.seed 42 \
  --agent.max-iterations 300 --agent.save-interval 25 \
  --agent.run-name late-flight-v1-pedestal200-warmstart-256 \
  --agent.logger tensorboard --agent.resume True \
  --agent.load-run warmstart-pedestal-model200 \
  --agent.load-checkpoint model_200.pt
```

Run directory:
`logs/rsl_rl/microduck_backflip_touchdown/2026-08-29_18-35-12_late-flight-v1-pedestal200-warmstart-256`.

**2026-08-29 18:39-18:45 IST — v29 braking objective correction.**

- Models 225 and 250 each produced only 1/64 landing latches and 0 strict
  holds on late-flight starts. Mean final rotation changed from 398.81 to
  397.63 degrees, far too slowly.
- Training telemetry showed late-pitch cost near -0.1 episode return while the
  policy could retain long positive survival return after an invalid impact.
  Rotation beyond 360 degrees had no direct cost, and the inherited
  body-contact termination required a valid landing latch before firing.
- Touchdown-only corrections: late-pitch weight -120 with a 300–350 degree
  gate; one-revolution overshoot cost weight -100 with 45 degree scale;
  landing-preparation and approach weights 100; body-contact weight -20; and
  immediate termination after flight-ending body-only contact whether or not a
  valid landing latched. Full-task and standing-evaluation rewards are
  unchanged.
- Regression tests cover the overshoot scale and post-flight termination. The
  updated 16-world smoke test had nonzero overshoot return (-1.4341), zero NaN
  termination, and completed normally.
- Resumed touchdown model 250 under the corrected MDP. Initial telemetry now
  exposes meaningful costs: late pitch -3.35, overshoot -5.40, mean episode
  length 9.35 steps, and mean reward -11.22. This proves failures are no longer
  rewarded by lying on the floor; it does not prove learning.

Resume command:

```bash
WARP_NUM_THREADS=12 .venv/bin/microduck-train \
  Mjlab-BackflipTouchdown-Flat-MicroDuck --gpu-ids None \
  --env.scene.num-envs 256 --env.seed 42 --agent.seed 42 \
  --agent.max-iterations 250 --agent.save-interval 25 \
  --agent.run-name late-flight-v2-braking-256 \
  --agent.logger tensorboard --agent.resume True \
  --agent.load-run 2026-08-29_18-35-12_late-flight-v1-pedestal200-warmstart-256 \
  --agent.load-checkpoint model_250.pt
```

Run directory:
`logs/rsl_rl/microduck_backflip_touchdown/2026-08-29_18-44-16_late-flight-v2-braking-256`.

**2026-08-29 18:46-19:05 IST — v30 checkpoint sweep, three-stage composition, and published cube attempt.**

- The corrected late-flight run was evaluated every 25 iterations from model
  275 through 400. Model 350 was the best strict precursor: 64/64 landing
  latches, 39/64 episodes reaching the low-angular-speed predicate, and a
  maximum strict-landing hold of 0.14 s. Later checkpoints improved the
  isolated low-speed rate (52/64 at model 400) but shortened the best strict
  hold to 0.10 s. No checkpoint met the required 0.50 s hold.
- A three-stage evaluator now supports launch, final-approach, and post-contact
  actors. Its first composition exposed an observation-distribution bug: the
  specialists received the parent episode's absolute phase and active-assist
  context even though both were trained from local phase zero without a
  spotter. The evaluator now clones the observation and rebases actor context
  slots 55 and 56 to specialist-local phase and zero assistance at each
  hand-off.
- Standing cube evaluation after rebasing used seed 42 and 64 worlds. It
  produced 6/64 rotations >=300 degrees, 4/64 valid feet-first landing
  latches, 0/64 strict 0.50 s landings, and no non-finite states. Thus the
  observation fix improved compositional validity but did not solve impact
  stabilization.
- Environment 44 was selected for visual evidence because it reached 359.65
  degrees and a valid feet-first latch. It took off at 0.34 s, crossed 300
  degrees at 0.66 s, activated the touchdown specialist near 340 degrees at
  0.70 s, and contacted at 0.74 s with 12.92 degrees tilt but 14.15 rad/s
  angular speed. It then collapsed forward; the maximum strict hold was 0 s.
- The verified 1280x720, 50 fps, 4.0 s H.264 video is
  `results/videos/v30-three-stage-rebased-env44/backflip-model_200-standing-step-0.mp4`.
  Its per-control-step trace is
  `results/traces/v30-three-stage-rebased-env44.json`. This is a diagnostic
  near-success, not a successful backflip landing and not hardware evidence.

Exact render command:

```bash
WARP_NUM_THREADS=2 .venv/bin/python scripts/eval_backflip.py \
  logs/rsl_rl/microduck_backflip_pedestal/2026-08-29_17-46-31_pedestal-v1-model50-warmstart-256/model_200.pt \
  --task-id Mjlab-Backflip-Pedestal-MicroDuck --num-envs 64 --seed 42 \
  --start-mode standing --assist-scale 1 \
  --recovery-checkpoint logs/rsl_rl/microduck_backflip_touchdown/2026-08-29_18-44-16_late-flight-v2-braking-256/model_350.pt \
  --recovery-switch-mode approach \
  --post-landing-checkpoint logs/rsl_rl/microduck_backflip_recovery/2026-08-29_18-07-19_impact-recovery-v1-model50-warmstart-256/model_150.pt \
  --render-env-index 44 \
  --video-dir results/videos/v30-three-stage-rebased-env44 \
  --trace-output results/traces/v30-three-stage-rebased-env44.json \
  --output results/research_pedestal_v30_three_stage_rebased_env44_render_seed42.json
```

**2026-08-29 19:06-19:12 IST — v31/v32 approach-gate sweep and integrated braking task.**

- A 3x64 standing-cube sweep moved the touchdown hand-off earlier: model 350
  at 320°/60°/0.42 m and 300°/80°/0.45 m, plus model 400 at
  310°/70°/0.45 m. With the inherited assisted residual clamp, all three were
  effectively identical to v30: 0 strict landings, 4/64 landing latches, and
  at most 0.02 s posture hold. This falsified the hypothesis that switching a
  few control ticks earlier was sufficient.
- Control-path inspection found that the evaluator selected the zero-assist
  touchdown actor but `BackflipResidualJointPositionAction` still limited its
  pre-contact targets to 5% authority because the launch episode retained
  assist eligibility. The evaluator now clears that eligibility only when the
  approach hand-off fires. The 3x64 full-authority repeat changed mechanics
  (best maximum contact rotation fell from 416.70° to 395.82° and maximum
  posture hold rose from 0.02 to 0.04 s) but still yielded 0/64 strict holds.
  Earlier model-350 control at 300° reduced landing latches to 2/64, showing
  that an abrupt actor swap also has a transition-distribution cost.
- Added `Mjlab-BackflipBrake-Pedestal-MicroDuck`, an integrated continuation
  inspired by segment sampling and iterative motion imitation in the research
  plan. It protects the assisted launch teacher through 280°, then restores
  full late-flight authority to the same phase-conditioned actor. Reset mix is
  75% cube standing / 25% ballistic late flight, progressing to 100% cube
  standing at iteration 200. Strong late-pitch, overshoot, landing-approach,
  and post-flight body-impact objectives are active from the first update.
- Focused tests: 37 passed. A 16-world warm-start smoke update completed at
  iteration 200/201 with no NaN termination and nonzero late-pitch (-2.5683),
  overshoot (-4.2748), approach (+0.2630), and preparation (+1.1897) returns.
  Mean episode length was 8.60 steps and strict-success return remained zero;
  this proves the intended learning signals execute, not that they solve the
  maneuver.

Planned integrated run command:

```bash
WARP_NUM_THREADS=12 .venv/bin/microduck-train \
  Mjlab-BackflipBrake-Pedestal-MicroDuck --gpu-ids None \
  --env.scene.num-envs 256 --env.seed 42 --agent.seed 42 \
  --agent.max-iterations 300 --agent.save-interval 25 \
  --agent.run-name integrated-late-braking-v1-256 \
  --agent.logger tensorboard --agent.resume True \
  --agent.load-run warmstart-pedestal-model200 \
  --agent.load-checkpoint model_200.pt
```

## Logging protocol for subsequent entries

For each new checkpoint or variant, append:

1. timestamp, run directory, checkpoint and source/diff identity;
2. one falsifiable hypothesis and the exact code/config delta;
3. exact training/evaluation commands, seeds, episode counts, and start mode;
4. takeoff, >=300, >=340, first-contact-feet, landing-latch, strict stable,
   body-contact, nonfinite, rotation and timing evidence;
5. trace/video paths when visual or mechanical diagnosis is used;
6. whether the hypothesis was supported, rejected, or remains uncertain;
7. the next decision and why CPU time is justified.

Training reward is diagnostic only. No run is promoted without strict
standing-start evaluator evidence, and no simulation result is promoted to a
physical-robot claim.
