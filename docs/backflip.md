# MicroDuck backflip

`Mjlab-Backflip-Flat-MicroDuck` is an episodic standing backflip: leave the
ground, rotate backward through one revolution in the sagittal plane, touch
down on the feet, and settle upright. It is not considered trained merely
because PPO reward rises.

The continuously updated hypotheses, commands, failures, checkpoint metrics,
and decisions are recorded in the
[backflip experiment log](backflip_experiment_log.md). Paper-derived design
rationale is kept separately in [backflip research](backflip_research.md).

## What counts

The task state machine requires all of the following, in order:

1. The feet supported the robot at the start.
2. The whole robot became collision-free above 0.135 m trunk height.
3. Backward pitch accumulated during one uninterrupted flight. The first
   ground recontact freezes the rotation frontier, so a bounce cannot be
   combined with a second hop. Ground rolls receive zero rotation credit,
   forward rotation cannot advance the frontier, and corkscrew rotation is
   continuously attenuated.
4. At least 320 degrees of airborne backward rotation occurred before an
   upright feet contact can latch a landing phase. This latch grants no final
   landing reward and cannot pass evaluation until the frozen airborne
   frontier reaches a true 360 degrees.
5. Evaluation requires a true 360-degree airborne revolution and then the
   robot to remain continuously on its
   feet for 0.5 seconds, within 20 degrees of upright, above 0.095 m trunk
   height, and below 2 rad/s angular speed. A transient feet touch followed by
   a fall is not a stable success.

Supported crouch and mid-flight reset states are reverse-curriculum training
data: the former teaches preload-to-launch extension and the latter teaches
the landing half. They never count as full evaluation episodes;
`scripts/eval_backflip.py` forces ordinary standing starts.

### Elevated cube curriculum

`Mjlab-Backflip-Pedestal-MicroDuck` starts ordinary trials on a 25 cm collision
cube above a continuous lower floor. The extra drop increases available flight
time. A landing counts only after the robot clears the cube by 4 cm, contacts
the lower floor feet-first after the required rotation, and satisfies the same
0.5-second stability gate. Mid-flight and recovery lessons spawn beside the
cube on the lower floor; they cannot intersect or land on the launch surface.

This task deliberately permits a small asymmetric backward translation. Its
virtual spotter adds a temporary 5 N backward force during discovery. The
force is annealed with the other assistance, and evaluation can set the entire
spotter scale to zero. A pedestal result is reported separately and is not
called a flat-ground backflip.

`Mjlab-BackflipTouchdown-Flat-MicroDuck` isolates the measured remaining
failure without skipping impact physics. It resets while still airborne late
in the revolution, then requires the policy to brake, choose its landing leg
configuration, contact the feet, absorb the impact, and hold. Its curriculum
expands from 330–350° at 12–18 rad/s to 280–355° at 10–22 rad/s. This differs
from a post-contact recovery reset, which cannot teach pre-impact foot
placement and was shown not to transfer to the actual cube touchdown.

`Mjlab-BackflipBrake-Pedestal-MicroDuck` is the integrated fine-tuning stage.
It warm-starts the cube launch actor, protects the assisted nominal-PD teacher
through 280°, then restores full action authority for extension, braking, and
impact absorption. Its reset mix starts at 75% complete cube trajectories and
25% late-flight lessons, becomes 100% cube starts by iteration 200, and uses
the touchdown specialist's late-spin, overshoot, preparation, and body-impact
objectives. This avoids both failure modes observed in policy composition:
synthetic post-contact states omit the real leg geometry, while a separate
late-flight actor cannot repair a launch trajectory if the assisted residual
clamp still limits it to 5% authority.

The current discovery task also uses an EFGCL-style virtual spotter: a 16 N
upward force and 1.40 Nm backward-pitch torque from 0.30 to 0.40 seconds. Its
scale drops only after a 60% strict landing rate over an evaluation window.
The policy observes bounded episode phase and assistance scale in two existing
body-command slots, preserving the 61-D network shape. Strict evaluation sets
the assistance scale to zero. Launch shaping is multiplied by
`1 - assist_scale`, so the actor cannot receive credit for force supplied by
the spotter; rotation and landing rewards remain active. See
[the research rationale](backflip_research.md).

The assisted phase also penalizes action magnitude in proportion to the
spotter scale, because zero action tracks the nominal PD pose that produced a
57.8% landing latch in the open-loop sweep. A separate pitch-rate cost activates
only after 300 degrees to teach extension/braking without weakening takeoff.
The action term additionally treats PPO as a bounded residual controller: it
has 5% target-displacement authority during fully assisted launch/flight,
switches to 100% immediately after recontact for recovery, and linearly regains
100% pre-contact authority as assistance reaches zero. Strict evaluation is
therefore the ordinary full-authority policy, not a permanently constrained
controller.
The integrated pedestal-braking variant additionally releases full authority
after 280°; the spotter pulse has already ended by then, so this changes only
the actor's ability to brake and place its feet before impact.

`Mjlab-BackflipReference-Pedestal-MicroDuck` is the iterative-reference
landing stage. Its resets sample complete `qpos`/`qvel` states captured when
the integrated model-225 cube actor crossed 260°, with root position made
terrain-origin-relative and the actor's phase observation continued from the
captured 0.56–0.70 s launch time. This preserves the actual joint geometry,
momentum, and randomized launch distribution while making the same difficult
approach repeatable. Scores from this task are RSI diagnostics, never complete
standing-start backflips; learned continuation weights must be returned to the
full cube task and pass its standing battery.
Unassisted reverse-curriculum worlds retain full authority regardless of the
global stage. Assistance can decay only after the same continuous 0.5-second
stable hold used by the evaluator; a one-frame feet touch is not curriculum
success.

`Mjlab-BackflipReferenceEarly-Pedestal-MicroDuck` moves the reference slice
back to the real launch actor's 180-degree apex crossing, giving the policy
about 0.27 s to untuck and remove angular energy. The mat variant lands onto a
raised 18 cm compliant surface, 7 cm below the cube top. This is an explicitly
easier curriculum milestone, not a replacement for the cube-to-floor or flat
task.

`Mjlab-BackflipReferenceMatLanding-Pedestal-MicroDuck` samples exact
full-revolution foot-contact simulator states and trains only the recovery
half. It restores action history and uses a local phase. Checkpoint 530 has
demonstrated the strict 0.5 s hold on 2/64 captured-touchdown trials, but that
does not count as a complete flip. `Mjlab-BackflipReferenceMatMixed-Pedestal-
MicroDuck` trains one actor on a 50/50 mixture of the real apex states and
those exact touchdown states so approach and recovery gradients can be
distilled into a single policy before returning to full cube starts.

Current verified curriculum milestone: mixed-training checkpoint 510,
evaluated with `WARP_NUM_THREADS=1`, has 2/64 strict successes at seed 45.
The best completes 369.49 degrees and holds every landing predicate for 2.02
seconds. This is reproducible apex-to-compliant-mat evidence. It is explicitly
not a complete takeoff from the cube, an unassisted flat-ground result, or
physical-robot evidence.

## Reproduce training

First run the CPU tests and the mandatory small smoke test:

```bash
uv sync
uv run --with pytest python -m pytest tests/
CUDA_VISIBLE_DEVICES='' uv run microduck-train Mjlab-Backflip-Flat-MicroDuck \
  --gpu-ids None --env.scene.num-envs 64 --agent.max-iterations 5 \
  --agent.logger tensorboard
```

For CUDA training:

```bash
uv run microduck-train Mjlab-Backflip-Flat-MicroDuck \
  --env.scene.num-envs 4096 --agent.max-iterations 5000
```

Apple Silicon note: PyTorch can expose the Metal/MPS device, but this task's
physics runs through NVIDIA Warp. Warp 1.12 in the reproducible environment
accepts CPU and CUDA devices only; constructing `ManagerBasedRlEnv` with
`device="mps"` fails with `ValueError: Invalid device identifier: mps`.
Moving only PPO to MPS is not a useful shortcut: in the measured 256-env CPU
run, physics collection takes about 8.1 seconds per iteration while learning
takes about 0.23 seconds.

The experimental CuMetal probe is a narrower, measured result. It captures a
real Warp-generated CUDA `add_one` kernel, compiles the selected forward entry
through `/Users/lulzx/work/cumetal`, and executes it on Metal. On the Apple M4
Pro, numerical readback was `[0,1,2,3,4,5,6,7] -> [1,2,3,4,5,6,7,8]`. Reproduce
the compiler boundary with:

```bash
CUMETAL_CACHE_DIR=/tmp/cumetal-warp-cache \
  uv run python scripts/probe_cumetal_warp.py \
  --output-dir results/cumetal_warp_probe --arch 80
```

This does **not** yet make `wp.get_devices()` expose a CuMetal device or move
MuJoCo Warp physics to the Apple GPU. That requires a CuMetal-backed Warp core,
module/launch integration, and shared-memory Torch interop. Until those runtime
pieces are implemented and the full simulator passes numerical parity tests,
local training remains CPU-backed; the probe must not be described as
GPU-accelerated backflip training.

The unambiguous `microduck-train` executable is intentional. Both this project
and the `mjlab` dependency publish an executable named `train`, and a fresh uv
environment can select the dependency's entry point, which does not understand
`--hf-jobs`.

Evaluate a local checkpoint on domain-randomized standing starts:

```bash
uv run python scripts/eval_backflip.py \
  logs/rsl_rl/microduck_backflip/<run>/model_5000.pt \
  --num-envs 128 --seed 42 \
  --output results/backflip_eval_seed42.json
```

For the cube curriculum, select the task explicitly:

```bash
uv run python scripts/eval_backflip.py \
  logs/rsl_rl/microduck_backflip_pedestal/<run>/model_450.pt \
  --task-id Mjlab-Backflip-Pedestal-MicroDuck \
  --num-envs 128 --seed 42 --start-mode standing --assist-scale 0 \
  --output results/pedestal_eval_seed42.json
```

Before export, require three 128-episode batteries with seeds 42, 123, and 2026:

- takeoff rate at least 99%;
- at least 95% reach 340 degrees of airborne backward rotation;
- at least 90% achieve a stable landing;
- no NaN termination;
- no obvious head, trunk, or shoulder landings on recorded rollouts.

Then repeat on `Mjlab-Backflip-Flat-Backlash-MicroDuck`; require at least 80%
stable landings. These are project acceptance gates, not proof that the real
robot will succeed.

## Export and daemon rehearsal

Always export through the repository script so the observation normalizer is
baked into ONNX:

```bash
uv run python scripts/export.py Mjlab-Backflip-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/microduck_backflip/<run>/model_5000.pt \
  --num-envs 1 --onnx-file backflip.onnx
```

Rehearse the daemon observation/action contract in CPU MuJoCo, using the
existing episodic `roulade` slot and `R` trigger:

```bash
uv run python scripts/infer_policy.py \
  --walking walk.onnx --standing stand.onnx \
  --roulade backflip.onnx --roulade-duration 2.0 --new-cmd-obs
```

Do not copy the ONNX to a robot until this rehearsal loads a 61-input,
14-output model and visually repeats the accepted behavior.

## Physical-robot safety gate

A backflip is a high-impact maneuver. Simulation, ONNX export, or a daemon
load is not hardware validation. The first physical attempt must be treated as
an experiment with an operator able to cut torque immediately.

Prerequisites:

- inspect feet, shells, fasteners, servo horns, cable routing, battery
  retention, and neck/head clearances;
- use a fully charged healthy battery and record voltage, servo temperature,
  current, IMU, commanded targets, and loop deadline counters;
- use a clear padded area with no people, animals, furniture, or hard edges in
  the fall envelope;
- use an overhead fall-arrest line that cannot enter the joints, with enough
  slack not to assist the jump but short enough to prevent a head-first floor
  impact;
- have one operator on the trigger and another on an immediate torque-off;
- begin with a single attempt, then inspect hardware and logs before any
  repeat. Never chain the trigger during bring-up.

Runtime compatibility must be resolved explicitly. The training policy is
unfiltered, while the current `microduck` daemon applies its global head/leg
low-pass to every network, including the `roulade` slot. Do one of the
following before hardware use:

1. add and test a dedicated unfiltered backflip/episodic slot in `robotd`; or
2. for an isolated padded test configuration, set `head_lowpass = 1.0` and
   `legs_lowpass = 1.0`, and do not use that configuration for ordinary
   walking policies trained with filtering.

The daemon's limp-fall predictor already excludes a busy roulade window; that
exclusion must remain active for the full backflip duration or it will cut gain
mid-rotation. Set the episodic duration from measured simulation timing with a
small landing margin, not by guessing. Keep the normal joint clamps, bus error
handling, deadman, battery shutdown, and telemetry enabled.

Stop after any body/head contact, cable snag, missed control deadline, bus
error burst, unexpected current/temperature rise, damaged part, or behavior
outside the simulation envelope. A failed attempt is not a reason to increase
action scale or gain without a new simulation experiment.

Only a witnessed, logged physical trial can support a statement that the real
MicroDuck performed a backflip. Until then, report simulation success only.
