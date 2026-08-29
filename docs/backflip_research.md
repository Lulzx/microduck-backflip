# Research plan: standing MicroDuck backflip

## Objective and present evidence

The acceptance objective is one uninterrupted standing-start maneuver:
supported preload, collision-free takeoff, at least 340 degrees of backward
airborne rotation, feet-first contact, and 0.5 seconds of stable upright hold.
Reverse-curriculum starts and assisted trajectories are training mechanisms,
not successful evaluations.

The best strict checkpoint before this research pass (`model_1175`, seed 42,
128 episodes) takes off in every episode but reaches at most 274.0 degrees. Its
mean takeoff vertical speed is 0.774 m/s, mean takeoff backward pitch rate is
9.13 rad/s, and mean uninterrupted flight is 0.147 s. Every episode eventually
has body-only ground contact; only 2.34% make feet the first contacting part.
This localizes the failure before landing control: the robot needs more flight
time and earlier angular impulse, then a distinct extension/recovery phase.

## First-principles decomposition

1. **Feasibility.** Confirm that available joint travel, actuator torque, and
   battery/power limits can generate the required vertical and angular impulse.
2. **Exploration.** Let PPO experience complete trajectories early enough for
   its critic to assign useful values to preload, push, tuck, and extension.
3. **Launch.** Couple vertical impulse and backward angular impulse while the
   feet still support the body; neither a straight jump nor a fast ground roll
   is useful.
4. **Flight.** Reduce pitch inertia early by tucking, preserve angular momentum,
   and begin extension before recontact. A mild corkscrew is acceptable, but
   pitch credit remains attenuated as the lateral axis leaves the sagittal
   plane.
5. **Landing.** Make feet-first contact a separate phase, then dissipate angular
   energy without head/trunk contact.
6. **Robustness and transfer.** Only after nominal strict success, add mass,
   friction, actuator, latency, and sensor uncertainty and distill to deployable
   observations.
7. **Acceptance.** Assistance is always zero in strict batteries and videos.

## Primary research and implications

- **EFGCL (2026)** applies a heuristic external force during early training,
  waits for a 60% success rate, and reduces assistance in 0.01 increments. PPO
  alone failed its backflip and lateral-flip tasks; assisted curricula learned
  them. The force did not need precise tuning. This directly targets our
  exploration and flight-time failures without requiring a reference motion.
  <https://arxiv.org/abs/2605.10063>
- **ZEST (2026)** independently combines a model-based assistive wrench with
  difficulty-adaptive sampling. It reports a continuous Spot backflip and finds
  residual reference actions, privileged critic state, actuator modeling, and
  per-segment failure sampling important. This is the next step if a simple
  spotter discovers a flip but not a reliable landing.
  <https://arxiv.org/abs/2602.00401>
- **Spatio-Temporal Motion Retargeting (2024)** first makes a reference
  kinematically feasible, then adjusts timing under dynamics constraints before
  RL tracking; it deployed backflips on four quadruped morphologies. This argues
  against asking a tracking policy to repair an infeasible hand-authored pose
  sequence.
  <https://arxiv.org/abs/2404.11557>
- **BeyondMimic (2025)** uses compact Cartesian tracking rewards, adaptive phase
  sampling, and termination around unrecoverable tracking errors for agile
  motion. If spotter-only training plateaus, the fallback should be an optimized
  reference and phase-conditioned residual tracker, not more scalar reward
  terms.
  <https://arxiv.org/abs/2508.08241>
- **ASAP (2025)** learns a residual action model from real trajectories and
  fine-tunes the simulator policy with that mismatch model. This is relevant
  only after simulated success and instrumented, safety-tethered hardware data;
  broad domain randomization alone can make an agile policy conservative.
  <https://arxiv.org/abs/2502.01143>
- **Stage-Wise Reward Shaping for Acrobatic Robots (2024)** models a backflip
  explicitly as Stand-Sit-Jump-Air-Land and changes objectives at physical
  transitions: loss of foot support enters Air and first foot contact enters
  Land. In Land it optimizes base height, balance, low base velocity, and pose
  as separate objectives, while treating body collision and hardware limits as
  constraints. This is directly relevant to our multiplicative landing score,
  whose gradient collapses when any one factor is poor.
  <https://arxiv.org/abs/2409.15755>
- **Flip Stunts using Iterative Motion Imitation (2026)** uses 50% Reference
  State Initialization, holds the phase at its final value after touchdown,
  and activates still-balance terms after tracking ends. It recursively turns
  each policy rollout into the next, more dynamically feasible reference and
  reports 93% success. Critically, it tightens touchdown velocity, power,
  torque, joint, and collision constraints only after exploration succeeds.
  <https://arxiv.org/abs/2603.27944>
- **LineRides (2026)** avoids exact timing: a spatial guideline plus a sequence
  of key orientations pays monotonic progress toward 90, 180, 270, and 360
  degree attitudes. It also deliberately specifies a non-level landing pitch
  to make one contact arrive first and absorb impact. This supports the user's
  allowance for an asymmetric, slightly axial MicroDuck flip rather than
  forcing bilateral symmetry.
  <https://arxiv.org/abs/2605.05110>

## Ordered experiments

1. Add a success-gated virtual spotter. For the 0.737 kg robot, raising takeoff
   from about 0.8 to 1.5 m/s in 0.10 s requires roughly 5.2 N of extra upward
   force. Its nominal pitch inertia is about 0.0050 kg m2, so adding 4 rad/s in
   the same interval requires about 0.20 Nm as a small corrective impulse. An
   open-loop randomized sweep showed that this was not enough to expose full
   trajectories. A 16 N and 1.40 Nm pulse from 0.30--0.40 s produced 100%
   takeoff, 100% >=300 degrees, 62.5% >=340 degrees, 84.4% feet-first first
   contact, and 57.8% landing latches over 128 nominal-PD trials. Use this as
   the initial spotter, expose bounded time and assist scale to the policy, and
   decay only after a 60% landing rate in an evaluation window.
   While spotting is active, multiply takeoff, vertical-speed, preload, and
   launch-quality shaping by `1 - assist_scale`. Otherwise PPO is paid for
   impulse supplied by the external wrench and can maximize return while
   counteracting the desired flip. Rotation, tuck, feet-first landing, and
   stable recovery remain fully rewarded at every curriculum stage.
2. Train from scratch or reset observation-normalizer statistics because the
   previously zero command slots now carry phase and assistance. Evaluate every
   saved checkpoint with assistance forced to zero.
   The first progress-only checkpoint (`v19/model_25`, 64 trials) validated the
   new objective boundary: assisted takeoff was 100%, 40.6% exceeded 300
   degrees, 12.5% exceeded 340 degrees, and 96.9% made a foot the first ground
   contact, while all launch-shaping returns stayed zero. Strict takeoff was
   still 0%. By `model_50`, 96.9% exceeded 360 degrees but mean rotation had
   grown to 558 degrees and stable landing remained 0%. Therefore preserve the
   open-loop-validated nominal PD pose in proportion to assist scale and tax
   backward pitch rate only after 300 degrees; both constraints leave an
   asymmetric/corkscrew solution available.
   The scalar prior alone later became bimodal (roughly 40 or 500--600
   degrees). Following ZEST's residual-controller structure, the next stage
   directly bounds policy target displacement to 20% at full assistance and
   restores it linearly to 100% at zero assistance. A deterministic untrained
   checkpoint under that residual action geometry already gives 100% takeoff,
   98.4% >=300 degrees, 56.3% >=340 degrees, 81.3% foot-first initial contact,
   and 51.6% landing latches over 64 randomized trials. That is the useful
   teacher distribution; PPO's remaining job is late-flight braking/recovery
   and then replacing the wrench as authority grows.
   In the first residual run, assisted checkpoint 25 reached 82.8% >=340
   degrees and 39.1% landing latches, but still 0% 0.5-second stable holds; by
   checkpoint 75 it had regressed to overspin. Episode accounting showed about
   16 units for rotation versus only 0.2--0.5 for post-contact standing.
   Recovery must therefore be lexicographically dominant: increase the latched
   upright-standing and stable-success weights by 10x while leaving rotation
   bounded after one revolution.
   The curriculum originally counted the landing latch while strict evaluation
   required a continuous 0.5-second hold. These are now the same state-machine
   event: 25 consecutive 50 Hz control steps with feet contact, >=340 degrees,
   trunk height >=0.095 m, <20 degrees tilt, and <2 rad/s angular speed. Only
   this stable latch can decay assistance. Reverse mid-flight worlds receive
   full residual action authority because they receive no external wrench, and
   their early sampling share is 30% while recovery is the bottleneck.
   An episode-wide 20% residual limit still let PPO corrupt the teacher before
   contact while restricting recovery afterward. The residual is therefore
   phase-specific: 5% authority for assisted launch/flight, then 100% on the
   first control step after recontact. Unassisted reverse states and strict
   evaluation always have 100% authority.
   Add a dedicated 20% early recovery slice initialized upright on the feet
   with bounded tilt and residual linear/angular velocity. Its landing latch is
   seeded but its 0.5-second stability latch is not, it receives full action
   authority, and it is excluded from assistance-decay statistics. This makes
   impact recovery directly learnable before requiring a complete approach.
   The first recovery-slice run did not improve the strict endpoint: models
   25, 50, and 75 all had 0% stable holds over 32 recovery starts. At model 50,
   a deterministic trace showed the policy immediately generating 8.6 rad/s
   yaw, losing foot-only support around 0.38 s, and settling at 0.060 m on its
   body. This is not an exploration shortage; it is a post-impact objective
   problem. Following the stage-wise and IMI evidence, split the post-contact
   product into independent upright, height, angular-stillness, and foot-support
   rewards. Add one-shot frontier credit for each newly achieved control step
   of the 25-step stable hold, while preserving the strict 25-step success
   latch and success-gated assistance decay.
3. If strict rotation remains below 300 degrees, sweep assist timing/impulse and
   inspect whether the policy learns to replace the external impulse as it is
   removed. Reject a curriculum that succeeds only while assisted.
4. If rotation crosses 340 degrees but feet-first landing is poor, introduce an
   optimized five-phase reference (stand, crouch, launch, tuck, extend) and train
   a phase-conditioned residual tracker with adaptive sampling of the failure
   bins.
5. Only after robust nominal success, fit actuator/power/delay parameters and
   run backlash plus multi-seed randomized batteries before any physical trial.
