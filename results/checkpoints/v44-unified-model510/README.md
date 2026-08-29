# v44 unified apex-to-mat checkpoint

This checkpoint is the first reproducible strict success for the
**reverse-curriculum apex-to-compliant-mat stage**. It is not a complete cube
takeoff, flat-ground result, or physical-robot result.

- task: `Mjlab-BackflipReferenceMat-Pedestal-MicroDuck`
- checkpoint: `model_510.pt`
- deterministic acceptance: `WARP_NUM_THREADS=1`, seed 45, 64 worlds
- result: 64/64 true revolutions and feet-first contacts; 2/64 strict 0.5 s
  landings
- best: environment 14, 369.487 degrees, 2.02 s strict hold
- video: `../../videos/v44-unified-model510-strict-seed45-env14-warp1/`
- evaluation: `../../research_reference_mat_unified_v44_model510_64_seed45_env14_warp1_rendered.json`

Reproduce:

```bash
WARP_NUM_THREADS=1 PYTHONPATH=src uv run --python .venv/bin/python --no-sync \
  scripts/eval_backflip.py \
  results/checkpoints/v44-unified-model510/model_510.pt \
  --task-id Mjlab-BackflipReferenceMat-Pedestal-MicroDuck \
  --num-envs 64 --seed 45 --start-mode task --assist-scale 0 \
  --render-env-index 14 \
  --video-dir results/videos/v44-unified-model510-strict-seed45-env14-warp1
```

SHA-256:

```text
acc3b5217c65aa78012317a53ce04be909432bdc885acf5087954afba8e7be71  model_510.pt
c86cc56eac4fa49d5670c790b0b41da8adb6f2d80fc20e2ea316a5ace53189ab  model_510.onnx
a141d6761f9064cfae325cb242877bf854842f2c746a393188f24cb73d689603  env.yaml
1026b14446a5fbc0caf74a7a2fd9c03b075b235b8330e908744c4b986395ff4b  agent.yaml
```
