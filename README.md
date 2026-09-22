<img width="1536" height="1024" alt="ChatGPT Image Sep 22, 2026, 09_34_07 AM" src="https://github.com/user-attachments/assets/d7563b82-43c3-4bd3-850e-07258706fab1" />


# PHI-SPIKE — Cessna 182

## What this is

This repo trains and evaluates PHI-SPIKE — a physics-informed spiking
neural network for aircraft fault diagnosis — on a JSBSim-simulated
Cessna 182.

## What we used it for

This is the **primary debugging ground** for the whole PHI-SPIKE
project. Every core numerical-stability fix in the codebase — the
JSBSim trim bug, the LIF layer shape bug, the membrane-potential
overflow, the gradient explosion — was found, diagnosed, and fixed
*here*, on this aircraft, before being ported to the other airframes
(`pa28`, `c172x_retrim`). If you're looking for the most
battle-tested, most thoroughly verified version of this pipeline,
it's this one.

The research question behind it: does coupling a physics-consistency
residual *into a spiking neuron's membrane dynamics* — not just into
the training loss — produce a meaningfully better, more spike-efficient
fault classifier than treating physics as an ordinary regularizer?
Five ablation variants isolate that question:

| Variant | Physics in membrane dynamics | Physics loss | Temporal loss | Learnable gate |
|---|:---:|:---:|:---:|:---:|
| `vanilla_snn` | – | – | – | – |
| `snn_physics_loss` | – | ✓ | – | – |
| `snn_membrane_conditioning` | ✓ | ✓ | – | – |
| `snn_temporal_physics` | ✓ | ✓ | ✓ | – |
| `full_phi_spike` | ✓ | ✓ | ✓ | ✓ |

10-class fault taxonomy: `healthy`, `sensor_bias`, `sensor_drift`,
`sensor_stuck`, `sensor_noise`, `actuator_degradation`,
`engine_performance_loss`, `control_surface_lag`, `fuel_flow_anomaly`,
`vibration_anomaly`.

## Verified trim envelope

2,000–6,000 ft altitude, 85–115 kt airspeed. Outside this range,
JSBSim's trim solver may fail or produce physically invalid telemetry.
This is the widest, most forgiving envelope of the three aircraft in
this project — c182 was the easiest of the three to get working
cleanly.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install numpy==2.4.6 scipy==1.16.3 pandas==2.3.3 jsbsim==1.3.1 \
            tqdm==4.67.1 pyarrow==22.0.0 scikit-learn==1.8.0 xgboost==3.2.0
```
Developed and verified against Python 3.11.14.

## Quickstart

Always smoke-test before a real run — this repo's own history is the
reason why (see below).

```bash
python -m training.train_campaign \
  --epochs 3 --batch-size 16 --device cpu --seeds 0 \
  --variants vanilla_snn --n-clean 6 --sequence-len 40 --aircraft c182 \
  --save-dir experiments/smoke_test
```

Real campaign:
```bash
python -m training.train_campaign \
  --epochs 40 --batch-size 32 --device cpu --seeds 0,1,2,3,4 \
  --variants vanilla_snn,snn_physics_loss,snn_membrane_conditioning,snn_temporal_physics,full_phi_spike \
  --n-clean 40 --sequence-len 80 --aircraft c182 \
  --save-dir experiments/full_campaign/c182
```
Launch with `nohup ... &; disown`, not `tmux` — see below.

## The debugging history (why this repo exists in its current form)

This pipeline did not work cleanly on the first attempt. In order:

1. **JSBSim trim failed silently.** `propulsion/engine/set-running` is
   not a real JSBSim property — the write silently succeeds but reads
   back `0.0`. The engine never actually starts, so `do_trim()` has no
   thrust to balance and fails. Fixed with the correct property:
   `propulsion/set-running = -1`.

2. **Trim was discarded on the first control input.** Even after fix
   #1, setting `fcs/elevator-cmd-norm` etc. to a raw value (instead of
   trim-value + delta) threw away the trim solution immediately,
   reproducing the same aircraft-tumbling telemetry trimming was
   supposed to prevent.

3. **The LIF layer silently scrambled batch and time.**
   `LIFLayer.forward()` had an auto-transpose heuristic that fired on
   nearly every real batch, undoing the caller's already-correct
   `(T, B, F)` tensor layout. Symptom: training stuck near
   majority-class accuracy for many epochs. Fixed by removing the
   heuristic.

4. **Membrane potential had no upper bound.** Neither `LIFLayer` nor
   `PhysicsConditionedLIF` clamped their internal state. At full
   training scale (`n_clean=40`, ~10,000–20,000 batches/epoch), this
   compounded to numerical overflow — confirmed by a climbing
   `[skipped N non-finite batches]` count that reached 100% of every
   batch by epoch 2. Fixed with `torch.clamp(..., -50.0, 50.0)` on
   `i_syn` and `v` every timestep.

5. **Gradient norm-clipping alone wasn't enough.** Direct
   instrumentation caught raw gradients reaching into the *trillions*
   before clipping — occasionally overflowing outright, which
   norm-clipping can't catch after the fact. Fixed by adding
   `clip_grad_value_(..., 5.0)` before the existing norm clip.

Fixes 3–5 together were confirmed — via real training runs, not
assumption — to eliminate the crash entirely: zero non-finite batches
across all 5 variants, 8 epochs each, at the scale that previously
locked up by epoch 2.

## Before you trust any result from this repo

- Run the 3-epoch smoke test above first. `[skipped N non-finite
  batches]` should read `0` or not appear. If it climbs epoch over
  epoch, stop — do not proceed to a full campaign.
- Full scale (`n_clean=40`) is memory-heavy — confirmed to OOM-kill on
  a <8GB box (verified via `dmesg`, not hypothetical). Check `free -h`
  an hour or two into a real run.
- Use `nohup ... &; disown` for long runs. Both an entire tmux server
  dying and multiple accidental `Ctrl+C` kills cost real time during
  development — `nohup` has no equivalent failure mode.

## Repository structure

```
simulation/       JSBSim aircraft wrapper, trim, telemetry generation
fault_injection/  10-class fault taxonomy, randomized onset/severity injection
physics/          Physics-consistency residual computation
encoding/         Telemetry -> spike encoding
models/
  vanilla_snn/    Baseline LIF network (no physics coupling)
  phi_spike/      Physics-conditioned LIF + the 4 physics-aware variants
datasets/         Dataset assembly, envelope config, windowing
training/         train_campaign.py -- main entry point
baselines/        Non-spiking comparison models
robustness/       Noise/dropout/severity robustness sweeps
edge_emulation/   Streaming inference latency benchmarks
evaluation/       Metrics aggregation, figures
verification/     Pipeline sanity checks
dashboard/        Results visualization server
```

## License

JSBSim is LGPL-licensed — check compatibility before choosing a
license for this repository if redistributing.

## Citation

_(add once published)_
