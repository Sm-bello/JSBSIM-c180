# Cloud run config — c182 (Cessna Skylane)

Verified this session:
- Trim: reliable within altitude 2000-6000ft, airspeed 85-115kt — wider
  and easier than pa28.
- 60s hold test at 4000ft/105kt: altitude drifted 0.1ft over 60s, pitch/roll
  did not move after settling. Cleanest result of any aircraft checked
  this session.
- `generate_clean_trajectories(aircraft='c182')` verified 10/10 through
  the validity gate after the envelope patch (see CHANGES_trim_fix_pass.md).

## Cloud command
```bash
cd ~/PHI-SPIKE
source .venv/bin/activate
python -m training.train_campaign \
  --epochs 40 --batch-size 32 --device cpu --seeds 0,1,2,3,4 \
  --variants vanilla_snn,snn_physics_loss,snn_membrane_conditioning,snn_temporal_physics,full_phi_spike \
  --n-clean 40 --sequence-len 80 \
  --aircraft c182 \
  --save-dir experiments/full_campaign/c182
```

## Before you launch the real campaign
Same recommendation as pa28 — run --quick first:
```bash
python -m training.train_campaign --quick --aircraft c182 \
  --save-dir experiments/smoke_test/c182
```

## Still open
- Same caveat as pa28: only the dataset factory got the per-aircraft
  envelope fix. Robustness/fault-severity scripts haven't been checked
  against c182's envelope specifically, though c182's wider range makes
  a mismatch less likely than pa28's.
