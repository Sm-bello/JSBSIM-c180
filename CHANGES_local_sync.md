# Local workspace sync — cloud fixes applied

Audited this local zip against every fix verified on the cloud today.

## Already present (no action needed)
- JSBSim trim fix (propulsion/set-running + do_trim + trim-relative control)
- LIF layer shape bug fix (auto-transpose heuristic already absent)
- Per-aircraft envelope/validity gate in datasets/factory.py (c182, pa28 entries present)

## Was missing, now applied
1. Membrane potential / synaptic current clamp -- neither
   models/vanilla_snn/lif_snn.py nor models/phi_spike/phi_snn.py had
   this (lif_snn.py had 0 clamps, phi_snn.py had only the original
   pre-existing residual_true clamp). Added torch.clamp(..., -50.0,
   50.0) on i_syn and v every timestep, both files -- matches the
   cloud-verified fix exactly.
2. Gradient value-clipping in training/train_campaign.py -- only
   clip_grad_norm_ was present. Added clip_grad_value_(..., 5.0)
   immediately before it.

## Verified after patching (this workspace, not assumed from elsewhere)
Syntax clean on all 3 files. Fix counts confirmed: 2 clamps in
lif_snn.py, 4 in phi_snn.py, 2 grad-clip calls in train_campaign.py --
matching the cloud-verified pattern exactly.

Real training run, n_clean=8, 3 epochs, aircraft=c182:
```
vanilla_snn     skips_per_epoch=[0, 0, 0]  test_acc=0.5754  test_f1=0.2095
full_phi_spike  skips_per_epoch=[0, 0, 0]  test_acc=0.5864  test_f1=0.2147
```
Zero non-finite batches, both variants, real learning.

## Not yet done
- Not verified at full scale (n_clean=40) on this specific local copy.
  Run the standard 3-epoch smoke test at full scale before committing
  to a real campaign, same as every other repo in this project.
