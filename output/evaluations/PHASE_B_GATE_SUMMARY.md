# Phase B gate summary (2026-08-18)

Phase B rebuilt the MS data representation and the objective's regularization,
then re-ran the preregistered calibration gate. **The gate does not pass.** It
also does not reproduce the Phase A failure mode: the reason is now a compute
budget too small to demonstrate benefit, not a representation that cannot work.

All numbers: 10k-molecule store (8000 train / 1000 val / 1000 test molecules),
800 labeled molecules, 5 probe seeds, converged L-BFGS probes, `general`
embedding, compared against a random initialization of the identical encoder.

## What changed

Absolute 0-1000 Da m/z grid; precursor (exact mass) removed from the grid and
from `continuous_metadata`; technique-aware per-sample augmentation; masking
ratio applied to informative patches only; general variance/covariance
ungated; ¹H chemical-shift scale corrected to 12 ppm; anti-aliased NMR
decimation. Stores rebuilt (`pilot-phaseb-10k`, `massbank-phaseb`);
`chemistry_weight` set to 0 so the run is purely self-supervised.

## Gate criteria

| criterion | result |
|---|---|
| MS lift meaningfully above prevalence floor | **fail** — MS+ 0.067, MS− 0.079 vs random 0.064, 0.080 |
| MassBank transfer above random, CI excluding zero | **fail** — all four strata cross zero |

MassBank, epoch 12 vs random: MS+ **+0.0025** [−0.0022,+0.0069] and **+0.0033**
[−0.0007,+0.0072] (scaffold-unseen); MS− −0.0035 and −0.0042, both crossing
zero. At 2 epochs all four strata were *significantly below* random; at 12
epochs MS+ has turned positive and MS− is no longer significantly negative.

## The decisive evidence: an epoch trend, not a verdict on the fixes

Mean macro-AUPRC across five families against the random control:

| epoch | steps | mean | delta vs random |
|---|---|---|---|
| random | 0 | 0.1891 | — |
| 2 | 666 | 0.1798 | −0.0093 |
| 4 | 1332 | 0.1778 | −0.0112 |
| 8 | 2664 | 0.1836 | −0.0054 |
| 12 | 3996 | 0.1852 | **−0.0038** [−0.0083,+0.0002] |

Monotonic improvement after epoch 4, still climbing at the end, mean CI now
touching zero. Per family at epoch 12: ¹H NMR **+0.0044 above random**
(CI excludes zero), ¹³C +0.0031 and MS+ +0.0003 (ties), MS− −0.0035, and IR
**−0.0235 below**. Phase A's 40,000-step model beat random by +0.027 on IR and
+0.033 on ¹³C, so the objective does surpass random given roughly ten times
this budget. A 4,000-step calibration cannot settle whether Phase B helps.

## Findings that stand on their own

1. **The mass shortcut is gone, and it was load-bearing.** MS+↔MS− batch
   retrieval recall@1 fell from 0.708 (Phase A) to 0.000. That retrieval was
   solved by molecular mass, not by spectral content.
2. **Formula distillation was silently preventing rank collapse.** Removing it
   for a clean self-supervised test exposed the collapse: with
   `variance_weight=0` the `general` embedding falls to **effective rank 5.1**.
   The regularizer this project had gated behind a detector that cannot fire is
   what holds rank at 42-61. Phase A's healthy-looking rank of 50.7 was
   produced by the 12-d formula regression target, not by the JEPA objective.
3. **The regularizer is not free.** At weight 1.0 it costs ~0.007 mean AUPRC
   versus weight 0; at 0.1 it preserves rank (42.6) at lower distortion.
4. **IR is the hard case.** Random projections of a dense 4096-point absorbance
   trace are a strong baseline (raw PCA reached 0.283 in Phase A). No Phase B
   configuration has matched it.

## Recommendation

Do not scale on this evidence, and do not discard the Phase B fixes on it
either — the gate as specified cannot separate them from the budget. The
cheapest informative next step is one properly sized calibration: the same
10k store at the Phase A step count (~40k steps, roughly 12 hours on this
hardware, or fewer epochs on a larger store), re-gated against the same random
control. If ¹H/¹³C keep climbing and MS reaches parity there, Phase B is
working and the claim narrows honestly to dense spectroscopy. If MS is still at
random after a fair budget, the JEPA objective is not extracting chemistry from
~5-12 fragment peaks and MS needs a different formulation (peak-set encoder or
supervised fragment modeling), not more compute.

Artifacts: `phaseb-*.json` in this directory; configs
`configs/phaseb-{calibration,novar,12ep}-10k.json`; stores
`data/processed/{pilot-phaseb-10k,massbank-phaseb}`; protocol addendum in
`docs/EXPERIMENT_PROTOCOL.md`.
