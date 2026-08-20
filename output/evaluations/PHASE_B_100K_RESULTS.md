# Phase B at full scale — 100k molecules, 12 epochs (2026-08-20)

Supersedes the gate verdict in `PHASE_B_GATE_SUMMARY.md`, which reported a
10k-molecule calibration. That calibration failed because 4,000 steps is too
few, not because the Phase B repairs were wrong. At Phase A's own budget
(100,000 molecules, 12 epochs, 39,996 steps, 13.8 h) **both gate criteria pass
and the result improves monotonically across every snapshot.**

Run: `runs/phaseb-100k-12ep`, config `configs/phaseb-100k-12ep.json`,
`chemistry_weight=0` (pure self-supervision, no formula distillation),
`variance_weight=0.1`. Final loss 0.8408 against Phase A's 1.597 at the
identical step count. Effective rank 38 → 95 over training; Phase A sat near 20
at matched steps.

## Simulated probes (800 labels, 5 seeds, converged probes, same split)

| family | random | raw PCA | ep4 | ep8 | ep12 | vs random | vs PCA | lift |
|---|---|---|---|---|---|---|---|---|
| IR | 0.2211 | **0.2928** | 0.2564 | 0.2893 | 0.2958 | +0.0747 | +0.0030 (tie) | 0.131→0.244 |
| ¹H NMR | 0.1718 | 0.1928 | 0.2074 | 0.2166 | 0.2191 | +0.0474 | **+0.0263** | 0.085→0.154 |
| ¹³C NMR | 0.1669 | 0.1910 | 0.2030 | 0.2217 | 0.2243 | +0.0574 | **+0.0333** | 0.074→0.160 |
| MS+ | 0.1653 | 0.1636 | 0.1819 | 0.1907 | 0.1932 | +0.0279 | **+0.0296** | 0.062→0.107 |
| MS− | 0.1752 | 0.1819 | 0.1814 | 0.1907 | 0.1927 | +0.0174 | **+0.0108** | 0.077→0.105 |
| **mean** | 0.1801 | 0.2044 | 0.2060 | 0.2218 | **0.2250** | **+0.0450** | **+0.0206** | |

vs random 95% CI [+0.0369, +0.0532]; vs PCA [+0.0147, +0.0256]. Phase A's mean
advantage over random after the same 40k steps was +0.0136, and raw PCA there
*matched* the encoder exactly (0.1886 vs 0.1892).

## MassBank experimental transfer (91,611 real spectra)

| stratum | prevalence | random | ep4 | ep8 | ep12 | delta | 95% CI |
|---|---|---|---|---|---|---|---|
| MS+ / all | 0.1296 | 0.1356 | 0.1427 | 0.1453 | 0.1510 | **+0.0155** | [+0.0115,+0.0198] |
| MS+ / molecule-unseen | 0.1297 | 0.1358 | 0.1428 | 0.1453 | 0.1510 | **+0.0152** | [+0.0114,+0.0193] |
| MS+ / scaffold-unseen | 0.1316 | 0.1367 | 0.1444 | 0.1475 | 0.1533 | **+0.0165** | [+0.0137,+0.0200] |
| MS− / all | 0.1392 | 0.1463 | 0.1533 | 0.1542 | 0.1549 | **+0.0086** | [+0.0039,+0.0133] |
| MS− / molecule-unseen | 0.1393 | 0.1464 | 0.1532 | 0.1541 | 0.1548 | **+0.0084** | [+0.0037,+0.0131] |
| MS− / scaffold-unseen | 0.1421 | 0.1515 | 0.1543 | 0.1552 | 0.1570 | +0.0055 | [−0.0003,+0.0113] |

Five of six strata clear zero, every stratum climbs monotonically ep4→ep12, and
the effect survives on scaffold-unseen molecules. Phase A's figure was +0.00104
with CI [−0.0057,+0.0078] — indistinguishable from a random encoder — while its
simulated score *rose*, the sim-to-real inversion that made the whole pilot
untrustworthy. Phase B moves both in the same direction.

## Held-out cross-technique retrieval (3,000-molecule pool, chance R@1 0.00033)

| pair | R@1 | R@10 | median rank | chance median |
|---|---|---|---|---|
| H1→C13 | 0.2423 | 0.6000 | **6** | 1500 |
| MS+→MS− | 0.1370 | 0.4987 | **11** | 1500 |
| IR→H1 | 0.0877 | 0.3437 | 25 | 1500 |
| IR→C13 | 0.0687 | 0.2750 | 39 | 1500 |
| IR→MS+ | 0.0057 | 0.0417 | 494 | 1500 |
| H1→MS+ | 0.0037 | 0.0357 | 465 | 1500 |
| C13→MS+ | 0.0033 | 0.0347 | 458 | 1500 |
| (other spectroscopy↔MS) | 0.0037–0.0040 | ~0.03 | 534–551 | 1500 |

The MS+→MS− result is the load-bearing one: 0.137 R@1 and median rank 11
achieved **without** the precursor mass, where Phase A's superficially better
0.708 batch recall was mass lookup. Spectroscopy↔MS is now genuinely above
chance (median rank ~460–550 against 1500) where Phase A sat *worse* than
chance at 2200–2775. The bridge exists but stays an order of magnitude weaker
than within-cluster retrieval.

## What has not improved

- **Technique is still 100% linearly decodable from `general`** (chance 0.333),
  unchanged from Phase A. This is close to tautological — `acquisition_embedding`
  is added directly to the summary token — but it means the probed
  representation remains modality-partitioned. `aligned` sits at 0.399 against
  0.333 chance, i.e. nearly modality-free.
- **IR ties raw PCA** (+0.0030, CI spans zero). A dense 4096-point absorbance
  trace appears close to linearly sufficient for functional groups; pretraining
  adds nothing measurable there. This is a finding about IR, not a defect.
- **MS− scaffold-unseen** remains a tie (+0.0055, CI touches zero).

## Standing limitations

One pretraining seed; the five seeds vary only the labeled subsample. Effects
are modest in absolute terms (+0.0155 macro-AUPRC on a 0.13 prevalence floor,
roughly 2.5x the random encoder's margin over prevalence). Every comparison
here is against random initialization and raw PCA — **the preregistered go
criterion is against same-capacity single-technique JEPAs and MOMENT, and
neither has been run**, so the formal protocol gate remains unsatisfied.
SpecTeach is still blocked, so IR and NMR have no experimental validation.

## Where this leaves the claim

Supported: one coordinate-aware encoder, pretrained self-supervised on
heterogeneous 1-D analytical signals, produces representations that beat both
random initialization and a raw-signal PCA at a 1% label budget on NMR and
MS/MS, and that advantage transfers to real experimental mass spectra including
unseen scaffolds. Not supported: any claim about IR, where PCA ties; any
"foundation model" framing at 22M parameters and one seed; and the
preregistered universality claim, which needs the specialist and MOMENT arms.

Next, in order of value per hour: a second pretraining seed to bound
seed variance; the five single-technique specialist JEPAs (the actual gate);
then MOMENT in an isolated environment, or a protocol amendment dropping it.
