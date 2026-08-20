# Phase A gate summary (2026-08-16)

Phase A fixed the measurement stack before touching the model: all probes are
now fit to convergence (L-BFGS logistic regression with validation-selected L2),
the supervised CNN baseline is refit per seed and scored on the identical
3,000-molecule test population as the frozen-probe audit, chance levels are
empirical, and two missing controls were run (untrained random encoder and a
raw-signal PCA-256 linear probe under the identical protocol). MOMENT remains
unexecuted (environment conflict; see the protocol addendum) and the decision
gate now records that explicitly.

All numbers below: fraction 0.08 of the 10k-molecule train subset (800 labeled
molecules), 5 probe seeds, general embedding, linear probe, random sampling,
37-label functional groups, mean prevalence floor 0.127.

## Simulated probing (mean macro-AUPRC / mean normalized lift)

| family    | random init | raw PCA-256 | JEPA 100k ep8 | supervised CNN |
|-----------|------------|-------------|---------------|----------------|
| IR        | 0.219/0.13 | 0.283/0.21  | 0.246/0.19    | **0.334/0.28** |
| H1 NMR    | 0.165/0.08 | 0.185/0.10  | 0.192/0.12    | **0.219/0.15** |
| C13 NMR   | 0.164/0.08 | 0.185/0.10  | 0.197/0.13    | **0.238/0.17** |
| MS+       | **0.165**/0.07 | 0.142/0.02 | 0.155/0.04 | 0.155/0.04     |
| MS−       | **0.165**/0.07 | 0.148/0.04 | 0.156/0.05 | 0.160/0.06     |
| **mean**  | 0.176/0.09 | 0.189/0.09  | 0.189/0.11    | **0.221/0.14** |

Paired per-seed JEPA-minus-random deltas (bootstrap 95% CI):
IR +0.027 [+0.026,+0.028], H1 +0.026 [+0.025,+0.027], C13 +0.033
[+0.032,+0.035], MS+ **−0.010** [−0.011,−0.009], MS− **−0.009**
[−0.010,−0.007].

## Experimental transfer (MassBank, converged probes, fraction 0.01)

Pretrained-vs-random deltas all cross zero or favor random; on MS− the random
encoder scores highest of all three encoders (0.1555 vs 0.1514 for the 100k
model). Scaling 10k → 100k still does not help on real data.

## Reading

1. **Probe under-fitting was not the story.** Converged probes moved the JEPA
   mean by +0.0003. The frozen representation genuinely contains this little
   linearly decodable chemistry.
2. **Dense spectroscopy learned something real**: +0.026 to +0.033 over the
   random encoder with tight CIs. But it is roughly half the gap to a small
   supervised CNN at the same 800 labels, and on IR a plain PCA of the raw
   signal beats the pretrained embedding outright (0.283 vs 0.246).
3. **MS pretraining is actively harmful**: the pretrained encoder is
   significantly *below its own random initialization* on both MS families.
   This is consistent with the Phase B diagnosis — the per-molecule
   (0, precursor) m/z grid, the ~99.7%-zero bins making 50% masking trivial,
   and baseline/noise augmentations that are non-physical for centroided
   spectra. Training taught the encoder to discard MS content.
4. **The random encoder beats PCA on MS** (0.165 vs 0.142–0.148). The random
   encoder still receives `continuous_metadata` containing the
   precursor-derived exact mass; even random projections keep it linearly
   recoverable. This is a direct measurement of the metadata shortcut inside
   the probed `general` representation.
5. **Cross-technique retrieval with fair pools** (3,000 molecules, one record
   each): H1→C13 is real (median rank 28 vs chance 1500; R@10 0.30);
   MS+→MS− (rank 35) is explainable by the mass shortcut; every
   spectroscopy↔MS pair sits exactly at chance. The aligned modality-shortcut
   accuracy is 0.467 against the correct 0.333 record-prior chance (the
   earlier "repaired to 0.41 vs 0.20 chance" framing overstated the repair).

## Decision

Stop tuning the current objective; the Phase A gate confirms Phase B is
warranted and tells it exactly what to fix first:

1. Rebuild the MS representation (absolute m/z grid or peak tokens,
   occupancy-aware masking, physical augmentations) — the current one is worse
   than no pretraining.
2. Remove the precursor/exact-mass features from any representation that is
   probed or aligned; re-measure the MS+→MS− retrieval afterward.
3. Treat IR+NMR as the demonstrated-signal track; a universal claim including
   MS is not supported by any current measurement.
4. Any future go/no-go needs the specialist JEPA baselines per family and an
   executable MOMENT adapter (isolated environment), or a formal protocol
   amendment dropping MOMENT.

Artifacts: `phaseA-*.json` in this directory; code changes in
`probes.py`, `probe_audit.py`, `evaluation.py`, `supervised_evaluation.py`,
`experimental_evaluation.py`, `baselines.py`, `reporting.py`, `cli.py`;
protocol addendum in `docs/EXPERIMENT_PROTOCOL.md`.
