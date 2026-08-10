# Post-Pilot Alignment Repair Amendment

## Why this amendment exists

The first 100,000-molecule pilot completed, but its aligned projection failed the
secondary endpoints. Cross-technique retrieval was at chance, the contrastive
loss remained near its random-batch value, and acquisition technique was 99.6%
linearly decodable from the aligned embedding. This document records a
post-failure repair; it does not rewrite the preregistered protocol.

## Diagnosed causes

1. Pair sampling operated on records. Three collision-energy records in each
   MS/MS family made MS-positive/MS-negative 33% of all pairs, while individual
   IR/NMR pairs appeared only about 2.8% of the time.
2. Alignment used the strongly augmented, 50%-masked JEPA context view.
3. Each molecule supplied only one cross-technique positive per epoch.
4. The aligned projection consumed the general summary token, which explicitly
   received acquisition, axis, unit, and continuous metadata.
5. Thresholded variance regularization allowed unit-normalized embeddings to
   settle at a large common direction near the collapse threshold.

## Repair

- Choose acquisition families before records; collision energy cannot weight a
  family or become a false negative.
- Use a complete, lightly augmented online view for alignment.
- Put one view from all five acquisition families in the same batch and treat
  all four other same-molecule views as positives.
- Keep a metadata-rich general token, but use a separate alignment summary token.
  Patch and alignment tokens receive coordinate content but no categorical
  technique embedding and cannot attend to the general token. The general token
  may still attend to all content. This remains one shared Transformer.
- Apply running batch centering, variance, covariance, common-direction, and
  acquisition-centroid controls throughout aligned training.
- Log positive/negative cosine, cosine margin, batch Recall@1, centroid norm,
  positives per anchor, and each acquisition-pair frequency to W&B.

## Verification gates

The repair must pass these gates before a full pilot:

1. A tiny five-view overfit test reaches greater than 95% retrieval in evaluation
   mode. The checked test reaches 100%.
2. Full-size MPS calibration completes without memory instability.
3. Calibration alignment loss falls below its random multi-positive reference.
4. Held-out aligned technique accuracy falls materially below the original 99.6%.
5. At least one nontrivial held-out cross-technique retrieval direction exceeds
   chance.

The 256-molecule full-model calibration passed all five gates on 2026-08-10. It
ran 160 steps at approximately 94 spectra/s and 16.5 GB MPS driver allocation.
Aligned technique accuracy was 56.4%. Held-out IR/NMR Recall@1 was 14-21% versus
3.6% chance, and MS-positive/MS-negative Recall@1 was 14%.

## Repaired pilot

`configs/pilot-multiview-mps.json` runs eight epochs over the 100,000-molecule
corpus with five views per molecule, batch size 24, alignment weight 0.2, and the
same frozen evaluation suite. Eight epochs process approximately the same number
of spectra as the original 20-epoch two-view pilot.

The repaired pilot is still subject to the original go/no-go criteria. Improved
retrieval alone is not sufficient to scale the corpus or claim universal
pretraining.
