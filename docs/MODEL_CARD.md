# JEPAnalytics Model Card

## Model

The default encoder has a shared patch projection over 4,096 intensity bins,
128 patches, 12 pre-norm Transformer layers, hidden dimension 384, six attention
heads, and no technique-specific backbone branches. A metadata token incorporates
axis, unit, acquisition, physical range, preprocessing statistics, precursor mass,
collision energy, sampling interval, and original axis orientation.

The model returns:

- `general`: a 384-dimensional representation used for downstream probes.
- `aligned`: a normalized 256-dimensional projection trained for same-molecule
  cross-technique retrieval.
- `patches`: 128 local 384-dimensional representations.

## Training objectives

The online encoder sees a physically perturbed signal with 40–60% of patches
masked. An EMA target encoder sees a complete, lightly perturbed view. The latent
predictor estimates target patch representations only at masked locations.

Following the first-pilot failure analysis, aligned training uses one lightly
augmented spectrum from every available acquisition family for each molecule.
Every other same-molecule view is a positive in a multi-positive contrastive
loss. The general token retains acquisition metadata. Patch and alignment tokens
do not receive categorical acquisition embeddings and cannot attend back to the
metadata-rich general token, preventing the direct technique shortcut while
retaining one shared Transformer. Running batch centering, aligned variance and
covariance control, and acquisition-centroid penalties are active throughout
aligned training.

Embedding standard deviation, effective rank, positive and negative cosine,
batch retrieval, centroid norm, uniformity loss, and all ten acquisition-pair
frequencies are recorded at every step.

## Intended use

This model is intended for research on representation transfer, few-shot
functional-group classification, cross-modal retrieval, and robustness. It is not
validated for compound identification, clinical decisions, quality-release
decisions, regulatory submissions, or unsupervised identification of unknowns.

## Limitations and risks

- The primary pretraining corpus is simulated and may teach simulator artifacts.
- A shared model can learn technique identity without learning transferable
  chemistry; the modality-shortcut test must accompany downstream results.
- Multi-view alignment increases per-molecule compute and does not guarantee
  that weakly related techniques such as IR and MS/MS become molecule-identifying.
- Robust per-spectrum scaling discards absolute instrument response.
- Rasterizing centroided MS/MS peaks introduces a resolution choice.
- Coordinate perturbations are deliberately small and do not represent every
  calibration or sample-preparation failure.
- The encoder covers pure-compound 1-D IR, NMR, and MS/MS only. It excludes
  mixtures, chromatography, PXRD, raw FIDs, and 2-D spectra.

## Reporting requirements

Every checkpoint must be accompanied by the canonical dataset manifest, split
digest, training configuration, run manifest, five-seed probe results, retrieval
results, robustness results, modality-shortcut accuracy, and license audit.
