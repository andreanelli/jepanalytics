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
Paired spectra from the same molecule receive a symmetric contrastive alignment
loss. Embedding standard deviation and effective rank are recorded at every step;
variance regularization activates only when a configured collapse threshold is
crossed.

## Intended use

This model is intended for research on representation transfer, few-shot
functional-group classification, cross-modal retrieval, and robustness. It is not
validated for compound identification, clinical decisions, quality-release
decisions, regulatory submissions, or unsupervised identification of unknowns.

## Limitations and risks

- The primary pretraining corpus is simulated and may teach simulator artifacts.
- A shared model can learn technique identity without learning transferable
  chemistry; the modality-shortcut test must accompany downstream results.
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

