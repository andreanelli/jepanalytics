# JEPAnalytics

JEPAnalytics is a paper-oriented feasibility framework for testing whether one
coordinate-aware Transformer can learn reusable representations of IR, NMR, and
MS/MS signals. It implements a JEPA-style latent prediction objective, paired
cross-technique alignment, controlled baselines, leakage-resistant splits, and
the preregistered few-shot evaluation.

This repository contains research code, not a pretrained foundation model. No
third-party spectral data is bundled or downloaded automatically.

## What is implemented

- A validated `SpectralSignal` API covering physical coordinates, intensities,
  technique metadata, provenance, molecule/scaffold identifiers, and labels.
- Peak-list-aware resampling, robust scaling, and physically plausible signal
  augmentations without reversal or arbitrary elastic warping.
- One shared 20–25M parameter encoder with general, aligned, and patch-level
  embeddings.
- EMA target JEPA training with mixed contiguous, random, and peak-centered
  masks; symmetric cross-modal alignment; collapse monitoring.
- A raw-intensity masked-autoencoder control using the identical backbone.
- Memory-mappable canonical data stores with hashes and automatic molecule,
  scaffold, replicate, and acquisition leakage checks.
- Frozen few-shot functional-group probes, retrieval, modality-shortcut, and
  robustness evaluations plus the preregistered go/no-go calculation.
- Opt-in MassBank and JCAMP-DX import helpers guarded by a data-license audit.

## Quick start

Python 3.11 or 3.12 is required.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'

jepanalytics prepare-synthetic data/processed/smoke --molecules 64 --bins 256
jepanalytics verify-store data/processed/smoke
jepanalytics pretrain --config configs/smoke.json
jepanalytics evaluate probes \
  --checkpoint runs/smoke/best.pt \
  --data data/processed/smoke \
  --output runs/smoke/probes.json \
  --probe-epochs 30 \
  --device cpu
```

Inspect the full encoder before allocating a training run:

```bash
jepanalytics inspect-model --config configs/pilot.json
```

On an Apple Silicon machine with Metal available, use
`configs/pilot-mps.json`. Its batch size of 64 was selected by the checked M4
Max calibration; `configs/mps-calibration.json` reproduces the compatibility
and throughput check before a long run.

## Public API

```python
import numpy as np
from jepanalytics import (
    AcquisitionFamily,
    AxisType,
    AxisUnit,
    SpectralSignal,
    UniversalSpectrumEncoder,
)

signal = SpectralSignal(
    coordinate=np.linspace(400, 4000, 1800),
    intensity=np.zeros(1800),
    axis_type=AxisType.WAVENUMBER,
    axis_unit=AxisUnit.INVERSE_CENTIMETER,
    acquisition=AcquisitionFamily.IR,
    molecule_id="example-inchikey",
    source="example",
    source_id="example-001",
)

model = UniversalSpectrumEncoder()
embeddings = model.encode(signal)
print(embeddings.general.shape, embeddings.aligned.shape, embeddings.patches.shape)
```

## Real data

The canonical importer accepts JSON Lines records containing `coordinate`,
`intensity`, enum names for `axis_type`, `axis_unit`, and `acquisition`, plus
the identifiers described in [`docs/DATA_CARD.md`](docs/DATA_CARD.md).

```bash
jepanalytics prepare-jsonl exported-signals.jsonl data/processed/pilot --bins 4096
```

The approved native upstream Parquet schema can be converted directly. The
adapter performs two passes so the
100,000-molecule stratified selection does not load the 18.6 GB archive into
memory, stores all six MS/MS collision-energy records, and assigns scaffold-safe
splits before writing signals:

```bash
pip install -e '.[chem,data]'
jepanalytics prepare-neurips upstream/data data/processed/pilot \
  --smarts resources/alberts-2024-functional-groups.json \
  --exclude-molecules data/external/specteach-inchikeys.txt \
  --molecules 100000
```

The approved uses and their constraints are recorded in
[`docs/LICENSE_AUDIT.json`](docs/LICENSE_AUDIT.json). A new release requires a
fresh audit. MassBank records must still pass their individual `LICENSE` fields,
and this project does not redistribute source or derived spectral datasets.

The complete experimental matrix and success criteria are frozen in
[`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md).

## Research positioning

The intended claim is narrower than being the first universal spectral model.
Relevant precedents include [MOMENT](https://arxiv.org/abs/2402.03885),
[SpectroGen](https://arxiv.org/abs/2407.16094), and
[SECS](https://www.nature.com/articles/s41467-026-73846-y). JEPAnalytics tests
the distinct hypothesis that one self-supervised, coordinate-aware 1-D backbone
improves label efficiency across heterogeneous analytical techniques.
