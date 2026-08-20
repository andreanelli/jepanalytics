# Preregistered Feasibility Protocol

## Primary hypothesis

A jointly pretrained JEPA encoder improves frozen 1% label functional-group
prediction over both a same-capacity single-technique JEPA and MOMENT by at least
0.03 macro-AUPRC without materially harming any acquisition family.

The primary metric is macro-AUPRC. Macro-F1 is secondary. Fractions are 1%, 5%,
10%, and 100% of training molecules. Seeds are 11, 17, 23, 31, and 47.

## Fixed experiment matrix

1. Untrained shared Transformer.
2. Five single-acquisition JEPA runs using acquisition IDs 0–4.
3. Shared masked raw-intensity autoencoder.
4. TS2Vec and MOMENT using their published open checkpoints and preprocessing,
   with exact versions recorded in the run manifest.
5. Supervised 1-D CNN and gradient-boosted tree baselines.
6. Universal JEPA with alignment weights 0, 0.05, and 0.2.
7. Leave-one-technique-out JEPA pretraining followed by a frozen 1% probe.

Alignment weight is selected between 0.05 and 0.2 using validation macro-AUPRC;
weight 0 remains an ablation and cannot be selected as the aligned model.

## Secondary endpoints

- Cross-technique molecule retrieval: Recall@1/5/10 and median rank.
- Simulated-to-experimental probes on SpecTeach.
- Experimental NMR and MS probing on nmrshiftdb2 and MassBank.
- Embedding cosine stability under coordinate shift, broadening, baseline drift,
  noise, and resolution loss.
- Linear technique prediction from both general and aligned embeddings.

## Go/no-go rule

At the 1% setting, proceed to full-corpus scaling only when all conditions hold:

- The mean improvement over the better specialist/MOMENT baseline is at least
  0.03 macro-AUPRC and its bootstrap 95% lower bound is above zero.
- At least four of five acquisition families improve.
- No family regresses by more than 0.02.
- SpecTeach improves over specialist baselines for at least two of IR, NMR, and MS.

`jepanalytics decide` applies these rules without discretionary overrides.

## Required ablation diagnostics after failure

If the result is no-go, report modality shortcut accuracy, embedding effective
rank, per-technique gradients or loss curves, alignment-weight sensitivity, and
whether lightweight technique adapters recover the specialist gap. Do not scale
or describe the model as universal solely because cross-modal retrieval succeeds.


## Addendum (2026-08-16): status of external baselines

This addendum records execution status; it does not alter any preregistered
threshold above.

- **MOMENT was never executed.** `momentfm` (0.1.4) requires `numpy==1.25.2`
  and `transformers==4.33.3`, which conflict with this project's pinned
  environment (`numpy>=1.26`, torch 2.13). Running it requires an isolated
  environment and a dedicated embedding-extraction adapter that do not yet
  exist. Until then, `jepanalytics decide` records
  `moment_baseline_available: false` and can never return "go": the primary
  preregistered comparison is unexecuted, not passed.
- **TS2Vec was never executed** for the same reason (no adapter).
- **SpecTeach remains blocked** on the proprietary Mnova container format, so
  the fourth go condition is unevaluable; the only experimental validation to
  date is MassBank MS/MS.
- A raw-signal PCA linear-probe control (`jepanalytics evaluate raw-pca`) was
  added as an executable untrained-feature baseline under the identical probe
  protocol and label budget.

## Addendum (2026-08-18): Phase B representation and objective repair

Phase A established that probe optimization was not the limiting factor and
that MS pretraining was *harmful* — the pretrained encoder scored below its own
random initialization on both MS families. Phase B changes the inputs and the
objective accordingly. Every item below alters the pretraining distribution, so
Phase B checkpoints are not comparable to Phase A checkpoints and the stores
were rebuilt rather than migrated.

1. **Absolute m/z grid.** Mass spectra were rasterized onto a per-molecule
   `(0, precursor)` window, so bin *i* denoted a different m/z for every
   molecule and absolute fragment masses were unlearnable. They now use a fixed
   `--ms-mz-range`, default 0–1000 Da, recorded in the store manifest. Peaks
   outside the window are dropped and counted
   (`ProcessedSignal.out_of_range_fraction`).
2. **Exact-mass shortcut removed.** `precursor_mz` is derived from
   `ExactMolWt ± 1.007276`. It no longer defines the rasterization window and
   is excluded from `continuous_metadata` unless
   `--include-precursor-metadata` is passed. With the fixed grid, the three
   coordinate channels that feed every content token are now constant across
   MS records.
3. **Technique-aware augmentation.** Centroided spectra receive peak dropout,
   per-peak abundance jitter, and whole-bin m/z miscalibration; baseline drift,
   additive Gaussian noise, broadening, and resolution loss are applied only to
   dense traces. All draws are per sample — broadening and resolution were
   previously drawn once per batch, so with five views per molecule every view
   received identical treatment.
4. **Occupancy-aware masking.** The mask ratio now applies to informative
   patches. A predicted MS/MS spectrum occupies roughly five of 128 patches, so
   uniform masking hid ~55 identically empty patches per record and the
   objective was satisfiable by emitting "empty patch".
5. **Regularizer and axis fixes.** General variance/covariance is applied
   unconditionally: it was gated on a collapse detector that cannot fire behind
   the output LayerNorm, so across the entire 100k run its weighted
   contribution was exactly zero. Chemical-shift normalization is now
   acquisition dependent (12 ppm for ¹H, 250 ppm for ¹³C) — a single 250 ppm
   constant compressed the ¹H window to 0.04 while ¹³C spanned ~0.92. Dense
   traces longer than the target grid are aggregated per bin instead of
   point-sampled, so single-sample NMR lines survive decimation.

Correction to the Phase A write-up: the aligned variance floor
(`alignment_target_std = 0.04`) was described as mathematically unreachable.
It is reachable — it is a hinge at roughly 0.64x the isotropic per-feature
std of an L2-normalized 256-d embedding (1/sqrt(256) = 0.0625), and it read
zero because the aligned space was near-isotropic, i.e. healthy. The genuinely
dead path was the gated general variance/covariance term in item 5. Training
now logs `aligned_isotropic_std` so this headroom is visible.
