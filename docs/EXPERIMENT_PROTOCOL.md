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

