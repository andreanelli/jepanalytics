"""Command-line interface for the complete feasibility workflow."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

import torch

from .adapters import load_license_audit
from .baselines import evaluate_raw_signal_pca_probe, external_baseline_status
from .chemistry import build_formula_targets
from .config import load_training_config
from .data import build_jsonl_store, build_synthetic_store, verify_canonical_store
from .evaluation import (
    build_embedding_cache,
    evaluate_cross_modal_retrieval,
    evaluate_few_shot_probes,
    evaluate_modality_shortcut,
    evaluate_robustness,
    write_evaluation,
)
from .experimental_evaluation import evaluate_experimental_transfer
from .manifest import write_json_atomic
from .massbank import DEFAULT_ALLOWED_LICENSES, build_massbank_store
from .model import EncoderConfig, UniversalSpectrumEncoder
from .neurips import build_neurips_store
from .preprocessing import DEFAULT_MS_MZ_RANGE
from .probe_audit import evaluate_probe_audit
from .reporting import go_no_go_decision, render_markdown_report
from .supervised_evaluation import evaluate_supervised_cnn_ceiling
from .training import train


def _json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _split_limits(args: argparse.Namespace) -> dict[str, int]:
    return {
        split: maximum
        for split, maximum in (
            ("train", args.max_train_molecules),
            ("validation", args.max_validation_molecules),
            ("test", args.max_test_molecules),
        )
        if maximum is not None
    }


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jepanalytics")
    subcommands = parser.add_subparsers(dest="command", required=True)

    synthetic = subcommands.add_parser("prepare-synthetic", help="build deterministic smoke data")
    synthetic.add_argument("output")
    synthetic.add_argument("--molecules", type=int, default=64)
    synthetic.add_argument("--bins", type=int, default=256)
    synthetic.add_argument("--labels", type=int, default=8)
    synthetic.add_argument("--seed", type=int, default=17)

    jsonl = subcommands.add_parser("prepare-jsonl", help="canonicalize JSONL interchange data")
    jsonl.add_argument("input")
    jsonl.add_argument("output")
    jsonl.add_argument("--bins", type=int, default=4096)

    neurips = subcommands.add_parser(
        "prepare-neurips", help="build the 100k-molecule pilot from upstream Parquet"
    )
    neurips.add_argument("input")
    neurips.add_argument("output")
    neurips.add_argument("--smarts", required=True, help="published functional-group SMARTS JSON")
    neurips.add_argument("--audit", default="docs/LICENSE_AUDIT.json")
    neurips.add_argument("--molecules", type=int, default=100_000)
    neurips.add_argument("--bins", type=int, default=4096)
    neurips.add_argument("--seed", type=int, default=17)
    neurips.add_argument(
        "--ms-mz-range",
        nargs=2,
        type=float,
        default=DEFAULT_MS_MZ_RANGE,
        metavar=("LOW", "HIGH"),
        help="absolute m/z window every mass spectrum is rasterized onto",
    )
    neurips.add_argument(
        "--include-precursor-metadata",
        action="store_true",
        help="feed precursor m/z to the encoder; it is derived from the exact "
        "molecular mass and makes probing partly solvable without the spectrum",
    )
    neurips.add_argument("--exclude-molecules")
    neurips.add_argument("--exclude-scaffolds")
    neurips.add_argument("--strict-scaffold", action="store_true")

    massbank = subcommands.add_parser(
        "prepare-massbank", help="build a licensed experimental MS2 evaluation store"
    )
    massbank.add_argument("input")
    massbank.add_argument("output")
    massbank.add_argument("--smarts", required=True)
    massbank.add_argument("--audit", default="docs/LICENSE_AUDIT.json")
    massbank.add_argument("--release", required=True)
    massbank.add_argument("--release-commit", required=True)
    massbank.add_argument("--source-archive")
    massbank.add_argument("--reference-data")
    massbank.add_argument("--bins", type=int, default=4096)
    massbank.add_argument(
        "--allowed-license", action="append", dest="allowed_licenses"
    )
    massbank.add_argument(
        "--ms-mz-range",
        nargs=2,
        type=float,
        default=DEFAULT_MS_MZ_RANGE,
        metavar=("LOW", "HIGH"),
        help="must match the pretraining store so bins carry the same m/z",
    )
    massbank.add_argument(
        "--include-precursor-metadata", action="store_true"
    )

    formula_targets = subcommands.add_parser(
        "prepare-formula-targets",
        help="build formula-composition targets aligned to a canonical store",
    )
    formula_targets.add_argument("--data", required=True)
    formula_targets.add_argument("--parquet", required=True)
    formula_targets.add_argument("--output", required=True)

    verify = subcommands.add_parser("verify-store", help="verify hashes and split leakage")
    verify.add_argument("data_root")

    pretrain = subcommands.add_parser("pretrain", help="run JEPA or MAE pretraining")
    pretrain.add_argument("--config", required=True)
    pretrain.add_argument("--resume", help="checkpoint to resume without editing the config")

    embed = subcommands.add_parser(
        "embed", help="cache frozen embeddings for repeated evaluations"
    )
    embed.add_argument("--checkpoint", required=True)
    embed.add_argument("--data", required=True)
    embed.add_argument("--output", required=True)
    embed.add_argument("--device", default="auto")
    embed.add_argument("--batch-size", type=int, default=256)
    embed.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=("train", "validation", "test"),
    )
    embed.add_argument(
        "--include-patch-pool",
        action="store_true",
        help="also cache the mean-pooled metadata-free patch representation",
    )
    embed.add_argument("--max-train-molecules", type=int)
    embed.add_argument("--max-validation-molecules", type=int)
    embed.add_argument("--max-test-molecules", type=int)
    embed.add_argument("--subset-seed", type=int, default=20260810)

    inspect_model = subcommands.add_parser("inspect-model", help="report model parameter counts")
    inspect_model.add_argument("--config", help="training config; defaults to the full encoder")

    random_checkpoint = subcommands.add_parser(
        "init-random-checkpoint", help="write an untrained controlled baseline"
    )
    random_checkpoint.add_argument("output")
    random_checkpoint.add_argument("--config", help="training config; defaults to full encoder")

    evaluate = subcommands.add_parser("evaluate", help="run preregistered evaluations")
    evaluate_sub = evaluate.add_subparsers(dest="evaluation", required=True)
    for name in ("probes", "probe-audit", "retrieval", "shortcut", "robustness"):
        current = evaluate_sub.add_parser(name)
        current.add_argument("--checkpoint", required=True)
        current.add_argument("--data", required=True)
        current.add_argument("--output", required=True)
        current.add_argument("--device", default="auto")
    evaluate_sub.choices["probes"].add_argument("--probe-epochs", type=int, default=500)
    for name in ("probes", "probe-audit", "retrieval", "shortcut"):
        evaluate_sub.choices[name].add_argument("--embeddings")
    audit = evaluate_sub.choices["probe-audit"]
    audit.add_argument(
        "--representations",
        nargs="+",
        choices=("general", "aligned", "general_aligned", "patch_pool"),
        default=("general", "aligned", "general_aligned", "patch_pool"),
    )
    audit.add_argument(
        "--probe-kinds", nargs="+", choices=("linear", "mlp"), default=("linear", "mlp")
    )
    audit.add_argument(
        "--sampling", nargs="+", choices=("random", "coverage"), default=("random", "coverage")
    )
    audit.add_argument("--fraction", type=float, default=0.01)
    audit.add_argument("--seeds", nargs="+", type=int, default=(11, 17, 23, 31, 47))
    audit.add_argument("--tuning-seed", type=int, default=17)
    audit.add_argument("--minimum-positives", type=int, default=5)
    audit.add_argument("--probe-epochs", type=int, default=500)
    audit.add_argument("--max-train-molecules", type=int)
    audit.add_argument("--max-validation-molecules", type=int)
    audit.add_argument("--max-test-molecules", type=int)
    audit.add_argument("--subset-seed", type=int, default=20260810)
    evaluate_sub.choices["retrieval"].add_argument("--split", default="test")
    evaluate_sub.choices["shortcut"].add_argument(
        "--representation", choices=("general", "aligned"), default="general"
    )
    evaluate_sub.choices["robustness"].add_argument("--split", default="test")
    supervised_cnn = evaluate_sub.add_parser("supervised-cnn")
    supervised_cnn.add_argument("--data", required=True)
    supervised_cnn.add_argument("--output", required=True)
    supervised_cnn.add_argument("--device", default="auto")
    supervised_cnn.add_argument("--fraction", type=float, default=0.01)
    supervised_cnn.add_argument("--seeds", nargs="+", type=int, default=(11, 17, 23, 31, 47))
    supervised_cnn.add_argument("--tuning-seed", type=int, default=17)
    supervised_cnn.add_argument("--epochs", type=int, default=100)
    supervised_cnn.add_argument("--batch-size", type=int, default=128)
    supervised_cnn.add_argument(
        "--acquisitions", nargs="+", type=int, choices=range(5), default=tuple(range(5))
    )
    supervised_cnn.add_argument("--max-train-molecules", type=int)
    supervised_cnn.add_argument("--max-validation-molecules", type=int)
    supervised_cnn.add_argument("--max-test-molecules", type=int)
    supervised_cnn.add_argument("--subset-seed", type=int, default=20260810)
    raw_pca = evaluate_sub.add_parser("raw-pca")
    raw_pca.add_argument("--data", required=True)
    raw_pca.add_argument("--output", required=True)
    raw_pca.add_argument("--device", default="auto")
    raw_pca.add_argument("--components", type=int, default=256)
    raw_pca.add_argument("--fraction", type=float, default=0.01)
    raw_pca.add_argument("--seeds", nargs="+", type=int, default=(11, 17, 23, 31, 47))
    raw_pca.add_argument("--tuning-seed", type=int, default=17)
    raw_pca.add_argument("--probe-epochs", type=int, default=500)
    raw_pca.add_argument("--fit-sample", type=int, default=20000)
    raw_pca.add_argument("--max-train-molecules", type=int)
    raw_pca.add_argument("--max-validation-molecules", type=int)
    raw_pca.add_argument("--max-test-molecules", type=int)
    raw_pca.add_argument("--subset-seed", type=int, default=20260810)
    experimental = evaluate_sub.add_parser("experimental-transfer")
    experimental.add_argument("--checkpoint", required=True)
    experimental.add_argument("--simulated-data", required=True)
    experimental.add_argument("--simulated-embeddings", required=True)
    experimental.add_argument("--experimental-data", required=True)
    experimental.add_argument("--experimental-embeddings", required=True)
    experimental.add_argument("--output", required=True)
    experimental.add_argument("--device", default="auto")
    experimental.add_argument(
        "--representation",
        choices=("general", "aligned", "general_aligned", "patch_pool"),
        default="general",
    )
    experimental.add_argument("--fraction", type=float, default=0.01)
    experimental.add_argument("--seeds", nargs="+", type=int, default=(11, 17, 23, 31, 47))
    experimental.add_argument("--tuning-seed", type=int, default=17)
    experimental.add_argument("--probe-epochs", type=int, default=500)

    baselines = subcommands.add_parser(
        "baseline-status", help="report optional MOMENT and TS2Vec integrations"
    )

    licenses = subcommands.add_parser("license-audit", help="show recorded data-license status")
    licenses.add_argument("--audit", default="docs/LICENSE_AUDIT.json")

    decision = subcommands.add_parser("decide", help="apply preregistered go/no-go rules")
    decision.add_argument("--candidate", required=True)
    decision.add_argument("--specialist", required=True)
    decision.add_argument(
        "--moment",
        help="MOMENT baseline report; omitting it records the primary "
        "comparison as unexecuted, which forces a no-go",
    )
    decision.add_argument("--fraction", type=float, default=0.01)
    decision.add_argument("--specteach", help="JSON mapping IR/NMR/MS to improvements")
    decision.add_argument("--output", required=True)

    report = subcommands.add_parser("report", help="render result JSON files as Markdown")
    report.add_argument("inputs", nargs="+")
    report.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-synthetic":
        manifest = build_synthetic_store(
            args.output,
            n_molecules=args.molecules,
            n_bins=args.bins,
            n_labels=args.labels,
            seed=args.seed,
        )
        _print({"manifest": str(manifest.resolve())})
        return 0
    if args.command == "prepare-jsonl":
        manifest = build_jsonl_store(args.input, args.output, n_bins=args.bins)
        _print({"manifest": str(manifest.resolve())})
        return 0
    if args.command == "prepare-neurips":
        manifest = build_neurips_store(
            args.input,
            args.output,
            smarts_definitions=args.smarts,
            license_audit=args.audit,
            max_molecules=args.molecules,
            n_bins=args.bins,
            seed=args.seed,
            excluded_molecules_file=args.exclude_molecules,
            excluded_scaffolds_file=args.exclude_scaffolds,
            strict_scaffold=args.strict_scaffold,
            ms_mz_range=tuple(args.ms_mz_range),
            include_precursor_metadata=args.include_precursor_metadata,
        )
        _print({"manifest": str(manifest.resolve())})
        return 0
    if args.command == "prepare-massbank":
        manifest = build_massbank_store(
            args.input,
            args.output,
            smarts_definitions=args.smarts,
            license_audit=args.audit,
            release=args.release,
            release_commit=args.release_commit,
            source_archive=args.source_archive,
            reference_data=args.reference_data,
            allowed_licenses=args.allowed_licenses or DEFAULT_ALLOWED_LICENSES,
            n_bins=args.bins,
            ms_mz_range=tuple(args.ms_mz_range),
            include_precursor_metadata=args.include_precursor_metadata,
        )
        _print({"manifest": str(manifest.resolve())})
        return 0
    if args.command == "prepare-formula-targets":
        manifest = build_formula_targets(args.data, args.parquet, args.output)
        _print({"manifest": str(manifest.resolve())})
        return 0
    if args.command == "verify-store":
        result = verify_canonical_store(args.data_root)
        _print(result)
        return 0 if result["valid"] else 1
    if args.command == "pretrain":
        config = load_training_config(args.config)
        if args.resume:
            # An explicit resume restores the complete experiment state from the
            # checkpoint.  It therefore supersedes any encoder-only warm start
            # recorded in the original launch configuration.
            config = replace(
                config,
                initialize_from=None,
                resume_from=args.resume,
            )
        checkpoint = train(config)
        _print({"checkpoint": str(checkpoint.resolve())})
        return 0
    if args.command == "embed":
        manifest = build_embedding_cache(
            args.checkpoint,
            args.data,
            args.output,
            splits=args.splits,
            batch_size=args.batch_size,
            device=args.device,
            include_patch_pool=args.include_patch_pool,
            max_molecules_per_split=_split_limits(args),
            subset_seed=args.subset_seed,
        )
        _print({"manifest": str(manifest.resolve())})
        return 0
    if args.command in {"inspect-model", "init-random-checkpoint"}:
        config = load_training_config(args.config).encoder if args.config else EncoderConfig()
        model = UniversalSpectrumEncoder(config)
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        if args.command == "inspect-model":
            _print(
                {
                    "encoder_parameters": trainable,
                    "target_range": [20_000_000, 25_000_000],
                    "within_target": 20_000_000 <= trainable <= 25_000_000,
                    "config": config.to_dict(),
                }
            )
        else:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": -1,
                    "global_step": 0,
                    "encoder_config": config.to_dict(),
                    "training_config": {"objective": "random"},
                    "online_encoder": model.state_dict(),
                },
                output,
            )
            _print({"checkpoint": str(output.resolve()), "encoder_parameters": trainable})
        return 0
    if args.command == "evaluate":
        if args.evaluation == "experimental-transfer":
            result = evaluate_experimental_transfer(
                args.checkpoint,
                args.simulated_data,
                args.simulated_embeddings,
                args.experimental_data,
                args.experimental_embeddings,
                representation=args.representation,
                fraction=args.fraction,
                seeds=args.seeds,
                tuning_seed=args.tuning_seed,
                epochs=args.probe_epochs,
                device=args.device,
            )
        elif args.evaluation == "supervised-cnn":
            result = evaluate_supervised_cnn_ceiling(
                args.data,
                fraction=args.fraction,
                seeds=args.seeds,
                tuning_seed=args.tuning_seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                acquisitions=args.acquisitions,
                max_molecules_per_split=_split_limits(args) or None,
                subset_seed=args.subset_seed,
                device=args.device,
            )
        elif args.evaluation == "raw-pca":
            result = evaluate_raw_signal_pca_probe(
                args.data,
                components=args.components,
                fraction=args.fraction,
                seeds=args.seeds,
                tuning_seed=args.tuning_seed,
                epochs=args.probe_epochs,
                fit_sample=args.fit_sample,
                max_molecules_per_split=_split_limits(args) or None,
                subset_seed=args.subset_seed,
                device=args.device,
            )
        elif args.evaluation == "probes":
            result = evaluate_few_shot_probes(
                args.checkpoint,
                args.data,
                probe_epochs=args.probe_epochs,
                device=args.device,
                embedding_cache=args.embeddings,
            )
        elif args.evaluation == "probe-audit":
            if not args.embeddings:
                raise ValueError("probe-audit requires --embeddings with validation and patch_pool")
            result = evaluate_probe_audit(
                args.checkpoint,
                args.data,
                args.embeddings,
                representations=args.representations,
                probe_kinds=args.probe_kinds,
                sampling_modes=args.sampling,
                fraction=args.fraction,
                seeds=args.seeds,
                tuning_seed=args.tuning_seed,
                minimum_positives=args.minimum_positives,
                epochs=args.probe_epochs,
                max_molecules_per_split=_split_limits(args),
                subset_seed=args.subset_seed,
                device=args.device,
            )
        elif args.evaluation == "retrieval":
            result = evaluate_cross_modal_retrieval(
                args.checkpoint,
                args.data,
                split=args.split,
                device=args.device,
                embedding_cache=args.embeddings,
            )
        elif args.evaluation == "shortcut":
            result = evaluate_modality_shortcut(
                args.checkpoint,
                args.data,
                representation=args.representation,
                device=args.device,
                embedding_cache=args.embeddings,
            )
        else:
            result = evaluate_robustness(
                args.checkpoint, args.data, split=args.split, device=args.device
            )
        write_evaluation(result, args.output)
        _print({"output": str(Path(args.output).resolve()), "kind": result["kind"]})
        return 0
    if args.command == "baseline-status":
        _print([asdict(status) for status in external_baseline_status()])
        return 0
    if args.command == "license-audit":
        _print(load_license_audit(args.audit))
        return 0
    if args.command == "decide":
        specteach = _json(args.specteach) if args.specteach else None
        result = go_no_go_decision(
            _json(args.candidate),
            _json(args.specialist),
            _json(args.moment) if args.moment else None,
            fraction=args.fraction,
            specteach_improvements=specteach,
        )
        write_json_atomic(result, args.output)
        _print(result)
        return 0 if result["decision"] == "go" else 2
    if args.command == "report":
        output = render_markdown_report([_json(path) for path in args.inputs], args.output)
        _print({"report": str(output.resolve())})
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
