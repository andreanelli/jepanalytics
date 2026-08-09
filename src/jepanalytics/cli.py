"""Command-line interface for the complete feasibility workflow."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

import torch

from .adapters import load_license_audit
from .baselines import external_baseline_status
from .config import load_training_config
from .data import build_jsonl_store, build_synthetic_store, verify_canonical_store
from .evaluation import (
    evaluate_cross_modal_retrieval,
    evaluate_few_shot_probes,
    evaluate_modality_shortcut,
    evaluate_robustness,
    write_evaluation,
)
from .manifest import write_json_atomic
from .model import EncoderConfig, UniversalSpectrumEncoder
from .neurips import build_neurips_store
from .reporting import go_no_go_decision, render_markdown_report
from .training import train


def _json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


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
    neurips.add_argument("--exclude-molecules")
    neurips.add_argument("--exclude-scaffolds")
    neurips.add_argument("--strict-scaffold", action="store_true")

    verify = subcommands.add_parser("verify-store", help="verify hashes and split leakage")
    verify.add_argument("data_root")

    pretrain = subcommands.add_parser("pretrain", help="run JEPA or MAE pretraining")
    pretrain.add_argument("--config", required=True)
    pretrain.add_argument("--resume", help="checkpoint to resume without editing the config")

    inspect_model = subcommands.add_parser("inspect-model", help="report model parameter counts")
    inspect_model.add_argument("--config", help="training config; defaults to the full encoder")

    random_checkpoint = subcommands.add_parser(
        "init-random-checkpoint", help="write an untrained controlled baseline"
    )
    random_checkpoint.add_argument("output")
    random_checkpoint.add_argument("--config", help="training config; defaults to full encoder")

    evaluate = subcommands.add_parser("evaluate", help="run preregistered evaluations")
    evaluate_sub = evaluate.add_subparsers(dest="evaluation", required=True)
    for name in ("probes", "retrieval", "shortcut", "robustness"):
        current = evaluate_sub.add_parser(name)
        current.add_argument("--checkpoint", required=True)
        current.add_argument("--data", required=True)
        current.add_argument("--output", required=True)
        current.add_argument("--device", default="auto")
    evaluate_sub.choices["probes"].add_argument("--probe-epochs", type=int, default=150)
    evaluate_sub.choices["retrieval"].add_argument("--split", default="test")
    evaluate_sub.choices["shortcut"].add_argument(
        "--representation", choices=("general", "aligned"), default="general"
    )
    evaluate_sub.choices["robustness"].add_argument("--split", default="test")

    baselines = subcommands.add_parser(
        "baseline-status", help="report optional MOMENT and TS2Vec integrations"
    )

    licenses = subcommands.add_parser("license-audit", help="show recorded data-license status")
    licenses.add_argument("--audit", default="docs/LICENSE_AUDIT.json")

    decision = subcommands.add_parser("decide", help="apply preregistered go/no-go rules")
    decision.add_argument("--candidate", required=True)
    decision.add_argument("--specialist", required=True)
    decision.add_argument("--moment", required=True)
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
        )
        _print({"manifest": str(manifest.resolve())})
        return 0
    if args.command == "verify-store":
        result = verify_canonical_store(args.data_root)
        _print(result)
        return 0 if result["valid"] else 1
    if args.command == "pretrain":
        config = load_training_config(args.config)
        if args.resume:
            config = replace(config, resume_from=args.resume)
        checkpoint = train(config)
        _print({"checkpoint": str(checkpoint.resolve())})
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
        if args.evaluation == "probes":
            result = evaluate_few_shot_probes(
                args.checkpoint,
                args.data,
                probe_epochs=args.probe_epochs,
                device=args.device,
            )
        elif args.evaluation == "retrieval":
            result = evaluate_cross_modal_retrieval(
                args.checkpoint, args.data, split=args.split, device=args.device
            )
        elif args.evaluation == "shortcut":
            result = evaluate_modality_shortcut(
                args.checkpoint,
                args.data,
                representation=args.representation,
                device=args.device,
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
            _json(args.moment),
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
