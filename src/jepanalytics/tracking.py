"""Optional experiment tracking without weakening local reproducibility."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import TrainingConfig
from .manifest import write_json_atomic


def _wandb_metrics(record: Mapping[str, Any]) -> dict[str, float]:
    mapping = {
        "loss": "train/total_loss",
        "jepa_loss": "train/jepa_loss",
        "alignment_loss": "train/alignment_loss",
        "reconstruction_loss": "train/reconstruction_loss",
        "variance_loss": "regularization/variance_loss",
        "covariance_loss": "regularization/covariance_loss",
        "embedding_std": "representation/general_std",
        "effective_rank": "representation/general_effective_rank",
        "aligned_embedding_std": "representation/aligned_std",
        "aligned_effective_rank": "representation/aligned_effective_rank",
        "collapsed": "representation/collapse_flag",
        "target_signal_residual_norm": "representation/target_signal_residual_norm",
        "masked_fraction": "data/masked_fraction",
        "learning_rate": "optimization/learning_rate",
        "gradient_norm": "optimization/gradient_norm",
        "step_seconds": "performance/step_seconds",
        "molecule_pairs_per_second": "performance/molecule_pairs_per_second",
        "spectra_per_second": "performance/spectra_per_second",
        "epoch_progress": "progress/epoch_fraction",
        "global_progress": "progress/run_fraction",
        "eta_hours": "progress/eta_hours",
    }
    payload = {
        target: float(record[source])
        for source, target in mapping.items()
        if source in record
    }
    payload["trainer/epoch"] = float(record["epoch"])
    payload["trainer/global_step"] = float(record["step"])
    if "elapsed_seconds" in record:
        payload["progress/elapsed_hours"] = float(record["elapsed_seconds"]) / 3600.0
    for source, target in (
        ("accelerator_allocated_bytes", "system/accelerator_allocated_gib"),
        ("accelerator_peak_allocated_bytes", "system/accelerator_peak_allocated_gib"),
        ("accelerator_driver_allocated_bytes", "system/accelerator_driver_allocated_gib"),
    ):
        if source in record:
            payload[target] = float(record[source]) / 1024**3
    return payload


@dataclass(slots=True)
class WandbTracker:
    """Thin W&B adapter; disabled runs do not import the SDK."""

    run: Any | None
    log_every: int
    state_path: Path
    is_new: bool = False

    @classmethod
    def start(
        cls,
        config: TrainingConfig,
        *,
        output: Path,
        run_manifest: Mapping[str, Any],
        dataset_manifest: Mapping[str, Any],
        encoder_parameters: int,
        train_molecules: int,
        steps_per_epoch: int,
    ) -> "WandbTracker":
        state_path = output / "wandb_run.json"
        if not config.wandb_enabled or config.wandb_mode == "disabled":
            return cls(None, config.log_every, state_path)

        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "W&B logging is enabled but wandb is not installed; install the project dependencies"
            ) from error

        is_new = not state_path.exists()
        run_config = config.to_dict()
        run_config["dataset"] = {
            "manifest_sha256": run_manifest["dataset_manifest_sha256"],
            "n_records": dataset_manifest.get("n_records"),
            "n_labels": dataset_manifest.get("n_labels"),
            "n_bins": dataset_manifest.get("n_bins"),
            "selected_molecules": dataset_manifest.get("provenance", {}).get(
                "selected_molecules"
            ),
            "split_sha256": dataset_manifest.get("provenance", {}).get("split_sha256"),
        }
        run_config["execution"] = {
            "git_revision": run_manifest.get("git_revision"),
            "encoder_parameters": encoder_parameters,
            "train_molecules": train_molecules,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": config.epochs * steps_per_epoch,
        }
        run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            id=config.wandb_run_id,
            name=config.wandb_run_name,
            group=config.wandb_group,
            tags=list(config.wandb_tags),
            config=run_config,
            dir=str(output.resolve()),
            mode=config.wandb_mode,
            resume="allow",
            force=config.wandb_mode == "online",
            save_code=True,
        )
        if run is None:
            raise RuntimeError("wandb.init() did not return a run")
        run.define_metric("trainer/global_step")
        run.define_metric("*", step_metric="trainer/global_step")
        run.summary.update(
            {
                "dataset_manifest_sha256": run_manifest["dataset_manifest_sha256"],
                "git_revision": run_manifest.get("git_revision"),
                "encoder_parameters": encoder_parameters,
                "train_molecules": train_molecules,
                "steps_per_epoch": steps_per_epoch,
                "total_steps": config.epochs * steps_per_epoch,
            }
        )
        write_json_atomic(
            {
                "entity": run.entity,
                "project": run.project,
                "id": run.id,
                "name": run.name,
                "url": run.url,
            },
            state_path,
        )
        return cls(run, config.log_every, state_path, is_new=is_new)

    def log_step(self, record: Mapping[str, Any], *, force: bool = False) -> None:
        if self.run is None:
            return
        step = int(record["step"])
        if not force and step % self.log_every:
            return
        self.run.log(_wandb_metrics(record), step=step)

    def backfill(self, metrics_path: Path) -> int:
        """Upload an existing local history once when W&B is first attached."""

        if self.run is None or not self.is_new or not metrics_path.exists():
            return 0
        logged = 0
        with metrics_path.open() as handle:
            for line in handle:
                record = json.loads(line)
                if int(record["step"]) % self.log_every == 0:
                    self.log_step(record, force=True)
                    logged += 1
        self.run.summary["backfilled_history_points"] = logged
        return logged

    def log_epoch(
        self, *, epoch: int, global_step: int, epoch_loss: float, best_loss: float
    ) -> None:
        if self.run is None:
            return
        self.run.log(
            {
                "trainer/global_step": float(global_step),
                "trainer/epoch": float(epoch),
                "train/epoch_loss": float(epoch_loss),
                "train/best_epoch_loss": float(best_loss),
            },
            step=global_step,
        )
        self.run.summary.update(
            {
                "last_completed_epoch": epoch,
                "last_global_step": global_step,
                "best_epoch_loss": best_loss,
            }
        )

    def finish(self, summary: Mapping[str, Any]) -> None:
        if self.run is None:
            return
        self.run.summary.update(dict(summary))
        self.run.finish()
