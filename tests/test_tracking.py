from pathlib import Path

import pytest

from jepanalytics.config import TrainingConfig
from jepanalytics.tracking import WandbTracker, _wandb_metrics


def test_wandb_metrics_are_grouped_and_convert_memory_to_gib():
    payload = _wandb_metrics(
        {
            "epoch": 2,
            "step": 41,
            "loss": 1.25,
            "effective_rank": 12.0,
            "accelerator_allocated_bytes": 2 * 1024**3,
            "elapsed_seconds": 1800,
        }
    )
    assert payload["trainer/global_step"] == 41
    assert payload["train/total_loss"] == 1.25
    assert payload["representation/general_effective_rank"] == 12
    assert payload["system/accelerator_allocated_gib"] == 2
    assert payload["progress/elapsed_hours"] == 0.5


def test_wandb_enabled_requires_stable_run_id():
    with pytest.raises(ValueError, match="wandb_run_id"):
        TrainingConfig(data_root="data", output_dir="run", wandb_enabled=True)


def test_disabled_tracker_does_not_require_sdk(tmp_path: Path):
    config = TrainingConfig(data_root="data", output_dir="run")
    tracker = WandbTracker.start(
        config,
        output=tmp_path,
        run_manifest={"dataset_manifest_sha256": "a" * 64},
        dataset_manifest={},
        encoder_parameters=10,
        train_molecules=4,
        steps_per_epoch=1,
    )
    assert tracker.run is None
