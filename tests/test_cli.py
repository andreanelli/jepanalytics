from pathlib import Path

from jepanalytics import cli
from jepanalytics.config import TrainingConfig


def test_explicit_resume_supersedes_warm_start(monkeypatch, tmp_path: Path):
    original = TrainingConfig(
        data_root="data",
        output_dir="run",
        initialize_from="warm-start.pt",
    )
    captured = {}

    monkeypatch.setattr(cli, "load_training_config", lambda _: original)

    def fake_train(config):
        captured["config"] = config
        return tmp_path / "best.pt"

    monkeypatch.setattr(cli, "train", fake_train)

    result = cli.main(
        [
            "pretrain",
            "--config",
            "config.json",
            "--resume",
            "last.pt",
        ]
    )

    assert result == 0
    assert captured["config"].resume_from == "last.pt"
    assert captured["config"].initialize_from is None
