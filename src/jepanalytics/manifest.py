"""Machine-readable provenance for datasets, runs, and reports."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(cwd: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def build_run_manifest(
    *,
    config: Mapping[str, Any],
    dataset_manifest: str | Path,
    repository: str | Path,
) -> dict[str, Any]:
    dataset_manifest = Path(dataset_manifest)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": dict(config),
        "dataset_manifest": str(dataset_manifest.resolve()),
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "mps_available": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
        },
        "git_revision": _git_revision(Path(repository)),
    }


def write_json_atomic(payload: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)

