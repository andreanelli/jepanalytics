import json
from pathlib import Path

import pytest

from jepanalytics.adapters import read_jcamp_xy, require_approved_license


def test_license_gate_is_fail_closed(tmp_path: Path):
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps({"sources": [{"id": "data", "status": "pending_verification"}]})
    )
    with pytest.raises(PermissionError):
        require_approved_license(audit, "data")


def test_reads_explicit_jcamp_pairs(tmp_path: Path):
    path = tmp_path / "signal.dx"
    path.write_text(
        "##TITLE=Example\n##XFACTOR=1\n##YFACTOR=0.5\n##XYDATA=(XY..XY)\n1, 2\n2, 4\n3, 6\n##END=\n"
    )
    x, y, metadata = read_jcamp_xy(path)
    assert x.tolist() == [1, 2, 3]
    assert y.tolist() == [1, 2, 3]
    assert metadata["TITLE"] == "Example"

