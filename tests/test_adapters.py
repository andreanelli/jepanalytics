import json
from pathlib import Path

import numpy as np
import pytest

from jepanalytics.adapters import SmartsLabeler, read_jcamp_xy, require_approved_license


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


def test_published_functional_group_resource_is_complete():
    pytest.importorskip("rdkit")
    from rdkit import Chem

    resource = Path(__file__).parents[1] / "resources" / "alberts-2024-functional-groups.json"
    labeler = SmartsLabeler(resource)
    assert len(labeler.names) == 37
    molecule = Chem.MolFromSmiles("CCO")
    assert np.array_equal(labeler("CCO"), labeler.label_molecule(molecule))
