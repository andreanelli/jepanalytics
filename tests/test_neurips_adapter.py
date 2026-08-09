import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("rdkit")
import pyarrow as pa
import pyarrow.parquet as pq

from jepanalytics.data import CanonicalSpectraDataset, verify_canonical_store
from jepanalytics.neurips import (
    MoleculeCandidate,
    build_neurips_store,
    stratified_candidate_sample,
)


def test_stratified_sample_handles_more_strata_than_requested_molecules():
    candidates = {
        f"mol-{index}": MoleculeCandidate(
            row_id=index,
            smiles="CCO",
            molecule_id=f"mol-{index}",
            scaffold_id=f"scaffold-{index}",
            heavy_atoms=5 + index * 10,
            exact_mass=46.0,
            labels=np.eye(3, dtype=np.float32)[index],
        )
        for index in range(3)
    }
    selected = stratified_candidate_sample(candidates, max_molecules=2, seed=17)
    assert len(selected) == 2


def test_native_neurips_schema_conversion(tmp_path: Path):
    rows = 2
    payload = {
        "smiles": ["CCO", "c1ccccc1"],
        "__index_level_0__": [101, 102],
        "molecular_formula": ["C2H6O", "C6H6"],
        "ir_spectra": [[0.0] * 1800 for _ in range(rows)],
        "h_nmr_spectra": [[0.0] * 10_000 for _ in range(rows)],
        "c_nmr_spectra": [[0.0] * 10_000 for _ in range(rows)],
    }
    for polarity in ("positive", "negative"):
        for energy in (10, 20, 40):
            payload[f"msms_cfmid_{polarity}_{energy}ev"] = [
                [[31.0, 20.0], [45.0, 100.0]],
                [[39.0, 10.0], [77.0, 100.0]],
            ]
    parquet = tmp_path / "upstream.parquet"
    pq.write_table(pa.table(payload), parquet)
    smarts = tmp_path / "functional-groups.json"
    smarts.write_text(
        json.dumps(
            {
                "functional_groups": [
                    {"name": "alcohol", "smarts": "[OX2H]"},
                    {"name": "arene", "smarts": "c1ccccc1"},
                ]
            }
        )
    )
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "id": "multimodal-spectroscopic-dataset",
                        "status": "approved_for_training",
                        "license": "test fixture",
                    }
                ]
            }
        )
    )
    output = tmp_path / "canonical"
    build_neurips_store(
        parquet,
        output,
        smarts_definitions=smarts,
        license_audit=audit,
        max_molecules=2,
        n_bins=64,
    )
    verification = verify_canonical_store(output)
    assert verification["valid"]
    assert verification["n_records"] == 18
    dataset = CanonicalSpectraDataset(output)
    assert dataset.arrays["labels"].shape == (18, 2)
    assert set(dataset.arrays["acquisition"].tolist()) == {0, 1, 2, 3, 4}
