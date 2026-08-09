#!/usr/bin/env python3
"""Resolve a local SpecTeach compound-list export to training exclusions.

The input JSON is intentionally local/ignored because the source workbook is
not redistributed. Each row must contain ``CAS#`` and may contain
``Compound/Filename`` and ``Formula``. PubChem responses are cached in the
output directory so interrupted runs are resumable and auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from jepanalytics.adapters import chemistry_identifiers


PUBCHEM_PROPERTY_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{identifier}/"
    "property/CanonicalSMILES,IsomericSMILES,InChIKey/JSON"
)


def _query_pubchem(cas_number: str, retries: int = 4) -> dict[str, Any] | None:
    url = PUBCHEM_PROPERTY_URL.format(identifier=urllib.parse.quote(cas_number, safe=""))
    request = urllib.request.Request(url, headers={"User-Agent": "JEPAnalytics/0.1"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
            return payload["PropertyTable"]["Properties"][0]
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code not in {429, 500, 502, 503, 504} or attempt + 1 == retries:
                raise
            time.sleep(2**attempt)
        except (TimeoutError, urllib.error.URLError):
            if attempt + 1 == retries:
                raise
            time.sleep(2**attempt)
    return None


def _smiles(properties: dict[str, Any]) -> str:
    for key in ("SMILES", "IsomericSMILES", "ConnectivitySMILES", "CanonicalSMILES"):
        value = properties.get(key)
        if value:
            return str(value)
    raise ValueError(f"PubChem response has no SMILES field: {sorted(properties)}")


def resolve(
    input_json: Path,
    output_root: Path,
    delay: float = 0.22,
    overrides_path: Path | None = None,
) -> dict[str, Any]:
    rows = json.loads(input_json.read_text())
    overrides = json.loads(overrides_path.read_text()) if overrides_path else {}
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = output_root / "pubchem-cache"
    cache_root.mkdir(exist_ok=True)
    resolved: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        cas_number = str(row.get("CAS#") or "").strip()
        if not cas_number:
            unresolved.append({"row": row, "reason": "missing CAS number"})
            continue
        cache_path = cache_root / f"{cas_number}.json"
        if cache_path.exists():
            properties = json.loads(cache_path.read_text())
        else:
            properties = _query_pubchem(cas_number)
            if properties is not None:
                cache_path.write_text(json.dumps(properties, indent=2, sort_keys=True) + "\n")
            if index + 1 < len(rows):
                time.sleep(delay)
        structure_source = "PubChem"
        if properties is None and cas_number in overrides:
            properties = {"CanonicalSMILES": overrides[cas_number], "CID": None}
            structure_source = "manual override"
        if properties is None:
            unresolved.append({"row": row, "reason": "not found in PubChem"})
            continue
        try:
            smiles, inchikey, scaffold = chemistry_identifiers(_smiles(properties))
        except ValueError as exc:
            unresolved.append({"row": row, "reason": str(exc), "pubchem": properties})
            continue
        resolved.append(
            {
                "compound": row.get("Compound/Filename"),
                "formula_reported": row.get("Formula"),
                "cas_number": cas_number,
                "pubchem_cid": properties.get("CID"),
                "structure_source": structure_source,
                "smiles": smiles,
                "inchikey": inchikey,
                "scaffold": scaffold,
            }
        )

    resolved_path = output_root / "specteach-resolved.json"
    unresolved_path = output_root / "specteach-unresolved.json"
    inchikey_path = output_root / "specteach-inchikeys.txt"
    scaffold_path = output_root / "specteach-scaffolds.txt"
    resolved_path.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n")
    unresolved_path.write_text(json.dumps(unresolved, indent=2, sort_keys=True) + "\n")
    inchikey_path.write_text("\n".join(sorted({row["inchikey"] for row in resolved})) + "\n")
    scaffold_path.write_text("\n".join(sorted({row["scaffold"] for row in resolved})) + "\n")
    return {
        "input": str(input_json.resolve()),
        "input_sha256": hashlib.sha256(input_json.read_bytes()).hexdigest(),
        "records": len(rows),
        "resolved": len(resolved),
        "unresolved": len(unresolved),
        "unique_inchikeys": len({row["inchikey"] for row in resolved}),
        "unique_scaffolds": len({row["scaffold"] for row in resolved}),
        "outputs": {
            "resolved": str(resolved_path.resolve()),
            "unresolved": str(unresolved_path.resolve()),
            "inchikeys": str(inchikey_path.resolve()),
            "scaffolds": str(scaffold_path.resolve()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--delay", type=float, default=0.22)
    parser.add_argument("--overrides", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            resolve(args.input_json, args.output_root, args.delay, args.overrides),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
