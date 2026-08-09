# JEPAnalytics Data Card

## Canonical representation

Every input record contains:

- `coordinate`: physical positions before resampling.
- `intensity`: matching one-dimensional signal values.
- `axis_type`: `WAVENUMBER`, `CHEMICAL_SHIFT`, or `MASS_TO_CHARGE`.
- `axis_unit`: `INVERSE_CENTIMETER`, `PPM`, or `MZ`.
- `acquisition`: `IR`, `H1_NMR`, `C13_NMR`, `MSMS_POSITIVE`, or
  `MSMS_NEGATIVE`.
- `molecule_id`: canonical structure identifier, normally an InChIKey.
- `source` and `source_id`: dataset and immutable record identifiers.
- `scaffold_id`: a standardized Bemis–Murcko scaffold.
- Optional multi-hot `labels` and acquisition `metadata`.

The JSONL interchange format uses those field names directly. Enum values are
their uppercase names. MS/MS peak lists set `metadata.representation` to
`peak_list`; dense traces use `dense`.

## Canonical store

Preparation produces memory-mappable `.npy` arrays, vocabularies, and a manifest
with a SHA-256 digest for every array. Intensities are stored as float16 after
resampling and scaling; metadata and labels use float32. Molecule, source, and
scaffold identifiers are integer-coded against checked-in vocabularies.

## Splitting and decontamination

- Canonicalize structures and deduplicate by InChIKey before sampling.
- Assign complete Bemis–Murcko scaffold groups to 80/10/10 splits.
- Keep all modalities, replicate measurements, ion modes, and collision energies
  for one molecule in the same split.
- Remove exact SpecTeach structures from simulated training.
- Produce a second experimental result after removing all matching SpecTeach
  scaffolds from simulated training.
- Run `jepanalytics verify-store` before every experiment. Training must stop on
  a hash mismatch or leakage finding.

## Bias and coverage

The simulated source is derived from patent chemistry and is not representative
of all natural products, organometallics, polymers, salts, mixtures, or instrument
conditions. Public experimental databases have uneven technique, instrument,
compound-class, and metadata coverage. Metrics must therefore be reported by
acquisition family and source, not only as a pooled average.

## Licensing

No upstream dataset is redistributed in this repository. See
`docs/LICENSE_AUDIT.json`; the current entries were approved by the project owner
on 2026-08-09 and must be reviewed again for a different release or use. Each
retained MassBank record must still pass its own license filter.
