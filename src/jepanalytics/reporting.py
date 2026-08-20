"""Paper-oriented aggregation and preregistered go/no-go decision."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .manifest import write_json_atomic
from .metrics import bootstrap_interval


def _values_by_acquisition(
    report: Mapping[str, Any],
    *,
    fraction: float,
    metric: str = "macro_auprc",
    representation: str = "general",
    probe_kind: str = "linear",
    sampling: str = "random",
) -> dict[str, list[float]]:
    """Group one metric by family from a single evaluation cell.

    Probe-audit reports contain several representations, probe kinds, and
    sampling modes; pooling across those cells (as the previous
    implementation did) blends incomparable numbers into one mean.  Rows
    without those keys (supervised CNN, legacy probes) pass through.
    """

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in report["results"]:
        if abs(float(row.get("fraction", fraction)) - fraction) > 1e-9:
            continue
        if row.get("representation", representation) != representation:
            continue
        if row.get("probe_kind", probe_kind) != probe_kind:
            continue
        if row.get("sampling", sampling) != sampling:
            continue
        if metric not in row or row[metric] is None:
            continue
        grouped[row["acquisition"]].append(float(row[metric]))
    return dict(grouped)


def _family_differences(
    candidate_values: Mapping[str, list[float]],
    baseline_reports: Mapping[str, Mapping[str, list[float]]],
) -> tuple[list[float], dict[str, dict[str, float]]]:
    differences: list[float] = []
    families: dict[str, dict[str, float]] = {}
    for family in sorted(candidate_values):
        current = np.asarray(candidate_values[family])
        baselines = {}
        for name, values in baseline_reports.items():
            if family not in values:
                raise ValueError(
                    f"baseline report {name!r} has no rows for family {family!r} "
                    "at the requested fraction and cell"
                )
            baselines[name] = np.asarray(values[family])
        baseline_mean = max(float(np.mean(values)) for values in baselines.values())
        family_difference = current - baseline_mean
        differences.extend(family_difference.tolist())
        families[family] = {
            "candidate": float(current.mean()),
            "candidate_std": float(current.std(ddof=1)) if current.size > 1 else 0.0,
            "baseline": baseline_mean,
            "baseline_stds": {
                name: float(values.std(ddof=1)) if values.size > 1 else 0.0
                for name, values in baselines.items()
            },
            "difference": float(family_difference.mean()),
        }
    return differences, families


def go_no_go_decision(
    candidate: Mapping[str, Any],
    specialist: Mapping[str, Any],
    moment: Mapping[str, Any] | None = None,
    *,
    fraction: float = 0.01,
    specteach_improvements: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Apply the preregistered decision rule at an explicit label fraction.

    The pass/fail thresholds stay on macro-AUPRC differences exactly as
    preregistered, but the report leads with normalized lift when available:
    on a 0.127 mean prevalence floor, raw AUPRC overstates how much signal a
    probe extracts.  When the MOMENT baseline has not been run the decision
    can never be "go" — the primary comparison is then unexecuted, and the
    output says so instead of silently comparing against the specialist only.
    """

    baseline_reports = {
        "specialist": _values_by_acquisition(specialist, fraction=fraction)
    }
    if moment is not None:
        baseline_reports["moment"] = _values_by_acquisition(moment, fraction=fraction)
    candidate_values = _values_by_acquisition(candidate, fraction=fraction)
    if not candidate_values:
        raise ValueError(
            f"candidate report has no general/linear/random rows at fraction {fraction}"
        )
    differences, family_results = _family_differences(
        candidate_values, baseline_reports
    )
    interval = bootstrap_interval(np.asarray(differences))
    improved = sum(row["difference"] > 0 for row in family_results.values())
    no_large_regression = all(
        row["difference"] >= -0.02 for row in family_results.values()
    )
    experimental = dict(specteach_improvements or {})
    experimental_pass = (
        len(experimental) >= 3
        and sum(value > 0 for value in experimental.values()) >= 2
    )
    checks = {
        "moment_baseline_available": moment is not None,
        "mean_improvement_at_least_0.03": interval.estimate >= 0.03,
        "bootstrap_lower_bound_above_zero": interval.lower > 0.0,
        "at_least_four_families_improve": improved >= 4,
        "no_family_regresses_over_0.02": no_large_regression,
        "specteach_improves_two_of_three_techniques": experimental_pass,
    }

    lift_summary: dict[str, Any] | None = None
    candidate_lift = _values_by_acquisition(
        candidate, fraction=fraction, metric="macro_normalized_lift"
    )
    if candidate_lift:
        lift_baselines = {
            name: _values_by_acquisition(
                report, fraction=fraction, metric="macro_normalized_lift"
            )
            for name, report in (("specialist", specialist), ("moment", moment))
            if report is not None
        }
        lift_baselines = {name: values for name, values in lift_baselines.items() if values}
        if lift_baselines and all(
            family in values
            for values in lift_baselines.values()
            for family in candidate_lift
        ):
            lift_differences, lift_families = _family_differences(
                candidate_lift, lift_baselines
            )
            lift_interval = bootstrap_interval(np.asarray(lift_differences))
            lift_summary = {
                "families": lift_families,
                "mean_difference_interval": {
                    "estimate": lift_interval.estimate,
                    "lower": lift_interval.lower,
                    "upper": lift_interval.upper,
                },
            }
        else:
            lift_summary = {
                "families": {
                    family: {"candidate": float(np.mean(values))}
                    for family, values in candidate_lift.items()
                },
                "note": "baseline reports lack macro_normalized_lift rows",
            }

    return {
        "decision": "go" if all(checks.values()) else "no-go",
        "headline_metric": "macro_normalized_lift",
        "fraction": fraction,
        "checks": checks,
        "normalized_lift": lift_summary,
        "mean_difference_interval": {
            "estimate": interval.estimate,
            "lower": interval.lower,
            "upper": interval.upper,
        },
        "families": family_results,
        "specteach_improvements": experimental,
    }


def render_markdown_report(sections: Sequence[Mapping[str, Any]], output: str | Path) -> Path:
    lines = ["# JEPAnalytics Feasibility Report", ""]
    for section in sections:
        lines.extend((f"## {section.get('kind', 'Result').replace('_', ' ').title()}", ""))
        if "decision" in section:
            lines.extend((f"Decision: **{section['decision'].upper()}**", ""))
        lines.extend(("```json", json.dumps(section, indent=2, sort_keys=True), "```", ""))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    return output
