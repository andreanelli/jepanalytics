"""Paper-oriented aggregation and preregistered go/no-go decision."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .manifest import write_json_atomic
from .metrics import bootstrap_interval


def _one_percent_by_acquisition(report: Mapping[str, Any]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in report["results"]:
        if abs(float(row["fraction"]) - 0.01) < 1e-9:
            grouped[row["acquisition"]].append(float(row["macro_auprc"]))
    return grouped


def go_no_go_decision(
    candidate: Mapping[str, Any],
    specialist: Mapping[str, Any],
    moment: Mapping[str, Any],
    *,
    specteach_improvements: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    candidate_values = _one_percent_by_acquisition(candidate)
    specialist_values = _one_percent_by_acquisition(specialist)
    moment_values = _one_percent_by_acquisition(moment)
    differences = []
    family_results = {}
    for family in sorted(candidate_values):
        current = np.asarray(candidate_values[family])
        specialist_mean = float(np.mean(specialist_values[family]))
        moment_mean = float(np.mean(moment_values[family]))
        baseline = max(specialist_mean, moment_mean)
        family_difference = current - baseline
        differences.extend(family_difference.tolist())
        family_results[family] = {
            "candidate": float(current.mean()),
            "baseline": baseline,
            "difference": float(family_difference.mean()),
        }
    interval = bootstrap_interval(np.asarray(differences))
    improved = sum(row["difference"] > 0 for row in family_results.values())
    no_large_regression = all(row["difference"] >= -0.02 for row in family_results.values())
    experimental = dict(specteach_improvements or {})
    experimental_pass = len(experimental) >= 3 and sum(value > 0 for value in experimental.values()) >= 2
    checks = {
        "mean_improvement_at_least_0.03": interval.estimate >= 0.03,
        "bootstrap_lower_bound_above_zero": interval.lower > 0.0,
        "at_least_four_families_improve": improved >= 4,
        "no_family_regresses_over_0.02": no_large_regression,
        "specteach_improves_two_of_three_techniques": experimental_pass,
    }
    return {
        "decision": "go" if all(checks.values()) else "no-go",
        "checks": checks,
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

