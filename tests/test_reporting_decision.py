import numpy as np
import pytest

from jepanalytics.evaluation import _one_record_per_molecule
from jepanalytics.reporting import go_no_go_decision


FAMILIES = ("IR", "H1_NMR", "C13_NMR", "MSMS_POSITIVE", "MSMS_NEGATIVE")


def _report(scores: dict[str, float], *, fraction: float = 0.01, **extra) -> dict:
    rows = []
    for family, score in scores.items():
        for seed in (11, 17):
            rows.append(
                {
                    "acquisition": family,
                    "fraction": fraction,
                    "seed": seed,
                    "macro_auprc": score,
                    "macro_normalized_lift": score / 2,
                    **extra,
                }
            )
    return {"results": rows}


def test_decision_requires_moment_baseline_to_go():
    candidate = _report({family: 0.30 for family in FAMILIES})
    specialist = _report({family: 0.20 for family in FAMILIES})
    decision = go_no_go_decision(
        candidate,
        specialist,
        None,
        specteach_improvements={"IR": 0.1, "NMR": 0.1, "MS": 0.1},
    )
    # Every numeric criterion passes, but the primary preregistered
    # comparison was never executed, so the decision cannot be a go.
    assert decision["decision"] == "no-go"
    assert decision["checks"]["moment_baseline_available"] is False
    assert decision["checks"]["mean_improvement_at_least_0.03"] is True
    assert decision["normalized_lift"]["families"]["IR"]["difference"] == pytest.approx(
        0.05
    )


def test_decision_goes_with_all_baselines_present():
    candidate = _report({family: 0.30 for family in FAMILIES})
    specialist = _report({family: 0.20 for family in FAMILIES})
    moment = _report({family: 0.22 for family in FAMILIES})
    decision = go_no_go_decision(
        candidate,
        specialist,
        moment,
        specteach_improvements={"IR": 0.1, "NMR": 0.1, "MS": 0.1},
    )
    assert decision["decision"] == "go"
    assert decision["families"]["IR"]["baseline"] == pytest.approx(0.22)


def test_decision_filters_fraction_and_audit_cells():
    candidate = _report({family: 0.30 for family in FAMILIES}, fraction=0.08)
    # Rows from other audit cells must not pollute the aggregate.
    candidate["results"].extend(
        _report(
            {family: 0.99 for family in FAMILIES},
            fraction=0.08,
            representation="aligned",
        )["results"]
    )
    candidate["results"].extend(
        _report({family: 0.99 for family in FAMILIES}, fraction=0.08, probe_kind="mlp")[
            "results"
        ]
    )
    specialist = _report({family: 0.20 for family in FAMILIES}, fraction=0.08)
    decision = go_no_go_decision(candidate, specialist, None, fraction=0.08)
    assert decision["families"]["IR"]["candidate"] == pytest.approx(0.30)
    with pytest.raises(ValueError, match="no general/linear/random rows"):
        go_no_go_decision(candidate, specialist, None, fraction=0.01)


def test_decision_errors_on_missing_baseline_family():
    candidate = _report({family: 0.30 for family in FAMILIES})
    specialist = _report({family: 0.20 for family in FAMILIES if family != "IR"})
    with pytest.raises(ValueError, match="IR"):
        go_no_go_decision(candidate, specialist, None)


def test_one_record_per_molecule_orders_and_dedupes():
    rng = np.random.default_rng(3)
    molecules = np.asarray([5, 5, 5, 7, 7, 9, 9, 9, 2])
    indices = np.arange(100, 109)
    wanted = np.asarray([2, 5, 9])
    rows = _one_record_per_molecule(indices, molecules, wanted, rng)
    assert rows.shape == (3,)
    np.testing.assert_array_equal(molecules[rows - 100], wanted)
