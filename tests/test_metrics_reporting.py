import numpy as np

from jepanalytics.metrics import bootstrap_interval, macro_auprc, macro_f1
from jepanalytics.reporting import go_no_go_decision


def test_metrics_are_exact_on_perfect_predictions():
    truth = np.array([[1, 0], [0, 1], [1, 0]], dtype=float)
    probability = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]])
    assert macro_auprc(truth, probability) == 1.0
    assert macro_f1(truth, probability) == 1.0
    interval = bootstrap_interval(np.ones(10), samples=100)
    assert interval.estimate == interval.lower == interval.upper == 1.0


def _report(value):
    families = ["IR", "H1_NMR", "C13_NMR", "MSMS_POSITIVE", "MSMS_NEGATIVE"]
    return {
        "results": [
            {"acquisition": family, "fraction": 0.01, "seed": seed, "macro_auprc": value}
            for family in families
            for seed in [11, 17, 23, 31, 47]
        ]
    }


def test_go_no_go_rules_are_machine_enforced():
    result = go_no_go_decision(
        _report(0.75),
        _report(0.65),
        _report(0.70),
        specteach_improvements={"IR": 0.01, "NMR": 0.02, "MS": -0.01},
    )
    assert result["decision"] == "go"
    failed = go_no_go_decision(
        _report(0.71),
        _report(0.70),
        _report(0.70),
        specteach_improvements={"IR": 0.01, "NMR": -0.02, "MS": -0.01},
    )
    assert failed["decision"] == "no-go"

