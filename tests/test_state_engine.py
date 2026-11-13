import pytest

from aeris.core.state_engine import (
    classify_regime,
    group_burden,
    severity_score,
    structural_validity,
    unit_burden,
)


def test_structural_validity_is_bounded():
    assert structural_validity(0.2, 0.4, 0.6) == pytest.approx(0.4)
    assert structural_validity(-1.0, 0.5, 2.0) == pytest.approx(0.5)


def test_severity_uses_four_documented_components():
    assert severity_score(0.2, 0.4, 0.6, 0.8) == pytest.approx(0.5)


def test_regime_classification():
    assert classify_regime(0.1, 0.1) == "stable"
    assert classify_regime(0.8, 0.5) == "transition"
    assert classify_regime(0.8, 0.9) == "collapse"


def test_group_burden():
    assert group_burden(3, 2) == 7


def test_unit_burden():
    assert unit_burden(4, 3) == 10


@pytest.mark.parametrize("counts", [(1.5, 0), (1, 0.5), (True, 1), (1, False), (-1, 0)])
def test_burden_rejects_fractional_boolean_or_negative_counts(counts):
    with pytest.raises(ValueError, match="non-negative integers"):
        group_burden(*counts)
    with pytest.raises(ValueError, match="non-negative integers"):
        unit_burden(*counts)


def test_invalid_calibration_thresholds_fail_closed():
    for kw in (
        {"transition_validity_threshold": float("nan")},
        {"collapse_validity_threshold": float("inf")},
        {"transition_validity_threshold": 0.8, "collapse_validity_threshold": 0.4},
        {"transition_severity_threshold": 0.9, "collapse_severity_threshold": 0.4},
        {"collapse_validity_threshold": -0.1},
    ):
        with pytest.raises(ValueError, match="threshold"):
            classify_regime(0.6, 0.5, **kw)


def test_overflowed_means_do_not_create_fake_valid_states():
    with pytest.raises(ValueError, match="validity mean"):
        structural_validity(1e308, 1e308, 1e308)
    with pytest.raises(ValueError, match="severity mean"):
        severity_score(1e308, 1e308, 1e308, 1e308)
