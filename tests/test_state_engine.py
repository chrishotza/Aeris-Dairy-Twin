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
