import math

from pcnn4.weighting import (boost, gate, include, negative_weight,
                             positive_weight_each, smoothstep)


def test_smoothstep_endpoints():
    assert smoothstep(-1.0) == 0.0
    assert smoothstep(0.0) == 0.0
    assert smoothstep(1.0) == 1.0
    assert smoothstep(2.0) == 1.0
    assert abs(smoothstep(0.5) - 0.5) < 1e-12


def test_gate_window():
    # chosen V* = x; alternatives below x-1 are out, at x fully in
    assert gate(-1.0) == 0.0
    assert gate(0.0) == 1.0
    assert gate(5.0) == 1.0
    assert 0.0 < gate(-0.5) < 1.0
    # monotonic over the ramp
    vals = [gate(-1.0 + 0.1 * i) for i in range(11)]
    assert all(a <= b for a, b in zip(vals, vals[1:]))


def test_boost_ramp():
    assert boost(0.0) == 1.0
    assert boost(-0.7) == 1.0            # no boost below the chosen V*
    assert abs(boost(0.5) - 10.0) < 1e-9  # significant gain -> 10x
    assert abs(boost(3.0) - 10.0) < 1e-9
    assert 1.0 < boost(0.25) < 10.0


def test_negative_weight_shape():
    # equal V* alternative weighs 1.0
    assert abs(negative_weight(0.0) - 1.0) < 1e-12
    # significantly better skipped alternative weighs 10x the equal one
    assert abs(negative_weight(0.5) / negative_weight(0.0) - 10.0) < 1e-9
    # slightly worse saves still count, but reduced
    assert 0.0 < negative_weight(-0.5) < 1.0
    assert negative_weight(-1.0) == 0.0


def test_include_window():
    assert include(0.0)
    assert include(-0.999)
    assert not include(-1.0)
    assert not include(-5.0)


def test_positive_balancing():
    # positives share the total negative weight
    each = positive_weight_each(4, 100.0)
    assert abs(each * 4 - 100.0) < 1e-9
    # floor kicks in when negatives are scarce
    assert positive_weight_each(4, 0.0) == 1.0


def test_boost_smooth():
    # no discontinuity around d = 0 or d = d_significant
    eps = 1e-6
    assert abs(boost(eps) - boost(0.0)) < 1e-3
    assert abs(boost(0.5 + eps) - boost(0.5 - eps)) < 1e-3
    assert not math.isnan(negative_weight(0.123))
