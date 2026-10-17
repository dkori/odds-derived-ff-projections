"""Lock the math for the demo chart modes: standard CDF and the 2-player diff.

These test the pure data transforms (not Plotly rendering) so they run without
the ``web`` extra. The JS front end ports these exact transforms, so these also
document the contract the JS must reproduce.
"""
from __future__ import annotations

import numpy as np
import pytest

from ffmc.sim.viz import _cdf_std_xy, _cdf_xy, diff_series


def _rec(name: str, mean: float, levels, values) -> dict:
    """Minimal store record with the fields the chart transforms read."""
    return {
        "name": name,
        "mean": mean,
        "median": float(np.interp(0.5, levels, values)),
        "quantiles": {"levels": list(levels), "values": list(values)},
    }


# A shared, strictly increasing quantile grid (as the slate writes).
_LEVELS = np.linspace(0.001, 0.999, 105)


def _lin_rec(name, mean, lo, hi):
    """A record whose quantile function is linear from lo..hi points."""
    values = lo + (hi - lo) * _LEVELS
    return _rec(name, mean, _LEVELS, values)


# --- standard CDF --------------------------------------------------------


def test_cdf_std_is_flipped_cdf_axes_swapped():
    rec = _lin_rec("A", 12.0, 2.0, 30.0)
    px, py = _cdf_xy(rec)          # flipped: x=percentile(0-100), y=points
    sx, sy = _cdf_std_xy(rec)      # standard: x=points, y=cum prob(0-1)
    # Standard CDF x == flipped y (points); standard y == flipped x / 100.
    assert np.allclose(sx, py)
    assert np.allclose(sy, px / 100.0)


def test_cdf_std_monotone_nondecreasing_prob():
    rec = _lin_rec("A", 12.0, 2.0, 30.0)
    x, y = _cdf_std_xy(rec)
    assert np.all(np.diff(y) >= -1e-9)          # cumulative prob non-decreasing
    assert 0.0 <= y.min() and y.max() <= 1.0    # a probability
    assert np.all(np.diff(x) >= -1e-9)          # points non-decreasing in level


def test_cdf_std_empty_on_bad_grid():
    bad = {"name": "x", "mean": 1.0, "median": 1.0,
           "quantiles": {"levels": [0.5], "values": [1.0]}}
    x, y = _cdf_std_xy(bad)
    assert x.size == 0 and y.size == 0


# --- two-player difference chart ----------------------------------------


def test_diff_orients_higher_mean_positive():
    # B has the higher mean, so B must be the positive direction regardless of
    # argument order.
    a = _lin_rec("A", 10.0, 0.0, 20.0)
    b = _lin_rec("B", 15.0, 5.0, 25.0)   # uniformly +5 over A
    x, diff, hi, lo = diff_series([("A", a), ("B", b)])
    assert hi == "B" and lo == "A"
    assert np.allclose(diff, 5.0)        # B - A = +5 at every percentile
    # Argument order must not change orientation.
    x2, diff2, hi2, lo2 = diff_series([("B", b), ("A", a)])
    assert hi2 == "B" and lo2 == "A"
    assert np.allclose(diff2, 5.0)


def test_diff_crossover_shows_negative_where_lower_mean_wins():
    # A has the higher mean but a LOWER floor: steep (boom/bust).
    # B has the lower mean but a higher floor: flat (safe).
    a = _lin_rec("A", 14.0, 0.0, 40.0)   # mean higher, floor 0, ceiling 40
    b = _lin_rec("B", 13.0, 8.0, 18.0)   # mean lower, floor 8, ceiling 18
    x, diff, hi, lo = diff_series([("A", a), ("B", b)])
    assert hi == "A" and lo == "B"
    # At the low end A trails B (B's higher floor) -> diff negative.
    assert diff[0] < 0
    # At the high end A beats B (A's higher ceiling) -> diff positive.
    assert diff[-1] > 0
    # Exactly one sign change (one crossover) for these linear curves.
    signs = np.sign(diff[np.abs(diff) > 1e-9])
    crossings = np.sum(np.diff(signs) != 0)
    assert crossings == 1


def test_diff_x_is_percentile_grid():
    a = _lin_rec("A", 10.0, 0.0, 20.0)
    b = _lin_rec("B", 11.0, 1.0, 21.0)
    x, diff, hi, lo = diff_series([("A", a), ("B", b)])
    assert x[0] == pytest.approx(0.1, abs=0.05)    # 0.001 * 100
    assert x[-1] == pytest.approx(99.9, abs=0.05)  # 0.999 * 100
    assert len(x) == len(diff) == len(_LEVELS)


def test_diff_requires_exactly_two_players():
    a = _lin_rec("A", 10.0, 0.0, 20.0)
    with pytest.raises(ValueError):
        diff_series([("A", a)])
    with pytest.raises(ValueError):
        diff_series([("A", a), ("A", a), ("A", a)])


def test_diff_interpolates_mismatched_grids():
    # B on a coarser grid than A: diff_series should interpolate B onto A's grid.
    a = _lin_rec("A", 10.0, 0.0, 20.0)
    coarse = np.linspace(0.001, 0.999, 20)
    b_vals = 2.0 + (22.0 - 2.0) * coarse   # uniformly +2 over A's line
    b = _rec("B", 12.0, coarse, b_vals)
    x, diff, hi, lo = diff_series([("A", a), ("B", b)])
    assert hi == "B" and lo == "A"
    assert len(diff) == len(_LEVELS)
    # B - A ≈ +2 everywhere (both are the same linear map, offset by 2).
    assert np.allclose(diff, 2.0, atol=0.1)
