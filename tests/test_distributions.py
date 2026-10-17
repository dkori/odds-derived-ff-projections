import numpy as np

import json
from pathlib import Path

from ffmc.distributions.ladder import (
    ContinuousLadderDistribution,
    DiscreteLadderDistribution,
    LadderPoint,
    TAIL_SHAPE_RECEIVING,
    TAIL_SHAPE_RUSHING,
)
from ffmc.distributions.poisson import (
    PoissonTouchdownDistribution,
    lambda_from_anytime_td,
)


def test_continuous_cdf_monotonic():
    pts = [
        LadderPoint(40.0, 0.80),
        LadderPoint(60.0, 0.55),
        LadderPoint(80.0, 0.30),
        LadderPoint(100.0, 0.12),
    ]
    d = ContinuousLadderDistribution(pts)
    xs = np.linspace(0, 200, 500)
    cdf = np.array([d.cdf(x) for x in xs])
    assert np.all(np.diff(cdf) >= -1e-9)  # non-decreasing
    assert cdf[0] >= 0.0 and cdf[-1] <= 1.0 + 1e-9


def test_continuous_sampler_recovers_prob_over():
    # P(X > 60) should be ~0.55 given the ladder point at 60.
    pts = [
        LadderPoint(40.0, 0.80),
        LadderPoint(60.0, 0.55),
        LadderPoint(80.0, 0.30),
        LadderPoint(100.0, 0.12),
    ]
    d = ContinuousLadderDistribution(pts)
    rng = np.random.default_rng(0)
    s = d.sample(200_000, rng)
    assert abs(np.mean(s > 60.0) - 0.55) < 0.02
    assert abs(np.mean(s > 80.0) - 0.30) < 0.02
    assert np.all(s >= 0.0)  # yards floor


def test_discrete_pmf_normalized_and_recovers_prob():
    # P(X > 4.5) = P(X >= 5) should be ~0.45.
    pts = [
        LadderPoint(2.5, 0.72),
        LadderPoint(4.5, 0.45),
        LadderPoint(6.5, 0.18),
    ]
    d = DiscreteLadderDistribution(pts)
    assert abs(d.pmf().sum() - 1.0) < 1e-9
    rng = np.random.default_rng(1)
    s = d.sample(200_000, rng)
    assert abs(np.mean(s >= 5) - 0.45) < 0.03
    assert np.all(s == np.floor(s))  # integer counts


def test_poisson_td_rate():
    # P(>=1) = 0.5  =>  lambda = ln 2
    assert abs(lambda_from_anytime_td(0.5) - np.log(2)) < 1e-9
    d = PoissonTouchdownDistribution(0.5)
    rng = np.random.default_rng(2)
    s = d.sample(200_000, rng)
    assert abs(np.mean(s >= 1) - 0.5) < 0.01


def test_discrete_nbinom_fits_all_anchors():
    """NB fit reproduces every posted anchor simultaneously (not just one)."""
    pts = [
        LadderPoint(2.5, 0.72),
        LadderPoint(4.5, 0.45),
        LadderPoint(6.5, 0.18),
    ]
    d = DiscreteLadderDistribution(pts)
    cdf = np.cumsum(d.pmf())
    for lp in pts:
        k = int(np.floor(lp.threshold))
        model_over = 1.0 - cdf[k]
        assert abs(model_over - lp.prob_over) < 0.04
    # Overdispersed relative to Poisson (variance/mean > 1).
    assert 1.0 + d.alpha * d.mu > 1.0


def test_discrete_upper_tail_decays_not_clamped():
    """Beyond the top posted line the PMF should decay, with no mass spike at k_max."""
    pts = [LadderPoint(2.5, 0.72), LadderPoint(4.5, 0.45), LadderPoint(6.5, 0.18)]
    d = DiscreteLadderDistribution(pts)
    pmf = d.pmf()
    # Tail past the highest posted integer (6) must be monotonically small,
    # and the final cell must not be an anomalous spike (the old clamp bug).
    top = 7
    assert np.all(np.diff(pmf[top:]) <= 1e-6)  # non-increasing tail
    assert pmf[-1] < pmf[top]                   # last cell is tiny, not a dump


def test_discrete_single_line_fallback():
    """A single posted line is underdetermined; fit still matches that point."""
    d = DiscreteLadderDistribution([LadderPoint(4.5, 0.45)])
    cdf = np.cumsum(d.pmf())
    assert abs((1.0 - cdf[4]) - 0.45) < 0.04
    assert abs(d.pmf().sum() - 1.0) < 1e-9


# --- yards upper tail (fat-tail fix, Bug B) -------------------------------


def _rb_rush_ladder():
    """A high-usage RB rushing ladder (P(X > threshold)), like a real workhorse."""
    return [
        LadderPoint(49.5, 0.927),
        LadderPoint(69.5, 0.758),
        LadderPoint(89.5, 0.553),
        LadderPoint(109.5, 0.358),
        LadderPoint(124.5, 0.260),
        LadderPoint(149.5, 0.121),
        LadderPoint(174.5, 0.065),
    ]


def test_steeper_tail_lightens_far_quantiles():
    """A larger tail_shape (faster-than-exponential decay) must pull the far
    upper quantiles DOWN relative to the plain-exponential tail (shape=1)."""
    pts = _rb_rush_ladder()
    exp_tail = ContinuousLadderDistribution(pts, tail_shape=1.0)
    steep = ContinuousLadderDistribution(pts, tail_shape=1.5)
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    s_exp = exp_tail.sample(300_000, rng1)
    s_steep = steep.sample(300_000, rng2)
    # Far tail is lighter (smaller) with the steeper shape.
    assert np.quantile(s_steep, 0.999) < np.quantile(s_exp, 0.999)
    assert s_steep.max() < s_exp.max()


def test_steeper_tail_preserves_posted_anchor_probs():
    """Steepening the *extrapolated* tail must not disturb the posted anchors
    (which live in the interpolated interior)."""
    pts = _rb_rush_ladder()
    d = ContinuousLadderDistribution(pts, tail_shape=1.5)
    rng = np.random.default_rng(1)
    s = d.sample(300_000, rng)
    for lp in pts[:-1]:  # interior anchors
        assert abs(np.mean(s > lp.threshold) - lp.prob_over) < 0.02


def test_steeper_tail_caps_far_ceiling():
    """The far ceiling of a big RB rushing line should be physically plausible:
    the 99.9th percentile stays well under ~320 yds (was ~330-540 with the old
    two-point exponential tail)."""
    d = ContinuousLadderDistribution(_rb_rush_ladder(), tail_shape=1.5)
    rng = np.random.default_rng(2)
    s = d.sample(300_000, rng)
    assert np.quantile(s, 0.999) < 320.0


# --- single-anchor fallback (main-line-only) ------------------------------


def test_single_anchor_flags_and_honors_line():
    """A lone main line: the distribution reports itself as single-anchor and
    still reproduces P(X > line)."""
    d = ContinuousLadderDistribution([LadderPoint(47.5, 0.50)])
    assert d.is_single_anchor is True
    assert d.n_anchors == 1
    rng = np.random.default_rng(0)
    s = d.sample(300_000, rng)
    assert abs(np.mean(s > 47.5) - 0.50) < 0.02


def test_single_anchor_tail_not_fabricated():
    """The single-anchor fallback must NOT fabricate an absurd ceiling. A 47.5-yd
    line at P=0.5 previously implied a ~315-yd 99th percentile (rate=1/L
    exponential); the pinned super-exponential form keeps q99 far tighter."""
    d = ContinuousLadderDistribution([LadderPoint(47.5, 0.50)], tail_shape=1.5)
    rng = np.random.default_rng(1)
    s = d.sample(300_000, rng)
    # Median sits at the line; q99 is plausible (well under the old ~315).
    assert abs(np.median(s) - 47.5) < 6.0
    assert np.quantile(s, 0.99) < 200.0


def test_single_anchor_lighter_than_old_exponential():
    """The new pinned form has a lighter upper tail than the old rate=1/L
    exponential fallback (tail_shape=1.0 with the same pin is heavier)."""
    pts = [LadderPoint(52.5, 0.50)]
    steep = ContinuousLadderDistribution(pts, tail_shape=1.5)
    flat = ContinuousLadderDistribution(pts, tail_shape=1.0)
    rng1 = np.random.default_rng(2)
    rng2 = np.random.default_rng(2)
    s_steep = steep.sample(300_000, rng1)
    s_flat = flat.sample(300_000, rng2)
    assert np.quantile(s_steep, 0.99) < np.quantile(s_flat, 0.99)


def test_measured_tail_shape_constants_are_calibrated_values():
    """The pooled constants are the values measured in
    analysis/yards_tail_shape_check.py (receiving ~1.15, rushing ~1.10), and
    are both near-exponential (< the old hand-picked 1.5)."""
    assert TAIL_SHAPE_RECEIVING == 1.15
    assert TAIL_SHAPE_RUSHING == 1.10
    # Rushing tail is at least as thin-decaying (lower exponent) as receiving,
    # matching the measured medians (rush 1.05 < rec 1.12).
    assert TAIL_SHAPE_RUSHING <= TAIL_SHAPE_RECEIVING < 1.5


def test_markets_wires_measured_tail_shape_per_stat():
    """build_player_models must pass the stat-appropriate measured constant:
    receiving_yards -> TAIL_SHAPE_RECEIVING, rushing_yards -> TAIL_SHAPE_RUSHING.
    (Guards against a future refactor silently reverting to a single default.)"""
    from ffmc.odds.markets import build_player_models

    payload = json.loads(
        (Path(__file__).parent.parent / "data/samples/sample_rb_event.json").read_text()
    )
    rb = build_player_models(payload, book_key="draftkings")["Sample RB"]
    assert rb.receiving_yards is not None
    assert rb.rushing_yards is not None
    assert rb.receiving_yards.tail_shape == TAIL_SHAPE_RECEIVING
    assert rb.rushing_yards.tail_shape == TAIL_SHAPE_RUSHING
