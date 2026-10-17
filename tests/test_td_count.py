"""Tests for the TD-count model, the one-sided TD de-vig, and TD-anchor parsing.

These cover the fat-upper-tail fix (see docs/DECISIONS.md, "TD count from a
ladder"): the anytime-TD prob is de-vigged, and a negative-binomial fit to the
TD ladder replaces the single-parameter Poisson so the multi-TD tail is priced
by the market rather than forced by one rate.
"""
import math

import numpy as np

from ffmc.distributions.poisson import lambda_from_anytime_td
from ffmc.distributions.td_count import TouchdownCountDistribution
from ffmc.odds.devig import (
    DEFAULT_TD_HOLD,
    american_to_implied_prob,
    devig_oneside_td_prob,
)


# --- one-sided TD de-vig ---------------------------------------------------


def test_td_devig_deflates_by_hold():
    # p_true = p_raw / (1 + hold)
    assert math.isclose(devig_oneside_td_prob(0.767, 0.06), 0.767 / 1.06, rel_tol=1e-9)
    # Default hold is applied when not specified.
    assert math.isclose(
        devig_oneside_td_prob(0.5), 0.5 / (1 + DEFAULT_TD_HOLD), rel_tol=1e-9
    )


def test_td_devig_zero_hold_is_identity():
    assert math.isclose(devig_oneside_td_prob(0.42, 0.0), 0.42, rel_tol=1e-9)


def test_td_devig_matches_gibbs_sanity_target():
    # WIP sanity target: raw 0.767/0.385 -> ~0.72 / ~0.36 at a 6% hold.
    assert abs(devig_oneside_td_prob(0.767) - 0.72) < 0.02
    assert abs(devig_oneside_td_prob(0.385) - 0.36) < 0.02


# --- negative-binomial TD-count fit ---------------------------------------


def _survival_from_pmf(pmf: np.ndarray, k: int) -> float:
    cdf = np.cumsum(pmf)
    return float(1.0 - cdf[k - 1])


def test_td_nbinom_reproduces_all_anchors():
    """NB fit to 4 de-vigged Gibbs anchors reproduces each P(>=k)."""
    raw = {
        1: american_to_implied_prob(-330),
        2: american_to_implied_prob(160),
        3: american_to_implied_prob(700),
        4: american_to_implied_prob(3000),
    }
    anchors = {k: devig_oneside_td_prob(v) for k, v in raw.items()}
    d = TouchdownCountDistribution(anchors)
    for k, target in anchors.items():
        assert abs(_survival_from_pmf(d.pmf(), k) - target) < 0.03
    # Mean is a plausible sub-1.5 for a high-usage back.
    assert 0.8 < d.expected_value() < 1.6


def test_td_nbinom_thinner_multi_td_tail_than_poisson():
    """With P(>=1) fixed, the anchored NB puts less mass on >=3 TDs than the
    single-parameter Poisson would (the fat-tail driver)."""
    anchors = {1: 0.72, 2: 0.30, 3: 0.09}
    nb = TouchdownCountDistribution(anchors)
    lam = lambda_from_anytime_td(anchors[1])
    from scipy.stats import poisson

    poisson_ge3 = float(1.0 - poisson.cdf(2, lam))
    nb_ge3 = _survival_from_pmf(nb.pmf(), 3)
    assert nb_ge3 < poisson_ge3


def test_td_single_anchor_poisson_fallback():
    """One anchor (anytime-TD only) falls back to the Poisson shape."""
    d = TouchdownCountDistribution({1: 0.5})
    # P(>=1) should match; mean ~ lambda = ln 2.
    assert abs(_survival_from_pmf(d.pmf(), 1) - 0.5) < 0.02
    assert abs(d.expected_value() - math.log(2)) < 0.05


def test_td_hard_cap_at_k_max():
    """Sampled counts never exceed k_max, and the PMF is normalized."""
    d = TouchdownCountDistribution({1: 0.9, 2: 0.7, 3: 0.5}, k_max=4)
    assert abs(d.pmf().sum() - 1.0) < 1e-9
    rng = np.random.default_rng(0)
    s = d.sample(100_000, rng)
    assert s.max() <= 4
    assert np.all(s == np.floor(s))  # integer counts


def test_td_invert_is_monotone_integer_ppf():
    """invert() must return non-decreasing integer counts (copula contract)."""
    d = TouchdownCountDistribution({1: 0.72, 2: 0.36, 3: 0.12})
    u = np.linspace(0.0, 1.0, 50)
    counts = d.invert(u)
    assert np.all(np.diff(counts) >= 0)
    assert np.all(counts == np.floor(counts))


# --- TD-anchor parsing from a payload -------------------------------------


def test_parser_builds_td_ladder_anchors():
    """anytime-TD (Yes) -> k=1; player_tds_over 'Over j.5' -> k=j+1, all de-vigged
    by the assumed hold (one-sided)."""
    from ffmc.odds.markets import _td_anchors

    payload = {
        "bookmakers": [
            {
                "key": "fanduel",
                "markets": [
                    {
                        "key": "player_anytime_td",
                        "outcomes": [
                            {"name": "Yes", "description": "Gibbs", "price": -330},
                        ],
                    },
                    {
                        "key": "player_tds_over",
                        "outcomes": [
                            {"name": "Over", "description": "Gibbs", "point": 1.5, "price": 160},
                            {"name": "Over", "description": "Gibbs", "point": 2.5, "price": 700},
                            {"name": "Over", "description": "Gibbs", "point": 3.5, "price": 3000},
                        ],
                    },
                ],
            }
        ]
    }
    anchors = _td_anchors(payload, "fanduel")["Gibbs"]
    assert sorted(anchors) == [1, 2, 3, 4]  # Over 1.5 -> P(>=2), etc.
    # k=1 is de-vigged anytime-TD (raw ~0.767 -> ~0.72).
    assert abs(anchors[1] - devig_oneside_td_prob(american_to_implied_prob(-330))) < 1e-9
    assert abs(anchors[2] - devig_oneside_td_prob(american_to_implied_prob(160))) < 1e-9
    # Survival is non-increasing in k.
    vals = [anchors[k] for k in sorted(anchors)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


def test_parser_prefers_two_sided_td_anchor_no_assumption():
    """A two-sided player_tds Over/Under for a k is de-vigged multiplicatively and
    preferred over the assumed-hold one-sided value for the same k."""
    from ffmc.odds.devig import devig_prob_from_pair
    from ffmc.odds.markets import _td_anchors

    payload = {
        "bookmakers": [
            {
                "key": "fanduel",
                "markets": [
                    {  # one-sided Over 1.5 -> k=2
                        "key": "player_tds_over",
                        "outcomes": [
                            {"name": "Over", "description": "X", "point": 1.5, "price": 160},
                        ],
                    },
                    {  # two-sided player_tds Over/Under 1.5 -> k=2 (preferred)
                        "key": "player_tds",
                        "outcomes": [
                            {"name": "Over", "description": "X", "point": 1.5, "price": 150},
                            {"name": "Under", "description": "X", "point": 1.5, "price": -180},
                        ],
                    },
                ],
            }
        ]
    }
    anchors = _td_anchors(payload, "fanduel")["X"]
    expected = devig_prob_from_pair(150, -180)
    assert abs(anchors[2] - expected) < 1e-9


# --- provenance wiring (model -> slate record) ----------------------------


def test_data_provenance_reports_ladder_and_missing_stats():
    """slate._data_provenance summarizes present/missing stats and TD richness
    from a built PlayerModel."""
    from ffmc.odds.markets import build_player_models
    from ffmc.sim.slate import _data_provenance

    payload = {
        "bookmakers": [
            {
                "key": "fanduel",
                "markets": [
                    {  # rushing only (a pure rusher: no receptions/rec-yds)
                        "key": "player_rush_yds",
                        "outcomes": [
                            {"name": "Over", "description": "RB", "point": 60.5, "price": -110},
                            {"name": "Under", "description": "RB", "point": 60.5, "price": -110},
                        ],
                    },
                    {"key": "player_anytime_td",
                     "outcomes": [{"name": "Yes", "description": "RB", "price": -150}]},
                    {"key": "player_tds_over",
                     "outcomes": [
                         {"name": "Over", "description": "RB", "point": 1.5, "price": 200},
                         {"name": "Over", "description": "RB", "point": 2.5, "price": 900},
                     ]},
                ],
            }
        ]
    }
    model = build_player_models(payload, book_key="fanduel")["RB"]
    prov = _data_provenance(model)
    assert prov["present"] == ["rushing_yards", "touchdowns"]
    assert set(prov["missing"]) == {"receptions", "receiving_yards"}
    assert prov["td"]["ladder"] is True
    assert prov["td"]["n_anchors"] == 3  # P>=1, P>=2, P>=3
    assert prov["td"]["model"] == "nbinom_ladder"


def test_data_provenance_flags_single_anchor_low_confidence():
    """A rec-yds line with only the main line (no alternates) -> low_confidence,
    with the anchor count recorded."""
    from ffmc.odds.markets import build_player_models
    from ffmc.sim.slate import _data_provenance

    payload = {
        "bookmakers": [
            {
                "key": "fanduel",
                "markets": [
                    {  # main line only, no _alternate ladder
                        "key": "player_reception_yds",
                        "outcomes": [
                            {"name": "Over", "description": "WR", "point": 47.5, "price": -110},
                            {"name": "Under", "description": "WR", "point": 47.5, "price": -110},
                        ],
                    },
                    {"key": "player_anytime_td",
                     "outcomes": [{"name": "Yes", "description": "WR", "price": -120}]},
                ],
            }
        ]
    }
    prov = _data_provenance(build_player_models(payload, book_key="fanduel")["WR"])
    assert prov["anchor_counts"]["receiving_yards"] == 1
    assert prov["single_anchor_stats"] == ["receiving_yards"]
    assert prov["low_confidence"] is True
