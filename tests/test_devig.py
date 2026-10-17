import math

from ffmc.odds.devig import (
    american_to_implied_prob,
    decimal_to_implied_prob,
    devig_pair,
    devig_prob_from_pair,
)


def test_american_to_implied_prob_negative():
    assert math.isclose(american_to_implied_prob(-110), 110 / 210, rel_tol=1e-9)


def test_american_to_implied_prob_positive():
    assert math.isclose(american_to_implied_prob(+120), 100 / 220, rel_tol=1e-9)


def test_decimal_to_implied_prob():
    assert math.isclose(decimal_to_implied_prob(2.0), 0.5, rel_tol=1e-9)


def test_devig_sums_to_one():
    r = devig_pair(-110, -110)
    assert math.isclose(r.over + r.under, 1.0, rel_tol=1e-12)
    assert math.isclose(r.over, 0.5, rel_tol=1e-9)
    assert r.overround > 1.0  # there was vig to remove


def test_devig_preserves_ratio():
    # Heavy favorite over: de-vigged over should stay > 0.5.
    r = devig_pair(-260, +200)
    assert r.over > 0.5
    assert math.isclose(r.over + r.under, 1.0, rel_tol=1e-12)


def test_devig_prob_from_pair_matches():
    assert devig_prob_from_pair(-140, 110) == devig_pair(-140, 110).over


# --- one-sided (Over-only) alt-ladder calibration -------------------------

from ffmc.odds.devig import (
    calibrate_oneside_ladder,
    oneside_power_exponent,
)


def test_power_exponent_pins_main_anchor():
    # pi chosen so p_raw ** pi == p_true
    p_raw, p_true = 0.526, 0.498
    pi = oneside_power_exponent(p_raw, p_true)
    assert math.isclose(p_raw ** pi, p_true, rel_tol=1e-9)
    assert pi > 1.0  # deflating (raw > true) requires exponent > 1 for p<1


def test_power_exponent_identity_on_degenerate():
    assert oneside_power_exponent(1.0, 0.5) == 1.0
    assert oneside_power_exponent(0.5, 0.0) == 1.0


def test_calibrate_without_anchor_is_identity():
    raw = {10.0: 0.8, 20.0: 0.5, 30.0: 0.2}
    out = calibrate_oneside_ladder(raw)
    assert out == raw


def test_calibrate_deflates_and_pins_main():
    raw = {14.5: 0.867, 43.5: 0.526, 149.5: 0.024}
    out = calibrate_oneside_ladder(raw, p_raw_main=0.526, p_true_main=0.498)
    # main line deflated to (approx) the de-vigged truth
    assert math.isclose(out[43.5], 0.498, rel_tol=1e-6)
    # every anchor deflated (raw carried vig, so calibrated <= raw)
    for line in raw:
        assert out[line] <= raw[line] + 1e-9
    # monotone decreasing preserved
    vals = [out[l] for l in sorted(out)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))
    # longshot (deep tail) deflated by a LARGER factor than a near-favorite
    fav_ratio = out[14.5] / raw[14.5]
    long_ratio = out[149.5] / raw[149.5]
    assert long_ratio < fav_ratio


def test_parser_builds_calibrated_multianchor_ladder_from_over_only_alts():
    """Regression: Over-only alt lines must yield a multi-anchor ladder pinned at
    the de-vigged main line (the single-anchor bug must not return)."""
    from ffmc.odds.markets import _ladder_for

    payload = {
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {  # main line: two-sided
                        "key": "player_reception_yds",
                        "outcomes": [
                            {"name": "Over", "description": "X", "point": 43.5, "price": -111},
                            {"name": "Under", "description": "X", "point": 43.5, "price": -113},
                        ],
                    },
                    {  # alt lines: Over-only ladder
                        "key": "player_reception_yds_alternate",
                        "outcomes": [
                            {"name": "Over", "description": "X", "point": 14.5, "price": -650},
                            {"name": "Over", "description": "X", "point": 69.5, "price": +180},
                            {"name": "Over", "description": "X", "point": 99.5, "price": +450},
                        ],
                    },
                ],
            }
        ]
    }
    ladders = _ladder_for(
        payload,
        ["player_reception_yds", "player_reception_yds_alternate"],
        "draftkings",
    )
    pts = ladders["X"]
    thresholds = [p.threshold for p in pts]
    # all four thresholds survive (not collapsed to a single anchor)
    assert thresholds == [14.5, 43.5, 69.5, 99.5]
    # main line pinned to its de-vigged value
    main = next(p for p in pts if p.threshold == 43.5)
    assert math.isclose(main.prob_over, devig_prob_from_pair(-111, -113), rel_tol=1e-9)
    # monotone decreasing survival
    probs = [p.prob_over for p in pts]
    assert all(a >= b for a, b in zip(probs, probs[1:]))
