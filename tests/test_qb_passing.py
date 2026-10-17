"""Tests for QB passing support: scoring, pass-TD ladder parsing, model build.

Covers the design invariants that matter for QBs:
  - passing yards (1 pt / 25 yds) and passing TDs (4 pt) score separately from
    the rushing/receiving anytime-TD count (6 pt), i.e. no double-count;
  - the pass-TD market is parsed as a two-sided-preferred NB ladder;
  - a QB model carries passing_yards + passing_tds alongside rushing stats, and
    simulates with all contributions added;
  - scoring files written before passing existed still load (backward compat).
"""
from __future__ import annotations

import numpy as np

from ffmc.scoring.config import ScoringConfig, StatLine, from_dict, half_ppr


def test_passing_scoring_math():
    """4 pts/pass TD, 1 pt/25 pass yds, with rush TD still 6 and separate."""
    cfg = ScoringConfig(
        points_per_reception=0.5,
        points_per_touchdown=6.0,
        points_per_passing_yard=0.04,
        points_per_passing_td=4.0,
    )
    line = StatLine(
        rushing_yards=30.0,   # 3.0
        touchdowns=1.0,       # 6.0 (rush/receive TD)
        passing_yards=250.0,  # 10.0
        passing_tds=2.0,      # 8.0
    )
    # 3.0 + 6.0 + 10.0 + 8.0 = 27.0
    assert abs(cfg.score(line) - 27.0) < 1e-9


def test_passing_td_and_rush_td_are_distinct():
    """A passing TD (4) and a rushing TD (6) are NOT pooled -- a QB who throws one
    and runs one in should score 10 from TDs, not 2*anything."""
    cfg = ScoringConfig(points_per_passing_td=4.0, points_per_touchdown=6.0)
    pass_only = cfg.score(StatLine(passing_tds=1.0))
    rush_only = cfg.score(StatLine(touchdowns=1.0))
    both = cfg.score(StatLine(passing_tds=1.0, touchdowns=1.0))
    assert abs(pass_only - 4.0) < 1e-9
    assert abs(rush_only - 6.0) < 1e-9
    assert abs(both - 10.0) < 1e-9


def test_scoring_file_backward_compatible_without_passing_keys():
    """A scoring dict written before passing existed still loads; passing points
    fall back to the sensible defaults (0.04 / 4.0)."""
    legacy = {
        "points_per_reception": 0.5,
        "points_per_receiving_yard": 0.1,
        "points_per_rushing_yard": 0.1,
        "points_per_touchdown": 6.0,
    }
    cfg = from_dict(legacy)
    assert cfg.points_per_passing_yard == 0.04
    assert cfg.points_per_passing_td == 4.0


def _qb_payload() -> dict:
    """A minimal FanDuel-shaped payload for one QB with pass yds + pass TDs +
    rushing + anytime-TD, mirroring the real market structure."""
    return {
        "bookmakers": [
            {
                "key": "fanduel",
                "markets": [
                    {
                        "key": "player_pass_yds",
                        "outcomes": [
                            {"name": "Over", "description": "QB", "point": 250.5, "price": -115},
                            {"name": "Under", "description": "QB", "point": 250.5, "price": -105},
                        ],
                    },
                    {
                        "key": "player_pass_yds_alternate",
                        "outcomes": [
                            {"name": "Over", "description": "QB", "point": 199.5, "price": -300},
                            {"name": "Over", "description": "QB", "point": 299.5, "price": 150},
                            {"name": "Over", "description": "QB", "point": 349.5, "price": 450},
                        ],
                    },
                    {
                        "key": "player_pass_tds",
                        "outcomes": [
                            {"name": "Over", "description": "QB", "point": 1.5, "price": -120},
                            {"name": "Under", "description": "QB", "point": 1.5, "price": 100},
                        ],
                    },
                    {
                        "key": "player_pass_tds_alternate",
                        "outcomes": [
                            {"name": "Over", "description": "QB", "point": 0.5, "price": -500},
                            {"name": "Over", "description": "QB", "point": 2.5, "price": 220},
                            {"name": "Over", "description": "QB", "point": 3.5, "price": 650},
                        ],
                    },
                    {
                        "key": "player_rush_yds",
                        "outcomes": [
                            {"name": "Over", "description": "QB", "point": 24.5, "price": -110},
                            {"name": "Under", "description": "QB", "point": 24.5, "price": -110},
                        ],
                    },
                    {
                        "key": "player_anytime_td",
                        "outcomes": [
                            {"name": "Yes", "description": "QB", "price": 180},
                        ],
                    },
                ],
            }
        ]
    }


def test_pass_td_anchors_two_sided_preferred_and_laddered():
    """player_pass_tds (two-sided) de-vigs multiplicatively; the alt Over rungs
    add more k anchors, giving a multi-anchor NB ladder."""
    from ffmc.odds.devig import devig_prob_from_pair, DevigMethod
    from ffmc.odds.markets import _pass_td_anchors

    anchors = _pass_td_anchors(_qb_payload(), "fanduel")["QB"]
    # Over 0.5->k1, Over 1.5->k2 (two-sided), Over 2.5->k3, Over 3.5->k4.
    assert sorted(anchors) == [1, 2, 3, 4]
    # k=2 came from the two-sided main line -> exact multiplicative de-vig.
    expected_k2 = devig_prob_from_pair(-120, 100, DevigMethod.MULTIPLICATIVE)
    assert abs(anchors[2] - expected_k2) < 1e-9
    # survival non-increasing in k
    vals = [anchors[k] for k in sorted(anchors)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


def test_qb_model_has_separate_passing_and_rushing_td_counts():
    from ffmc.odds.markets import build_player_models

    qb = build_player_models(_qb_payload(), book_key="fanduel")["QB"]
    assert qb.passing_yards is not None
    assert qb.passing_tds is not None
    assert qb.rushing_yards is not None
    assert qb.touchdowns is not None  # anytime (rush/receive) TD, distinct count
    # passing yards reproduce roughly the main 250.5 line region
    assert 180.0 < qb.passing_yards.expected_value() < 320.0
    # expected passing TDs is a real multi-anchor NB mean (~1-2)
    assert 0.8 < qb.passing_tds.expected_value() < 2.5
    # the anytime-TD (rush) count is a separate, smaller expectation
    assert qb.touchdowns.expected_value() < 1.0


def test_qb_simulation_adds_passing_contribution():
    """A QB's projected mean must reflect pass yds (0.04) + pass TD (4) on top of
    rushing and the anytime-TD count, and exceed what rushing alone would give."""
    from ffmc.odds.markets import build_player_models
    from ffmc.sim.montecarlo import simulate_player

    qb = build_player_models(_qb_payload(), book_key="fanduel")["QB"]
    cfg = half_ppr()  # includes passing defaults 0.04 / 4.0
    sd = simulate_player(qb, cfg, n_sims=40_000, seed=1)
    # ~250 pass yds*0.04 (=10) + ~1.3 pass TD*4 (~5) + rushing/anytime -> well above 12
    assert sd.mean > 12.0
    assert sd.floor < sd.mean < sd.ceiling


def test_non_qb_model_has_no_passing_stats():
    """A payload with no passing markets yields None passing marginals, so skill
    players are unaffected."""
    from ffmc.odds.markets import build_player_models

    payload = {
        "bookmakers": [
            {
                "key": "fanduel",
                "markets": [
                    {
                        "key": "player_reception_yds",
                        "outcomes": [
                            {"name": "Over", "description": "WR", "point": 60.5, "price": -110},
                            {"name": "Under", "description": "WR", "point": 60.5, "price": -110},
                        ],
                    },
                ],
            }
        ]
    }
    wr = build_player_models(payload, book_key="fanduel")["WR"]
    assert wr.passing_yards is None
    assert wr.passing_tds is None
