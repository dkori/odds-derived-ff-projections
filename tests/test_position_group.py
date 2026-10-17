"""Tests for the QB vs skill (WR/RB/TE) position-group split.

Quarterbacks are now modeled (passing marginals) and kept in the store, tagged
``position_group`` so the Flask app can compare them separately from skill
players. These tests cover:
  - ``classify_group`` keys off passing-stat presence (position-agnostic: it
    reads the posted markets, not a position label);
  - the slate record carries ``position_group`` and group-aware provenance;
  - ``ProjectionsStore`` lists/searches scoped to a group, including the
    legacy-record fallback (no ``position_group`` field -> infer from markets);
  - ``data_notes`` reports only group-relevant missing stats and surfaces the
    QB passing-TD badge.
"""
from __future__ import annotations

import json

from ffmc.odds.markets import build_player_models
from ffmc.sim.slate import _data_provenance, classify_group
from ffmc.sim.store import ProjectionsStore
from ffmc.sim.viz import data_notes


def _qb_payload():
    """A QB (passing + rushing) and a skill RB (rushing + TD) in one game."""
    return {
        "bookmakers": [
            {
                "key": "fanduel",
                "markets": [
                    {
                        "key": "player_pass_yds",
                        "outcomes": [
                            {"name": "Over", "description": "Josh Allen", "point": 249.5, "price": -110},
                            {"name": "Under", "description": "Josh Allen", "point": 249.5, "price": -110},
                        ],
                    },
                    {
                        "key": "player_pass_tds",
                        "outcomes": [
                            {"name": "Over", "description": "Josh Allen", "point": 1.5, "price": -120},
                            {"name": "Under", "description": "Josh Allen", "point": 1.5, "price": -110},
                        ],
                    },
                    {
                        "key": "player_rush_yds",
                        "outcomes": [
                            {"name": "Over", "description": "Josh Allen", "point": 39.5, "price": -110},
                            {"name": "Under", "description": "Josh Allen", "point": 39.5, "price": -110},
                            {"name": "Over", "description": "Saquon Barkley", "point": 79.5, "price": -110},
                            {"name": "Under", "description": "Saquon Barkley", "point": 79.5, "price": -110},
                        ],
                    },
                    {
                        "key": "player_anytime_td",
                        "outcomes": [
                            {"name": "Yes", "description": "Saquon Barkley", "price": -160},
                        ],
                    },
                ],
            }
        ]
    }


def test_classify_group_from_passing_markets():
    models = build_player_models(_qb_payload(), book_key="fanduel")
    assert classify_group(models["Josh Allen"]) == "qb"
    assert classify_group(models["Saquon Barkley"]) == "skill"


def test_provenance_is_group_aware():
    models = build_player_models(_qb_payload(), book_key="fanduel")
    qb_prov = _data_provenance(models["Josh Allen"])
    # QB: passing yards present; a QB is never flagged "missing receptions".
    assert "passing_yards" in qb_prov["present"]
    assert "receptions" not in qb_prov["missing"]
    assert "receiving_yards" not in qb_prov["missing"]
    assert "pass_td" in qb_prov  # QB-only provenance block

    rb_prov = _data_provenance(models["Saquon Barkley"])
    # Skill player: passing stats never appear in present or missing.
    assert "passing_yards" not in rb_prov["present"]
    assert "passing_yards" not in rb_prov["missing"]
    assert "passing_tds" not in rb_prov["missing"]
    assert "pass_td" not in rb_prov


def _mk_store(tmp_path, players):
    store_dir = tmp_path / "proj"
    week_dir = store_dir / "week9"
    week_dir.mkdir(parents=True)
    snap = {
        "week_label": "week9",
        "snapshot_id": "2026-10-01T0000Z",
        "players": players,
    }
    (week_dir / "latest.json").write_text(json.dumps(snap))
    (week_dir / "index.json").write_text(
        json.dumps({"week_label": "week9", "latest": snap["snapshot_id"],
                    "snapshots": [{"snapshot_id": snap["snapshot_id"], "file": "latest.json"}]})
    )
    return ProjectionsStore(store_dir)


def test_store_filters_by_group(tmp_path):
    players = {
        "Josh Allen": {"name": "Josh Allen", "position_group": "qb",
                       "markets_present": ["rushing_yards", "passing_yards", "passing_tds"]},
        "Saquon Barkley": {"name": "Saquon Barkley", "position_group": "skill",
                           "markets_present": ["rushing_yards", "touchdowns"]},
        "CeeDee Lamb": {"name": "CeeDee Lamb", "position_group": "skill",
                        "markets_present": ["receptions", "receiving_yards"]},
    }
    store = _mk_store(tmp_path, players)
    assert store.player_names("week9", group="qb") == ["Josh Allen"]
    assert store.player_names("week9", group="skill") == ["CeeDee Lamb", "Saquon Barkley"]
    assert len(store.player_names("week9")) == 3  # None -> all
    # Search is also scoped.
    assert store.search_players("week9", "a", group="qb") == ["Josh Allen"]
    assert "Josh Allen" not in store.search_players("week9", "a", group="skill")


def test_store_group_fallback_for_legacy_records(tmp_path):
    """A record without position_group is classified from markets_present."""
    players = {
        "Legacy QB": {"name": "Legacy QB",
                      "markets_present": ["rushing_yards", "passing_yards"]},
        "Legacy WR": {"name": "Legacy WR",
                      "markets_present": ["receptions", "receiving_yards"]},
    }
    store = _mk_store(tmp_path, players)
    assert ProjectionsStore.record_group(players["Legacy QB"]) == "qb"
    assert ProjectionsStore.record_group(players["Legacy WR"]) == "skill"
    assert store.player_names("week9", group="qb") == ["Legacy QB"]


def test_data_notes_surfaces_passing_for_qb():
    rec = {
        "position_group": "qb",
        "markets_present": ["rushing_yards", "passing_yards", "passing_tds"],
        "data_provenance": {
            "present": ["rushing_yards", "passing_yards", "passing_tds"],
            "missing": [],
            "td": {"available": False, "ladder": False, "n_anchors": 0, "model": None},
            "pass_td": {"available": True, "ladder": True, "n_anchors": 3,
                        "model": "nbinom_ladder"},
        },
    }
    notes = data_notes(rec)
    assert notes["group"] == "qb"
    assert "passing yds" in notes["present"]
    assert notes["pass_td"]["available"] is True
    assert "passing-TD ladder" in notes["summary"]


def test_percentile_at_inverts_stored_quantiles():
    """percentile_at(value) should recover the cumulative level of the stored
    dense quantile grid (the hover-percentile feature relies on this exactness)."""
    from ffmc.sim.viz import percentile_at

    # A simple monotonic CDF grid: value == 100*level, so pct(x) == x.
    rec = {
        "quantiles": {
            "levels": [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
            "values": [0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0],
        }
    }
    assert abs(float(percentile_at(rec, 10.0)) - 10.0) < 1e-6
    assert abs(float(percentile_at(rec, 50.0)) - 50.0) < 1e-6
    assert abs(float(percentile_at(rec, 90.0)) - 90.0) < 1e-6
    # Interpolates between grid nodes.
    assert abs(float(percentile_at(rec, 17.5)) - 17.5) < 1e-6
    # Clamps outside the grid range.
    assert float(percentile_at(rec, -5.0)) == 0.0
    assert float(percentile_at(rec, 200.0)) == 100.0
    # Degenerate grid -> nan (no crash).
    import numpy as np
    assert np.isnan(float(percentile_at({"quantiles": {"levels": [], "values": []}}, 5.0)))


def test_cdf_builder_from_quantile_grid():
    """CDF mode draws points-vs-cumulative-percent straight from the stored grid
    (no re-sim), monotone, 0..100, and selecting cdf never falls back to small
    multiples regardless of player count."""
    import numpy as np

    from ffmc.sim.viz import _cdf_xy, build_figure

    rec = {
        "mean": 19.3, "median": 19.3,
        "quantiles": {
            "levels": [0.0, 0.1, 0.5, 0.9, 1.0],
            "values": [0.0, 11.5, 19.3, 29.0, 40.0],
        },
    }
    x, y = _cdf_xy(rec)
    # Axes swapped: x is percentile (0..100), y is points.
    assert list(x) == [0.0, 10.0, 50.0, 90.0, 100.0]
    assert bool(np.all(np.diff(y) >= 0))  # points non-decreasing with percentile
    # points at the 50th percentile is the median.
    assert abs(float(np.interp(50.0, x, y)) - 19.3) < 1e-6

    store = {"week_label": "wk", "scoring": "half_ppr", "correlated": True, "book": "fd"}
    # 5 players in CDF mode must stay a single overlaid figure (5 traces), not
    # a small-multiples grid.
    series = [(f"P{i}", rec) for i in range(5)]
    fig = build_figure(store, series, mode="cdf")
    assert len(fig.data) == 5
    assert fig.layout.yaxis.title.text == "fantasy points"
    assert fig.layout.xaxis.title.text == "percentile"
