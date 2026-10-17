"""Tests for the finer distribution fields and the versioned store layout."""
from __future__ import annotations

import numpy as np

from ffmc.sim.montecarlo import (
    ScoreDistribution,
    default_quantile_levels,
)
from ffmc.sim.slate import write_snapshot
from ffmc.sim.store import ProjectionsStore


def _dist(seed: int = 0, n: int = 20_000) -> ScoreDistribution:
    rng = np.random.default_rng(seed)
    # A skewed, non-negative distribution resembling fantasy points.
    samples = np.abs(rng.normal(14.0, 8.0, size=n))
    return ScoreDistribution(name="Test Player", samples=samples)


# --- distribution fields -------------------------------------------------


def test_default_quantile_levels_dense_and_tail_heavy():
    levels = default_quantile_levels()
    # Far more than the old 9-point grid, sorted, unique, within (0,1).
    assert len(levels) > 90
    assert list(levels) == sorted(levels)
    assert len(set(levels.tolist())) == len(levels)
    assert levels[0] <= 0.001 and levels[-1] >= 0.999
    # Tail-dense: nodes exist below 0.05 and above 0.95.
    assert (levels < 0.05).sum() >= 5
    assert (levels > 0.95).sum() >= 5


def test_dense_quantiles_monotone_and_aligned():
    d = _dist()
    dq = d.dense_quantiles()
    assert len(dq["levels"]) == len(dq["values"])
    # Quantile function is non-decreasing.
    vals = dq["values"]
    assert all(vals[i] <= vals[i + 1] + 1e-6 for i in range(len(vals) - 1))
    # p50 should be close to the sample median.
    mid = dq["levels"].index(0.5)
    assert abs(dq["values"][mid] - d.median) < 0.5


def test_histogram_counts_sum_to_n():
    d = _dist()
    h = d.histogram(bins=64)
    assert len(h["counts"]) == 64
    assert len(h["edges"]) == 65
    assert h["n"] == d.samples.size
    # Overflow folded into the last bin -> counts sum to n exactly.
    assert sum(h["counts"]) == d.samples.size


def test_histogram_degenerate_all_zero():
    d = ScoreDistribution(name="Zero", samples=np.zeros(1000))
    h = d.histogram(bins=16)
    # No crash on a zero-width span; still sums to n.
    assert sum(h["counts"]) == 1000
    assert h["edges"][-1] > h["edges"][0]


def test_histogram_caps_at_q99_by_default():
    """The stored histogram spans to q99 (not q99.9), so a heavy upper tail no
    longer stretches the drawn x-axis. The full tail still lives in the dense
    quantile grid."""
    rng = np.random.default_rng(3)
    # Heavy-tailed sample: a few extreme values well past q99.
    samples = np.concatenate([np.abs(rng.normal(14, 8, 20_000)), [200.0, 300.0]])
    d = ScoreDistribution(name="Heavy", samples=samples)
    h = d.histogram()
    top_edge = h["edges"][-1]
    assert abs(top_edge - float(np.quantile(samples, 0.99))) < 1e-2
    # q99 is far below the extreme max, so the axis is not stretched to 300.
    assert top_edge < 100.0


def test_viz_density_grid_does_not_pad_past_edges():
    """_smooth_density must not extend the grid past the histogram's last bin
    center (the old one-bin pad painted a phantom hump from ~0 tail mass)."""
    from ffmc.sim.viz import _smooth_density

    centers = np.linspace(1.0, 50.0, 64)
    density = np.ones_like(centers) / centers.size
    grid, _ = _smooth_density(centers, density)
    assert grid[0] >= centers[0] - 1e-9
    assert grid[-1] <= centers[-1] + 1e-9


# --- data-provenance notes ------------------------------------------------


def test_data_notes_full_ladder_receiver():
    """A receiver with no rushing line and a TD ladder: missing rushing flagged,
    TD reported as a full ladder with its anchor count."""
    from ffmc.sim.viz import data_notes

    rec = {
        "markets_present": ["receptions", "receiving_yards", "touchdowns"],
        "data_provenance": {
            "present": ["receptions", "receiving_yards", "touchdowns"],
            "missing": ["rushing_yards"],
            "td": {"available": True, "ladder": True, "n_anchors": 3,
                   "model": "nbinom_ladder"},
        },
    }
    nd = data_notes(rec)
    assert "rushing yds" in nd["missing"]
    assert nd["td"]["ladder"] is True and nd["td"]["n_anchors"] == 3
    assert "no rushing yds" in nd["summary"]
    assert "full TD ladder (3 anchors)" in nd["summary"]


def test_data_notes_anytime_only_and_no_td():
    from ffmc.sim.viz import data_notes

    anytime = {
        "data_provenance": {
            "present": ["rushing_yards", "touchdowns"],
            "missing": ["receptions", "receiving_yards"],
            "td": {"available": True, "ladder": False, "n_anchors": 1,
                   "model": "poisson_anytime"},
        }
    }
    assert "anytime-TD only" in data_notes(anytime)["summary"]

    no_td = {
        "data_provenance": {
            "present": ["rushing_yards"],
            "missing": ["receptions", "receiving_yards", "touchdowns"],
            "td": {"available": False, "ladder": False, "n_anchors": 0, "model": None},
        }
    }
    nd = data_notes(no_td)
    assert nd["td"]["available"] is False
    assert "no TD line" in nd["summary"]
    # 'touchdowns' is not double-reported in the missing-stat list (the TD note
    # covers it).
    assert "touchdowns" not in nd["missing"]


def test_data_notes_legacy_record_fallback():
    """Older snapshots have only markets_present; TD richness reports unknown."""
    from ffmc.sim.viz import data_notes

    legacy = {"markets_present": ["receptions", "receiving_yards", "touchdowns"]}
    nd = data_notes(legacy)
    assert "receptions" in nd["present"]
    assert nd["td"]["available"] is True
    assert nd["td"]["ladder"] is None  # unknown for legacy
    assert "ladder unknown" in nd["summary"]


def test_data_notes_low_confidence_single_anchor():
    """A continuous stat with a single anchor (main line only) is flagged
    low-confidence with a warning in the summary."""
    from ffmc.sim.viz import data_notes

    rec = {
        "data_provenance": {
            "present": ["receptions", "receiving_yards", "touchdowns"],
            "missing": ["rushing_yards"],
            "td": {"available": True, "ladder": True, "n_anchors": 3,
                   "model": "nbinom_ladder"},
            "anchor_counts": {"receptions": 1, "receiving_yards": 1},
            "single_anchor_stats": ["receiving_yards"],
            "low_confidence": True,
        }
    }
    nd = data_notes(rec)
    assert nd["low_confidence"] is True
    assert "receiving_yards" in nd["single_anchor_stats"]
    assert "low confidence" in nd["summary"]
    assert "main line only" in nd["summary"]


def test_data_notes_full_data_not_low_confidence():
    from ffmc.sim.viz import data_notes

    rec = {
        "data_provenance": {
            "present": ["receptions", "receiving_yards", "rushing_yards", "touchdowns"],
            "missing": [],
            "td": {"available": True, "ladder": True, "n_anchors": 4,
                   "model": "nbinom_ladder"},
            "anchor_counts": {"receptions": 6, "receiving_yards": 10, "rushing_yards": 8},
            "single_anchor_stats": [],
            "low_confidence": False,
        }
    }
    nd = data_notes(rec)
    assert nd["low_confidence"] is False
    assert "low confidence" not in nd["summary"]


# --- versioned store layout ----------------------------------------------


def _store_dict(week: str, snapshot_id: str, players=("A", "B")) -> dict:
    return {
        "week_label": week,
        "snapshot_id": snapshot_id,
        "generated_at": f"{snapshot_id}",
        "book": "draftkings",
        "scoring": "ppr",
        "correlated": True,
        "n_sims": 1000,
        "games": [{"event_id": "e1"}],
        "players": {p: {"name": p, "mean": 10.0} for p in players},
    }


def test_write_snapshot_preserves_prior_runs(tmp_path):
    week = "week9"
    s1 = _store_dict(week, "2026-11-01T1000Z")
    s2 = _store_dict(week, "2026-11-03T1000Z", players=("A", "B", "C"))

    p1 = write_snapshot(s1, week, store_dir=tmp_path)
    p2 = write_snapshot(s2, week, store_dir=tmp_path)

    # Both immutable snapshots exist; the first was NOT overwritten.
    assert p1.exists() and p2.exists() and p1 != p2

    store = ProjectionsStore(tmp_path)
    snaps = store.list_snapshots(week)
    assert [s["snapshot_id"] for s in snaps] == [
        "2026-11-01T1000Z",
        "2026-11-03T1000Z",
    ]
    # latest.json points at the newest run.
    assert store.latest_snapshot_id(week) == "2026-11-03T1000Z"
    assert len(store.load(week)["players"]) == 3
    # Older snapshot is still readable by id.
    assert len(store.load(week, "2026-11-01T1000Z")["players"]) == 2


def test_write_snapshot_idempotent_index(tmp_path):
    week = "week9"
    s = _store_dict(week, "2026-11-01T1000Z")
    write_snapshot(s, week, store_dir=tmp_path)
    write_snapshot(s, week, store_dir=tmp_path)  # same id again
    snaps = ProjectionsStore(tmp_path).list_snapshots(week)
    assert len(snaps) == 1  # de-duped by snapshot_id


def test_store_reader_search_and_names(tmp_path):
    week = "week9"
    s = _store_dict(week, "2026-11-01T1000Z", players=("Josh Allen", "Josh Jacobs", "AJ Brown"))
    write_snapshot(s, week, store_dir=tmp_path)
    store = ProjectionsStore(tmp_path)

    assert store.list_weeks() == ["week9"]
    assert store.player_names(week) == ["AJ Brown", "Josh Allen", "Josh Jacobs"]
    # Prefix matches ordered before substring matches.
    res = store.search_players(week, "josh")
    assert res[:2] == ["Josh Allen", "Josh Jacobs"]
    assert store.get_player(week, "AJ Brown")["mean"] == 10.0
    assert store.get_player(week, "Nobody") is None
