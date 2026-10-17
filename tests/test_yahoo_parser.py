"""Tests for the Yahoo matchup parser + fetcher (offline paths).

Validated against a committed synthetic fixture that mirrors the real matchup
page's mirrored 11-column table structure. The real page is private and
~1 MB (gitignored), so these tests use the small hand-built fixture.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ffmc.yahoo import parse_matchup_html, YahooFetcher, YahooFetchError
from ffmc.yahoo.parser import PlayerLine

FIXTURE = Path(__file__).parent / "fixtures" / "yahoo_matchup_sample.html"


@pytest.fixture
def html() -> str:
    return FIXTURE.read_text()


@pytest.fixture
def page(html):
    return parse_matchup_html(html, week=5)


def _by_name(page, name) -> PlayerLine:
    matches = [p for p in page.players if p.name == name]
    assert matches, f"{name} not parsed"
    return matches[0]


# ---- parser ---------------------------------------------------------------

def test_parses_all_players_and_skips_total(page):
    # 3 roster slots x 2 teams = 6 players; the TOTAL footer row is skipped.
    assert len(page.players) == 6
    assert all(isinstance(p, PlayerLine) for p in page.players)
    assert "Total" not in {p.name for p in page.players}


def test_week_override_wins(html):
    assert parse_matchup_html(html, week=5).week == 5
    assert parse_matchup_html(html, week=9).week == 9


def test_left_player_proj_and_actual(page):
    alice = _by_name(page, "Alice Arm")
    assert alice.team_side == "home"
    assert alice.projected == pytest.approx(20.20)
    assert alice.actual == pytest.approx(29.76)
    assert alice.nfl_team == "Dal"       # title-case team abbrev
    assert alice.position == "QB"
    assert alice.roster_slot == "QB"


def test_right_player_proj_and_actual(page):
    bob = _by_name(page, "Bob Back")
    assert bob.team_side == "away"
    assert bob.projected == pytest.approx(19.27)
    assert bob.actual == pytest.approx(15.80)
    assert bob.nfl_team == "GB"          # upper-case team abbrev
    assert bob.position == "QB"


def test_composite_roster_slot(page):
    carol = _by_name(page, "Carol Catch")
    assert carol.roster_slot == "W/R/T"
    assert carol.position == "WR"


def test_missing_actual_parses_to_none(page):
    # Dave's actual cell is "-" (game not yet played) -> None, proj still present.
    dave = _by_name(page, "Dave Deep")
    assert dave.actual is None
    assert dave.projected == pytest.approx(11.90)


def test_bench_slot_detected(page):
    erin = _by_name(page, "Erin End")
    frank = _by_name(page, "Frank Flat")
    assert erin.roster_slot == "BN"
    assert frank.roster_slot == "BN"


def test_both_sides_present(page):
    sides = {p.team_side for p in page.players}
    assert sides == {"home", "away"}
    assert sum(p.team_side == "home" for p in page.players) == 3
    assert sum(p.team_side == "away" for p in page.players) == 3


def test_malformed_html_yields_no_players():
    page = parse_matchup_html("<html><body>no table here</body></html>", week=1)
    assert page.players == []


# ---- fetcher (offline) ----------------------------------------------------

def test_parse_file_writes_snapshot(tmp_path):
    fetcher = YahooFetcher(snapshot_dir=tmp_path, raw_dir=tmp_path / "raw")
    page = fetcher.parse_file(FIXTURE, week=5)
    assert len(page.players) == 6

    snap = fetcher.snapshot_path(5)
    assert snap == tmp_path / "week5" / "matchup.json"
    assert snap.exists()

    data = json.loads(snap.read_text())
    assert data["week"] == 5
    assert len(data["players"]) == 6
    first = data["players"][0]
    assert set(first) == {
        "player_id", "name", "nfl_team", "position",
        "roster_slot", "team_side", "projected", "actual", "owner",
    }


def test_season_namespaces_snapshot(tmp_path):
    fetcher = YahooFetcher(season=2025, snapshot_dir=tmp_path, raw_dir=tmp_path / "raw")
    fetcher.parse_file(FIXTURE, week=12)
    snap = fetcher.snapshot_path(12)
    assert snap == tmp_path / "2025" / "week12" / "matchup.json"
    assert snap.exists()


def test_parse_file_no_write(tmp_path):
    fetcher = YahooFetcher(snapshot_dir=tmp_path, raw_dir=tmp_path / "raw")
    fetcher.parse_file(FIXTURE, week=7, write=False)
    assert not fetcher.snapshot_path(7).exists()


def test_live_fetch_without_cookie_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("YAHOO_COOKIE", raising=False)
    fetcher = YahooFetcher(snapshot_dir=tmp_path, raw_dir=tmp_path / "raw")
    with pytest.raises(YahooFetchError):
        # no cached raw HTML + no cookie -> must raise, never hit the network
        fetcher.fetch_html("https://example.invalid/matchup?week=1")


def test_soft_block_detection():
    from ffmc.yahoo.fetcher import looks_like_matchup
    assert looks_like_matchup('<table id="statTable1">')
    assert not looks_like_matchup("<html>please sign in</html>")


# --- team-name extraction (owner enrichment) ----------------------------

from ffmc.yahoo.parser import parse_matchup_team_names


def test_team_names_home_then_away():
    html = (
        '<div><a class="F-link" '
        'href="https://football.fantasysports.yahoo.com/f1/123456/10">'
        'Team Cheeseberders</a></div>'
        '<div><a class="F-link" '
        'href="https://football.fantasysports.yahoo.com/f1/123456/9">'
        "Buck-ee's Beaver Nuggies</a></div>"
    )
    home, away = parse_matchup_team_names(html)
    assert home == "Team Cheeseberders"
    assert away == "Buck-ee's Beaver Nuggies"


def test_team_names_dedup_repeated_link():
    # A repeated home link (e.g. a sub-nav) must not become the "away" team.
    html = (
        '<a href="https://x/f1/5/1">Alpha</a>'
        '<a href="https://x/f1/5/1">Alpha</a>'
        '<a href="https://x/f1/5/2">Beta</a>'
    )
    assert parse_matchup_team_names(html) == ("Alpha", "Beta")


def test_team_names_missing_degrades_to_none():
    assert parse_matchup_team_names("<div>no team links here</div>") == (None, None)
