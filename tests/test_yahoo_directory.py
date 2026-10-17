"""Tests for the Yahoo position-directory scraper (parser + pagination + union).

All offline: a tiny HTML fixture and an injected no-op sleep / fake fetch, so
nothing touches the network.
"""
from __future__ import annotations

import random

import pytest

from ffmc.yahoo.directory import (
    YahooDirectory,
    DirectoryEntry,
    PlayerRow,
    parse_players_page,
    parse_players_page_rows,
    build_players_url,
    PAGE_SIZE,
)


def _page(players, *, with_table=True):
    """Minimal players-list HTML. ``players`` = [(id, name)]; id='' -> omit attr."""
    if not with_table:
        return "<html><body><p>no players</p></body></html>"
    rows = []
    for pid, name in players:
        idattr = f' data-ys-playerid="{pid}"' if pid else ""
        rows.append(
            f'<tr><td class="Ta-start">'
            f'<a class="name"{idattr} href="/nfl/{pid or 0}/" title="{name}">{name}</a>'
            f"</td></tr>"
        )
    return (
        "<html><body><table class='Table'><tbody>"
        + "".join(rows)
        + "</tbody></table></body></html>"
    )


def test_parse_players_page_basic():
    html = _page([("1", "Alice Back"), ("2", "Bob Front")])
    rows = parse_players_page(html, "WR")
    assert rows == [("1", "Alice Back"), ("2", "Bob Front")]


def test_parse_players_page_empty_is_stop_signal():
    assert parse_players_page(_page([], with_table=False), "WR") == []


def _rich_page(rows):
    """Players-list HTML mimicking the real row markup for the rich parser.

    ``rows`` = [(pid, name, proj_text, detail_text)]. ``proj_text`` goes in the
    ``td.pts.Selected`` cell (the selected stat = projection); ``detail_text``
    is the player-detail subtitle ('Phi - TE' pre-game, 'Final W 41-7 @ Den'
    post-game).
    """
    trs = []
    for pid, name, proj, detail in rows:
        idattr = f' data-ys-playerid="{pid}"' if pid else ""
        trs.append(
            "<tr>"
            f'<td class="Ta-start player">'
            f'<a class="name"{idattr} href="/nfl/players/{pid or 0}" title="{name}">{name}</a>'
            f'<span class="ysf-player-detail Fz-xxs">{detail}</span>'
            "</td>"
            f'<td class="Alt Ta-end Nowrap pts Selected">{proj}</td>'
            "</tr>"
        )
    return (
        "<html><body><table class='Table'><tbody>"
        + "".join(trs)
        + "</tbody></table></body></html>"
    )


def test_rich_parser_extracts_projection_and_team_pregame():
    html = _rich_page([("30995", "Dallas Goedert", "11.42", "Phi - TE")])
    rows = parse_players_page_rows(html, "TE")
    assert len(rows) == 1
    r = rows[0]
    assert isinstance(r, PlayerRow)
    assert r.player_id == "30995"
    assert r.name == "Dallas Goedert"
    assert r.projected == 11.42
    assert r.nfl_team == "Phi"


def test_rich_parser_degrades_team_postgame_but_keeps_projection():
    # Post-game: detail is a game-status string with no 'Team - Pos'.
    html = _rich_page([("1", "Some Guy", "0.00", "Final L 3-19 @ Den")])
    r = parse_players_page_rows(html, "WR")[0]
    assert r.projected == 0.0
    assert r.nfl_team is None  # no guess from a game-status subtitle


def test_rich_parser_handles_missing_projection_cell():
    # A row with no selected-stat cell -> projected None, still parses identity.
    html = _page([("7", "No Proj")])
    r = parse_players_page_rows(html, "RB")[0]
    assert r.name == "No Proj"
    assert r.projected is None


def test_rich_parser_strips_comma_thousands():
    html = _rich_page([("1", "Season Total Guy", "2,236", "Phi - QB")])
    assert parse_players_page_rows(html, "QB")[0].projected == 2236.0


def _greedy_page(header_titles, body_rows):
    """Players HTML with a 2-row thead (second aligns to body) + data rows.

    ``header_titles`` = column titles for the aligned header row. ``body_rows``
    = [(pid, name, [cell_text,...])] aligned positionally to header_titles. The
    a.name anchor goes in the cell whose header title is 'Offense'; the
    'Fantasy Points' cell gets the pts.Selected class.
    """
    hrow = "".join(f'<th title="{t}">{t}</th>' for t in header_titles)
    thead = f"<thead><tr><th>grp</th></tr><tr>{hrow}</tr></thead>"
    trs = []
    for pid, name, cells in body_rows:
        tds = []
        for i, txt in enumerate(cells):
            title = header_titles[i].lower()
            if title == "offense":
                idattr = f' data-ys-playerid="{pid}"' if pid else ""
                tds.append(
                    f'<td class="player"><a class="name"{idattr} '
                    f'href="/nfl/players/{pid}" title="{name}">{name}</a></td>'
                )
            elif title == "fantasy points":
                tds.append(f'<td class="pts Selected">{txt}</td>')
            else:
                tds.append(f"<td>{txt}</td>")
        trs.append("<tr>" + "".join(tds) + "</tr>")
    return (
        "<html><body><table class='Table'>"
        + thead
        + "<tbody>" + "".join(trs) + "</tbody></table></body></html>"
    )


def test_greedy_capture_keys_by_header():
    headers = ["Offense", "Fantasy Points", "Passing Yards", "Passing Touchdowns",
               "Interceptions", "Rushing Yards", "Receptions", "Receiving Yards",
               "Percent player is rostered in Yahoo leagues"]
    body = [("111", "Josh Allen",
             ["Josh Allen", "24.1", "271", "2", "0", "39", "0", "0", "99%"])]
    rows = parse_players_page_rows(_greedy_page(headers, body), "QB")
    assert len(rows) == 1
    r = rows[0]
    assert r.projected == 24.1          # from the Selected cell
    s = r.stats
    assert s["fantasy_points"] == 24.1  # same column, greedily captured
    assert s["pass_yds"] == 271.0
    assert s["pass_tds"] == 2.0
    assert s["interceptions"] == 0.0
    assert s["rush_yds"] == 39.0
    assert s["percent_rostered"] == "99%"   # % kept as string
    assert "offense" not in s               # structural column skipped
    assert s["pass_yds"] == 271.0


def test_preseason_and_actual_headers_map_to_rank_keys():
    # In the players-list view, "Pre-Season"/"Actual" columns are RANKS, not
    # points (verified: distinct integers, Gibbs = preseason #1). Guard the
    # mapping so a future rename back to *_points would fail loudly.
    headers = ["Offense", "Fantasy Points", "Pre-Season", "Actual"]
    body = [("1", "Top Pick", ["Top Pick", "23.8", "1", "1"])]
    r = parse_players_page_rows(_greedy_page(headers, body), "RB")[0]
    assert r.stats.get("preseason_rank") == 1.0
    assert r.stats.get("actual_rank") == 1.0
    assert "preseason_points" not in r.stats
    assert "actual_points" not in r.stats


def test_greedy_capture_unmapped_header_slugified():
    headers = ["Offense", "Fantasy Points", "Snap Count %Weird"]
    body = [("1", "Guy", ["Guy", "10.0", "55"])]
    r = parse_players_page_rows(_greedy_page(headers, body), "RB")[0]
    # unmapped header is slugified, not dropped (greedy)
    assert r.stats.get("snap_count_weird") == 55.0


def test_greedy_capture_absent_when_header_misaligned():
    # No thead -> no keys -> stats empty, but identity/projection still parse.
    html = _rich_page([("1", "No Header", "12.3", "Phi - WR")])
    r = parse_players_page_rows(html, "WR")[0]
    assert r.projected == 12.3
    assert r.stats == {}


def test_build_carries_projection_and_team():
    rb0 = _rich_page([("7", "Flex Guy", "14.1", "Dal - RB")])
    wr0 = _rich_page([("7", "Flex Guy", "14.1", "Dal - WR")])
    pages = {
        build_players_url("123456", "RB", 0, season=2025): rb0,
        build_players_url("123456", "WR", 0, season=2025): wr0,
    }
    d = _directory_with(pages)
    d.positions = ("RB", "WR")
    entries = {e.player_id: e for e in d.build()}
    e = entries["7"]
    assert e.positions == ("RB", "WR")
    assert e.projected == 14.1
    assert e.nfl_team == "Dal"
    assert e.to_dict()["projected"] == 14.1


def test_parse_players_page_id_fallback_to_href():
    # Real pages always have some id-bearing rows (that's how the players table
    # is detected). A row missing data-ys-playerid falls back to its /nfl/<id>/
    # href. Include one normal row so the table is recognized.
    html = _page([("1", "Normal"), ("", "No Attr")])
    html = html.replace('href="/nfl/0/"', 'href="/nfl/987/"')
    rows = parse_players_page(html, "RB")
    assert rows == [("1", "Normal"), ("987", "No Attr")]


def test_build_players_url_shape():
    url = build_players_url("123456", "WR", 50, season=2025, stat1="S_PW_18")
    assert "/2025/f1/123456/players" in url
    assert "pos=WR" in url and "count=50" in url and "stat1=S_PW_18" in url


class _FakeFetcher:
    """Stand-in for YahooFetcher: serves canned pages, counts live fetches."""

    def __init__(self, pages_by_url):
        self.pages_by_url = pages_by_url
        self.cookie = "fake"
        self.live_fetches = 0
        self._cache = {}

    def is_cached(self, url):
        return url in self._cache

    def _raw_path(self, url):  # not used in tests (never cache-hit path)
        raise AssertionError("should not read raw path in these tests")


def _directory_with(pages):
    """Build a YahooDirectory whose _fetch_one serves `pages` (dict url->html)."""
    d = YahooDirectory(
        league_id="123456",
        season=2025,
        sleep=lambda s: None,              # no real waiting
        rng=random.Random(0),
    )

    calls = {"n": 0}

    def fake_fetch_one(url, *, force_refresh=False):
        calls["n"] += 1
        return pages.get(url, _page([], with_table=False)), False

    d._fetch_one = fake_fetch_one  # type: ignore[assignment]
    d._calls = calls
    return d


def test_scrape_position_stops_after_consecutive_empties():
    # Full page 0, full page 1, then TWO empties -> stop. 55 rows total.
    p0 = _page([(str(i), f"P{i}") for i in range(PAGE_SIZE)])
    p1 = _page([(str(100 + i), f"Q{i}") for i in range(PAGE_SIZE)])
    pages = {
        build_players_url("123456", "RB", 0, season=2025): p0,
        build_players_url("123456", "RB", PAGE_SIZE, season=2025): p1,
        # pages 2 and 3 default to empty (fake fetch serves empty for unknown)
    }
    d = _directory_with(pages)
    d.stop_after_empties = 2
    rows = d.scrape_position("RB")
    assert len(rows) == 2 * PAGE_SIZE
    assert all(hasattr(r, "name") for r in rows)  # PlayerRow objects
    # fetched page0, page1, empty page2, empty page3 -> 4 calls then stop
    assert d._calls["n"] == 4


def test_scrape_position_tolerates_interior_empty_page():
    # Full page0, EMPTY page1 (interior gap), full page2, then two empties.
    # Mirrors the real TE list (empty at count=100, resumes at 125).
    p0 = _page([(str(i), f"P{i}") for i in range(PAGE_SIZE)])
    p2 = _page([(str(200 + i), f"R{i}") for i in range(PAGE_SIZE)])
    pages = {
        build_players_url("123456", "TE", 0, season=2025): p0,
        # page 1 (count=25) absent -> served empty by the fake
        build_players_url("123456", "TE", 2 * PAGE_SIZE, season=2025): p2,
    }
    d = _directory_with(pages)
    d.stop_after_empties = 2
    rows = d.scrape_position("TE")
    names = {r.name for r in rows}
    # the interior empty did NOT stop us: page2's players are present
    assert "R0" in names and "P0" in names
    assert len(rows) == 2 * PAGE_SIZE


def test_build_unions_dual_eligibility():
    # Player "7" shows up in BOTH the RB and WR searches -> positions {RB, WR}.
    rb0 = _page([("7", "Flex Guy"), ("8", "Pure RB")])
    wr0 = _page([("7", "Flex Guy"), ("9", "Pure WR")])
    pages = {
        build_players_url("123456", "RB", 0, season=2025): rb0,
        build_players_url("123456", "WR", 0, season=2025): wr0,
    }
    d = _directory_with(pages)
    d.positions = ("RB", "WR")
    entries = {e.player_id: e for e in d.build()}
    assert entries["7"].positions == ("RB", "WR")
    assert entries["8"].positions == ("RB",)
    assert entries["9"].positions == ("WR",)
