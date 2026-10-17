"""Scrape a Yahoo Fantasy *players* page into a position directory.

Why this exists
---------------
The matchup page (``parser.py``) does not carry a *bench* player's position for
the 2025 layout — only starters' positions are recoverable (from their roster
slot). The counterfactual lineup optimizer needs every rostered player's
**slot eligibility**, bench included. Yahoo's league *players* page is keyed by
position via the ``pos=`` query param, so scraping it per position (QB/RB/WR/TE)
yields a clean ``player_id -> {positions}`` directory.

Eligibility is a *set*: a player eligible at both RB and WR appears in both the
RB and WR searches (Yahoo filters that list by lineup-slot eligibility), so we
**union** the positions found for each ``player_id``. That union is exactly the
FLEX/multi-position signal the optimizer needs.

The players list paginates by a ``count`` row-offset that steps by the page
size (25). An out-of-range ``count`` returns an empty table — that is the stop
condition. Position comes from the *query param*, not the row markup.

Politeness / reproducibility
----------------------------
Mirrors ``crawler.LeagueCrawler``: a jittered delay is applied **only before an
actual live fetch** (never before a cache hit, so re-runs are instant), with
bounded exponential-backoff retry on transient errors and an immediate halt on
a soft block. ``sleep``/``rng`` are injectable so tests run with no real wait.
The whole pagination loop (sleeps included) lives *inside* this class, so one
run reproduces the entire directory — callers never drive pagination by hand.
"""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

import requests

try:
    from bs4 import BeautifulSoup
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "ffmc.yahoo requires beautifulsoup4 + lxml. "
        "Install with: pip install -e '.[yahoo]'"
    ) from exc

from .fetcher import YahooFetcher, YahooBlockedError, YahooFetchError

_BASE = "https://football.fantasysports.yahoo.com"
PAGE_SIZE = 25
POSITIONS = ("QB", "RB", "WR", "TE")


@dataclass(frozen=True)
class DirectoryEntry:
    """One player's identity + the set of lineup positions they're eligible at,
    plus the Yahoo weekly projection, NFL team, and the greedily-captured raw
    stat row (keyed by header) for later analytics."""

    player_id: str
    name: str
    positions: tuple[str, ...]  # sorted union, e.g. ("RB", "WR")
    projected: Optional[float] = None   # Yahoo weekly projected points
    nfl_team: Optional[str] = None      # e.g. "Phi" (None if not on the page)
    stats: dict = field(default_factory=dict)  # full raw stat row (greedy)

    def to_dict(self) -> dict:
        return {"player_id": self.player_id, "name": self.name,
                "positions": list(self.positions),
                "projected": self.projected, "nfl_team": self.nfl_team,
                "stats": self.stats}


def build_players_url(
    league_id: str,
    position: str,
    count: int,
    *,
    season: Optional[int] = None,
    stat1: str = "S_PW_18",
) -> str:
    """URL for one page of the league players list, filtered to ``position``.

    ``count`` is the row offset (0, 25, 50, ...). ``stat1`` selects the stat
    context; the directory only needs identity + eligibility, so its exact value
    is immaterial as long as it's stable across pages (keeps the cache keys
    deterministic). ``season`` prefixes the path for a past season.
    """
    season_seg = f"/{season}" if season is not None else ""
    return (
        f"{_BASE}{season_seg}/f1/{league_id}/players"
        f"?status=ALL&eteam=ALL&fteam=NONE&pos={position}&cut_type=9"
        f"&stat1={stat1}&myteam=0&sort=PTS&sdir=3&count={count}"
    )


def _player_id_in(cell) -> str:
    """First data-ys-playerid anywhere in a cell, else '' (id lives on the name
    anchor in some rows and a sibling note/video anchor in others)."""
    if cell is None:
        return ""
    el = cell.select_one("[data-ys-playerid]")
    return el.get("data-ys-playerid", "") if el else ""


def _players_table(soup):
    """The table that actually lists players (the first one with a player id)."""
    for t in soup.find_all("table"):
        if t.select_one("[data-ys-playerid]"):
            return t
    return None


# "Phi - TE", "GB - WR", "Was - QB" in the player-detail subtitle (pre-game).
# Post-game the same span shows a game-status string ("Final W 41-7 ...") with
# no "- POS", so this simply doesn't match and nfl_team degrades to None.
_DETAIL_TEAM_POS_RE = re.compile(
    r"\b([A-Za-z]{2,3})\s*-\s*(QB|RB|WR|TE|K|DEF|D/ST)\b"
)


def _num_cell(text: str) -> Optional[float]:
    """Parse a numeric points cell; '-'/''/non-numeric -> None.

    Yahoo renders thousands with commas (e.g. '2,236') for season-total stats;
    the projected-week-points stat is a small decimal ('14.72'), so strip commas
    before parsing to be safe.
    """
    if text is None:
        return None
    text = text.strip().replace(",", "")
    if not text or text in {"-", "\u2013", "\u2014"}:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(m.group()) if m else None


def _row_projected(tr) -> Optional[float]:
    """The selected-stat value for a players-list row.

    The players list is sorted by (and highlights) one stat column, marked with
    both ``pts`` and ``Selected`` classes on its ``<td>``. When the page is
    fetched with the projected-week-points stat selected (``stat1`` =
    ``S_PW_<week>``), that cell IS the player's Yahoo weekly projection. Read
    ``td.pts.Selected`` first, then fall back to any ``td.Selected``.
    """
    sel = tr.select_one("td.pts.Selected") or tr.select_one("td.Selected")
    if sel is None:
        return None
    return _num_cell(sel.get_text(strip=True))


def _row_nfl_team(cell) -> Optional[str]:
    """The player's own NFL team abbreviation from the player cell, if present.

    Pre-game rows carry a 'Team - Pos' subtitle (e.g. 'Phi - TE'); return the
    team. Post-game rows show a game-status string instead, so this returns
    ``None`` (team unknown from this markup) rather than guessing.
    """
    if cell is None:
        return None
    m = _DETAIL_TEAM_POS_RE.search(cell.get_text(" ", strip=True))
    return m.group(1) if m else None


# Normalize a Yahoo header title to a stable snake_case stat key. Anything not
# listed is slugified generically, so new/renamed columns are still captured
# (greedy): we never silently drop a column, we just key it by its slug.
_HEADER_KEY_MAP = {
    "fantasy points": "fantasy_points",   # the SELECTED/sorted stat column
    # NOTE: in the players-list view these two columns are RANKS, not points
    # (verified: distinct integers 1..N; Gibbs = preseason #1, drafted 1st
    # overall). Yahoo's ambiguous "Pre-Season"/"Actual" headers mean rank here.
    "pre-season": "preseason_rank",
    "actual": "actual_rank",
    "percent player is rostered in yahoo leagues": "percent_rostered",
    "games played": "games_played",
    "bye week": "bye_week",
    "roster status": "roster_status",
    "passing yards": "pass_yds",
    "passing touchdowns": "pass_tds",
    "interceptions": "interceptions",
    "rushing attempts": "rush_att",
    "rushing yards": "rush_yds",
    "rushing touchdowns": "rush_tds",
    "targets": "targets",
    "receptions": "receptions",
    "receiving yards": "rec_yds",
    "receiving touchdowns": "rec_tds",
    "return touchdowns": "return_tds",
    "2-point conversions": "two_point_conversions",
    "fumbles lost": "fumbles_lost",
}

# Header titles that are structural, not stats (skip when building the dict).
_HEADER_SKIP = {"", "offense", "highlight"}


def _slugify_header(title: str) -> str:
    """Fallback snake_case key for an unmapped header title."""
    s = re.sub(r"[^a-z0-9]+", "_", title.strip().lower()).strip("_")
    return s or "col"


def _header_keys(table) -> Optional[list[str]]:
    """Per-column stat keys from the header row that aligns 1:1 with body cells.

    Yahoo's players table has a two-row ``thead``; the *second* row has one cell
    per data column (the first row is group spanners like Passing/Rushing). We
    pick the header row whose cell count matches the body row width, then map
    each title through ``_HEADER_KEY_MAP`` (falling back to a slug). Returns
    ``None`` if no header row aligns, so the caller can skip stat capture rather
    than mis-align columns.
    """
    thead = table.find("thead")
    body = table.find("tbody")
    if thead is None or body is None:
        return None
    first_body = body.find("tr", recursive=False)
    if first_body is None:
        return None
    nbody = len(first_body.find_all("td", recursive=False))
    for hr in thead.find_all("tr"):
        cells = hr.find_all(["th", "td"], recursive=False)
        if len(cells) != nbody:
            continue
        keys = []
        for c in cells:
            title = (c.get("title") or c.get_text(" ", strip=True) or "").strip()
            low = title.lower()
            keys.append(_HEADER_KEY_MAP.get(low, _slugify_header(title)))
        return keys
    return None


def _clean_value(text: str):
    """Clean one stat cell: '-'/'' -> None; strip commas; keep '%' strings.

    Numbers are returned as ``float`` (comma thousands stripped); a percentage
    like '42%' is kept as the string '42%' (the raw form is most useful for a
    later analytics DB, which can parse as needed). Non-numeric labels (e.g.
    'FA', 'W 1') are returned as their stripped string.
    """
    if text is None:
        return None
    t = text.strip()
    if not t or t in {"-", "\u2013", "\u2014"}:
        return None
    if t.endswith("%"):
        return t
    cleaned = t.replace(",", "")
    if re.fullmatch(r"-?\d+(?:\.\d+)?", cleaned):
        return float(cleaned)
    return t


def _row_stats(tr, keys: Optional[list[str]]) -> dict:
    """Greedily capture every stat cell in a row, keyed by header.

    Aligns body ``<td>`` cells to ``keys`` positionally. Structural columns
    (player cell, spacers) are skipped; empty cells map to ``None``. Returns an
    empty dict if we have no header keys (so capture is best-effort, never
    mis-aligned)."""
    if not keys:
        return {}
    tds = tr.find_all("td", recursive=False)
    out: dict = {}
    for i, td in enumerate(tds):
        if i >= len(keys):
            break
        key = keys[i]
        if key in _HEADER_SKIP or key == "col":
            continue
        # the player-name cell slugs to something like 'offense'/'' -> skip;
        # also skip the watch-list / highlight icon columns.
        val = _clean_value(td.get_text(" ", strip=True))
        if val is not None:
            out[key] = val
    return out


@dataclass(frozen=True)
class PlayerRow:
    """One parsed players-list row.

    ``projected`` is the selected-stat value (the Yahoo weekly projection when
    the page is fetched with the projected stat). ``stats`` greedily captures
    *every* tabular stat column Yahoo renders for the row, keyed by a normalized
    header name (e.g. ``pass_yds``, ``rec_tds``, ``percent_rostered``) — raw
    material for a future analytics DB. Values are floats where numeric, strings
    otherwise (e.g. '42%'), ``None`` for blank cells.
    """

    player_id: str
    name: str
    projected: Optional[float] = None
    nfl_team: Optional[str] = None
    stats: dict = field(default_factory=dict)


def parse_players_page_rows(html: str, position: str) -> list[PlayerRow]:
    """Parse one players-list page -> [PlayerRow] for ``position``.

    Richer than :func:`parse_players_page` (which returns bare (id, name)
    tuples): pulls the selected-stat **projection** (``td.pts.Selected``), the
    player's **NFL team** (pre-game subtitle), and **greedily captures every
    stat column** Yahoo renders (keyed by header title) into ``PlayerRow.stats``
    for later analytics. Returns an empty list when the page has no player table
    (the paginator's stop condition).
    """
    soup = BeautifulSoup(html, "lxml")
    table = _players_table(soup)
    if table is None:
        return []
    body = table.find("tbody")
    if body is None:
        return []

    keys = _header_keys(table)
    out: list[PlayerRow] = []
    for tr in body.find_all("tr", recursive=False):
        anchor = tr.select_one("a.name")
        if anchor is None:
            continue
        name = anchor.get("title") or anchor.get_text(strip=True)
        if not name:
            continue
        cell = anchor.find_parent("td")
        pid = _player_id_in(cell) or tr_id_fallback(tr, anchor)
        out.append(
            PlayerRow(
                player_id=pid,
                name=name,
                projected=_row_projected(tr),
                nfl_team=_row_nfl_team(cell),
                stats=_row_stats(tr, keys),
            )
        )
    return out


def parse_players_page(html: str, position: str) -> list[tuple[str, str]]:
    """Parse one players-list page -> [(player_id, name)] for ``position``.

    Backward-compatible thin wrapper over :func:`parse_players_page_rows` (kept
    for callers/tests that only need identity). Position is supplied by the
    caller (it's the query filter, not in the row). Returns an empty list when
    the page has no player table — the paginator's stop condition. Rows whose id
    can't be resolved are kept with id='' so the assembler can fall back to a
    name match and still see them.
    """
    return [(r.player_id, r.name) for r in parse_players_page_rows(html, position)]


def tr_id_fallback(tr, anchor) -> str:
    """Best-effort id when it's not on the name cell as ``data-ys-playerid``.

    On deeper players-list pages the top-player markup gives way to a plainer
    row whose id is only in the name anchor's href, e.g.
    ``https://sports.yahoo.com/nfl/players/34078`` (NOT the ``/nfl/<id>`` form).
    Try that first, then any ``data-ys-playerid`` elsewhere in the row, then the
    older ``/nfl/<id>`` shape.
    """
    import re
    href = anchor.get("href") or ""
    m = re.search(r"/nfl/players/(\d+)", href)
    if m:
        return m.group(1)
    el = tr.select_one("[data-ys-playerid]")
    if el:
        return el.get("data-ys-playerid", "")
    m = re.search(r"/nfl/(\d+)", href)
    return m.group(1) if m else ""


@dataclass
class YahooDirectory:
    """Paginate the league players list across positions into a directory.

    Reuses ``YahooFetcher`` for cookie-auth + URL-hash caching + soft-block
    detection. The polite jittered delay is applied only before a live fetch.
    """

    league_id: str
    season: Optional[int] = None
    fetcher: Optional[YahooFetcher] = None
    positions: tuple[str, ...] = POSITIONS
    stat1: str = "S_PW_18"
    max_pages_per_pos: int = 40          # safety cap (40*25 = 1000 players/pos)
    stop_after_empties: int = 2          # consecutive empty pages => end of list
                                         # (a single empty can be an interior gap)
    delay_range: tuple[float, float] = (2.0, 4.0)
    max_retries: int = 3
    backoff_base: float = 1.5
    sleep: Callable[[float], None] = time.sleep
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        if self.fetcher is None:
            self.fetcher = YahooFetcher(season=self.season)

    # ---- one page, cache-aware, with retry/backoff -----------------------
    def _fetch_one(self, url: str, *, force_refresh: bool) -> tuple[str, bool]:
        """Return (html, from_cache). Sleeps only before a real network fetch.

        NOTE: ``YahooFetcher.fetch_html`` guards with ``looks_like_matchup``
        (checks for 'statTable'), which a players page does NOT contain — so we
        must NOT route a live players fetch through it. We fetch directly here
        and detect a soft block ourselves (a players page without any player
        id / 'Sign in' wall). Cache hits still go through the fetcher so the
        cache key/layout stays identical to the rest of the package.
        """
        if self.fetcher.is_cached(url) and not force_refresh:
            # Read raw from cache without the matchup guard.
            return self.fetcher._raw_path(url).read_text(
                encoding="utf-8", errors="replace"), True

        if not self.fetcher.cookie:
            raise YahooFetchError(
                "No cached HTML and no cookie available. Set YAHOO_COOKIE or "
                "pass a fetcher with a cookie."
            )

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64; rv:151.0) "
                "Gecko/20100101 Firefox/151.0"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cookie": self.fetcher.cookie,
        }
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            lo, hi = self.delay_range
            self.sleep(self.rng.uniform(lo, hi))
            try:
                resp = self.fetcher._session.get(url, headers=headers, timeout=30)
                if resp.status_code != 200:
                    raise YahooFetchError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                html = resp.text
                if not _looks_like_players(html):
                    raise YahooBlockedError(
                        "Players page has no player table — likely a soft block, "
                        "throttle, or expired cookie (bounced to login)."
                    )
                self.fetcher._raw_path(url).write_text(html, encoding="utf-8")
                return html, False
            except YahooBlockedError:
                raise  # terminal — let the caller halt
            except (requests.RequestException, YahooFetchError) as e:
                last_exc = e
                self.sleep(self.backoff_base ** attempt + self.rng.uniform(0, 0.5))
        raise YahooFetchError(
            f"giving up on {url} after {self.max_retries} attempts: {last_exc}"
        )

    # ---- paginate one position -------------------------------------------
    def scrape_position(self, position: str, *, force_refresh: bool = False) -> list[PlayerRow]:
        """All :class:`PlayerRow` for a position (identity + projection + team).

        Yahoo's players list can contain an **isolated empty page in the middle**
        of the list (observed: TE count=100 is empty but 125-224 are full). So a
        single empty page is NOT a reliable end-of-list signal. We only stop
        after ``stop_after_empties`` *consecutive* empty pages (the true tail is
        a run of empties), capped by ``max_pages_per_pos``. De-duplicate by
        (id, name) in case a gap causes an overlap.
        """
        seen: set[tuple[str, str]] = set()
        rows: list[PlayerRow] = []
        consecutive_empty = 0
        for page in range(self.max_pages_per_pos):
            count = page * PAGE_SIZE
            url = build_players_url(
                self.league_id, position, count,
                season=self.season, stat1=self.stat1,
            )
            html, _ = self._fetch_one(url, force_refresh=force_refresh)
            page_rows = parse_players_page_rows(html, position)
            if not page_rows:
                consecutive_empty += 1
                if consecutive_empty >= self.stop_after_empties:
                    break  # a RUN of empties -> genuine end of list
                continue   # isolated empty -> keep going (interior gap)
            consecutive_empty = 0
            for row in page_rows:
                key = (row.player_id, row.name)
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
        return rows

    # ---- full directory (union across positions) -------------------------
    def build(self, *, force_refresh: bool = False) -> list[DirectoryEntry]:
        """Scrape all positions and union eligibility per player_id.

        Players keyed by ``player_id`` when present; a '' id falls back to a
        name key so eligibility still unions (rare — only ids we couldn't
        resolve). The projection and NFL team are carried from the first page a
        player appears on with those fields set (projections are identical
        across position searches for the same player). Returns entries sorted by
        name for stable output.
        """
        pos_by_key: dict[str, set[str]] = {}
        name_by_key: dict[str, str] = {}
        proj_by_key: dict[str, Optional[float]] = {}
        team_by_key: dict[str, Optional[str]] = {}
        stats_by_key: dict[str, dict] = {}

        for position in self.positions:
            for row in self.scrape_position(position, force_refresh=force_refresh):
                key = row.player_id or f"name:{row.name.lower()}"
                pos_by_key.setdefault(key, set()).add(position)
                name_by_key.setdefault(key, row.name)
                # keep the first non-None projection / team we see for a player
                if proj_by_key.get(key) is None and row.projected is not None:
                    proj_by_key[key] = row.projected
                if team_by_key.get(key) is None and row.nfl_team is not None:
                    team_by_key[key] = row.nfl_team
                # merge the greedy stat row (first-seen wins per key; identical
                # across position searches for the same player)
                if row.stats:
                    merged = stats_by_key.setdefault(key, {})
                    for k, v in row.stats.items():
                        merged.setdefault(k, v)

        entries = [
            DirectoryEntry(
                player_id=("" if key.startswith("name:") else key),
                name=name_by_key[key],
                positions=tuple(sorted(pos_by_key[key])),
                projected=proj_by_key.get(key),
                nfl_team=team_by_key.get(key),
                stats=stats_by_key.get(key, {}),
            )
            for key in pos_by_key
        ]
        entries.sort(key=lambda e: e.name)
        return entries

    def write(self, entries: list[DirectoryEntry], out_path: Path | str) -> Path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "league_id": self.league_id,
            "season": self.season,
            "positions_scraped": list(self.positions),
            "n_entries": len(entries),
            "entries": [e.to_dict() for e in entries],
        }
        out.write_text(json.dumps(payload, indent=2))
        return out


def _looks_like_players(html: str) -> bool:
    """True if the HTML looks like a players list (has a player id or the table)."""
    return "data-ys-playerid" in html or "players?status" in html


def load_directory(path: Path | str) -> dict[str, DirectoryEntry]:
    """Load a written directory -> {player_id: DirectoryEntry} (ids only)."""
    data = json.loads(Path(path).read_text())
    out: dict[str, DirectoryEntry] = {}
    for e in data["entries"]:
        if e["player_id"]:
            out[e["player_id"]] = DirectoryEntry(
                player_id=e["player_id"], name=e["name"],
                positions=tuple(e["positions"]),
                projected=e.get("projected"),
                nfl_team=e.get("nfl_team"),
                stats=e.get("stats", {}),
            )
    return out
