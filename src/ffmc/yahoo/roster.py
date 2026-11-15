"""Parse a Yahoo Fantasy *team roster* page and crawl a league's rosters.

Why this exists (and why it's separate from ``parser.py``): in Nov 2026 Yahoo
moved the *matchup* page (``/f1/<lg>/matchup``) to client-side rendering, so the
old server-rendered player/projection table (``statTable1``) is gone — a matchup
page now carries only a manager head-to-head summary, no player rows. The
*per-team roster* page (``/f1/<lg>/<mid>``) still renders server-side with the
familiar ``statTable`` markup and all 15-17 rostered players, so the crawler now
sources league membership from there instead.

One wrinkle: the roster page's own "Projected Points" column empties out as each
player's game locks on game day (Yahoo swaps projection for live/for actual), so
reading projections straight off the roster page is a race against kickoff. We
avoid that entirely: the roster page gives us **membership + owner** (player ids,
which never disappear), and the projection is joined in from the already-captured
**position directory** (``directory.py``) by player id — an exact id join, no
fuzzy matching, and the directory is scraped pre-lock. The matchup parser in
``parser.py`` is left intact (older cached seasons still parse through it).

Output shape matches the matchup crawler's ``roster.json`` exactly
(``{season, week, league_id, blocked, players:[{player_id, name, position,
roster_slot, projected, actual, owner}]}``) so ``backtest/dataset.py`` and the
demo build consume it with no changes.
"""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
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

# "BSB and Friends - CeeDeed Rivalry | Fantasy Football | Yahoo! Sports"
# The viewed team's display name is the segment between the league name and the
# first pipe. (A persistent left-nav element shows the LOGGED-IN user's team on
# every page, so we must NOT read the team name from the body — only the title
# reflects whose roster is being viewed.)
_TITLE_TEAM_RE = re.compile(r"^\s*.*?\s*-\s*(.*?)\s*\|\s*Fantasy", re.IGNORECASE)

_POS_TOKENS = {"QB", "RB", "WR", "TE", "K", "DEF", "D/ST", "FB"}


@dataclass(frozen=True)
class RosterPlayer:
    """One rostered player parsed off a team roster page."""

    player_id: str
    name: str
    position: Optional[str]      # NFL position, e.g. "QB"
    roster_slot: Optional[str]   # the slot they occupy, e.g. "W/R/T" or "BN"

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "name": self.name,
            "position": self.position,
            "roster_slot": self.roster_slot,
        }


def team_name_from_title(html: str) -> Optional[str]:
    """The viewed team's display name from the page ``<title>`` (or None)."""
    soup = BeautifulSoup(html, "lxml")
    title = soup.find("title")
    if not title:
        return None
    m = _TITLE_TEAM_RE.match(title.get_text())
    return m.group(1).strip() if m else None


def _row_position(tds) -> Optional[str]:
    """Roster-slot/position label from the first cell (the slot column)."""
    if not tds:
        return None
    txt = tds[0].get_text(strip=True)
    return txt or None


def parse_roster_html(html: str) -> list[RosterPlayer]:
    """Parse a team roster page into its rostered players (membership only).

    Reads every ``statTable`` on the page (offense, plus the separate K and DST
    tables) and pulls each row's player id + name + slot. Projections/actuals are
    intentionally NOT read here (see module docstring — the roster page's
    projection column is unreliable on game day); they're joined later from the
    directory by player id.
    """
    soup = BeautifulSoup(html, "lxml")
    out: list[RosterPlayer] = []
    seen: set[str] = set()
    for table in soup.find_all("table", id=re.compile(r"^statTable\d+$")):
        body = table.find("tbody")
        if body is None:
            continue
        for tr in body.find_all("tr", recursive=False):
            anchor = tr.select_one("a.name")
            if anchor is None:
                continue
            name = anchor.get("title") or anchor.get_text(strip=True)
            if not name:
                continue
            idel = tr.select_one("[data-ys-playerid]")
            pid = idel.get("data-ys-playerid", "") if idel else ""
            key = pid or name
            if key in seen:
                continue
            seen.add(key)
            tds = tr.find_all("td", recursive=False)
            slot = _row_position(tds)
            # Position: the slot cell holds the roster slot (QB/BN/W-R-T). The
            # player's true NFL position rides in the name cell's detail text
            # ("Jax - QB"); fall back to the slot if it's a plain position token.
            pos = None
            detail = anchor.find_parent("td")
            if detail is not None:
                m = re.search(r"\b(QB|RB|WR|TE|K|DEF|D/ST)\b", detail.get_text(" ", strip=True))
                if m:
                    pos = m.group(1)
            if pos is None and slot in _POS_TOKENS:
                pos = slot
            out.append(RosterPlayer(player_id=pid, name=name, position=pos, roster_slot=slot))
    return out


def build_roster_url(league_id: str, mid: int, season: Optional[int] = None) -> str:
    """URL for a single team's roster page."""
    season_seg = f"/{season}" if season is not None else ""
    return f"{_BASE}{season_seg}/f1/{league_id}/{mid}"


def load_directory_projections(path: Path | str) -> dict[str, float]:
    """Map player_id -> Week-N projected points from a position-directory scrape.

    The directory (``scripts/scrape_position_directory.py``) is captured pre-lock
    and keyed by the same ``player_id`` the roster page exposes, so this join is
    exact. Entries without a player id or projection are skipped.
    """
    data = json.loads(Path(path).read_text())
    out: dict[str, float] = {}
    for e in data.get("entries", []):
        pid = e.get("player_id")
        proj = e.get("projected")
        if pid and proj is not None:
            out[str(pid)] = float(proj)
    return out


@dataclass
class RosterCrawlResult:
    """Outcome of a league roster crawl (shape-compatible with the old crawler)."""

    season: Optional[int]
    week: int
    league_id: str
    players: list[dict]          # already-joined player dicts (roster.json rows)
    pages_fetched: int
    pages_from_cache: int
    teams_seen: int
    blocked: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "season": self.season,
            "week": self.week,
            "league_id": self.league_id,
            "pages_fetched": self.pages_fetched,
            "pages_from_cache": self.pages_from_cache,
            "teams_seen": self.teams_seen,
            "blocked": self.blocked,
            "error": self.error,
            "n_players": len(self.players),
            "players": self.players,
        }


@dataclass
class RosterCrawler:
    """Crawl all teams' roster pages and join directory projections by id.

    ``directory_projs`` maps player_id -> projection (from
    :func:`load_directory_projections`). A player with no directory projection is
    still emitted with ``projected=None`` so the record is faithful; the dataset
    builder drops those (nothing to compare), exactly as before.
    """

    league_id: str
    directory_projs: dict[str, float]
    season: Optional[int] = None
    fetcher: Optional[YahooFetcher] = None
    n_teams: int = 10
    delay_range: tuple[float, float] = (2.0, 4.0)
    max_retries: int = 3
    backoff_base: float = 1.5
    sleep: Callable[[float], None] = time.sleep
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        if self.fetcher is None:
            self.fetcher = YahooFetcher(season=self.season)

    def _fetch_one(self, url: str, *, force_refresh: bool) -> tuple[str, bool]:
        if self.fetcher.is_cached(url) and not force_refresh:
            return self.fetcher.fetch_html(url), True
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            lo, hi = self.delay_range
            self.sleep(self.rng.uniform(lo, hi))
            try:
                return self.fetcher.fetch_html(url, force_refresh=force_refresh), False
            except YahooBlockedError:
                raise
            except (requests.RequestException, YahooFetchError) as e:
                last_exc = e
                self.sleep(self.backoff_base ** attempt + self.rng.uniform(0, 0.5))
        raise YahooFetchError(
            f"giving up on {url} after {self.max_retries} attempts: {last_exc}"
        )

    def crawl_week(self, week: int, *, force_refresh: bool = False) -> RosterCrawlResult:
        by_id: dict[str, dict] = {}
        by_name: dict[str, dict] = {}
        fetched = cached = teams = 0
        blocked = False
        error: Optional[str] = None

        for mid in range(1, self.n_teams + 1):
            url = build_roster_url(self.league_id, mid, self.season)
            try:
                html, from_cache = self._fetch_one(url, force_refresh=force_refresh)
            except YahooBlockedError as e:
                blocked = True
                error = f"soft block at mid={mid}: {e}"
                break
            except YahooFetchError as e:
                error = f"fetch failed at mid={mid}: {e}"
                break

            cached += int(from_cache)
            fetched += int(not from_cache)
            teams += 1

            owner = team_name_from_title(html) or f"Team {mid}"
            for rp in parse_roster_html(html):
                proj = self.directory_projs.get(rp.player_id)
                row = {
                    "player_id": rp.player_id,
                    "name": rp.name,
                    "position": rp.position,
                    "roster_slot": rp.roster_slot,
                    "projected": proj,
                    "actual": None,  # actuals come from nflverse at grade time
                    "owner": owner,
                }
                if rp.player_id:
                    by_id.setdefault(rp.player_id, row)
                else:
                    by_name.setdefault(rp.name, row)

        players = list(by_id.values())
        seen_names = {r["name"] for r in players}
        players += [r for n, r in by_name.items() if n not in seen_names]

        return RosterCrawlResult(
            season=self.season, week=week, league_id=self.league_id,
            players=players, pages_fetched=fetched, pages_from_cache=cached,
            teams_seen=teams, blocked=blocked, error=error,
        )

    def write_roster(self, result: RosterCrawlResult, out_dir: Path | str = "data/yahoo") -> Path:
        base = Path(out_dir)
        if self.season is not None:
            base = base / str(self.season)
        out = base / f"week{result.week}" / "roster.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result.to_dict(), indent=2))
        return out
