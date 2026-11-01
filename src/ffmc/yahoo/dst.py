"""Yahoo weekly DST projections (and actuals) via the league *players* list.

Why the players list, not the matchup crawler
---------------------------------------------
The matchup pages only show DSTs rostered in a given league, a biased sample
(mostly the good defenses and streamers). The players list with ``status=ALL``
shows every DST, and ``stat1=S_PW_<week>`` makes the highlighted/selected column
Yahoo's weekly *projection* for that week; ``stat1=S_W_<week>`` is the same page
showing the realized *actual* instead (confirmed by the league owner).

Why a DST-specific parser (not ``YahooDirectory``)
--------------------------------------------------
The DST players table has a *different* markup from the QB/RB/WR/TE table that
``ffmc.yahoo.directory`` parses: rows carry no ``data-ys-playerid``, identity is
the team (the name anchor links to ``/nfl/teams/<slug>/``), and the row exposes
the eight per-stat projected components as their own columns (Points Allowed,
Sack, Safety, Interception, Fumble Recovery, Touchdown, Block Kick, Kick/Punt
Return). So we parse the DST table directly here. We still reuse
:class:`ffmc.yahoo.fetcher.YahooFetcher` for cookie auth and URL-hash caching
(the cache key includes ``stat1``, so projection/actual and each week cache
separately), and we mirror its polite-delay / soft-block / retry behavior.

The eight components are the inputs the stage-2 "stats-variant" models add on
top of the bottom-line projection ``p`` (DEFENSE_MATCHUP_RATINGS.md §4).
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

from .directory import build_players_url, PAGE_SIZE
from .fetcher import YahooFetcher, YahooBlockedError, YahooFetchError
from .league_ids import resolve_league_id

DEF_POSITION = "DEF"
DATA_ROOT = Path("data/dst_matchup")

# The eight per-stat projected component columns, in the order Yahoo renders
# them in the DST players table (verified on a live 2025 page). The last column
# is "Kickoff and Punt Return Touchdowns" (a return-TD count worth 6, NOT yards).
COMPONENT_KEYS = (
    "points_allowed", "sack", "safety", "interception",
    "fumble_recovery", "touchdown", "block_kick", "return_td",
)


# --- team href slug -> nflverse abbreviation --------------------------------
# The DST name anchor links to ``https://sports.yahoo.com/nfl/teams/<slug>/``.
# The slug is a stable, unambiguous team key (unlike the bare nickname), so we
# map it straight to the nflverse play-by-play abbreviation. Rams/Chargers slugs
# are ``la-rams``/``la-chargers``; nflverse uses LA (Rams) / LAC (Chargers).
_SLUG_TO_ABBR: dict[str, str] = {
    "arizona": "ARI", "atlanta": "ATL", "baltimore": "BAL", "buffalo": "BUF",
    "carolina": "CAR", "chicago": "CHI", "cincinnati": "CIN", "cleveland": "CLE",
    "dallas": "DAL", "denver": "DEN", "detroit": "DET", "green-bay": "GB",
    "houston": "HOU", "indianapolis": "IND", "jacksonville": "JAX",
    "kansas-city": "KC", "las-vegas": "LV", "la-chargers": "LAC",
    "la-rams": "LA", "los-angeles-chargers": "LAC", "los-angeles-rams": "LA",
    "miami": "MIA", "minnesota": "MIN", "new-england": "NE",
    "new-orleans": "NO", "ny-giants": "NYG", "ny-jets": "NYJ",
    "new-york-giants": "NYG", "new-york-jets": "NYJ",
    "philadelphia": "PHI", "pittsburgh": "PIT", "san-francisco": "SF",
    "seattle": "SEA", "tampa-bay": "TB", "tennessee": "TEN",
    "washington": "WAS",
    # historical slugs, in case a past season renders the old name
    "oakland": "LV", "san-diego": "LAC", "st-louis": "LA",
    "washington-football-team": "WAS",
}

# Fold historical/variant pbp abbreviations to the nflverse-dominant form so a
# join against play-by-play lines up. outcomes.py applies the same folding.
_ABBR_ALIASES = {"LAR": "LA", "WFT": "WAS", "OAK": "LV", "SD": "LAC", "STL": "LA"}

_TEAM_HREF_RE = re.compile(r"/nfl/teams/([a-z0-9-]+)/")
# The opponent is the second anchor in the ysf-game-status span: a
# ``/pointsagainst?...`` link whose title starts with the opponent's uppercase
# abbreviation (e.g. "TEN gives up ..."). The sibling game-status anchor ends
# with "vs" (this team hosted) or "@" (this team was away).
_OPP_TITLE_ABBR_RE = re.compile(r"^([A-Z]{2,3})\b")
_HOMEAWAY_RE = re.compile(r"(vs|@)\s*$")


def slug_to_abbr(slug: Optional[str]) -> Optional[str]:
    """Map a Yahoo team href slug to an nflverse abbreviation, else ``None``."""
    if not slug:
        return None
    return _SLUG_TO_ABBR.get(slug.strip().lower())


def normalize_abbr(abbr: str) -> str:
    """Fold historical/variant abbreviations to the nflverse-dominant form."""
    return _ABBR_ALIASES.get(abbr, abbr)


def _num(text: Optional[str]) -> Optional[float]:
    """Parse a numeric points cell; '-'/''/non-numeric -> None (commas stripped)."""
    if text is None:
        return None
    t = text.strip().replace(",", "")
    if not t or t in {"-", "\u2013", "\u2014"}:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", t)
    return float(m.group()) if m else None


@dataclass(frozen=True)
class DstRow:
    """One DST's scraped value for a (season, week), already teamed.

    ``value`` is the selected-stat cell (the bottom-line projection under
    ``S_PW_<week>``, or the realized actual under ``S_W_<week>``). ``components``
    is the eight per-stat projected columns keyed by :data:`COMPONENT_KEYS`
    (populated on projection pages; on an actual page they are realized per-stat
    values, which we keep for the reconciliation check).
    """

    team: Optional[str]           # nflverse abbr, None if slug didn't resolve
    name: str                     # Yahoo nickname ("Browns")
    slug: Optional[str]           # team href slug ("cleveland")
    value: Optional[float]        # selected-stat value (projection or actual)
    components: dict = field(default_factory=dict)
    opponent: Optional[str] = None   # nflverse abbr of the offense faced, None on a bye
    home: Optional[bool] = None      # True if this DST's team hosted, from vs/@

    def to_dict(self) -> dict:
        return {"team": self.team, "name": self.name, "slug": self.slug,
                "value": self.value, "components": self.components,
                "opponent": self.opponent, "home": self.home}


def _dst_table(soup):
    """The DST players table: the first table whose body rows link to
    ``/nfl/teams/<slug>/`` (the DST name anchor). Returns ``None`` if absent
    (the paginator's stop condition / a soft-block page)."""
    for t in soup.find_all("table"):
        body = t.find("tbody")
        if body is None:
            continue
        if body.find("a", href=_TEAM_HREF_RE):
            return t
    return None


def _parse_opponent(tr) -> tuple[Optional[str], Optional[bool]]:
    """Opponent nflverse abbr + home flag from a row's game-status cell.

    The opponent and home/away are rendered in the player cell's
    ``ysf-game-status`` span (not a stat column): a ``vs``/``@`` game-status
    anchor followed by a ``pointsagainst`` link whose ``title`` starts with the
    opponent's uppercase abbreviation. Returns ``(None, None)`` on a bye or when
    the status isn't present (e.g. a pre-game page before the matchup renders).
    """
    status = tr.find("span", class_="ysf-game-status")
    if status is None:
        return None, None
    links = status.find_all("a")
    if not links:
        return None, None
    gs = links[0].get_text(" ", strip=True)
    ha_m = _HOMEAWAY_RE.search(gs)
    home = None if ha_m is None else (ha_m.group(1) == "vs")
    opponent = None
    if len(links) > 1:
        title = links[1].get("title", "") or ""
        tm = _OPP_TITLE_ABBR_RE.match(title)
        raw = tm.group(1) if tm else links[1].get_text(strip=True).upper()
        opponent = normalize_abbr(raw) if raw else None
    return opponent, home


def parse_dst_page(html: str) -> list[DstRow]:
    """Parse one DST players-list page into [DstRow].

    Column layout (verified live): the selected stat is ``td.pts.Selected``; the
    eight component columns are the last eight numeric data cells before the
    trailing spacer. We index components from the END of the row so a layout
    shift in the leading identity/rank columns doesn't misalign them. The
    opponent faced and home/away come from the game-status cell (§ opponent
    extraction), so the whole modeling table is Yahoo-only (no schedule file).
    """
    soup = BeautifulSoup(html, "lxml")
    table = _dst_table(soup)
    if table is None:
        return []
    body = table.find("tbody")
    if body is None:
        return []

    out: list[DstRow] = []
    for tr in body.find_all("tr", recursive=False):
        a = tr.find("a", href=_TEAM_HREF_RE)
        if a is None:
            continue
        m = _TEAM_HREF_RE.search(a.get("href") or "")
        slug = m.group(1) if m else None
        nick = a.get_text(strip=True)
        abbr = slug_to_abbr(slug)

        sel = tr.select_one("td.pts.Selected") or tr.select_one("td.Selected")
        value = _num(sel.get_text(strip=True)) if sel else None

        # Components: the last 8 cells before the trailing 'Spacer' td.
        tds = tr.find_all("td", recursive=False)
        data_tds = [td for td in tds if "Spacer" not in (td.get("class") or [])]
        comp_cells = data_tds[-len(COMPONENT_KEYS):] if len(data_tds) >= len(COMPONENT_KEYS) else []
        components = {
            k: _num(td.get_text(strip=True))
            for k, td in zip(COMPONENT_KEYS, comp_cells)
        } if comp_cells else {}

        opponent, home = _parse_opponent(tr)

        out.append(DstRow(
            team=normalize_abbr(abbr) if abbr else None,
            name=nick, slug=slug, value=value, components=components,
            opponent=opponent, home=home,
        ))
    return out


def _stat1_for(week: int, *, kind: str) -> str:
    """Yahoo stat context key. kind='proj' -> S_PW_<week> (projection);
    kind='actual' -> S_W_<week> (realized)."""
    if kind == "proj":
        return f"S_PW_{week}"
    if kind == "actual":
        return f"S_W_{week}"
    raise ValueError(f"kind must be 'proj' or 'actual', got {kind!r}")


@dataclass
class DstScraper:
    """Paginate the DST players list for one (season, week, kind).

    Reuses ``YahooFetcher`` for cookie auth + URL-hash caching + soft-block
    detection; the polite jittered delay is applied only before a live fetch, so
    re-runs read instantly from cache. ``sleep``/``rng`` are injectable for
    tests.
    """

    season: int
    week: int
    kind: str = "proj"
    league_id: Optional[str] = None
    fetcher: Optional[YahooFetcher] = None
    max_pages: int = 8                   # 32 DSTs / 25 per page -> 2 pages + stop
    stop_after_empties: int = 2
    delay_range: tuple[float, float] = (2.0, 4.0)
    max_retries: int = 3
    backoff_base: float = 1.5
    sleep: Callable[[float], None] = time.sleep
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        self.league_id = self.league_id or resolve_league_id(season=self.season)
        if self.fetcher is None:
            self.fetcher = YahooFetcher(season=self.season)

    def _fetch_one(self, url: str, *, force_refresh: bool) -> str:
        if self.fetcher.is_cached(url) and not force_refresh:
            return self.fetcher._raw_path(url).read_text(
                encoding="utf-8", errors="replace")
        if not self.fetcher.cookie:
            raise YahooFetchError(
                "No cached HTML and no cookie available. Set YAHOO_COOKIE.")
        headers = {
            "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64; rv:151.0) "
                           "Gecko/20100101 Firefox/151.0"),
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
                if resp.status_code in (999, 429, 403):
                    # Yahoo's rate-limit / bot-block response. Retrying a few
                    # seconds apart won't clear the cooldown window, so surface
                    # it as a block and let the caller back off, rather than
                    # burning retries and then skipping the week.
                    raise YahooBlockedError(
                        f"HTTP {resp.status_code}: Yahoo rate-limit/bot block "
                        f"(cooldown in effect). {resp.text[:120]}")
                if resp.status_code != 200:
                    raise YahooFetchError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                html = resp.text
                if "data-ys-playerid" not in html and "players?status" not in html:
                    raise YahooBlockedError(
                        "DST players page looks empty — likely a soft block, "
                        "throttle, or expired cookie (bounced to login).")
                self.fetcher._raw_path(url).write_text(html, encoding="utf-8")
                return html
            except YahooBlockedError:
                raise
            except (requests.RequestException, YahooFetchError) as e:
                last_exc = e
                self.sleep(self.backoff_base ** attempt + self.rng.uniform(0, 0.5))
        raise YahooFetchError(
            f"giving up on {url} after {self.max_retries} attempts: {last_exc}")

    def scrape(self, *, force_refresh: bool = False) -> list[DstRow]:
        stat1 = _stat1_for(self.week, kind=self.kind)
        seen: set[tuple[Optional[str], str]] = set()
        rows: list[DstRow] = []
        consecutive_empty = 0
        for page in range(self.max_pages):
            url = build_players_url(
                self.league_id, DEF_POSITION, page * PAGE_SIZE,
                season=self.season, stat1=stat1)
            html = self._fetch_one(url, force_refresh=force_refresh)
            page_rows = parse_dst_page(html)
            if not page_rows:
                consecutive_empty += 1
                if consecutive_empty >= self.stop_after_empties:
                    break
                continue
            consecutive_empty = 0
            for r in page_rows:
                key = (r.slug, r.name)
                if key not in seen:
                    seen.add(key)
                    rows.append(r)
        return rows


def scrape_dst_week(*, season: int, week: int, kind: str = "proj",
                    league_id: Optional[str] = None,
                    force_refresh: bool = False) -> list[DstRow]:
    """Scrape one (season, week) DST players-list page (projection or actual)."""
    return DstScraper(season=season, week=week, kind=kind,
                      league_id=league_id).scrape(force_refresh=force_refresh)


def write_dst_week(rows: list[DstRow], *, season: int, week: int,
                   kind: str = "proj", root: Path | str = DATA_ROOT) -> Path:
    fname = "dst_projections.json" if kind == "proj" else "dst_actuals.json"
    out = Path(root) / str(season) / f"week{week}" / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "season": season, "week": week, "kind": kind, "n_rows": len(rows),
        "rows": [r.to_dict() for r in rows],
    }, indent=2))
    return out


def load_dst_week(*, season: int, week: int, kind: str = "proj",
                  root: Path | str = DATA_ROOT) -> list[DstRow]:
    fname = "dst_projections.json" if kind == "proj" else "dst_actuals.json"
    path = Path(root) / str(season) / f"week{week}" / fname
    data = json.loads(Path(path).read_text())
    return [DstRow(team=r["team"], name=r["name"], slug=r.get("slug"),
                   value=r["value"], components=r.get("components", {}),
                   opponent=r.get("opponent"), home=r.get("home"))
            for r in data["rows"]]
