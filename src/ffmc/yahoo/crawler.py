"""Crawl a full Yahoo Fantasy league's matchup pages for a given week.

Each matchup page renders *both* teams' rosters, so to collect every rostered
player in an N-team league we loop ``mid1 = 1..N`` and dedupe players by
``player_id``. (You only strictly need to visit each distinct matchup, i.e.
~N/2 pages, but visiting all N and deduping is simpler and self-correcting.)

Design goals, in priority order:
  1. **Never corrupt the dataset on a block.** A 200 response with no matchup
     table is a soft block / login wall, not an empty week. We detect it and
     *halt the whole run* rather than record empty rosters.
  2. **Resume cheaply.** Every page is cached by URL, so a re-run only fetches
     what's missing. Get blocked at team 7, fix it, re-run — teams 1–6 don't
     re-fetch.
  3. **Be polite.** A jittered delay is applied *only before an actual network
     fetch* (never before a cache hit), so re-runs are instant and live runs
     don't hammer Yahoo. Volume is tiny (~N pages/week).
  4. **Tolerate transient failures.** Bounded retry with exponential backoff +
     jitter on timeouts / 5xx / 429, then give up and report.

The ``sleep`` and ``rng`` seams are injectable so tests run instantly and
deterministically with no real waiting.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import requests

from .fetcher import YahooFetcher, YahooBlockedError, YahooFetchError
from .parser import MatchupPage, PlayerLine, parse_matchup_html, parse_matchup_teams, parse_matchup_team_names

# Yahoo NFL fantasy matchup URL. The /<season>/ segment is optional (current
# season omits it); we always include it for past seasons.
_BASE = "https://football.fantasysports.yahoo.com"


def build_matchup_url(league_id: str, week: int, mid1: int, season: Optional[int] = None) -> str:
    season_seg = f"/{season}" if season is not None else ""
    return (
        f"{_BASE}{season_seg}/f1/{league_id}/matchup"
        f"?week={week}&mid1={mid1}&module=matchup"
    )


@dataclass
class CrawlResult:
    season: Optional[int]
    week: int
    league_id: str
    players: list[PlayerLine]
    pages_fetched: int          # live network fetches
    pages_from_cache: int
    teams_seen: int
    blocked: bool = False
    error: Optional[str] = None
    pages_skipped: int = 0       # opponent pages skipped (already covered)

    def to_dict(self) -> dict:
        return {
            "season": self.season,
            "week": self.week,
            "league_id": self.league_id,
            "pages_fetched": self.pages_fetched,
            "pages_from_cache": self.pages_from_cache,
            "pages_skipped": self.pages_skipped,
            "teams_seen": self.teams_seen,
            "blocked": self.blocked,
            "error": self.error,
            "n_players": len(self.players),
            "players": [p.to_dict() for p in self.players],
        }


@dataclass
class LeagueCrawler:
    """Crawl all teams in a league for a week into a deduped player list."""

    league_id: str
    season: Optional[int] = None
    fetcher: Optional[YahooFetcher] = None
    n_teams: int = 10
    delay_range: tuple[float, float] = (2.0, 4.0)   # jittered seconds between live fetches
    max_retries: int = 3
    backoff_base: float = 1.5
    # injectable seams (tests pass no-op sleep + seeded rng)
    sleep: Callable[[float], None] = time.sleep
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        if self.fetcher is None:
            self.fetcher = YahooFetcher(season=self.season)

    # ---- planning (no network) -------------------------------------------
    def plan_urls(self, week: int) -> list[tuple[int, str, bool]]:
        """Return [(mid1, url, is_cached)] for every team — pure, no network."""
        out = []
        for mid1 in range(1, self.n_teams + 1):
            url = build_matchup_url(self.league_id, week, mid1, self.season)
            out.append((mid1, url, self.fetcher.is_cached(url)))
        return out

    # ---- one page with retry/backoff -------------------------------------
    def _fetch_one(self, url: str, *, force_refresh: bool) -> tuple[str, bool]:
        """Return (html, from_cache). Raises YahooBlockedError on a soft block.

        Applies the polite delay only when a live fetch is actually needed.
        Retries transient transport errors with exponential backoff + jitter.
        """
        if self.fetcher.is_cached(url) and not force_refresh:
            return self.fetcher.fetch_html(url), True

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            lo, hi = self.delay_range
            self.sleep(self.rng.uniform(lo, hi))
            try:
                return self.fetcher.fetch_html(url, force_refresh=force_refresh), False
            except YahooBlockedError:
                raise  # a block is terminal — do not retry, let caller halt
            except (requests.RequestException, YahooFetchError) as e:
                last_exc = e
                # exponential backoff before the next attempt
                self.sleep(self.backoff_base ** attempt + self.rng.uniform(0, 0.5))
        raise YahooFetchError(
            f"giving up on {url} after {self.max_retries} attempts: {last_exc}"
        )

    # ---- crawl a week -----------------------------------------------------
    def crawl_week(self, week: int, *, force_refresh: bool = False) -> CrawlResult:
        by_id: dict[str, PlayerLine] = {}
        by_name_fallback: dict[str, PlayerLine] = {}
        fetched = cached = teams = 0
        blocked = False
        error: Optional[str] = None
        covered: set[int] = set()        # team ids already pulled (self or opponent)
        skipped = 0

        for mid1 in range(1, self.n_teams + 1):
            if mid1 in covered:
                # already saw this team as someone's opponent — skip the
                # redundant page (each matchup renders both rosters).
                skipped += 1
                continue
            url = build_matchup_url(self.league_id, week, mid1, self.season)
            try:
                html, from_cache = self._fetch_one(url, force_refresh=force_refresh)
            except YahooBlockedError as e:
                blocked = True
                error = f"soft block at mid1={mid1}: {e}"
                break
            except YahooFetchError as e:
                error = f"fetch failed at mid1={mid1}: {e}"
                break

            cached += int(from_cache)
            fetched += int(not from_cache)
            teams += 1

            # mark both teams in this matchup as covered so we skip the opponent
            self_id, opp_id = parse_matchup_teams(html)
            covered.add(mid1)
            if self_id is not None:
                covered.add(self_id)
            if opp_id is not None:
                covered.add(opp_id)

            page = parse_matchup_html(html, week=week)
            home_owner, away_owner = parse_matchup_team_names(html)
            import dataclasses
            for p in page.players:
                # Attach the owning fantasy-team name by matchup side (home=left,
                # away=right). PlayerLine is frozen, so replace to set owner.
                owner = home_owner if p.team_side == "home" else away_owner
                if owner is not None and p.owner is None:
                    p = dataclasses.replace(p, owner=owner)
                # dedupe by stable player_id; fall back to name when id missing
                if p.player_id:
                    if p.player_id not in by_id:
                        by_id[p.player_id] = p
                else:
                    by_name_fallback.setdefault(p.name, p)

        players = list(by_id.values())
        # include name-only players that never appeared with an id
        seen_names = {p.name for p in players}
        players += [p for n, p in by_name_fallback.items() if n not in seen_names]

        return CrawlResult(
            season=self.season, week=week, league_id=self.league_id,
            players=players, pages_fetched=fetched, pages_from_cache=cached,
            teams_seen=teams, blocked=blocked, error=error,
            pages_skipped=skipped,
        )

    def write_roster(self, result: CrawlResult, out_dir: Path | str = "data/yahoo") -> Path:
        base = Path(out_dir)
        if self.season is not None:
            base = base / str(self.season)
        out = base / f"week{result.week}" / "roster.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        import json
        out.write_text(json.dumps(result.to_dict(), indent=2))
        return out
