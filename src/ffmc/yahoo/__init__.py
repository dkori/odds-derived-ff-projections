"""Personal-use Yahoo Fantasy matchup scraper + parser.

Yahoo retired third-party Fantasy *API* access (apps get app-level 403s as of
mid-2026) and the public API never exposed general per-player projections
anyway. The league matchup *page*, however, is server-rendered HTML that
contains, for every roster slot, both Yahoo's **projected** points and the
player's **actual** points for that week — and for already-played weeks of the
current season Yahoo keeps the pre-game projection next to the actual.

So we scrape the matchup page with a logged-in session cookie (personal use,
low volume) and parse the mirrored matchup table out of the HTML. This gives a
weekly {player -> (projected, actual)} snapshot we can line up against our own
odds-derived projections store for an accuracy comparison.

This package is deliberately small and self-contained:
  - ``parser``  : pure HTML -> structured rows (no network, fully testable)
  - ``fetcher`` : cached, cookie-authenticated page fetch + snapshot writer

NOTE on history: this only works going *forward* from the current season. Yahoo
does not version projections, so you cannot reconstruct what it projected in a
past season's week — the projection column is only meaningful for weeks of the
live season. Collect weekly as the season progresses.
"""
from __future__ import annotations

from .parser import MatchupPage, PlayerLine, parse_matchup_html, parse_matchup_team_names
from .fetcher import YahooFetcher, YahooFetchError, YahooBlockedError, looks_like_matchup
from .crawler import LeagueCrawler, CrawlResult, build_matchup_url
from .directory import (
    DirectoryEntry,
    PlayerRow,
    YahooDirectory,
    parse_players_page,
    parse_players_page_rows,
    build_players_url,
    load_directory,
)

__all__ = [
    "MatchupPage",
    "PlayerLine",
    "parse_matchup_html",
    "parse_matchup_team_names",
    "YahooFetcher",
    "YahooFetchError",
    "YahooBlockedError",
    "looks_like_matchup",
    "LeagueCrawler",
    "CrawlResult",
    "build_matchup_url",
    "DirectoryEntry",
    "PlayerRow",
    "YahooDirectory",
    "parse_players_page",
    "parse_players_page_rows",
    "build_players_url",
    "load_directory",
]
