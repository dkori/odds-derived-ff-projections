"""Cached, cookie-authenticated fetch of a Yahoo Fantasy matchup page.

Personal use only. Yahoo's ToS makes authenticated scraping a gray area; keep it
low-volume (a handful of requests per week), cache aggressively, and never
redistribute the raw pages. Raw HTML is cached under ``data/yahoo/raw/`` and
parsed snapshots under ``data/yahoo/[<season>/]week<week>/``.

Auth is a session cookie copied from a logged-in browser. Supply it via the
``YAHOO_COOKIE`` environment variable (preferred — keeps it out of shell
history and argv) or pass it explicitly. The cookie is sensitive: it is a full
account session. Treat it like a password, don't commit it, and rotate it
(log out/in) if it leaks.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from .parser import MatchupPage, parse_matchup_html

DEFAULT_RAW_DIR = Path("data/yahoo/raw")
DEFAULT_SNAPSHOT_DIR = Path("data/yahoo")

# A browser-ish UA; Yahoo serves the full page to this. The real auth is the cookie.
_DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:151.0) Gecko/20100101 Firefox/151.0"
)


class YahooFetchError(RuntimeError):
    pass


class YahooBlockedError(YahooFetchError):
    """Raised when a 200 response has no matchup table (soft block / login wall).

    Distinct from a transport error so the crawler can halt the whole run rather
    than mistake it for an empty week.
    """


def looks_like_matchup(html: str) -> bool:
    """True if the HTML contains a matchup stat table (vs a login/soft-block page)."""
    return "statTable" in html


@dataclass
class YahooFetcher:
    """Fetch + cache a matchup page, then parse it into a weekly snapshot.

    ``season`` (e.g. 2025) namespaces snapshots so past-season weeks don't
    collide with current-season ones. Leave it ``None`` for the current season
    (snapshots land directly under ``data/yahoo/week<week>/``, back-compatible).
    """

    cookie: Optional[str] = None
    season: Optional[int] = None
    raw_dir: Path | str = DEFAULT_RAW_DIR
    snapshot_dir: Path | str = DEFAULT_SNAPSHOT_DIR
    session: Optional[requests.Session] = None

    def __post_init__(self) -> None:
        self.cookie = self.cookie or os.environ.get("YAHOO_COOKIE")
        self.raw_dir = Path(self.raw_dir)
        self.snapshot_dir = Path(self.snapshot_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._session = self.session or requests.Session()

    # ---- raw HTML (cached) ------------------------------------------------
    def _raw_path(self, url: str) -> Path:
        """Cache key is the full URL hash, so season/week/team are all captured."""
        digest = hashlib.sha256(url.encode()).hexdigest()[:12]
        return self.raw_dir / f"matchup_{digest}.html"

    def is_cached(self, url: str) -> bool:
        return self._raw_path(url).exists()

    def fetch_html(self, url: str, *, force_refresh: bool = False) -> str:
        """Return matchup-page HTML, from cache unless ``force_refresh``.

        A live fetch requires a cookie; a cache hit does not. Raises
        :class:`YahooBlockedError` if a fetched page lacks a matchup table.
        """
        cache_file = self._raw_path(url)
        if cache_file.exists() and not force_refresh:
            return cache_file.read_text(encoding="utf-8", errors="replace")

        if not self.cookie:
            raise YahooFetchError(
                "No cached HTML and no cookie available. Set YAHOO_COOKIE "
                "(copied from a logged-in browser session) or pass cookie=..., "
                "or point at a saved HTML file instead."
            )

        headers = {
            "User-Agent": _DEFAULT_UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cookie": self.cookie,
        }
        resp = self._session.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            raise YahooFetchError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        html = resp.text
        if not looks_like_matchup(html):
            raise YahooBlockedError(
                "Fetched page has no matchup table — likely a soft block, "
                "throttle, or expired cookie (bounced to login). Stop, wait, "
                "refresh the cookie, and resume (cached pages won't re-fetch)."
            )
        cache_file.write_text(html, encoding="utf-8")
        return html

    # ---- snapshot (parsed) ------------------------------------------------
    def _week_dir(self, week: int) -> Path:
        base = self.snapshot_dir
        if self.season is not None:
            base = base / str(self.season)
        return base / f"week{week}"

    def snapshot_path(self, week: int) -> Path:
        return self._week_dir(week) / "matchup.json"

    def write_snapshot(self, page: MatchupPage, week: int) -> Path:
        out = self.snapshot_path(week)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(page.to_dict(), indent=2))
        return out

    def fetch_week(
        self,
        url: str,
        week: int,
        *,
        force_refresh: bool = False,
        write: bool = True,
    ) -> MatchupPage:
        """Fetch (or load cached) HTML for a week, parse, and optionally snapshot."""
        html = self.fetch_html(url, force_refresh=force_refresh)
        page = parse_matchup_html(html, week=week)
        if write:
            self.write_snapshot(page, week)
        return page

    def parse_file(
        self, html_path: Path | str, week: int, *, write: bool = True
    ) -> MatchupPage:
        """Parse a locally saved matchup HTML file (no network) into a snapshot."""
        html = Path(html_path).read_text(encoding="utf-8", errors="replace")
        page = parse_matchup_html(html, week=week)
        if write:
            self.write_snapshot(page, week)
        return page
