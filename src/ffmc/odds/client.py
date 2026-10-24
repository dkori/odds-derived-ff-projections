"""The Odds API client with on-disk caching.

The free tier is quota-limited (a few hundred requests/month), so every response
is cached to ``data/cache/`` keyed by the request. During development you work
almost entirely from cache; a live call is only made on a cache miss (or when
``force_refresh`` is set).

Player props live on the *event-odds* endpoint, not the bulk odds endpoint:
    GET /v4/sports/{sport}/events/{event_id}/odds

The exact market keys (e.g. player_receptions, player_reception_yds and their
_alternate variants, player_anytime_td) should be confirmed against the live API
once you have a key; they are centralized in ffmc.odds.markets.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

import requests

BASE_URL = "https://api.the-odds-api.com/v4"
DEFAULT_CACHE_DIR = Path("data/cache")


class OddsAPIError(RuntimeError):
    """An Odds API request failed.

    Carries the HTTP ``status_code`` (when the failure was an HTTP response)
    so callers can distinguish a per-event 404 -- a game whose props have been
    pulled because it already kicked off -- from auth/quota errors (401/429)
    that should still fail the whole run. ``status_code`` is ``None`` when the
    error isn't tied to an HTTP response (e.g. a missing API key).
    """

    def __init__(self, *args, status_code: Optional[int] = None) -> None:
        super().__init__(*args)
        self.status_code = status_code


class OddsClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = session or requests.Session()

    def _cache_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{digest}.json"

    def _get(self, path: str, params: dict[str, Any], *, force_refresh: bool = False) -> Any:
        cache_key = path + "?" + json.dumps(params, sort_keys=True)
        cache_file = self._cache_path(cache_key)

        if cache_file.exists() and not force_refresh:
            return json.loads(cache_file.read_text())

        if not self.api_key:
            raise OddsAPIError(
                "No cached response and ODDS_API_KEY is not set. "
                "Set the env var or work from a fixture in data/samples/."
            )

        params = {**params, "apiKey": self.api_key}
        resp = self._session.get(f"{BASE_URL}{path}", params=params, timeout=30)
        if resp.status_code != 200:
            raise OddsAPIError(
                f"HTTP {resp.status_code}: {resp.text[:200]}",
                status_code=resp.status_code,
            )
        data = resp.json()
        cache_file.write_text(json.dumps(data))
        # Surface quota usage if the API returned it.
        remaining = resp.headers.get("x-requests-remaining")
        last = resp.headers.get("x-requests-last")
        if remaining is not None:
            cost = f" (last call cost {last})" if last is not None else ""
            print(f"[odds-api] requests remaining this period: {remaining}{cost}")
        return data

    def list_events(self, sport: str = "americanfootball_nfl", **kw) -> list[dict]:
        """Upcoming events (games) for a sport. Cheap; does not include props."""
        return self._get(f"/sports/{sport}/events", {}, **kw)

    def event_player_props(
        self,
        event_id: str,
        markets: list[str],
        sport: str = "americanfootball_nfl",
        regions: str = "us",
        bookmakers: Optional[str] = None,
        odds_format: str = "american",
        **kw,
    ) -> dict:
        """Player-prop odds for a single event.

        ``markets`` is a comma-joined list of market keys (see ffmc.odds.markets).
        ``bookmakers`` optionally restricts to a single book, e.g. 'draftkings'.
        """
        params: dict[str, Any] = {
            "regions": regions,
            "markets": ",".join(markets),
            "oddsFormat": odds_format,
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
        return self._get(f"/sports/{sport}/events/{event_id}/odds", params, **kw)

    # ---- historical endpoints (PAID plan; 10x cost per market) -----------
    def historical_events(
        self,
        date: str,
        sport: str = "americanfootball_nfl",
        commence_time_from: Optional[str] = None,
        commence_time_to: Optional[str] = None,
        **kw,
    ) -> dict:
        """Historical events snapshot at ``date`` (ISO8601).

        Cost: 1 credit (0 if no events found). Returns a wrapper
        ``{timestamp, previous_timestamp, next_timestamp, data: [events]}``.
        Use this to find event ids + verify games before fetching paid odds.
        """
        params: dict[str, Any] = {"date": date}
        if commence_time_from:
            params["commenceTimeFrom"] = commence_time_from
        if commence_time_to:
            params["commenceTimeTo"] = commence_time_to
        return self._get(f"/historical/sports/{sport}/events", params, **kw)

    def historical_event_odds(
        self,
        event_id: str,
        date: str,
        markets: list[str],
        sport: str = "americanfootball_nfl",
        regions: str = "us",
        bookmakers: Optional[str] = None,
        odds_format: str = "american",
        **kw,
    ) -> dict:
        """Historical odds for one event at ``date``. PAID: 10 x markets x regions.

        Returns ``{timestamp, previous_timestamp, next_timestamp, data: {event}}``.
        """
        params: dict[str, Any] = {
            "date": date,
            "regions": regions,
            "markets": ",".join(markets),
            "oddsFormat": odds_format,
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
        return self._get(
            f"/historical/sports/{sport}/events/{event_id}/odds", params, **kw
        )
