"""Tests for the Yahoo league crawler.

Uses a fake fetcher so no network or real sleeping happens. The crawler's
``sleep`` seam is a no-op and ``rng`` is seeded for determinism.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest

from ffmc.yahoo.crawler import LeagueCrawler, build_matchup_url
from ffmc.yahoo.fetcher import YahooBlockedError, YahooFetchError

FIXTURE = Path(__file__).parent / "fixtures" / "yahoo_matchup_sample.html"


class FakeFetcher:
    """Stand-in for YahooFetcher with scripted per-URL responses.

    responses: dict[url -> html | Exception]. ``cached_urls`` marks which URLs
    are "already on disk" (so the crawler treats them as cache hits).
    """

    def __init__(self, responses, cached_urls=None, season=None):
        self.responses = responses
        self.cached_urls = set(cached_urls or [])
        self.season = season
        self.calls = []  # URLs actually "fetched" live

    def is_cached(self, url):
        return url in self.cached_urls

    def fetch_html(self, url, *, force_refresh=False):
        if not self.is_cached(url) or force_refresh:
            self.calls.append(url)
        val = self.responses[url]
        if isinstance(val, Exception):
            raise val
        return val


def _html():
    return FIXTURE.read_text()


def _no_sleep(_seconds):
    return None


def _crawler(fetcher, n_teams=2, season=None):
    return LeagueCrawler(
        league_id="999", season=season, fetcher=fetcher, n_teams=n_teams,
        sleep=_no_sleep, rng=random.Random(0),
    )


def test_build_url_includes_season_and_params():
    url = build_matchup_url("123456", week=12, mid1=3, season=2025)
    assert "/2025/f1/123456/matchup" in url
    assert "week=12" in url and "mid1=3" in url
    # current season omits the year segment
    assert "/2025/" not in build_matchup_url("123456", week=12, mid1=3)


def test_plan_urls_reports_cache_status():
    urls = {build_matchup_url("999", 12, m): _html() for m in (1, 2)}
    cached = [build_matchup_url("999", 12, 1)]
    crawler = _crawler(FakeFetcher(urls, cached_urls=cached), n_teams=2)
    plan = crawler.plan_urls(12)
    assert len(plan) == 2
    assert plan[0][2] is True    # mid1=1 cached
    assert plan[1][2] is False   # mid1=2 not cached


def test_dedupes_players_across_teams():
    # Both teams return the same fixture (6 players). Deduped by id -> 6, not 12.
    urls = {build_matchup_url("999", 12, m): _html() for m in (1, 2)}
    crawler = _crawler(FakeFetcher(urls), n_teams=2)
    result = crawler.crawl_week(12)
    assert result.teams_seen == 2
    assert len(result.players) == 6
    assert not result.blocked


def test_soft_block_halts_run_without_corrupting():
    # team1 ok, team2 soft-blocked -> stop, keep team1's players, flag blocked.
    u1 = build_matchup_url("999", 12, 1)
    u2 = build_matchup_url("999", 12, 2)
    u3 = build_matchup_url("999", 12, 3)
    responses = {
        u1: _html(),
        u2: YahooBlockedError("no statTable"),
        u3: _html(),  # should never be reached
    }
    fetcher = FakeFetcher(responses)
    crawler = _crawler(fetcher, n_teams=3)
    result = crawler.crawl_week(12)
    assert result.blocked is True
    assert "mid1=2" in result.error
    assert result.teams_seen == 1          # halted after the block
    assert len(result.players) == 6        # team1 data preserved, not empty
    assert u3 not in fetcher.calls         # did not continue past the block


def test_no_real_sleep_and_delay_only_on_live_fetch():
    # All cached -> crawler must not sleep at all and must not "fetch" live.
    sleeps = []
    urls = {build_matchup_url("999", 12, m): _html() for m in (1, 2)}
    fetcher = FakeFetcher(urls, cached_urls=list(urls))
    crawler = LeagueCrawler(
        league_id="999", fetcher=fetcher, n_teams=2,
        sleep=lambda s: sleeps.append(s), rng=random.Random(0),
    )
    result = crawler.crawl_week(12)
    assert fetcher.calls == []     # nothing fetched live
    assert sleeps == []            # no delay before cache hits
    assert result.pages_from_cache == 2
    assert result.pages_fetched == 0


def test_retry_then_success(monkeypatch):
    # First attempt raises a transient error, second succeeds.
    u1 = build_matchup_url("999", 12, 1)
    seq = [YahooFetchError("timeout"), _html()]

    class FlakyFetcher(FakeFetcher):
        def fetch_html(self, url, *, force_refresh=False):
            self.calls.append(url)
            val = seq.pop(0)
            if isinstance(val, Exception):
                raise val
            return val

    fetcher = FlakyFetcher({u1: None})
    crawler = _crawler(fetcher, n_teams=1)
    result = crawler.crawl_week(12)
    assert len(result.players) == 6
    assert result.error is None


def test_skips_opponent_already_covered():
    # Page mid1=1 reports it covers teams (1, 2); mid1=2 must then be skipped.
    html_12 = _html().replace("<body>", "<body>mid1=1 mid2=2 ")
    u1 = build_matchup_url("999", 12, 1)
    u2 = build_matchup_url("999", 12, 2)
    responses = {u1: html_12, u2: _html()}
    fetcher = FakeFetcher(responses)
    crawler = _crawler(fetcher, n_teams=2)
    result = crawler.crawl_week(12)
    assert result.teams_seen == 1          # only team1's page visited
    assert result.pages_skipped == 1       # team2 skipped as covered opponent
    assert u2 not in fetcher.calls         # opponent page never fetched
    assert len(result.players) == 6


def test_write_roster(tmp_path):
    urls = {build_matchup_url("999", 12, m, season=2025): _html() for m in (1, 2)}
    crawler = _crawler(FakeFetcher(urls, season=2025), n_teams=2, season=2025)
    result = crawler.crawl_week(12)
    path = crawler.write_roster(result, out_dir=tmp_path)
    assert path == tmp_path / "2025" / "week12" / "roster.json"
    import json
    data = json.loads(path.read_text())
    assert data["season"] == 2025 and data["week"] == 12
    assert data["n_players"] == 6
