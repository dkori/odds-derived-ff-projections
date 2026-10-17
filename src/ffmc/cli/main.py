"""Offline CLI for fantasy-montecarlo.

Examples:
    # Simulate every player in a cached/sample event payload (no API key):
    ffmc simulate --event data/samples/sample_event.json --scoring ppr --sims 50000

    # List upcoming NFL events (needs ODDS_API_KEY or a cached response):
    ffmc events

    # Fetch player props for one event and simulate (needs ODDS_API_KEY):
    ffmc fetch --event-id <id> --book draftkings --scoring ppr
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ffmc.odds.client import OddsClient
from ffmc.odds.markets import ALL_MARKETS, build_player_models
from ffmc.scoring.config import from_preset, resolve_scoring
from ffmc.sim.montecarlo import simulate_player


def _resolve_scoring(args):
    """CLI scoring resolution: --scoring-file wins, else --scoring preset (if the
    user set one), else the default scoring file (0.5 PPR, auto-created)."""
    preset = args.scoring if getattr(args, "scoring", None) else None
    return resolve_scoring(
        scoring_file=getattr(args, "scoring_file", None), preset=preset
    )


def _print_distribution(dist, thresholds: list[float]) -> None:
    s = dist.summary()
    print(f"\n{s['name']}")
    print(f"  mean {s['mean']:>6}   median {s['median']:>6}   sd {s['std']:>5}")
    print(f"  floor(p10) {s['floor_p10']:>6}   ceiling(p90) {s['ceiling_p90']:>6}")
    if thresholds:
        probs = "   ".join(
            f">{t}: {dist.prob_over(t) * 100:4.1f}%" for t in thresholds
        )
        print(f"  {probs}")


def _load_event(path: str) -> dict:
    return json.loads(Path(path).read_text())


def cmd_simulate(args: argparse.Namespace) -> int:
    payload = _load_event(args.event)
    models = build_player_models(payload, book_key=args.book)
    if not models:
        print("No players parsed from payload (check --book / market keys).")
        return 1
    scoring, _label = _resolve_scoring(args)
    thresholds = [float(t) for t in args.thresholds.split(",")] if args.thresholds else []
    _run_and_print(models, scoring, args, thresholds)
    return 0


def cmd_events(args: argparse.Namespace) -> int:
    client = OddsClient()
    events = client.list_events(sport=args.sport)
    for e in events:
        print(f"{e['id']}  {e['commence_time']}  {e['away_team']} @ {e['home_team']}")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    client = OddsClient()
    payload = client.event_player_props(
        event_id=args.event_id,
        markets=ALL_MARKETS,
        sport=args.sport,
        bookmakers=args.book,
    )
    models = build_player_models(payload, book_key=args.book)
    scoring, _label = _resolve_scoring(args)
    thresholds = [float(t) for t in args.thresholds.split(",")] if args.thresholds else []
    _run_and_print(models, scoring, args, thresholds)
    return 0


def cmd_slate(args: argparse.Namespace) -> int:
    from datetime import datetime

    from ffmc.sim.slate import precompute_slate, select_events
    from ffmc.odds.client import OddsClient as _Client

    def _parse_dt(s):
        return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None

    commence_from = _parse_dt(args.commence_from)
    commence_to = _parse_dt(args.commence_to)

    client = _Client()
    # events endpoint is free; use it to preview scope + cost before spending.
    events = client.list_events(sport=args.sport)
    if args.event_ids:
        wanted = set(args.event_ids.split(","))
        events = [e for e in events if e["id"] in wanted]
    else:
        events = select_events(events, commence_from, commence_to)

    n_markets = len(ALL_MARKETS)
    worst_case = len(events) * n_markets  # 1 region/book
    print(f"[slate] {len(events)} games in scope for '{args.week_label}'")
    for e in events:
        print(f"    {e['commence_time']}  {e['away_team']} @ {e['home_team']}")
    print(
        f"[slate] worst-case cost if all uncached: "
        f"{len(events)} games x up to {n_markets} markets requested x 1 book "
        f"= up to {worst_case} credits. Books return only the markets they price "
        f"(~8/game typical), so real cost is lower; cached games cost 0."
    )

    if args.dry_run:
        print("[slate] --dry-run: not fetching or simulating.")
        return 0

    scoring, scoring_label = _resolve_scoring(args)
    result = precompute_slate(
        scoring=scoring,
        scoring_name=scoring_label,
        week_label=args.week_label,
        client=client,
        book=args.book,
        sport=args.sport,
        commence_from=commence_from,
        commence_to=commence_to,
        n_sims=args.sims,
        seed=args.seed,
        correlated=args.correlated,
        force_refresh=args.force_refresh,
        event_ids=args.event_ids.split(",") if args.event_ids else None,
        store_dir=args.store_dir,
        exclude_qbs=args.exclude_qbs,
    )
    print(
        f"\n[slate] done: {result.n_players} players across {result.n_games} games "
        f"-> {result.store_path}"
    )
    all_excluded = sorted({q for g in result.games for q in g.get("qbs_excluded", [])})
    if all_excluded:
        print(f"[slate] excluded {len(all_excluded)} QBs: {', '.join(all_excluded)}")
    elif args.exclude_qbs:
        print("[slate] no listed QBs found in this slate to exclude.")
    for g in result.games:
        qb = g.get("qbs_excluded") or []
        qb_note = f"  (−{len(qb)} QB)" if qb else ""
        print(
            f"    {g['game']:36s} qualified {g['players_qualified']:>3}"
            f" / {g['players_total']:>3} parsed{qb_note}"
        )
    return 0


def cmd_crawl(args: argparse.Namespace) -> int:
    from ffmc.yahoo import LeagueCrawler, YahooFetcher
    from ffmc.yahoo.league_ids import resolve_league_id, LeagueIdError

    try:
        league_id = resolve_league_id(explicit=args.league_id, season=args.season)
    except LeagueIdError as e:
        print(f"[crawl] {e}")
        return 1

    fetcher = YahooFetcher(cookie=args.cookie, season=args.season)
    crawler = LeagueCrawler(
        league_id=league_id,
        season=args.season,
        fetcher=fetcher,
        n_teams=args.teams,
        delay_range=(args.delay_min, args.delay_max),
    )
    weeks = [int(w) for w in str(args.weeks).split(",")]

    if args.dry_run:
        print(f"[crawl] DRY RUN — league {league_id}, season {args.season}, "
              f"{args.teams} teams, weeks {weeks}")
        total_fetch = total_cached = 0
        for week in weeks:
            plan = crawler.plan_urls(week)
            cached = sum(1 for *_, c in plan if c)
            tofetch = len(plan) - cached
            total_fetch += tofetch
            total_cached += cached
            print(f"\n  week {week}: {len(plan)} pages ({cached} cached, {tofetch} to fetch)")
            for mid1, url, is_c in plan:
                print(f"    [{'cache' if is_c else 'FETCH'}] mid1={mid1:<2} {url}")
        print(f"\n[crawl] would make {total_fetch} live request(s); "
              f"{total_cached} already cached. No network performed.")
        return 0

    rc = 0
    for week in weeks:
        result = crawler.crawl_week(week, force_refresh=args.force_refresh)
        path = crawler.write_roster(result, out_dir=args.out_dir)
        status = "BLOCKED" if result.blocked else ("ERROR" if result.error else "ok")
        print(f"[crawl] week {week}: {len(result.players)} players from "
              f"{result.teams_seen} team page(s) "
              f"({result.pages_fetched} fetched, {result.pages_from_cache} cached) "
              f"-> {path}  [{status}]")
        if result.error:
            print(f"         {result.error}")
        if result.blocked:
            print("[crawl] halting remaining weeks due to soft block. "
                  "Wait, refresh cookie, and re-run (cached pages won't re-fetch).")
            rc = 2
            break
    return rc


def cmd_yahoo(args: argparse.Namespace) -> int:
    from ffmc.yahoo import YahooFetcher, YahooFetchError

    fetcher = YahooFetcher(
        cookie=args.cookie, season=args.season, snapshot_dir=args.out_dir
    )
    try:
        if args.html_file:
            page = fetcher.parse_file(args.html_file, week=args.week, write=not args.no_write)
            source = args.html_file
        elif args.url:
            page = fetcher.fetch_week(
                args.url, week=args.week,
                force_refresh=args.force_refresh, write=not args.no_write,
            )
            source = args.url
        else:
            print("Provide either --html-file (offline) or --url (live, needs cookie).")
            return 1
    except YahooFetchError as e:
        print(f"[yahoo] fetch error: {e}")
        return 1

    starters = [p for p in page.players if p.roster_slot not in {"BN", "IR"}]
    print(f"[yahoo] week {page.week}: {len(page.players)} players "
          f"({len(starters)} starters) from {source}")
    print(f"{'SIDE':<5}{'SLOT':<6}{'PLAYER':<22}{'TEAM/POS':<9}{'PROJ':>7}{'ACTUAL':>8}")
    for p in page.players:
        proj = "-" if p.projected is None else f"{p.projected:.2f}"
        act = "-" if p.actual is None else f"{p.actual:.2f}"
        tp = f"{p.nfl_team or '?'}/{p.position or '?'}"
        print(f"{p.team_side:<5}{str(p.roster_slot):<6}{p.name:<22}{tp:<9}{proj:>7}{act:>8}")

    if not args.no_write:
        print(f"\n[yahoo] snapshot written: {fetcher.snapshot_path(args.week)}")
    return 0


def _run_and_print(models, scoring, args, thresholds) -> None:
    mode = "correlated (Gaussian copula)" if args.correlated else "independent"
    print(f"[mode: {mode}]")
    for name in sorted(models):
        model = models[name]
        dist = simulate_player(
            model,
            scoring,
            n_sims=args.sims,
            seed=args.seed,
            correlated=args.correlated,
        )
        _print_distribution(dist, thresholds)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ffmc", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--scoring",
        default=None,
        help="scoring preset: ppr | half_ppr | standard. If omitted (and no "
        "--scoring-file), the default scoring file config/scoring.json is used "
        "(auto-created at 0.5 PPR).",
    )
    common.add_argument(
        "--scoring-file",
        default=None,
        help="path to a JSON scoring config (overrides --scoring)",
    )
    common.add_argument("--sims", type=int, default=50_000)
    common.add_argument("--seed", type=int, default=None)
    common.add_argument("--book", default=None, help="bookmaker key, e.g. draftkings")
    common.add_argument(
        "--thresholds", default="", help="comma list of point thresholds for P(>x)"
    )
    common.add_argument("--sport", default="americanfootball_nfl")
    common.add_argument(
        "--correlated",
        action="store_true",
        help="couple stats with a Gaussian copula (measured nflverse correlations); "
        "default is independent",
    )


    sp = sub.add_parser("simulate", parents=[common], help="simulate from a saved event payload")
    sp.add_argument("--event", required=True, help="path to event odds JSON")
    sp.set_defaults(func=cmd_simulate)

    ep = sub.add_parser("events", parents=[common], help="list upcoming events")
    ep.set_defaults(func=cmd_events)

    fp = sub.add_parser("fetch", parents=[common], help="fetch props for an event and simulate")
    fp.add_argument("--event-id", required=True)
    fp.set_defaults(func=cmd_fetch)

    slp = sub.add_parser(
        "slate", parents=[common], help="precompute a slate's projections store"
    )
    slp.add_argument(
        "--week-label", required=True, help="label for the store, e.g. 'week4'"
    )
    slp.add_argument(
        "--commence-from", default=None, help="ISO8601 lower bound on kickoff time"
    )
    slp.add_argument(
        "--commence-to", default=None, help="ISO8601 upper bound on kickoff time"
    )
    slp.add_argument(
        "--event-ids", default=None, help="comma list of event ids (overrides window)"
    )
    slp.add_argument(
        "--store-dir", default="data/projections", help="output directory"
    )
    slp.add_argument(
        "--force-refresh",
        action="store_true",
        help="bypass odds cache (spends credits)",
    )
    slp.add_argument(
        "--dry-run",
        action="store_true",
        help="list games and estimate credit cost without fetching/simulating",
    )
    slp.add_argument(
        "--exclude-qbs",
        action="store_true",
        help="drop quarterbacks (via config/qb_exclusions.json). Default is to "
        "KEEP QBs now that passing is modeled; they are tagged position_group=qb "
        "so the front end compares them separately from skill players.",
    )
    slp.set_defaults(func=cmd_slate)

    yp = sub.add_parser(
        "yahoo",
        help="parse a Yahoo matchup page (saved HTML or live URL) into a weekly "
        "projected-vs-actual snapshot under data/yahoo/<week>/",
    )
    yp.add_argument("--week", type=int, required=True, help="NFL week number")
    yp.add_argument(
        "--season", type=int, default=None,
        help="season year (e.g. 2025) to namespace the snapshot; omit for the "
        "current season",
    )
    yp.add_argument(
        "--html-file", default=None,
        help="path to a saved matchup HTML page (offline; no cookie needed)",
    )
    yp.add_argument(
        "--url", default=None,
        help="live matchup URL, e.g. "
        "https://football.fantasysports.yahoo.com/f1/<league>/matchup?week=N&mid1=M",
    )
    yp.add_argument(
        "--cookie", default=None,
        help="Yahoo session cookie (defaults to $YAHOO_COOKIE). Sensitive — "
        "prefer the env var over passing it on the command line.",
    )
    yp.add_argument(
        "--out-dir", default="data/yahoo", help="snapshot output directory"
    )
    yp.add_argument(
        "--force-refresh", action="store_true",
        help="re-fetch even if raw HTML is cached (live mode only)",
    )
    yp.add_argument(
        "--no-write", action="store_true", help="print only; do not write a snapshot"
    )
    yp.set_defaults(func=cmd_yahoo)

    cp = sub.add_parser(
        "crawl",
        help="crawl all teams in a Yahoo league for one or more weeks into a "
        "deduped roster.json per week (data/yahoo/<season>/week<week>/)",
    )
    cp.add_argument("--league-id", default=None,
                    help="Yahoo league id, e.g. 123456. If omitted, resolved by "
                         "--season from config/league_ids.json (or $FFMC_LEAGUE_ID).")
    cp.add_argument("--weeks", required=True, help="comma list of weeks, e.g. 12,13,14")
    cp.add_argument("--season", type=int, default=None, help="season year, e.g. 2025")
    cp.add_argument("--teams", type=int, default=10, help="number of teams in the league")
    cp.add_argument("--cookie", default=None, help="Yahoo cookie (defaults to $YAHOO_COOKIE)")
    cp.add_argument("--out-dir", default="data/yahoo", help="snapshot output directory")
    cp.add_argument("--delay-min", type=float, default=2.0, help="min seconds between live fetches")
    cp.add_argument("--delay-max", type=float, default=4.0, help="max seconds between live fetches")
    cp.add_argument("--force-refresh", action="store_true", help="re-fetch cached pages")
    cp.add_argument(
        "--dry-run", action="store_true",
        help="list the URLs that would be fetched and cache-hit counts; no network",
    )
    cp.set_defaults(func=cmd_crawl)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
