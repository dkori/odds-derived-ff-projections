"""Slate precompute: turn a week's worth of games into a projections store.

This is the backend batch job behind the browser front-end. It does NOT run at
request time. It:

1. Lists the events (games) for a slate window via the free ``/events`` endpoint.
2. For each game, fetches the player-prop markets for one book (event-odds
   endpoint, cached to ``data/cache/``).
3. Parses every player and simulates the ones with **sufficient data** (at least
   one of receptions / receiving-yards / rushing-yards -- this drops anytime-TD-
   only entries like linemen, QBs, and D/ST that are noise for v1 skill-player
   scope).
4. Writes one combined projections JSON that the front end reads and fuzzy-
   matches player names against.

Cost model (see docs/DECISIONS.md): ``cost = [unique markets returned] x
[regions]`` per game. We now request the primary usage markets plus the TD
ladder (``player_anytime_td`` + ``player_tds_over`` + optional two-sided
``player_tds`` / ``player_rush_reception_tds``). Books return only the subset
they price, so real cost is ~8 credits/game (not the length of the request
list); a ~16-game slate is ~112-130 credits. ``/events`` is free. The client
caches responses, so re-running a slate after a full fetch costs 0 additional
credits.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ffmc.odds.client import OddsClient
from ffmc.odds.markets import ALL_MARKETS, build_player_models
from ffmc.scoring.config import ScoringConfig, scoring_trio
from ffmc.sim.montecarlo import PlayerModel, ScoreDistribution, simulate_player_multi

DEFAULT_STORE_DIR = Path("data/projections")
DEFAULT_QB_EXCLUSIONS = Path("config/qb_exclusions.json")


def _normalize_name(name: str) -> str:
    """Case/punctuation-insensitive key for robust name matching.

    Lowercases, drops everything except ASCII letters/digits/spaces, and
    collapses whitespace. So "C.J. Stroud", "CJ Stroud", and "c j stroud" all
    map to the same key, and "Michael Penix Jr." matches "Michael Penix Jr".
    """
    out = []
    for ch in name.lower():
        if ch.isalnum() or ch.isspace():
            out.append(ch)
    return " ".join("".join(out).split())


def load_qb_exclusions(
    path: Path | str | None = DEFAULT_QB_EXCLUSIONS,
) -> set[str]:
    """Load the QB denylist as a set of normalized names.

    Returns an empty set if the file is missing (filter is then a no-op). v1
    excludes QBs because it models only skill-player prop markets and does not
    fetch/simulate passing stats, so a QB's projection here is only their
    rushing+TD floor and is structurally wrong. The Odds API exposes no position
    field and QBs are market-indistinguishable from run-only RBs, so this is a
    maintained list rather than an odds-derived rule. See config/qb_exclusions.json.
    """
    if path is None:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    data = json.loads(p.read_text())
    return {_normalize_name(n) for n in data.get("quarterbacks", [])}



def has_sufficient_data(model: PlayerModel) -> bool:
    """A player qualifies if they have at least one usage/yardage line.

    Anytime-TD alone is not enough: that market lists everyone who could score
    (linemen, D/ST), so a TD-only entry is noise. Requiring a
    receptions/receiving-yds/rushing-yds line keeps the ~150-250 skill players
    the books price for usage; a passing-yds/passing-TD line qualifies a
    quarterback (whose score is dominated by passing even with no rush line).
    """
    return any(
        d is not None
        for d in (
            model.receptions,
            model.receiving_yards,
            model.rushing_yards,
            model.passing_yards,
            model.passing_tds,
        )
    )


def _stat_presence(model: PlayerModel) -> list[str]:
    present = []
    if model.receptions is not None:
        present.append("receptions")
    if model.receiving_yards is not None:
        present.append("receiving_yards")
    if model.rushing_yards is not None:
        present.append("rushing_yards")
    if model.touchdowns is not None:
        present.append("touchdowns")
    if model.passing_yards is not None:
        present.append("passing_yards")
    if model.passing_tds is not None:
        present.append("passing_tds")
    return present


# The stats we model, for a stable "present/missing" provenance report. The
# passing stats come last so skill-player records read the same as before.
_ALL_STATS = [
    "receptions",
    "receiving_yards",
    "rushing_yards",
    "touchdowns",
    "passing_yards",
    "passing_tds",
]


def classify_group(model: PlayerModel) -> str:
    """Position group for a player, derived purely from the posted markets.

    A player with a passing line (passing yards or passing TDs) is a
    quarterback; everyone else is a skill player (WR/RB/TE). This stays
    position-agnostic in spirit -- there is no position label, only the stats the
    book actually priced -- but gives the front end a clean QB vs skill split so
    QBs (whose scores are dominated by passing) are compared against each other
    rather than mixed in with receivers and runners.
    """
    if model.passing_yards is not None or model.passing_tds is not None:
        return "qb"
    return "skill"


def _td_source(model: PlayerModel) -> dict:
    """Describe how the TD marginal was built (for data-provenance notes).

    Returns a small dict:
        {"available": bool,          # any TD market at all
         "n_anchors": int,           # count TD survival anchors P(>=k) parsed
         "ladder": bool,             # >=2 anchors -> negative-binomial fit
         "model": "nbinom_ladder" | "poisson_anytime" | None}
    A single anchor means only anytime-TD (P>=1) was posted, so the model fell
    back to Poisson; >=2 means a real TD ladder (player_tds_over etc.) pinned the
    multi-TD tail with a negative binomial.
    """
    td = model.touchdowns
    anchors = getattr(td, "anchors", None) if td is not None else None
    n = len(anchors) if anchors else (1 if td is not None else 0)
    if td is None:
        return {"available": False, "n_anchors": 0, "ladder": False, "model": None}
    ladder = n >= 2
    return {
        "available": True,
        "n_anchors": n,
        "ladder": ladder,
        "model": "nbinom_ladder" if ladder else "poisson_anytime",
    }


def _continuous_anchor_counts(model: PlayerModel) -> dict:
    """Anchor count per continuous/discrete ladder stat (data-completeness signal).

    A continuous stat (yards) reconstructed from a SINGLE anchor (main line only,
    no alternates -- common early in the week) can't have its shape measured, so
    the distribution's tail is an assumption, not data. We record the anchor
    count so the front end can flag those low-confidence stats. Returns e.g.
    ``{"receiving_yards": 1, "rushing_yards": 8, "receptions": 6}`` (TDs handled
    separately in ``_td_source``).
    """
    out: dict[str, int] = {}
    for stat, dist in (
        ("receptions", model.receptions),
        ("receiving_yards", model.receiving_yards),
        ("rushing_yards", model.rushing_yards),
        ("passing_yards", model.passing_yards),
    ):
        if dist is None:
            continue
        # Both ladder classes expose their anchor list via .points.
        pts = getattr(dist, "points", None)
        if pts is not None:
            out[stat] = len(pts)
    return out


def _pass_td_source(model: PlayerModel) -> dict:
    """Describe how the passing-TD marginal was built (QB provenance note).

    Mirrors ``_td_source`` but for the separate passing-TD count: >=2 anchors
    means a real ``player_pass_tds`` ladder (negative binomial); a single anchor
    is the fallback. Returns the same shape as ``_td_source`` so the front end
    renders it with the identical badge logic.
    """
    ptd = model.passing_tds
    anchors = getattr(ptd, "anchors", None) if ptd is not None else None
    n = len(anchors) if anchors else (1 if ptd is not None else 0)
    if ptd is None:
        return {"available": False, "n_anchors": 0, "ladder": False, "model": None}
    ladder = n >= 2
    return {
        "available": True,
        "n_anchors": n,
        "ladder": ladder,
        "model": "nbinom_ladder" if ladder else "poisson_anytime",
    }


def _data_provenance(model: PlayerModel) -> dict:
    """Per-player data-provenance summary for the front end.

    Records which of the four modeled stats had posted lines, which were missing,
    how the TD marginal was built (full ladder vs anytime-only), the per-stat
    anchor counts, and a ``low_confidence`` flag when any continuous stat rests
    on a single anchor (main line only -> assumed shape, misleading tail). Lets
    the UI explain *why* a projection looks the way it does and de-emphasize
    thin-data players.
    """
    present = _stat_presence(model)
    anchor_counts = _continuous_anchor_counts(model)
    # Yards stats with a single anchor are low-confidence (fabricated tail).
    single_anchor_stats = [
        s for s in ("receiving_yards", "rushing_yards", "passing_yards")
        if anchor_counts.get(s) == 1
    ]
    # The "missing" list reports only stats relevant to the player's group so a
    # receiver isn't flagged as "missing passing yards" (and vice versa). TD
    # counts are reported via the dedicated td/pass_td blocks, not here.
    group = classify_group(model)
    if group == "qb":
        relevant = ("rushing_yards", "passing_yards")
    else:
        relevant = ("receptions", "receiving_yards", "rushing_yards")
    prov = {
        "present": present,
        "missing": [s for s in relevant if s not in present],
        "td": _td_source(model),
        "anchor_counts": anchor_counts,
        "single_anchor_stats": single_anchor_stats,
        "low_confidence": bool(single_anchor_stats),
    }
    # Only attach the passing-TD provenance for QBs (keeps skill records lean).
    if group == "qb":
        prov["pass_td"] = _pass_td_source(model)
    return prov


def _dist_block(dist: ScoreDistribution) -> dict:
    """The stored distribution block for one scoring: summary + curve + histogram.

    This is the per-scoring payload the front end reads. ``summary()`` already
    rounds the scalars; the dense quantile grid and 64-bin histogram are what the
    app draws the survival/density curves from without re-simulating.
    """
    s = dist.summary()
    return {
        "mean": s["mean"],
        "median": s["median"],
        "std": s["std"],
        "floor_p10": s["floor_p10"],
        "ceiling_p90": s["ceiling_p90"],
        "quantiles": dist.dense_quantiles(),
        "histogram": dist.histogram(),
    }


@dataclass
class SlateResult:
    """Outcome of a slate precompute run (for reporting)."""

    week_label: str
    generated_at: str
    book: Optional[str]
    scoring: str
    correlated: bool
    n_games: int
    n_players: int
    games: list[dict] = field(default_factory=list)
    store_path: Optional[str] = None


def select_events(
    events: list[dict],
    commence_from: Optional[datetime] = None,
    commence_to: Optional[datetime] = None,
) -> list[dict]:
    """Filter events to a slate window by commence_time (inclusive bounds)."""
    out = []
    for e in events:
        ct = e.get("commence_time")
        if ct is None:
            continue
        dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
        if commence_from and dt < commence_from:
            continue
        if commence_to and dt > commence_to:
            continue
        out.append(e)
    return out


def precompute_slate(
    scoring: ScoringConfig,
    scoring_name: str,
    *,
    week_label: str,
    client: Optional[OddsClient] = None,
    book: Optional[str] = "draftkings",
    sport: str = "americanfootball_nfl",
    commence_from: Optional[datetime] = None,
    commence_to: Optional[datetime] = None,
    n_sims: int = 50_000,
    seed: Optional[int] = None,
    correlated: bool = True,
    store_dir: Path | str = DEFAULT_STORE_DIR,
    force_refresh: bool = False,
    event_ids: Optional[list[str]] = None,
    exclude_qbs: bool = False,
    qb_exclusions_path: Path | str | None = DEFAULT_QB_EXCLUSIONS,
) -> SlateResult:
    """Run the full slate precompute and write a combined projections store.

    ``event_ids`` optionally restricts to specific games (else all events in the
    commence window are used). ``force_refresh`` bypasses the odds cache (spends
    credits); by default cached payloads are reused for free.

    Quarterbacks are now modeled (passing marginals) and are KEPT by default,
    tagged with ``position_group == "qb"`` so the front end can compare them
    separately from skill players (WR/RB/TE, ``"skill"``). Pass
    ``exclude_qbs=True`` to drop the quarterbacks listed in ``qb_exclusions_path``
    (legacy behaviour from before passing was modeled).
    """
    client = client or OddsClient()
    qb_names = load_qb_exclusions(qb_exclusions_path) if exclude_qbs else set()

    events = client.list_events(sport=sport)  # free endpoint
    if event_ids:
        wanted = set(event_ids)
        events = [e for e in events if e["id"] in wanted]
    else:
        events = select_events(events, commence_from, commence_to)

    players_out: dict[str, dict] = {}
    games_report: list[dict] = []

    for e in events:
        payload = client.event_player_props(
            event_id=e["id"],
            markets=ALL_MARKETS,
            sport=sport,
            bookmakers=book,
            force_refresh=force_refresh,
        )
        models = build_player_models(payload, book_key=book)
        qualifying = {n: m for n, m in models.items() if has_sufficient_data(m)}
        # Drop excluded QBs (v1 has no passing model; see load_qb_exclusions).
        excluded = [n for n in qualifying if _normalize_name(n) in qb_names]
        for n in excluded:
            del qualifying[n]

        for name, model in qualifying.items():
            # Sample ONCE, score under the canonical trio (standard/0.5/full PPR)
            # plus — if the store's own scoring isn't one of those three — that
            # config too, so the flat top-level fields always reflect the store's
            # declared scoring. Same draws across all configs (consistent).
            configs = dict(scoring_trio())
            configs[scoring_name] = scoring  # overwrites if already in the trio
            dists = simulate_player_multi(
                model, configs, n_sims=n_sims, seed=seed, correlated=correlated
            )
            # The trio block the scoring toggle reads.
            scorings_block = {label: _dist_block(dists[label]) for label in scoring_trio()}
            # Flat top-level fields = the store's declared scoring (back-compat:
            # the apps/build-scripts that predate the toggle read these).
            flat = _dist_block(dists[scoring_name])
            record = {
                "name": name,
                "mean": flat["mean"],
                "median": flat["median"],
                "std": flat["std"],
                "floor_p10": flat["floor_p10"],
                "ceiling_p90": flat["ceiling_p90"],
                "quantiles": flat["quantiles"],
                "histogram": flat["histogram"],
                "scorings": scorings_block,
                "markets_present": _stat_presence(model),
                "position_group": classify_group(model),
                "data_provenance": _data_provenance(model),
                "game": f"{e.get('away_team')} @ {e.get('home_team')}",
                "commence_time": e.get("commence_time"),
                "event_id": e["id"],
            }
            # If a player somehow appears in two games (shouldn't within a week),
            # keep the one with more markets.
            prev = players_out.get(name)
            if prev is None or len(record["markets_present"]) > len(
                prev["markets_present"]
            ):
                players_out[name] = record

        games_report.append(
            {
                "event_id": e["id"],
                "game": f"{e.get('away_team')} @ {e.get('home_team')}",
                "commence_time": e.get("commence_time"),
                "players_qualified": len(qualifying),
                "players_total": len(models),
                "qbs_excluded": excluded,
            }
        )

    generated_at = datetime.now(timezone.utc).isoformat()
    snapshot_id = _snapshot_id(generated_at)
    store = {
        "week_label": week_label,
        "snapshot_id": snapshot_id,
        "generated_at": generated_at,
        "sport": sport,
        "book": book,
        "scoring": scoring_name,
        "scorings_available": list(scoring_trio().keys()),
        "scoring_default": scoring_name if scoring_name in scoring_trio() else "half_ppr",
        "correlated": correlated,
        "n_sims": n_sims,
        "games": games_report,
        "players": players_out,
    }

    store_path = write_snapshot(store, week_label, store_dir=store_dir)

    return SlateResult(
        week_label=week_label,
        generated_at=generated_at,
        book=book,
        scoring=scoring_name,
        correlated=correlated,
        n_games=len(events),
        n_players=len(players_out),
        games=games_report,
        store_path=str(store_path),
    )


def _slug(label: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in label.lower()).strip("_")


def _snapshot_id(generated_at: str) -> str:
    """Compact UTC timestamp id, e.g. '2026-09-29T2119Z' (filename-safe)."""
    dt = datetime.fromisoformat(generated_at)
    return dt.strftime("%Y-%m-%dT%H%M%SZ")


def write_snapshot(
    store: dict, week_label: str, *, store_dir: Path | str = DEFAULT_STORE_DIR
) -> Path:
    """Write an immutable, versioned snapshot for a week and refresh pointers.

    Layout (never overwrites a prior snapshot; preserves older weeks):

        <store_dir>/<week>/<snapshot_id>.json   # this run, immutable
        <store_dir>/<week>/latest.json          # copy of the newest snapshot
        <store_dir>/<week>/index.json           # manifest of all snapshots

    Returns the path of the written snapshot.
    """
    store_dir = Path(store_dir)
    week_dir = store_dir / _slug(week_label)
    week_dir.mkdir(parents=True, exist_ok=True)

    snapshot_id = store["snapshot_id"]
    snapshot_path = week_dir / f"{snapshot_id}.json"
    payload = json.dumps(store, indent=2)
    snapshot_path.write_text(payload)

    # latest.json is a plain copy (portable; no symlinks) of the newest run.
    (week_dir / "latest.json").write_text(payload)

    _update_index(week_dir, store)
    return snapshot_path


def _update_index(week_dir: Path, store: dict) -> None:
    """Append/refresh this snapshot's entry in the per-week index manifest."""
    index_path = week_dir / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:
        index = {"week_label": store["week_label"], "snapshots": []}

    entry = {
        "snapshot_id": store["snapshot_id"],
        "generated_at": store["generated_at"],
        "book": store.get("book"),
        "scoring": store.get("scoring"),
        "correlated": store.get("correlated"),
        "n_sims": store.get("n_sims"),
        "n_players": len(store.get("players", {})),
        "n_games": len(store.get("games", [])),
        "file": f"{store['snapshot_id']}.json",
    }
    # De-dupe by snapshot_id (idempotent re-writes), then sort newest-last.
    snaps = [s for s in index["snapshots"] if s.get("snapshot_id") != entry["snapshot_id"]]
    snaps.append(entry)
    snaps.sort(key=lambda s: s.get("generated_at", ""))
    index["snapshots"] = snaps
    index["latest"] = entry["snapshot_id"]
    index_path.write_text(json.dumps(index, indent=2))
