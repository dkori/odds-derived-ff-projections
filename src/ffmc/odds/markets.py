"""The Odds API market keys and a parser from raw payload -> PlayerModel.

Market keys (confirm against the live API with your key; centralized here so a
rename is a one-line change). The Odds API groups over/under outcomes under a
market; each outcome has a player ``description``, a ``name`` of "Over"/"Under",
a ``point`` (the line/threshold), and a ``price`` (the odds).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from ffmc.distributions.ladder import (
    ContinuousLadderDistribution,
    DiscreteLadderDistribution,
    LadderPoint,
    TAIL_SHAPE_PASSING,
    TAIL_SHAPE_RECEIVING,
    TAIL_SHAPE_RUSHING,
)
from ffmc.distributions.td_count import TouchdownCountDistribution
from ffmc.odds.devig import (
    DEFAULT_TD_HOLD,
    DevigMethod,
    devig_prob_from_pair,
)
from ffmc.sim.montecarlo import PlayerModel

# Main + alternate line markets we consume.
MARKET_RECEPTIONS = ["player_receptions", "player_receptions_alternate"]
MARKET_RECEIVING_YDS = ["player_reception_yds", "player_reception_yds_alternate"]
MARKET_RUSHING_YDS = ["player_rush_yds", "player_rush_yds_alternate"]
MARKET_ANYTIME_TD = ["player_anytime_td"]

# QB passing markets. Pass yards behave like the other continuous yards markets
# (two-sided main line + Over-only alternates). Pass TDs post a two-sided main
# line + an Over ladder, so they fit the same NB-from-ladder path as player_tds
# (and the two-sided anchor means no assumed-hold de-vig is needed).
MARKET_PASS_YDS = ["player_pass_yds", "player_pass_yds_alternate"]
MARKET_PASS_TDS = ["player_pass_tds", "player_pass_tds_alternate"]

# TD-count ladder markets (see docs/DECISIONS.md, "TD count from a ladder").
#   player_anytime_td       -> P(>=1 TD)  (Yes-only on every probed book)
#   player_tds_over         -> "Over k.5 TDs" -> P(>= k+1); one-sided (Over-only)
#   player_tds              -> two-sided Over/Under TDs (rare; de-vig if present)
#   player_rush_reception_tds (+ _alternate) -> Rush+Rec TD O/U (rare)
# The extra keys are attempted on every fetch; many books return only a subset,
# so absent keys are simply ignored by the parser.
MARKET_TDS_OVER = ["player_tds_over"]
MARKET_TDS_TWO_SIDED = [
    "player_tds",
    "player_rush_reception_tds",
    "player_rush_reception_tds_alternate",
]
MARKET_TD_LADDER = MARKET_ANYTIME_TD + MARKET_TDS_OVER + MARKET_TDS_TWO_SIDED

ALL_MARKETS = (
    MARKET_RECEPTIONS + MARKET_RECEIVING_YDS + MARKET_RUSHING_YDS + MARKET_TD_LADDER
    + MARKET_PASS_YDS + MARKET_PASS_TDS
)


def _collect_over_under(
    outcomes: list[dict],
) -> dict[str, dict[float, dict[str, float]]]:
    """Group outcomes -> {player: {point: {"Over": price, "Under": price}}}."""
    by_player: dict[str, dict[float, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for o in outcomes:
        player = o.get("description")
        point = o.get("point")
        side = o.get("name")  # "Over" / "Under" / "Yes" / "No"
        price = o.get("price")
        if player is None or price is None:
            continue
        if point is None:
            point = 0.5  # anytime-TD style Yes/No has no point
        by_player[player][float(point)][side] = float(price)
    return by_player


def _ladder_for(
    markets_payload: dict, market_keys: list[str], book_key: Optional[str]
) -> dict[str, list[LadderPoint]]:
    """Build per-player ladders of P(X > threshold) from prop lines.

    Alternate-line markets are posted **Over-only** (each alt line has an Over
    price but no paired Under). A one-sided Over price yields a raw P(X > point)
    that still carries the book's margin. Where the **main** line has both sides,
    we two-sided de-vig it to get one CLEAN anchor, then **calibrate the whole
    one-sided ladder to it** with a power transform (see
    ``calibrate_oneside_ladder``) -- this removes the measured Over-side hold,
    more so in the longshot tail (favorite-longshot bias). Without a two-sided
    main line, raw Over probs are used unchanged (no basis to de-vig).

    See docs/DECISIONS.md ("Over-only alt de-vig").
    """
    from ffmc.odds.devig import (
        american_to_implied_prob,
        calibrate_oneside_ladder,
    )

    ladders: dict[str, list[LadderPoint]] = {}

    # Group everything per player first: raw Over prob per threshold, plus any
    # clean de-vigged main-line anchor (threshold with both sides).
    raw_by_player: dict[str, dict[float, float]] = defaultdict(dict)
    main_anchor: dict[str, tuple[float, float, float]] = {}  # player -> (L, p_raw, p_true)

    for book in markets_payload.get("bookmakers", []):
        if book_key and book["key"] != book_key:
            continue
        for market in book.get("markets", []):
            if market["key"] not in market_keys:
                continue
            grouped = _collect_over_under(market.get("outcomes", []))
            for player, points in grouped.items():
                for point, sides in points.items():
                    if "Over" in sides:
                        p_raw = american_to_implied_prob(sides["Over"])
                        raw_by_player[player][point] = p_raw
                        if "Under" in sides:
                            p_true = devig_prob_from_pair(
                                sides["Over"],
                                sides["Under"],
                                DevigMethod.MULTIPLICATIVE,
                            )
                            # Prefer the main line: the two-sided threshold
                            # closest to P=0.5 is the "main" line.
                            cur = main_anchor.get(player)
                            if cur is None or abs(p_true - 0.5) < abs(cur[2] - 0.5):
                                main_anchor[player] = (point, p_raw, p_true)

    for player, raw in raw_by_player.items():
        anchor = main_anchor.get(player)
        if anchor is not None:
            _, p_raw_main, p_true_main = anchor
            calibrated = calibrate_oneside_ladder(
                raw, p_raw_main=p_raw_main, p_true_main=p_true_main
            )
            # Pin the main-line threshold to its exact de-vigged value.
            calibrated[anchor[0]] = p_true_main
        else:
            calibrated = calibrate_oneside_ladder(raw)
        pts = [LadderPoint(thr, calibrated[thr]) for thr in calibrated]
        ladders[player] = sorted(pts, key=lambda p: p.threshold)
    return ladders


def _td_anchors(
    markets_payload: dict, book_key: Optional[str], *, hold: Optional[float] = None
) -> dict[str, dict[int, float]]:
    """Extract per-player de-vigged TD-count survival anchors {k: P(X >= k)}.

    Sources (all mapped to a survival threshold ``k`` = P(X >= k)):
      * ``player_anytime_td`` "Yes"      -> k = 1
      * ``player_tds_over``   "Over j.5" -> k = ceil(j.5) = j + 1  (Over 1.5 -> 2)
      * two-sided ``player_tds`` / ``player_rush_reception_tds`` "Over j.5" -> k,
        de-vigged multiplicatively against the paired "Under" (no assumption).

    One-sided prices (Yes/Over with no paired opposite) are de-vigged by an
    assumed per-side ``hold`` (``devig_oneside_td_prob``; default 0.06). A
    two-sided anchor, when present for the same k, is preferred (no assumption).
    """
    from ffmc.odds.devig import (
        american_to_implied_prob,
        devig_oneside_td_prob,
    )

    if hold is None:
        hold = DEFAULT_TD_HOLD

    # player -> {k: (prob, is_two_sided)}; two-sided wins over one-sided for a k.
    acc: dict[str, dict[int, tuple[float, bool]]] = defaultdict(dict)

    def _put(player: str, k: int, prob: float, two_sided: bool) -> None:
        cur = acc[player].get(k)
        if cur is None or (two_sided and not cur[1]):
            acc[player][k] = (prob, two_sided)

    for book in markets_payload.get("bookmakers", []):
        if book_key and book["key"] != book_key:
            continue
        for market in book.get("markets", []):
            key = market["key"]
            if key not in MARKET_TD_LADDER:
                continue
            grouped = _collect_over_under(market.get("outcomes", []))
            for player, points in grouped.items():
                for point, sides in points.items():
                    if key in MARKET_ANYTIME_TD:
                        # Yes/No anytime-TD -> P(X >= 1).
                        if "Yes" in sides and "No" in sides:
                            p = devig_prob_from_pair(
                                sides["Yes"], sides["No"], DevigMethod.MULTIPLICATIVE
                            )
                            _put(player, 1, p, True)
                        elif "Yes" in sides:
                            p_raw = american_to_implied_prob(sides["Yes"])
                            _put(player, 1, devig_oneside_td_prob(p_raw, hold), False)
                    else:
                        # Over/Under count ladder. "Over j.5" -> P(X >= j+1).
                        k = int(point) + 1  # floor(point) + 1; point is a half-int
                        if "Over" in sides and "Under" in sides:
                            p = devig_prob_from_pair(
                                sides["Over"], sides["Under"], DevigMethod.MULTIPLICATIVE
                            )
                            _put(player, k, p, True)
                        elif "Over" in sides:
                            p_raw = american_to_implied_prob(sides["Over"])
                            _put(player, k, devig_oneside_td_prob(p_raw, hold), False)

    return {
        player: {k: prob for k, (prob, _) in ks.items()}
        for player, ks in acc.items()
    }


def _pass_td_anchors(
    markets_payload: dict, book_key: Optional[str]
) -> dict[str, dict[int, float]]:
    """Per-player passing-TD survival anchors {k: P(X >= k)} from pass-TD markets.

    ``player_pass_tds`` posts a two-sided Over/Under main line and
    ``player_pass_tds_alternate`` an Over ladder (Over 0.5/1.5/2.5[/3.5]).
    "Over j.5" -> P(X >= j+1). Unlike the anytime-TD market, pass TDs have a
    genuine two-sided main line, so we de-vig that pair multiplicatively (no
    assumed hold); Over-only alt rungs that lack a measured two-sided pair fall
    back to the raw implied prob. A two-sided anchor is preferred for any k.

    Returns the same {k: P(X>=k)} shape ``TouchdownCountDistribution`` consumes;
    with >=2 anchors it fits an NB, so the multi-pass-TD tail is market-priced.
    """
    from ffmc.odds.devig import american_to_implied_prob

    acc: dict[str, dict[int, tuple[float, bool]]] = defaultdict(dict)

    def _put(player: str, k: int, prob: float, two_sided: bool) -> None:
        cur = acc[player].get(k)
        if cur is None or (two_sided and not cur[1]):
            acc[player][k] = (prob, two_sided)

    for book in markets_payload.get("bookmakers", []):
        if book_key and book["key"] != book_key:
            continue
        for market in book.get("markets", []):
            if market["key"] not in MARKET_PASS_TDS:
                continue
            grouped = _collect_over_under(market.get("outcomes", []))
            for player, points in grouped.items():
                for point, sides in points.items():
                    k = int(point) + 1  # "Over j.5" -> P(X >= j+1)
                    if "Over" in sides and "Under" in sides:
                        p = devig_prob_from_pair(
                            sides["Over"], sides["Under"], DevigMethod.MULTIPLICATIVE
                        )
                        _put(player, k, p, True)
                    elif "Over" in sides:
                        _put(player, k, american_to_implied_prob(sides["Over"]), False)

    return {
        player: {k: prob for k, (prob, _) in ks.items()}
        for player, ks in acc.items()
    }


def build_player_models(
    markets_payload: dict,
    book_key: Optional[str] = None,
    *,
    td_hold: Optional[float] = None,
) -> dict[str, PlayerModel]:
    """Turn one event's odds payload into {player_name: PlayerModel}.

    ``td_hold`` is the assumed per-side hold used to de-vig one-sided TD prices
    (default ``DEFAULT_TD_HOLD``); pass 0 to use raw TD implied probs.
    """
    rec = _ladder_for(markets_payload, MARKET_RECEPTIONS, book_key)
    rec_yds = _ladder_for(markets_payload, MARKET_RECEIVING_YDS, book_key)
    rush_yds = _ladder_for(markets_payload, MARKET_RUSHING_YDS, book_key)
    td = _td_anchors(markets_payload, book_key, hold=td_hold)
    pass_yds = _ladder_for(markets_payload, MARKET_PASS_YDS, book_key)
    pass_td = _pass_td_anchors(markets_payload, book_key)

    players = (
        set(rec) | set(rec_yds) | set(rush_yds) | set(td)
        | set(pass_yds) | set(pass_td)
    )
    models: dict[str, PlayerModel] = {}
    for name in players:
        anchors = td.get(name)
        pass_td_anchors = pass_td.get(name)
        models[name] = PlayerModel(
            name=name,
            receptions=(
                DiscreteLadderDistribution(rec[name]) if rec.get(name) else None
            ),
            receiving_yards=(
                ContinuousLadderDistribution(
                    rec_yds[name], tail_shape=TAIL_SHAPE_RECEIVING
                )
                if rec_yds.get(name)
                else None
            ),
            rushing_yards=(
                ContinuousLadderDistribution(
                    rush_yds[name], tail_shape=TAIL_SHAPE_RUSHING
                )
                if rush_yds.get(name)
                else None
            ),
            touchdowns=(
                TouchdownCountDistribution(anchors) if anchors else None
            ),
            passing_yards=(
                ContinuousLadderDistribution(
                    pass_yds[name], tail_shape=TAIL_SHAPE_PASSING
                )
                if pass_yds.get(name)
                else None
            ),
            passing_tds=(
                TouchdownCountDistribution(pass_td_anchors)
                if pass_td_anchors
                else None
            ),
        )
    return models
