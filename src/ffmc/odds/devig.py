"""Odds conversion and de-vigging utilities.

A sportsbook's posted odds encode an *implied probability* for each outcome, but
those implied probabilities include the book's margin (the "vig" or overround),
so the two sides of an over/under do NOT sum to 1. Before we can treat the
numbers as a real probability distribution we have to remove that margin.

This module provides:
  - american_to_implied_prob: American odds -> raw implied probability
  - devig_pair:              remove vig from a single over/under pair
  - DevigMethod:             pluggable strategy (multiplicative now, room for Shin)

Everything here is pure and unit-testable; no network, no numpy required.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


def american_to_implied_prob(odds: float) -> float:
    """Convert American odds to a raw (vig-inclusive) implied probability.

    Positive odds (underdog):  P = 100 / (odds + 100)
    Negative odds (favorite):  P = -odds / (-odds + 100)

    >>> round(american_to_implied_prob(-110), 4)
    0.5238
    >>> round(american_to_implied_prob(+120), 4)
    0.4545
    """
    if odds == 0:
        raise ValueError("American odds of 0 are not valid")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)


def decimal_to_implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to a raw implied probability (1 / decimal)."""
    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be > 1.0")
    return 1.0 / decimal_odds


class DevigMethod(str, Enum):
    """Strategy for removing the bookmaker margin from a set of outcomes."""

    MULTIPLICATIVE = "multiplicative"
    # Future: SHIN = "shin", POWER = "power"


@dataclass(frozen=True)
class DevigResult:
    """De-vigged probabilities for the two sides of a threshold.

    ``over`` is P(X > line) and ``under`` is P(X <= line); they sum to 1.
    ``overround`` is the pre-devig sum (a diagnostic; e.g. 1.045 = 4.5% vig).
    """

    over: float
    under: float
    overround: float


def devig_pair(
    over_odds: float,
    under_odds: float,
    method: DevigMethod = DevigMethod.MULTIPLICATIVE,
) -> DevigResult:
    """Remove vig from a single over/under pair given as American odds.

    The two raw implied probabilities sum to ``overround`` (> 1). Multiplicative
    de-vigging simply rescales each so they sum to 1, preserving their ratio.

    >>> r = devig_pair(-110, -110)
    >>> round(r.over, 3), round(r.under, 3)
    (0.5, 0.5)
    >>> round(r.overround, 3)
    1.048
    """
    p_over_raw = american_to_implied_prob(over_odds)
    p_under_raw = american_to_implied_prob(under_odds)
    overround = p_over_raw + p_under_raw

    if method is DevigMethod.MULTIPLICATIVE:
        p_over = p_over_raw / overround
        p_under = p_under_raw / overround
    else:  # pragma: no cover - reserved for future methods
        raise NotImplementedError(f"De-vig method {method} not implemented yet")

    return DevigResult(over=p_over, under=p_under, overround=overround)


def devig_prob_from_pair(
    over_odds: float,
    under_odds: float,
    method: DevigMethod = DevigMethod.MULTIPLICATIVE,
) -> float:
    """Convenience: return just the de-vigged P(X > line) for an over/under pair."""
    return devig_pair(over_odds, under_odds, method).over


# --- one-sided (Over-only) alternate-ladder calibration -------------------
#
# Alternate prop lines are posted Over-only, so each anchor's implied prob still
# carries the book's margin and can't be de-vigged against a paired Under. But
# the MAIN line usually has both sides, giving one CLEAN, de-vigged anchor. We
# use it to calibrate the whole one-sided ladder:
#
#   power form:    p_calibrated = p_raw ** pi,   pi chosen so
#                  p_raw(main) ** pi == p_true(main)
#
# A power transform corrects longshots (deep-tail lines) more than favorites,
# which is the right direction for favorite-longshot bias -- and it's pinned by
# the single measured de-vig at the main line. If no two-sided main line is
# available we fall back to returning the raw probs unchanged. See
# docs/DECISIONS.md ("Over-only alt de-vig").


def oneside_power_exponent(p_raw_main: float, p_true_main: float) -> float:
    """Exponent pi such that ``p_raw_main ** pi == p_true_main``.

    Both args are probabilities in (0, 1). Returns 1.0 (identity) if inputs are
    degenerate (<=0, >=1, or a raw prob of ~1 that can't be exponentiated).
    """
    import math

    if not (0.0 < p_raw_main < 1.0) or not (0.0 < p_true_main < 1.0):
        return 1.0
    denom = math.log(p_raw_main)
    if denom == 0.0:
        return 1.0
    return math.log(p_true_main) / denom


def calibrate_oneside_ladder(
    raw_probs: dict[float, float],
    *,
    p_raw_main: float | None = None,
    p_true_main: float | None = None,
) -> dict[float, float]:
    """Calibrate a one-sided (Over-only) ladder of P(X > line) values.

    ``raw_probs`` maps line -> raw implied P(>line) (vig-inclusive). If a clean
    de-vigged main-line anchor is supplied (both ``p_raw_main`` and
    ``p_true_main``), a power transform pinned at that anchor is applied to every
    line -- this deflates the ladder by the *measured* Over-side hold, more so in
    the longshot tail. Without the anchor, the raw probs are returned unchanged
    (no basis to de-vig a purely one-sided market).

    Returned probs are clamped to (0, 1) and NOT re-sorted; the caller builds the
    ladder. The transform is monotone, so ordering is preserved.
    """
    if p_raw_main is None or p_true_main is None:
        return dict(raw_probs)
    pi = oneside_power_exponent(p_raw_main, p_true_main)
    out: dict[float, float] = {}
    for line, p in raw_probs.items():
        if 0.0 < p < 1.0:
            out[line] = min(max(p**pi, 1e-9), 1.0 - 1e-9)
        else:
            out[line] = min(max(p, 1e-9), 1.0 - 1e-9)
    return out


# --- touchdown-count de-vig (one-sided, assumed hold) ---------------------
#
# TD-count anchors come from two market shapes:
#   * player_anytime_td (Yes-only on every probed book) -> P(>=1 TD)
#   * player_tds_over  "Over k.5" (Over-only)           -> P(>= k+1)
# Neither exposes a paired opposite side we can multiplicatively de-vig
# against, and (unlike the yards ladder) there is no two-sided *main* TD line to
# pin a power transform to. So we make ONE explicit assumption: each posted TD
# probability carries a modest per-side hold, and we deflate the raw implied
# prob by that hold. See docs/DECISIONS.md ("TD count from a ladder").
#
# A per-side hold h means the raw Yes/Over implied prob is inflated by roughly
# (1 + h): p_raw ~= p_true * (1 + h). We invert that:  p_true = p_raw / (1 + h).
# Default h = 0.06 (a typical two-way TD Yes/No hold split per side); it is a
# single knob, not buried per-line, and is configurable at the call site.

DEFAULT_TD_HOLD = 0.06


def devig_oneside_td_prob(p_raw: float, hold: float = DEFAULT_TD_HOLD) -> float:
    """Deflate a one-sided TD implied prob by an assumed per-side ``hold``.

    ``p_raw`` is the vig-inclusive implied P(>= k) from a Yes/Over TD price.
    Returns ``p_raw / (1 + hold)``, clamped to (0, 1). With the default 6% hold a
    raw 0.767 -> ~0.723 and a raw 0.385 -> ~0.363 (matches the WIP sanity target
    for Gibbs). ``hold = 0`` is the identity (use the raw prob unchanged).
    """
    if hold < 0:
        raise ValueError("hold must be non-negative")
    p = p_raw / (1.0 + hold)
    return min(max(p, 1e-9), 1.0 - 1e-9)
