"""Position-agnostic correlation structure for the Gaussian copula.

Every correlation is a CONTINUOUS function of odds-derived usage quantities that
any player has (possibly zero). There is no position label and no "WR path" vs
"RB path": a player's profile *emerges* from their own prop lines. A pure
receiver has ~0 rushing volume, so rushing couplings vanish automatically; a
pure runner has high rush share, so they strengthen -- with everything in
between interpolated smoothly.

CALIBRATION (position-agnostic): fits below were estimated by POOLING all skill
players (WR/TE/RB/FB, 193 players, 2018-2023) with NO position filter, and
regressing each per-player rank correlation on continuous usage features. See
analysis/nflverse_correlation_check.py. Position was not used as an input at any
stage -- only as a way to ensure the pooled sample spanned the full range of
usage profiles.

Drivers (both computable from the marginals at runtime):
  rush_share = E[rushing yards] / (E[rushing yards] + E[receiving yards])
  ypc        = E[receiving yards] / E[receptions]   (deep vs short receiver)

Fitted relationships (Spearman rank correlations):
  rho(rush_yds, TD)   = 0.024 + 0.489 * rush_share      (R^2 = 0.64)
        ~0 for pure receivers, ~0.40 for pure runners: a runner's rushing yards
        and TDs travel together (goal-line volume); a receiver's don't.
  rho(rec_yds, TD)    = 0.364 - 0.265 * rush_share, with a YPC tilt for
        receiver-heavy players. Falls as a player becomes more run-dominant
        (their scoring shifts to the ground).
  rho(rush_yds,rec_yds) = 0.017 + 0.161 * rush_share    (R^2 = 0.12)
        and rho(rush_yds, receptions) tracks it (volume moves together).
  rho(receptions, receiving yards) = 0.80  (strong, flat across all players).
  rho(receptions, TD)              = ~0.28 (weak; gentle YPC tilt, clamped).

IMPORTANT: the odds do NOT *contain* these correlations (single-stat props never
price the joint). The usage features *predict* them via relationships calibrated
on history. Predictive power ranges from strong (rush_share -> rush<->TD, R^2
0.64) to modest (YPC -> rec_yds<->TD). Treat per-player outputs as
better-than-uniform estimates, not precise truths. TDs are a single pooled count
(rush + receive) because the anytime-TD market prices only one number.

Rank correlations are converted to the Gaussian copula's Pearson parameter via
the exact identity rho_pearson = 2*sin(pi*rho_spearman/6), so simulated RANK
correlations reproduce the measured values. Correlated mode is OPT-IN; the
default simulation is fully independent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Canonical stat order used everywhere in the copula matrix.
STATS = ("receptions", "receiving_yards", "rushing_yards", "touchdowns")

# --- receptions <-> receiving yards: strong, flat across all players ---------
RHO_REC_YDS = 0.80

# --- receptions <-> TDs: weak, gentle YPC tilt (fit R^2 ~ 0.04), clamped ------
REC_TD_INTERCEPT = 0.2370
REC_TD_SLOPE_YPC = 0.08932   # per (TD per 100 rec yds); see below usage
REC_TD_CLAMP = (0.24, 0.38)
REC_TD_DEFAULT = 0.28

# --- rush_yds <-> TD: continuous in rush_share (pooled, R^2 = 0.64) -----------
RUSH_TD_INTERCEPT = 0.024
RUSH_TD_SLOPE_SHARE = 0.489
RUSH_TD_CLAMP = (0.0, 0.45)

# --- rec_yds <-> TD: falls with rush_share; plus a YPC tilt for receivers -----
# Base (pooled) relationship in rush_share:
RECYDS_TD_INTERCEPT = 0.364
RECYDS_TD_SLOPE_SHARE = -0.265
# Receiver-side YPC tilt (only meaningful when rush_share is low). Applied as a
# small additive adjustment scaled by receiving involvement (1 - rush_share):
RECYDS_TD_YPC_SLOPE = 0.0217   # from the receiver-only YPC fit
RECYDS_TD_YPC_PIVOT = 12.0     # ~league-avg YPC; tilt is relative to this
RECYDS_TD_CLAMP = (0.05, 0.50)

# --- rush_yds <-> receiving yards / receptions: mild, rises with rush_share ---
RUSH_RECYDS_INTERCEPT = 0.017
RUSH_RECYDS_SLOPE_SHARE = 0.161
RUSH_RECYDS_CLAMP = (0.0, 0.30)


def spearman_to_gaussian(rho_s: float) -> float:
    """Convert a Spearman rank correlation to the Gaussian-copula Pearson rho.

    Exact identity for the Gaussian copula:
        rho_spearman = (6 / pi) * arcsin(rho_pearson / 2)
        => rho_pearson = 2 * sin(pi * rho_spearman / 6)
    """
    return 2.0 * math.sin(math.pi * rho_s / 6.0)


def _clamp(x: float, lo_hi: tuple[float, float]) -> float:
    lo, hi = lo_hi
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class PlayerFeatures:
    """Odds-computable usage features. All derived from the marginals; a player
    with no line for a stat contributes 0 expected value there.

        rush_share = E[rush yds] / (E[rush yds] + E[rec yds])   in [0, 1]
        ypc        = E[rec yds] / E[receptions]                 (None if no rec)
        td_per_100_yds = 100 * E[TD] / E[total yds]             (None if no yds)
    """

    rush_share: float = 0.0
    ypc: float | None = None
    td_per_100_yds: float | None = None

    @classmethod
    def from_expectations(
        cls,
        exp_receptions: float | None,
        exp_receiving_yards: float | None,
        exp_rushing_yards: float | None,
        exp_touchdowns: float | None,
    ) -> "PlayerFeatures":
        rec = exp_receptions or 0.0
        recy = exp_receiving_yards or 0.0
        rushy = exp_rushing_yards or 0.0
        td = exp_touchdowns or 0.0
        total_yds = rushy + recy

        rush_share = rushy / total_yds if total_yds > 0 else 0.0
        ypc = (recy / rec) if rec > 0 else None
        tdp = (100.0 * td / total_yds) if total_yds > 0 else None
        return cls(rush_share=rush_share, ypc=ypc, td_per_100_yds=tdp)


@dataclass(frozen=True)
class CorrelationSpec:
    """Resolved Spearman correlations for one player (all six pairs)."""

    rec_yds: float
    rec_td: float
    yds_td: float          # receiving yards <-> TD
    rush_recyds: float
    rush_rec: float
    rush_td: float

    @classmethod
    def from_features(cls, f: PlayerFeatures) -> "CorrelationSpec":
        """The single, position-agnostic mapping from usage -> correlations."""
        s = _clamp(f.rush_share, (0.0, 1.0))

        # rush_yds <-> TD: continuous in rush_share (vanishes for receivers).
        rush_td = _clamp(RUSH_TD_INTERCEPT + RUSH_TD_SLOPE_SHARE * s, RUSH_TD_CLAMP)

        # rec_yds <-> TD: base falls with rush_share; receiver YPC tilt scaled by
        # receiving involvement so it fades out as a player becomes run-heavy.
        recyds_td = RECYDS_TD_INTERCEPT + RECYDS_TD_SLOPE_SHARE * s
        if f.ypc is not None:
            recyds_td += (1.0 - s) * RECYDS_TD_YPC_SLOPE * (f.ypc - RECYDS_TD_YPC_PIVOT)
        yds_td = _clamp(recyds_td, RECYDS_TD_CLAMP)

        # receptions <-> TD: weak, gentle red-zone (TD-per-100-yds) tilt.
        if f.td_per_100_yds is not None:
            rec_td = _clamp(
                REC_TD_INTERCEPT + REC_TD_SLOPE_YPC * f.td_per_100_yds, REC_TD_CLAMP
            )
        else:
            rec_td = REC_TD_DEFAULT

        # rush <-> receiving: mild, rises with rush_share (0 for pure receiver).
        rush_recyds = _clamp(
            RUSH_RECYDS_INTERCEPT + RUSH_RECYDS_SLOPE_SHARE * s, RUSH_RECYDS_CLAMP
        )

        return cls(
            rec_yds=RHO_REC_YDS,
            rec_td=rec_td,
            yds_td=yds_td,
            rush_recyds=rush_recyds,
            rush_rec=rush_recyds,   # tracks rush<->rec_yds (volume moves together)
            rush_td=rush_td,
        )

    def gaussian_matrix(self) -> np.ndarray:
        """4x4 Gaussian-copula correlation matrix in STATS order.

        Order: (receptions, receiving_yards, rushing_yards, touchdowns).
        All six pairs come from the continuous usage mapping. Rank correlations
        are converted to the copula's Pearson parameter, then the matrix is
        projected to the nearest valid (PSD) correlation matrix so
        multivariate_normal never fails on a slightly non-PSD input.
        """
        s2g = spearman_to_gaussian
        # indices: 0=rec, 1=rec_yds, 2=rush_yds, 3=td
        m = np.eye(4)
        m[0, 1] = m[1, 0] = s2g(self.rec_yds)
        m[0, 3] = m[3, 0] = s2g(self.rec_td)
        m[1, 3] = m[3, 1] = s2g(self.yds_td)
        m[1, 2] = m[2, 1] = s2g(self.rush_recyds)
        m[0, 2] = m[2, 0] = s2g(self.rush_rec)
        m[2, 3] = m[3, 2] = s2g(self.rush_td)
        return _nearest_psd_correlation(m)


def _nearest_psd_correlation(m: np.ndarray) -> np.ndarray:
    """Clip negative eigenvalues and renormalize to a valid correlation matrix.

    With six nonzero off-diagonals (dual-threat players) the raw matrix can be
    slightly non-PSD, so this projection matters more than in the receiver-only
    case.
    """
    vals, vecs = np.linalg.eigh(m)
    if np.all(vals >= -1e-10):
        return m
    vals_clipped = np.clip(vals, 1e-8, None)
    psd = (vecs * vals_clipped) @ vecs.T
    d = np.sqrt(np.diag(psd))
    psd = psd / np.outer(d, d)
    np.fill_diagonal(psd, 1.0)
    return psd
