"""Empirically calibrate the UPPER-TAIL SHAPE of the continuous yards marginals
(receiving yards, rushing yards, QB passing yards) from real nflverse game logs.

------------------------------------------------------------------------------
WHY THIS EXISTS
------------------------------------------------------------------------------
In ``src/ffmc/distributions/ladder.py`` the continuous (yards) marginal is
reconstructed from a ladder of de-vigged prop lines. The INTERIOR of the CDF is
pinned by the posted lines (monotone PCHIP). Beyond the TOP posted line we have
exactly one known point of the survival function -- S(t_top) -- and the shape of
the decay above it is *underdetermined by the odds*. A single posted line gives
a quantile, not a mean, and for a right-skewed stat the mean lives partly in
that unobserved tail, so you cannot pin the tail by "matching the implied mean"
(circular) or "matching the median" (the median is in the interior, invariant
to the tail). The decay shape must be imported from outside the single player's
ladder.

The model currently decays the tail with a Weibull-style super-exponential form

    S(x) = S(t_top) * exp( -( rate * (x - t_top) ) ** shape )          (x > t_top)

where ``rate`` is fit from the last few anchors and ``shape`` ("tail_shape") is
a HAND-PICKED CONSTANT = 1.5 for every player and both stats. shape = 1 recovers
a plain exponential tail; shape > 1 decays faster (thinner ceiling). This script
replaces that guess with a MEASURED value, and tests whether a single pooled
constant per stat suffices or whether the shape should vary with an observable
feature.

------------------------------------------------------------------------------
THE IDENTIFICATION PROBLEM AND HOW WE SOLVE IT (read this)
------------------------------------------------------------------------------
The tail shape is unobservable at prediction time (the odds don't price it). To
borrow it from history we must regress it on a feature that is computable BOTH
ways with an IDENTICAL formula:

    - BACKWARD (fitting):   from a player's real game outcomes.
    - FORWARD (applying):   from the de-vigged ladder's reconstructed interior.

We therefore split each player's distribution into:

    INTERIOR  -- below a high per-player threshold. Observable both ways
                 (game-log quantiles backward; ladder quantiles forward).
    TAIL      -- above that threshold. Observable only BACKWARD (from game logs);
                 this is the quantity we are trying to learn.

We summarise the interior with scale-free SHAPE features (so different-volume
players are comparable), measure the tail shape from the exceedances, and fit

    tail_shape = g(interior_feature)                 per stat (rec / rush)

Forward, we read the same interior feature off the ladder and predict the shape.
No tail information is ever needed forward -- that is the whole point.

ASSUMPTION (weak, and testable here): players whose observable interior is
equally spread tend to have similarly shaped tails once scale is divided out.
If the fit has little slope / low R^2, the honest conclusion is "use one pooled
constant per stat" (still strictly better than a hand-picked 1.5). If it has
real slope, a conditional shape earns its complexity.

------------------------------------------------------------------------------
HOW WE MEASURE THE TAIL SHAPE (two independent summaries, cross-checked)
------------------------------------------------------------------------------
1. WEIBULL TAIL EXPONENT (``shape_weibull``) -- fits the EXACT functional form
   the model uses, so the number drops straight into ``tail_shape``:
       S(x) = exp( -(x / scale) ** shape )  for the exceedance e = x - u >= 0,
   i.e. we fit a 2-parameter Weibull to the exceedances over the threshold u by
   maximum likelihood (scipy ``weibull_min`` with loc=0). ``shape`` here is the
   model's ``tail_shape``: =1 exponential, >1 thinner.

2. GPD SHAPE (``xi``) -- the standard extreme-value-theory peaks-over-threshold
   summary (``genpareto`` fit to the same exceedances). Reported as an
   independent cross-check / sanity read on tail heaviness: xi < 0 => bounded,
   thin tail (expected for yards); xi ~ 0 => exponential; xi > 0 => heavy. It is
   NOT what the model consumes, but if the two disagree wildly the Weibull fit
   is suspect.

Threshold u: the player's empirical ~85th percentile (``TAIL_QUANTILE``), the
analog of "top posted line". Exceedances are the games above it. We require a
minimum number of exceedances per player (``MIN_EXCEEDANCES``) or the tail fit
is too noisy to trust and the player is dropped from the regression.

------------------------------------------------------------------------------
INTERIOR FEATURES (all scale-free, all computable from a ladder too)
------------------------------------------------------------------------------
Computed from the interior (<= threshold u) only, so they never peek at the tail:
    iqr_over_median      = (p75 - p25) / median          # robust dispersion
    interior_cv          = std / mean  (interior only)    # dispersion
    interior_skew_proxy  = (p75 - median) / (median - p25)  # interior asymmetry;
                           > 1 means the observable upper-middle is already
                           stretched -- a natural forward signal for a heavy tail.

We report which feature best predicts the tail shape (OLS slope + R^2) rather
than pre-committing to one.

------------------------------------------------------------------------------
REPRODUCIBILITY
------------------------------------------------------------------------------
    pip install -e ".[analysis]"            # pandas + pyarrow (+ scipy already)
    python analysis/yards_tail_shape_check.py

Pulls public nflverse weekly player stats (no API key). Seasons 2018-2023 (2024
deliberately left out as a held-out backtest year). Receiving and rushing are
fit SEPARATELY (different physical processes; we have ample data for each). A
PNG of tail-shape vs each interior feature is written to
``data/plots/`` if matplotlib is available (optional).

Output is a printed report; THIS SCRIPT CHANGES NO MODEL CODE. The decision of
pooled-constant vs conditional (and the chosen numbers) is made by reading this
report, then wired into ``ladder.py`` in a separate step.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import genpareto, weibull_min

# --- configuration (explicit, not magic) ------------------------------------
SEASONS = [2018, 2019, 2020, 2021, 2022, 2023]
URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
       "player_stats/player_stats_{yr}.parquet")

# A player needs enough games for both a stable interior and a measurable tail.
MIN_GAMES = 40
# Threshold for the "tail": the per-player empirical quantile that stands in for
# "top posted line". 0.85 keeps ~15% of games as exceedances.
TAIL_QUANTILE = 0.85
# Minimum number of games above the threshold to attempt a tail fit.
MIN_EXCEEDANCES = 8
# Only count games where the player was actually involved in that phase, so a
# pure runner's zero receiving games don't masquerade as a receiving tail.
MIN_STAT_VALUE = 1.0  # yards > 0 to count as a "used in this phase" game
# Passing: a QB who barely threw (injury/mop-up) is not a real passing game, so
# gate passing on a higher floor to keep the sample to genuine starts.
MIN_PASSING_VALUE = 50.0

STAT_CONFIG = {
    "receiving": {
        "yards_col": "receiving_yards", "gate_col": "receiving_yards",
        "positions": ["WR", "TE", "RB", "FB"], "gate_value": MIN_STAT_VALUE,
    },
    "rushing": {
        "yards_col": "rushing_yards", "gate_col": "rushing_yards",
        "positions": ["WR", "TE", "RB", "FB"], "gate_value": MIN_STAT_VALUE,
    },
    # Passing is QB-only; QB rushing yards reuse the skill-player rushing tail
    # (position-independent), so only passing yards need a QB-specific fit.
    "passing": {
        "yards_col": "passing_yards", "gate_col": "passing_yards",
        "positions": ["QB"], "gate_value": MIN_PASSING_VALUE,
    },
}


def load() -> pd.DataFrame:
    frames = [pd.read_parquet(URL.format(yr=y)) for y in SEASONS]
    df = pd.concat(frames, ignore_index=True)
    # Keep QBs too (for passing); per-stat position filters apply in fit_players.
    df = df[df["position"].isin(["WR", "TE", "RB", "FB", "QB"])].copy()
    for c in ["receiving_yards", "rushing_yards", "receptions", "carries",
              "passing_yards"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


@dataclass
class PlayerTailFit:
    player: str
    position: str
    games: int
    median: float
    # interior shape features (scale-free)
    iqr_over_median: float
    interior_cv: float
    interior_skew_proxy: float
    # measured tail shape
    shape_weibull: float   # the model's tail_shape (Weibull exponent)
    xi_gpd: float          # EVT cross-check
    n_exceed: int


def _interior_features(interior: np.ndarray) -> tuple[float, float, float]:
    """Scale-free dispersion/asymmetry summaries of the interior (<= threshold).

    Returns (iqr_over_median, interior_cv, interior_skew_proxy). Guarded against
    degenerate (near-constant) interiors.
    """
    p25, p50, p75 = np.percentile(interior, [25, 50, 75])
    med = max(p50, 1e-9)
    iqr_over_median = (p75 - p25) / med
    mean = interior.mean()
    cv = interior.std(ddof=1) / mean if mean > 1e-9 and len(interior) > 1 else np.nan
    upper = p75 - p50
    lower = p50 - p25
    skew_proxy = upper / lower if lower > 1e-9 else np.nan
    return float(iqr_over_median), float(cv), float(skew_proxy)


def _fit_tail(exceed: np.ndarray) -> tuple[float, float]:
    """Fit tail-shape summaries to exceedances e = (x - u) >= 0.

    Returns (shape_weibull, xi_gpd).
      shape_weibull: Weibull exponent of S(e) = exp(-(e/scale)**shape); this is
                     the model's ``tail_shape`` (fit with loc fixed at 0).
      xi_gpd:        GPD shape parameter (EVT peaks-over-threshold cross-check).
    """
    e = exceed[exceed > 0]
    if len(e) < MIN_EXCEEDANCES:
        return np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            shape_w, _loc, _scale = weibull_min.fit(e, floc=0.0)
        except Exception:
            shape_w = np.nan
        try:
            xi, _loc, _scale = genpareto.fit(e, floc=0.0)
        except Exception:
            xi = np.nan
    return float(shape_w), float(xi)


def fit_players(df: pd.DataFrame, stat: str) -> pd.DataFrame:
    cfg = STAT_CONFIG[stat]
    ycol, gcol = cfg["yards_col"], cfg["gate_col"]
    positions = cfg["positions"]
    gate_value = cfg["gate_value"]
    sub = df[df["position"].isin(positions)]
    fits: list[PlayerTailFit] = []
    for (player, pos), g in sub.groupby(["player_display_name", "position"]):
        vals = g[g[gcol] >= gate_value][ycol].to_numpy(float)
        if len(vals) < MIN_GAMES:
            continue
        u = float(np.quantile(vals, TAIL_QUANTILE))
        interior = vals[vals <= u]
        exceed = vals[vals > u] - u
        if len(interior) < 5 or len(exceed) < MIN_EXCEEDANCES:
            continue
        iqr_m, cv, skew = _interior_features(interior)
        shape_w, xi = _fit_tail(exceed)
        if not np.isfinite(shape_w):
            continue
        fits.append(PlayerTailFit(
            player=player, position=pos, games=len(vals),
            median=float(np.median(vals)),
            iqr_over_median=iqr_m, interior_cv=cv, interior_skew_proxy=skew,
            shape_weibull=shape_w, xi_gpd=xi, n_exceed=len(exceed),
        ))
    return pd.DataFrame([f.__dict__ for f in fits])


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Simple OLS y = a + b x. Returns (intercept, slope, r_squared)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    xm, ym = x.mean(), y.mean()
    sxx = float(np.sum((x - xm) ** 2))
    if sxx <= 0:
        return np.nan, np.nan, np.nan
    b = float(np.sum((x - xm) * (y - ym)) / sxx)
    a = float(ym - b * xm)
    yhat = a + b * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - ym) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return a, b, r2


def _weighted_summary(res: pd.DataFrame) -> tuple[float, float, float]:
    """Pooled-constant candidate: mean/median/exceedance-weighted-mean shape."""
    s = res["shape_weibull"].to_numpy(float)
    w = res["n_exceed"].to_numpy(float)
    mean = float(np.nanmean(s))
    median = float(np.nanmedian(s))
    wmean = float(np.nansum(s * w) / np.nansum(w))
    return mean, median, wmean


def report_stat(res: pd.DataFrame, stat: str) -> None:
    print(f"\n{'='*74}\n{stat.upper()} YARDS  --  tail-shape calibration "
          f"(n={len(res)} players, >= {MIN_GAMES} games)\n{'='*74}")
    if res.empty:
        print("  no players passed the filters.")
        return

    mean, median, wmean = _weighted_summary(res)
    print("Measured Weibull tail exponent (= model tail_shape):")
    print(f"    mean {mean:.3f}   median {median:.3f}   "
          f"exceedance-weighted mean {wmean:.3f}")
    print(f"    sd {res['shape_weibull'].std():.3f}   "
          f"range [{res['shape_weibull'].min():.2f}, {res['shape_weibull'].max():.2f}]")
    _current = {"receiving": "1.15 (measured)", "rushing": "1.10 (measured)",
                "passing": "1.15 (measured)"}.get(stat, "n/a")
    print(f"    (current constant in ladder.py for {stat} = {_current})")

    xi = res["xi_gpd"].to_numpy(float)
    print(f"\nGPD shape xi (EVT cross-check; <0 bounded/thin, 0 expo, >0 heavy):")
    print(f"    mean {np.nanmean(xi):.3f}   median {np.nanmedian(xi):.3f}   "
          f"share xi<0 = {np.mean(xi < 0):.0%}")

    print("\nDoes tail shape vary with an observable interior feature?")
    print("  (OLS  shape_weibull = a + b * feature;  low R^2 => use a constant)")
    for feat, label in [
        ("iqr_over_median", "IQR / median        "),
        ("interior_cv", "interior CV         "),
        ("interior_skew_proxy", "interior skew proxy "),
    ]:
        a, b, r2 = _ols(res[feat].to_numpy(float), res["shape_weibull"].to_numpy(float))
        print(f"    {label}  slope {b:+.4f}  intercept {a:+.3f}  R^2 {r2:.3f}")

    print("\n  Most / least spread-interior players (shape at the extremes):")
    srt = res.dropna(subset=["iqr_over_median"]).sort_values("iqr_over_median")
    for _, r in pd.concat([srt.head(3), srt.tail(3)]).iterrows():
        print(f"    {r['player'][:22]:<22} {r['position']:<3} "
              f"IQR/med {r['iqr_over_median']:.2f}  "
              f"shape {r['shape_weibull']:.2f}  xi {r['xi_gpd']:+.2f}  "
              f"n_exc {int(r['n_exceed'])}")


def _maybe_plot(results: dict[str, pd.DataFrame]) -> None:
    try:
        import os

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("\n(matplotlib not installed; skipping plot)")
        return
    os.makedirs("data/plots", exist_ok=True)
    feats = ["iqr_over_median", "interior_cv", "interior_skew_proxy"]
    fig, axes = plt.subplots(len(results), len(feats),
                             figsize=(4 * len(feats), 4 * len(results)),
                             squeeze=False)
    for i, (stat, res) in enumerate(results.items()):
        for j, feat in enumerate(feats):
            ax = axes[i][j]
            ax.scatter(res[feat], res["shape_weibull"], s=12, alpha=0.5)
            a, b, r2 = _ols(res[feat].to_numpy(float),
                            res["shape_weibull"].to_numpy(float))
            if np.isfinite(b):
                xs = np.linspace(np.nanmin(res[feat]), np.nanmax(res[feat]), 50)
                ax.plot(xs, a + b * xs, "r-", lw=1.5)
            ax.axhline(1.5, color="gray", ls="--", lw=1, label="current 1.5")
            ax.set_title(f"{stat}: shape vs {feat}\nR^2={r2:.3f}", fontsize=9)
            ax.set_xlabel(feat)
            ax.set_ylabel("Weibull tail shape")
    fig.tight_layout()
    out = "data/plots/yards_tail_shape.png"
    fig.savefig(out, dpi=110)
    print(f"\nPlot written to {out}")


def main() -> None:
    print("Loading nflverse weekly player stats "
          f"{SEASONS[0]}-{SEASONS[-1]} (public, no API key) ...")
    df = load()
    results: dict[str, pd.DataFrame] = {}
    for stat in STAT_CONFIG:
        results[stat] = fit_players(df, stat)
        report_stat(results[stat], stat)
    _maybe_plot(results)
    print("\nDiagnostic only -- no model code changed. Use the printed "
          "slopes/R^2 to decide pooled-constant vs conditional per stat.")


if __name__ == "__main__":
    main()
