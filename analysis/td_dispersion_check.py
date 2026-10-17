"""Empirically calibrate the DISPERSION of the touchdown-count distribution from
real nflverse game logs, to replace the Poisson fallback used when only the
anytime-TD market is posted.

------------------------------------------------------------------------------
WHY THIS EXISTS
------------------------------------------------------------------------------
``src/ffmc/distributions/td_count.py`` fits a negative binomial (NB) to the TD
ladder. With TWO OR MORE anchors (anytime + Over 1.5 / Over 2.5) the market
prices the multi-TD tail directly and the NB is well-identified -- that path is
fine and this script does NOT touch it.

The problem is the SINGLE-ANCHOR fallback: when a book posts only
``player_anytime_td`` = P(X >= 1) and no Over ladder, the code falls back to

    lambda = -ln(1 - P(X>=1)),   X ~ Poisson(lambda)      (alpha ~ 0)

Poisson has variance = mean (no free dispersion). Real per-game TD counts are
OVERDISPERSED (mostly 0, occasional multi-TD games), so for a fixed P(X>=1) the
Poisson shape places TOO MUCH mass on 2+ TDs -- it systematically overstates
multi-TD games (observed in the projections). We cannot see the dispersion from
one anchor; we must borrow it from history.

------------------------------------------------------------------------------
THE IDEA (same methodology as the yards tail-shape calibration)
------------------------------------------------------------------------------
The forward-observable quantity from the anytime market is the MEAN TD RATE:

    mu = -ln(1 - P(X>=1))        # same lambda the Poisson fallback computes

The unobservable-forward quantity is the DISPERSION alpha (variance = mu +
alpha * mu**2; alpha = 0 is Poisson). We measure alpha from history and regress
it on the one observable we will have forward:

    alpha = h(mu)                # dispersion as a function of mean rate

Backward (fitting): per player, from real per-game TD counts, estimate the mean
and the dispersion. Forward (applying): from the anytime market, solve mu, then
set alpha = h(mu) instead of ~0, giving a history-calibrated NB fallback that
keeps P(X>=1) exact but pulls the multi-TD mass down to realistic levels.

TDs are pooled (rushing + receiving) into one count because the anytime-TD
market prices a single number -- so we calibrate on total TDs per game, exactly
the quantity the model samples.

------------------------------------------------------------------------------
HOW DISPERSION IS ESTIMATED (and why we are careful)
------------------------------------------------------------------------------
NB dispersion is only identifiable when sample variance > mean, and per-player
season samples are small with tiny integer counts, so a per-player MLE is noisy
and often rails to the Poisson boundary. We therefore use robust, transparent
estimators and cross-check:

1. METHOD-OF-MOMENTS per player (reported, but noisy):
       alpha_mom = (var - mean) / mean**2      (clipped at 0; 0 => Poisson-like)
   Regressed on mean rate to see the trend h(mu).

2. POOLED DISPERSION (the number we actually want): rather than averaging noisy
   per-player alphas, we find the SINGLE alpha that best reproduces the real
   AGGREGATE multi-TD rates. Concretely, across all player-games we compare, at
   each player's own mean rate, the empirical P(X>=2 | X>=1) against the NB
   prediction, and choose the alpha minimizing squared error. This directly
   targets "don't overstate multi-TD games", which is the actual failure.
   We also report, for reference, what Poisson (alpha=0) predicts for the same
   quantity so the improvement is explicit.

The deliverable for wiring-in is: (a) a recommended pooled constant alpha for
the fallback, and (b) whether a mild alpha = h(mu) slope is worth it (low R^2 =>
use the constant).

------------------------------------------------------------------------------
REPRODUCIBILITY
------------------------------------------------------------------------------
    pip install -e ".[analysis]"
    python analysis/td_dispersion_check.py

Pulls public nflverse weekly player stats (no API key), seasons 2018-2023 (2024
held out). Diagnostic only; changes no model code.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import nbinom

SEASONS = [2018, 2019, 2020, 2021, 2022, 2023]
URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
       "player_stats/player_stats_{yr}.parquet")

# A player needs enough games to estimate a per-game mean/variance at all.
MIN_GAMES = 24
# Scorers only: a player who essentially never scores gives mu ~ 0 and no
# dispersion signal. Require a minimum total TDs to enter the dispersion fit.
MIN_TOTAL_TDS = 6


def load() -> pd.DataFrame:
    frames = [pd.read_parquet(URL.format(yr=y)) for y in SEASONS]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["position"].isin(["WR", "TE", "RB", "FB"])].copy()
    for c in ["rushing_tds", "receiving_tds"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    # Pooled TD count per game = rushing + receiving (the market prices one number).
    df["total_tds"] = df["rushing_tds"] + df["receiving_tds"]
    return df


@dataclass
class PlayerTDFit:
    player: str
    position: str
    games: int
    mean_rate: float          # empirical per-game mean TD (mu)
    var_rate: float
    alpha_mom: float          # method-of-moments dispersion (clipped >= 0)
    p_ge1: float              # empirical P(X>=1)
    p_ge2: float              # empirical P(X>=2)
    p_ge2_given_ge1: float    # empirical P(X>=2 | X>=1)


def _nbinom_params(mu: float, alpha: float) -> tuple[float, float]:
    mu = max(mu, 1e-6)
    alpha = max(alpha, 1e-9)
    n = 1.0 / alpha
    p = n / (n + mu)
    return n, p


def _nb_p_ge2_given_ge1(mu: float, alpha: float) -> float:
    """Model P(X>=2 | X>=1) for NB(mu, alpha). alpha->0 is the Poisson limit."""
    if alpha <= 1e-9:
        # Poisson limit
        p0 = np.exp(-mu)
        p1 = mu * np.exp(-mu)
    else:
        n, p = _nbinom_params(mu, alpha)
        p0 = nbinom.pmf(0, n, p)
        p1 = nbinom.pmf(1, n, p)
    denom = 1.0 - p0
    if denom <= 1e-12:
        return np.nan
    return float((denom - p1) / denom)


def fit_players(df: pd.DataFrame) -> pd.DataFrame:
    fits: list[PlayerTDFit] = []
    for (player, pos), g in df.groupby(["player_display_name", "position"]):
        counts = g["total_tds"].to_numpy(float)
        if len(counts) < MIN_GAMES or counts.sum() < MIN_TOTAL_TDS:
            continue
        mean = counts.mean()
        var = counts.var(ddof=1)
        if mean <= 1e-6:
            continue
        alpha_mom = max((var - mean) / (mean ** 2), 0.0)
        fits.append(PlayerTDFit(
            player=player, position=pos, games=len(counts),
            mean_rate=float(mean), var_rate=float(var),
            alpha_mom=float(alpha_mom),
            p_ge1=float(np.mean(counts >= 1)),
            p_ge2=float(np.mean(counts >= 2)),
            p_ge2_given_ge1=float(np.mean(counts >= 2) / max(np.mean(counts >= 1), 1e-9)),
        ))
    return pd.DataFrame([f.__dict__ for f in fits])


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
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


def _best_pooled_alpha(res: pd.DataFrame) -> tuple[float, float, float]:
    """Find the single alpha minimizing squared error between empirical and NB
    P(X>=2 | X>=1), evaluated at each player's own mean rate.

    Returns (alpha_star, rmse_at_alpha_star, rmse_poisson).
    """
    mu = res["mean_rate"].to_numpy(float)
    emp = res["p_ge2_given_ge1"].to_numpy(float)
    valid = np.isfinite(mu) & np.isfinite(emp)
    mu, emp = mu[valid], emp[valid]

    def rmse(alpha: float) -> float:
        pred = np.array([_nb_p_ge2_given_ge1(m, alpha) for m in mu])
        return float(np.sqrt(np.nanmean((pred - emp) ** 2)))

    res_opt = minimize_scalar(rmse, bounds=(1e-6, 5.0), method="bounded")
    alpha_star = float(res_opt.x)
    return alpha_star, rmse(alpha_star), rmse(0.0)


def report(res: pd.DataFrame) -> None:
    print(f"\n{'='*74}\nTOUCHDOWN COUNT  --  dispersion calibration "
          f"(n={len(res)} players, >= {MIN_GAMES} games, >= {MIN_TOTAL_TDS} TDs)"
          f"\n{'='*74}")
    if res.empty:
        print("  no players passed the filters.")
        return

    print("Per-game pooled TD rate (mu) and method-of-moments dispersion (alpha):")
    print(f"    mu     mean {res['mean_rate'].mean():.3f}   "
          f"range [{res['mean_rate'].min():.2f}, {res['mean_rate'].max():.2f}]")
    print(f"    alpha  mean {res['alpha_mom'].mean():.3f}   "
          f"median {res['alpha_mom'].median():.3f}   "
          f"share overdispersed (alpha>0) = {np.mean(res['alpha_mom'] > 1e-6):.0%}")
    print("    (Poisson fallback currently assumes alpha ~ 0)")

    a, b, r2 = _ols(res["mean_rate"].to_numpy(float),
                    res["alpha_mom"].to_numpy(float))
    print("\nDoes dispersion vary with mean rate?  (OLS alpha = a + b*mu)")
    print(f"    slope {b:+.4f}  intercept {a:+.3f}  R^2 {r2:.3f}   "
          f"(low R^2 => use a pooled constant alpha)")

    alpha_star, rmse_star, rmse_pois = _best_pooled_alpha(res)
    print("\nPooled alpha that best reproduces real P(X>=2 | X>=1):")
    print(f"    alpha* = {alpha_star:.3f}")
    print(f"    RMSE(P>=2|>=1)  NB(alpha*) = {rmse_star:.4f}   "
          f"vs Poisson(alpha=0) = {rmse_pois:.4f}")
    improve = (rmse_pois - rmse_star) / rmse_pois if rmse_pois > 0 else 0.0
    print(f"    => NB reduces multi-TD-rate error by {improve:.0%} vs Poisson")

    print("\n  Illustration at representative mean rates "
          "(empirical pooled vs model P(X>=2|X>=1)):")
    print(f"    {'mu':>5}  {'empir':>7}  {'Poisson':>8}  {'NB(a*)':>7}")
    for mu_show in [0.3, 0.5, 0.7, 0.9]:
        near = res[(res["mean_rate"] - mu_show).abs() < 0.1]
        emp = near["p_ge2_given_ge1"].mean() if len(near) else np.nan
        pois = _nb_p_ge2_given_ge1(mu_show, 0.0)
        nb = _nb_p_ge2_given_ge1(mu_show, alpha_star)
        emp_s = f"{emp:.3f}" if np.isfinite(emp) else "   n/a"
        print(f"    {mu_show:>5.1f}  {emp_s:>7}  {pois:>8.3f}  {nb:>7.3f}  "
              f"(n={len(near)})")


def _maybe_plot(res: pd.DataFrame, alpha_star: float) -> None:
    try:
        import os

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("\n(matplotlib not installed; skipping plot)")
        return
    os.makedirs("data/plots", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(res["mean_rate"], res["alpha_mom"], s=14, alpha=0.5)
    a, b, r2 = _ols(res["mean_rate"].to_numpy(float),
                    res["alpha_mom"].to_numpy(float))
    xs = np.linspace(res["mean_rate"].min(), res["mean_rate"].max(), 50)
    if np.isfinite(b):
        ax1.plot(xs, a + b * xs, "r-", lw=1.5, label=f"OLS R^2={r2:.3f}")
    ax1.axhline(0, color="gray", ls="--", lw=1, label="Poisson (alpha=0)")
    ax1.set_xlabel("mean TD rate mu")
    ax1.set_ylabel("method-of-moments alpha")
    ax1.set_title("Per-player dispersion vs mean rate")
    ax1.legend()

    mus = np.linspace(0.1, 1.2, 100)
    ax2.plot(mus, [_nb_p_ge2_given_ge1(m, 0.0) for m in mus], "b-",
             label="Poisson")
    ax2.plot(mus, [_nb_p_ge2_given_ge1(m, alpha_star) for m in mus], "g-",
             label=f"NB(alpha*={alpha_star:.2f})")
    ax2.scatter(res["mean_rate"], res["p_ge2_given_ge1"], s=12, alpha=0.4,
                color="k", label="empirical")
    ax2.set_xlabel("mean TD rate mu")
    ax2.set_ylabel("P(X>=2 | X>=1)")
    ax2.set_title("Multi-TD rate: Poisson overstates vs NB")
    ax2.legend()
    fig.tight_layout()
    out = "data/plots/td_dispersion.png"
    fig.savefig(out, dpi=110)
    print(f"\nPlot written to {out}")


def main() -> None:
    print("Loading nflverse weekly player stats "
          f"{SEASONS[0]}-{SEASONS[-1]} (public, no API key) ...")
    df = load()
    res = fit_players(df)
    report(res)
    if not res.empty:
        alpha_star, _, _ = _best_pooled_alpha(res)
        _maybe_plot(res, alpha_star)
    print("\nDiagnostic only -- no model code changed. Use alpha* (and the "
          "mu-slope R^2) to replace the Poisson single-anchor fallback.")


if __name__ == "__main__":
    main()
