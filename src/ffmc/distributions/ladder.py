"""Reconstruct a sampleable distribution from a ladder of prop lines.

A sportsbook posts, for a given stat, one or more over/under lines. After
de-vigging, each line ``t`` gives us a point on the *survival function*:

    S(t) = P(X > t)          equivalently   F(t) = P(X <= t) = 1 - S(t)

With several alternate lines we get several points on F. To draw random values
we need a full CDF, so we:

  1. sort the (threshold, F) points,
  2. enforce monotonicity (F must be non-decreasing),
  3. interpolate between known points (monotone PCHIP),
  4. extrapolate the tails beyond the posted range (parametric),
  5. inverse-transform sample: draw u ~ U(0,1), return F^{-1}(u).

Two flavors:
  - ContinuousLadderDistribution: for yards. Monotone PCHIP on the CDF interior,
    exponential survival tails beyond the posted range.
  - DiscreteLadderDistribution:   for receptions. Fits a negative-binomial to the
    posted CDF anchors, giving a coherent PMF between integers and a proper
    decaying upper tail (no uniform fill, no clamped residual mass).

Touchdowns are handled separately (see poisson.py) because books post only an
anytime-TD probability, not a ladder.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from scipy.stats import nbinom


@dataclass
class LadderPoint:
    """One posted line: threshold value and de-vigged P(X > threshold)."""

    threshold: float
    prob_over: float  # P(X > threshold), de-vigged, in (0, 1)


def _to_sorted_cdf_points(points: list[LadderPoint]) -> tuple[np.ndarray, np.ndarray]:
    """Return (thresholds, F) arrays sorted by threshold, F = 1 - P(over).

    Enforces that F is strictly non-decreasing (books' alternate lines can be
    mildly inconsistent after de-vigging; we clamp rather than fail).
    """
    if not points:
        raise ValueError("Need at least one ladder point to build a distribution")

    pts = sorted(points, key=lambda p: p.threshold)
    thresholds = np.array([p.threshold for p in pts], dtype=float)
    f = np.array([1.0 - p.prob_over for p in pts], dtype=float)

    if np.any(np.diff(thresholds) <= 0):
        raise ValueError("Ladder thresholds must be distinct")

    # Enforce monotone non-decreasing CDF (repair small inversions).
    f = np.maximum.accumulate(f)
    f = np.clip(f, 1e-9, 1.0 - 1e-9)
    return thresholds, f


# --- measured upper-tail shapes (see analysis/yards_tail_shape_check.py) ------
# The Weibull tail exponent used in ``_upper_survival`` was originally a
# hand-picked 1.5 for every player and stat. ``analysis/yards_tail_shape_check.py``
# fit the exponent to real nflverse exceedances (games above each player's ~85th
# percentile), 2018-2023, SEPARATELY for receiving and rushing:
#
#     receiving yards : median 1.12  (mean 1.17, sd 0.28, n=153)
#     rushing yards   : median 1.05  (mean 1.11, sd 0.32, n=49)
#
# The GPD cross-check gave xi < 0 for ~75-80% of players (bounded/thin tails),
# so a modest exponent is safe and the old 1.5 was too steep (it over-thinned the
# ceiling). The exponent did NOT vary with any observable interior-dispersion
# feature (all regressions R^2 <= 0.04), so a POOLED CONSTANT PER STAT is the
# honest choice -- a conditional shape would be fitting noise. See
# docs/DECISIONS.md ("Yards upper tail -- measured tail shape").
TAIL_SHAPE_RECEIVING: float = 1.15
TAIL_SHAPE_RUSHING: float = 1.10
# Passing yards (QB). Measured from nflverse QB logs the same way (n=28 QBs):
# Weibull exponent median 1.14 / mean 1.21, GPD ξ mean −0.83 with 96% of QBs
# bounded (ξ<0) — the LEAST heavy-tailed of the three yards stats (QB passing is
# the least skewed: high floor, no catch-dependent zeros). No interior-dispersion
# dependence (R²≈0.00), so a pooled constant again. Set to 1.15 (= measured
# median ~1.14, and matches receiving for consistency). Because pass-yds ladders
# are deep (9–11 alt anchors), the tail is low-stakes here: 1.15 vs 1.14 moves
# full QB projections by 0.00; even 1.0 vs 1.5 moves q99 by only ~8–16 yds.
TAIL_SHAPE_PASSING: float = 1.15


@dataclass
class ContinuousLadderDistribution:
    """Continuous distribution reconstructed from a ladder (for yards).

    Interior: monotone PCHIP interpolation of the CDF.
    Tails: exponential decay of the survival/CDF beyond the posted range, with a
    rate inferred from the outermost posted segment. Values are clamped at 0
    (a player can't gain negative yards in this model).
    """

    points: list[LadderPoint]
    floor: float = 0.0  # hard lower bound on the stat (yards >= 0)
    # --- upper-tail modeling knobs (see _upper_survival) ------------------
    # Beyond the top posted line we decay the survival function faster than a
    # plain exponential. We (a) fit the base rate by log-linear regression over
    # the last ``tail_fit_anchors`` points (robust to a single noisy top line)
    # and (b) raise the decay to a Weibull-style power ``tail_shape`` so far-tail
    # mass drops off super-exponentially. tail_shape=1.0 recovers a plain
    # exponential tail.
    #
    # ``tail_shape`` is MEASURED from history, not guessed: see the module-level
    # ``TAIL_SHAPE_RECEIVING`` (1.15) / ``TAIL_SHAPE_RUSHING`` (1.10) constants
    # and ``analysis/yards_tail_shape_check.py``. Callers (``odds/markets.py``)
    # pass the stat-appropriate constant explicitly. The default below is the
    # receiving value as a safe fallback for ad-hoc construction; it is close to
    # a plain exponential, consistent with the thin (xi<0) tails the data shows.
    # (The earlier 1.5 was a hand-picked over-correction; real exponents cluster
    # near 1.1.) See docs/DECISIONS.md ("Yards upper tail").
    tail_shape: float = TAIL_SHAPE_RECEIVING
    tail_fit_anchors: int = 4
    _thresholds: np.ndarray = field(init=False, repr=False)
    _f: np.ndarray = field(init=False, repr=False)
    _cdf: PchipInterpolator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._thresholds, self._f = _to_sorted_cdf_points(self.points)
        if len(self._thresholds) == 1:
            # Single line: we can't interpolate a shape. A book that posted only
            # the main line (no alternates, e.g. early in the week) gives us just
            # ONE point P(X > L). We can't infer the true shape from one anchor,
            # so we fall back to a conservative parametric form (see cdf /
            # _single_anchor_rate) whose scale is pinned to (L, F) and whose tail
            # uses the same faster-than-exponential ``tail_shape``. This is a
            # genuine assumption -- callers should treat single-anchor stats as
            # LOW-CONFIDENCE (see ``n_anchors`` / ``is_single_anchor``); the slate
            # flags them in data_provenance.
            self._cdf = None  # type: ignore[assignment]
        else:
            self._cdf = PchipInterpolator(
                self._thresholds, self._f, extrapolate=False
            )

    @property
    def n_anchors(self) -> int:
        """Number of posted ladder anchors (1 = main line only, low confidence)."""
        return len(self._thresholds)

    @property
    def is_single_anchor(self) -> bool:
        """True when only one line was posted -> shape is assumed, not measured."""
        return len(self._thresholds) == 1

    def _single_anchor_rate(self) -> float:
        """Decay rate for the single-anchor fallback, pinned to (L, F).

        We model the stat as a distribution with survival
        ``S(x) = exp(-(rate * x) ** tail_shape)`` and choose ``rate`` so the one
        posted anchor is honoured: ``S(L) = 1 - F``. Solving,
        ``rate = ((-ln(1-F)) ** (1/shape)) / L``. Using ``tail_shape`` (>1) here
        makes the tail decay faster than the old ``rate = 1/L`` exponential,
        which produced absurd ceilings from one point (e.g. a 47.5-yd line
        implied a ~315-yd 99th percentile). Tied to the actual line, not a bare
        reciprocal.
        """
        t0, f0 = self._thresholds[0], self._f[0]
        surv = max(1.0 - f0, 1e-9)
        shape = max(self.tail_shape, 1e-6)
        span = max(t0 - self.floor, 1e-6)
        return float((-np.log(surv)) ** (1.0 / shape) / span)


    # --- tail models -----------------------------------------------------
    def _lower_tail_rate(self) -> float:
        """Rate for exponential lower tail below the smallest threshold."""
        t0, f0 = self._thresholds[0], self._f[0]
        span = max(t0 - self.floor, 1e-6)
        # F(floor) -> 0, F(t0) = f0  =>  treat as exponential rise.
        return -np.log(max(1.0 - f0, 1e-9)) / span if f0 < 1 else 1.0 / span

    def _upper_tail_rate(self) -> float:
        """Base exponential decay rate for the upper tail.

        Fit by log-linear least squares of log(survival) vs threshold over the
        last ``tail_fit_anchors`` posted points (min 2). Using several anchors
        stabilizes the rate estimate against a single noisy top line; the
        *faster-than-exponential* decay itself comes from ``tail_shape`` in
        ``_upper_survival``, not from this rate.
        """
        n = len(self._thresholds)
        if n < 2:
            return 1.0 / max(self._thresholds[-1], 1.0)
        k = min(self.tail_fit_anchors, n)
        t = self._thresholds[-k:]
        s = 1.0 - self._f[-k:]  # survival at those thresholds
        # Guard against non-positive survival (clip already keeps f < 1).
        s = np.clip(s, 1e-12, 1.0)
        log_s = np.log(s)
        # Slope of log(survival) vs t is -rate; least-squares fit.
        t_mean = t.mean()
        denom = float(np.sum((t - t_mean) ** 2))
        if denom <= 0:
            return 1.0 / max(self._thresholds[-1], 1.0)
        slope = float(np.sum((t - t_mean) * (log_s - log_s.mean())) / denom)
        rate = -slope
        # A degenerate/positive slope (survival rising) is nonsensical; fall back
        # to the last-segment two-point estimate.
        if not np.isfinite(rate) or rate <= 0:
            tn, s_n = self._thresholds[-1], 1.0 - self._f[-1]
            t_prev, s_prev = self._thresholds[-2], 1.0 - self._f[-2]
            if s_prev > s_n > 0:
                return (np.log(s_prev) - np.log(s_n)) / (tn - t_prev)
            return 1.0 / max(tn, 1.0)
        return rate

    def _upper_survival(self, x: float) -> float:
        """Survival S(x) = P(X > x) for x beyond the top posted threshold.

        Weibull-style super-exponential decay pinned at the last anchor:
            S(x) = S_n * exp( -(rate * (x - tn)) ** tail_shape )
        With ``tail_shape > 1`` this drops faster than the plain exponential
        ``exp(-rate*(x-tn))`` (recovered at tail_shape == 1), pulling the far
        quantiles (q99+) down to physically plausible values.
        """
        tn = self._thresholds[-1]
        s_n = 1.0 - self._f[-1]
        rate = self._upper_tail_rate()
        z = max(rate * (x - tn), 0.0)
        shape = max(self.tail_shape, 1e-6)
        return float(s_n * np.exp(-(z ** shape)))


    def cdf(self, x: float) -> float:
        x = max(x, self.floor)
        if self._cdf is None:
            # Single-anchor fallback: S(x) = exp(-(rate*(x-floor))**shape), pinned
            # so S(L) = 1 - F at the one posted line (see _single_anchor_rate).
            # The tail_shape (>1) keeps this from fabricating an absurd ceiling
            # the way the old rate=1/L exponential did.
            rate = self._single_anchor_rate()
            shape = max(self.tail_shape, 1e-6)
            z = max(rate * (x - self.floor), 0.0)
            return float(1.0 - np.exp(-(z ** shape)))
        t0, tn = self._thresholds[0], self._thresholds[-1]
        if x <= t0:
            rate = self._lower_tail_rate()
            return float(self._f[0] * (1.0 - np.exp(-rate * (x - self.floor))) /
                         (1.0 - np.exp(-rate * (t0 - self.floor)) + 1e-12))
        if x >= tn:
            return float(1.0 - self._upper_survival(x))
        return float(self._cdf(x))

    def _upper_grid_hi(self) -> float:
        """Upper bound for dense grids: far enough out that S(hi) is negligible.

        Solve S * exp(-(rate*(hi-anchor))**shape) ~ 1e-6 for hi, so the grid spans
        essentially the whole tail without wasting nodes on ~0 mass.
        """
        shape = max(self.tail_shape, 1e-6)
        if self._cdf is None:
            # Single-anchor: S(x) = exp(-(rate*(x-floor))**shape) from floor.
            rate = max(self._single_anchor_rate(), 1e-9)
            reach = (np.log(1e6)) ** (1.0 / shape) / rate
            return float(self.floor + reach)
        tn = self._thresholds[-1]
        rate = max(self._upper_tail_rate(), 1e-9)
        # (rate*(hi-tn))**shape = -ln(1e-6) => hi = tn + (ln(1e6))**(1/shape)/rate
        reach = (np.log(1e6)) ** (1.0 / shape) / rate
        return float(tn + reach)

    def expected_value(self) -> float:
        """E[X] via the survival-function integral E[X] = integral_0^inf S(x) dx
        (valid because X >= floor >= 0), evaluated on a dense grid."""
        hi = self._upper_grid_hi()
        grid = np.linspace(self.floor, hi, 4000)
        surv = 1.0 - np.array([self.cdf(x) for x in grid])
        trap = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2.0 renamed it
        return float(self.floor + trap(surv, grid))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Inverse-transform sampling on a dense CDF grid."""
        u = rng.random(n)
        return self.invert(u)

    def invert(self, u: np.ndarray) -> np.ndarray:
        """Inverse CDF: map supplied uniforms in [0,1] to stat values.

        This is the entry point a copula uses to drive the marginal with its own
        (correlated) uniforms instead of fresh independent ones.
        """
        return self._invert(np.asarray(u, dtype=float))

    def _invert(self, u: np.ndarray) -> np.ndarray:
        # Build a dense grid spanning well past the posted range and invert.
        lo = self.floor
        hi = self._upper_grid_hi()
        grid = np.linspace(lo, hi, 2000)
        cdf_vals = np.array([self.cdf(x) for x in grid])
        # Ensure monotone for interp.
        cdf_vals = np.maximum.accumulate(cdf_vals)
        cdf_vals[0], cdf_vals[-1] = 0.0, 1.0
        return np.interp(u, cdf_vals, grid)


@dataclass
class DiscreteLadderDistribution:
    """Discrete count distribution reconstructed from a ladder (for receptions).

    Lines are posted at half-integers (e.g. 2.5, 4.5), so P(X > 2.5) = P(X >= 3)
    and each line pins a CDF anchor at the integer below it:
    P(X <= floor(line)). We fit a **negative-binomial** distribution to those
    anchors and use it as the PMF over integer counts 0..k_max.

    Why negative-binomial rather than interpolating the CDF:
      - It is a single coherent probability model, so it gives a principled
        shape *between* the posted integers (no arbitrary "uniform fill") and a
        proper, decaying *upper tail* beyond the highest posted line (not
        residual mass clamped at k_max).
      - Reception counts are overdispersed (variance > mean); NB is the standard
        overdispersed generalization of Poisson and reduces toward Poisson as
        dispersion vanishes.

    Fitting: we choose NB parameters (mean mu >= 0, dispersion alpha > 0) to
    minimize squared error between the NB CDF and the de-vigged anchor points.
    With only one anchor the problem is underdetermined, so we fall back to a
    Poisson-like fit (alpha -> small) that matches that single point.

    Parameterization (mean/dispersion):
        mean = mu,   variance = mu + alpha * mu**2
        In scipy's nbinom(n, p):  n = 1/alpha,  p = n / (n + mu).
    """

    points: list[LadderPoint]
    k_max: int = 30
    _pmf: np.ndarray = field(init=False, repr=False)
    _support: np.ndarray = field(init=False, repr=False)
    _mu: float = field(init=False, repr=False)
    _alpha: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        thresholds, f = _to_sorted_cdf_points(self.points)
        # A half-line at t pins P(X <= floor(t)) = F. Build (k, target_cdf) pairs.
        anchor_k = np.floor(thresholds).astype(int)
        target_cdf = f

        self._mu, self._alpha = self._fit_nbinom(anchor_k, target_cdf)

        support = np.arange(0, self.k_max + 1)
        n, p = self._nbinom_params(self._mu, self._alpha)
        pmf = nbinom.pmf(support, n, p)
        # Fold the (tiny) tail beyond k_max into the last cell so it sums to 1.
        tail = max(0.0, 1.0 - pmf.sum())
        pmf[-1] += tail
        pmf = np.clip(pmf, 0.0, None)
        pmf /= pmf.sum()
        self._support = support
        self._pmf = pmf

    @staticmethod
    def _nbinom_params(mu: float, alpha: float) -> tuple[float, float]:
        """Convert (mean, dispersion) to scipy nbinom (n, p)."""
        mu = max(mu, 1e-6)
        alpha = max(alpha, 1e-6)
        n = 1.0 / alpha
        p = n / (n + mu)
        return n, p

    def _fit_nbinom(
        self, anchor_k: np.ndarray, target_cdf: np.ndarray
    ) -> tuple[float, float]:
        """Least-squares fit of (mu, alpha) to the anchor CDF points."""

        def objective(theta: np.ndarray) -> float:
            mu, alpha = theta
            if mu <= 0 or alpha <= 0:
                return 1e9
            n, p = self._nbinom_params(mu, alpha)
            model_cdf = nbinom.cdf(anchor_k, n, p)
            return float(np.sum((model_cdf - target_cdf) ** 2))

        # Initial guess: mu from the median anchor, mild overdispersion.
        # If an anchor is near 0.5, its k is close to the median ~ mean.
        idx = int(np.argmin(np.abs(target_cdf - 0.5)))
        mu0 = max(float(anchor_k[idx]) + 0.5, 0.5)
        x0 = np.array([mu0, 0.3])

        if len(anchor_k) == 1:
            # Underdetermined: pin alpha small (near-Poisson) and solve mu so the
            # single anchor CDF matches. A 1-D search on mu is robust.
            k0, c0 = float(anchor_k[0]), float(target_cdf[0])
            alpha = 1e-3

            def obj_mu(mu_arr: np.ndarray) -> float:
                mu = mu_arr[0]
                if mu <= 0:
                    return 1e9
                n, p = self._nbinom_params(mu, alpha)
                return float((nbinom.cdf(k0, n, p) - c0) ** 2)

            res = minimize(obj_mu, x0=np.array([mu0]), method="Nelder-Mead")
            return float(max(res.x[0], 1e-6)), alpha

        res = minimize(objective, x0=x0, method="Nelder-Mead")
        mu, alpha = res.x
        return float(max(mu, 1e-6)), float(max(alpha, 1e-6))

    @property
    def mu(self) -> float:
        return self._mu

    @property
    def alpha(self) -> float:
        return self._alpha

    def pmf(self) -> np.ndarray:
        return self._pmf

    def expected_value(self) -> float:
        return float(np.dot(self._support, self._pmf))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.invert(rng.random(n))

    def invert(self, u: np.ndarray) -> np.ndarray:
        """Inverse CDF for the fitted discrete distribution.

        Maps uniforms to integer counts via the PMF's cumulative distribution
        (smallest k with CDF(k) >= u). Lets a copula drive receptions with its
        correlated uniforms while preserving this exact marginal.
        """
        u = np.asarray(u, dtype=float)
        cdf = np.cumsum(self._pmf)
        cdf[-1] = 1.0  # guard against fp drift so u ~ 1.0 maps in-range
        idx = np.searchsorted(cdf, u, side="left")
        idx = np.clip(idx, 0, len(self._support) - 1)
        return self._support[idx].astype(float)
