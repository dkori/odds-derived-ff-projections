"""Touchdown-count distribution fit to a TD ladder.

Books post TD scoring as a small ladder of one-sided prices:

    player_anytime_td  "Yes"      -> P(X >= 1)
    player_tds_over    "Over 1.5" -> P(X >= 2)
    player_tds_over    "Over 2.5" -> P(X >= 3)
    player_tds_over    "Over 3.5" -> P(X >= 4)

After de-vigging (see ``odds/devig.devig_oneside_td_prob``) these are SURVIVAL
probabilities S(k) = P(X >= k). With **two or more** anchors we fit a
**negative-binomial** count model to them -- the same overdispersed family the
receptions marginal uses (``DiscreteLadderDistribution``) -- so the multi-TD
tail is *priced by the market* (P(>=3), P(>=4) anchors) instead of being forced
by a single Poisson rate.

Why not Poisson: Poisson is one-parameter, so fixing P(>=1) fixes lambda and
therefore ALL higher probabilities. Empirically that inflates multi-TD games
(the fat-tail bug: a de-vigged P(>=1)~0.72 implies Poisson P(>=3)~11%, far above
the market's ~3-12%). NB has a second (dispersion) parameter, so it matches
P(>=1) AND P(>=2..k) simultaneously and decays realistically past the top anchor.

Fallback: with a SINGLE anchor (P(>=1) only, e.g. a book that posts anytime-TD
but no Over ladder) the fit is underdetermined, so we fall back to the classic
Poisson(lambda = -ln(1 - P(>=1))) shape.

A hard cap (default 4) is applied to sampled counts as a guard against absurd
draws; with an anchored NB fit this is rarely binding but keeps the tail sane.

Implements the same Protocol as the other marginals (``sample``, ``invert``,
``expected_value``) so ``sim/montecarlo`` needs no special-casing, and
``invert`` returns integer counts so the Gaussian copula can drive it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import nbinom

from ffmc.distributions.poisson import lambda_from_anytime_td


@dataclass
class TouchdownCountDistribution:
    """TD count from >=1 de-vigged survival anchors {k: P(X >= k)}.

    ``anchors`` maps an integer threshold ``k`` (>=1) to the de-vigged survival
    probability ``P(X >= k)``. With >=2 anchors a negative binomial is fit to
    them; with exactly one, a Poisson fallback anchored to P(>=1) is used.
    """

    anchors: dict[int, float]
    k_max: int = 4  # hard cap on sampled TD counts (guard)
    _pmf: np.ndarray = field(init=False, repr=False)
    _support: np.ndarray = field(init=False, repr=False)
    _mu: float = field(init=False, repr=False)
    _alpha: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.anchors:
            raise ValueError("TouchdownCountDistribution needs >=1 anchor")
        # Sort anchors by k; convert survival P(X>=k) to CDF F(k-1)=1-P(X>=k).
        ks = np.array(sorted(self.anchors), dtype=int)
        surv = np.array([self.anchors[int(k)] for k in ks], dtype=float)
        # Clean up any mild non-monotonicity from independent de-vig: survival
        # must be non-increasing in k.
        surv = np.minimum.accumulate(surv)
        surv = np.clip(surv, 1e-9, 1.0 - 1e-9)
        anchor_cdf_k = ks - 1  # "Over (k-0.5)" pins F(k-1) = 1 - P(X>=k)
        target_cdf = 1.0 - surv

        if len(ks) >= 2:
            self._mu, self._alpha = self._fit_nbinom(anchor_cdf_k, target_cdf)
        else:
            # Single anchor -> Poisson fallback (near-zero dispersion NB).
            lam = lambda_from_anytime_td(float(surv[0]))
            self._mu, self._alpha = max(lam, 1e-6), 1e-6

        support = np.arange(0, self.k_max + 1)
        n, p = self._nbinom_params(self._mu, self._alpha)
        pmf = nbinom.pmf(support, n, p)
        # Fold the tail beyond k_max into the last cell (hard cap) so it sums to 1.
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

        # Initial guess: a small positive mean, mild overdispersion. TD means are
        # sub-1 for most players, so start modestly.
        x0 = np.array([0.6, 0.5])
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
        """Inverse CDF: map uniforms to integer TD counts (capped at k_max).

        Mirrors ``DiscreteLadderDistribution.invert`` so the copula can drive TD
        counts with correlated uniforms while preserving this exact marginal.
        """
        u = np.asarray(u, dtype=float)
        cdf = np.cumsum(self._pmf)
        cdf[-1] = 1.0  # guard against fp drift so u ~ 1.0 maps in-range
        idx = np.searchsorted(cdf, u, side="left")
        idx = np.clip(idx, 0, len(self._support) - 1)
        return self._support[idx].astype(float)
