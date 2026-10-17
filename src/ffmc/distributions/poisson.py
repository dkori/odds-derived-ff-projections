"""Touchdown distribution from an anytime-TD probability.

Books typically post only "anytime TD" = P(scores at least 1 TD), not a full
ladder over 0/1/2/3 TDs. To sample a TD *count* we assume TDs arrive as a
Poisson process and back out the rate from that single probability:

    P(X >= 1) = 1 - P(X = 0) = 1 - e^{-lambda}
    =>  lambda = -ln(1 - P(X >= 1))

Then TD count ~ Poisson(lambda). This is a standard, defensible assumption but
it IS an assumption about the shape of the TD distribution (see README).

If a book ever exposes a full TD ladder, use DiscreteLadderDistribution instead.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def lambda_from_anytime_td(prob_anytime_td: float) -> float:
    """Poisson rate implied by P(scores >= 1 TD)."""
    p = min(max(prob_anytime_td, 1e-9), 1.0 - 1e-9)
    return float(-np.log(1.0 - p))


@dataclass
class PoissonTouchdownDistribution:
    """Poisson-distributed TD count derived from an anytime-TD probability."""

    prob_anytime_td: float

    @property
    def rate(self) -> float:
        return lambda_from_anytime_td(self.prob_anytime_td)

    def expected_value(self) -> float:
        return self.rate  # E[Poisson(lambda)] = lambda

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.poisson(self.rate, size=n).astype(float)

    def invert(self, u: np.ndarray) -> np.ndarray:
        """Inverse CDF of Poisson(rate): map uniforms to TD counts.

        Lets a copula drive TD counts with correlated uniforms while keeping the
        Poisson marginal exact.
        """
        from scipy.stats import poisson

        u = np.asarray(u, dtype=float)
        return poisson.ppf(u, self.rate).astype(float)
