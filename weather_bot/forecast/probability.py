"""Convert ensemble forecasts into probability distributions.

The empirical CDF over ensemble members is the foundation. We expose two
methods to compute exceedance probabilities:

  * `empirical`  — fraction of members above threshold, with a Laplace
                   smoothing of +0.5 / +1 to avoid hard 0 or 1 at the tails.
  * `gaussian`   — fit a normal distribution to the members. Useful for very
                   extreme thresholds where the empirical estimate becomes
                   degenerate, but biased if the true distribution is skewed.

For Polymarket pricing, prefer `empirical` near the body of the distribution
and `gaussian` (with caution) for tail thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from scipy import stats

from ..units import Unit, from_celsius, to_celsius
from .fetcher import EnsembleForecast


@dataclass
class TempDistribution:
    """Empirical distribution of daily max temperature from ensemble members."""

    location_name: str
    target_date: date
    members: np.ndarray  # daily max per member, °C

    @property
    def n_members(self) -> int:
        return int(np.sum(~np.isnan(self.members)))

    @property
    def mean(self) -> float:
        return float(np.nanmean(self.members))

    @property
    def std(self) -> float:
        return float(np.nanstd(self.members, ddof=1))

    def quantile(self, q: float) -> float:
        return float(np.nanquantile(self.members, q))

    def prob_above(
        self,
        threshold: float,
        unit: Unit = "C",
        method: str = "empirical",
    ) -> float:
        threshold_c = to_celsius(threshold, unit)
        clean = self.members[~np.isnan(self.members)]
        if method == "empirical":
            # Laplace smoothing: prevents degenerate 0/1 at the tails of a
            # finite ensemble. With 51 members, a "0/51" event becomes 0.5/52.
            return float((np.sum(clean > threshold_c) + 0.5) / (len(clean) + 1))
        if method == "gaussian":
            mu, sigma = clean.mean(), clean.std(ddof=1)
            if sigma == 0:
                return 1.0 if threshold_c < mu else 0.0
            return float(1.0 - stats.norm.cdf(threshold_c, loc=mu, scale=sigma))
        raise ValueError(f"Unknown method: {method!r}")

    def prob_below(
        self,
        threshold: float,
        unit: Unit = "C",
        method: str = "empirical",
    ) -> float:
        return 1.0 - self.prob_above(threshold, unit, method)

    def prob_in_range(
        self,
        low: float,
        high: float,
        unit: Unit = "C",
        method: str = "empirical",
    ) -> float:
        return self.prob_above(low, unit, method) - self.prob_above(high, unit, method)

    def prob_in_bucket(
        self,
        center: float,
        unit: Unit = "C",
        method: str = "empirical",
    ) -> float:
        """Probability that the observation rounds to the integer `center`.

        Polymarket buckets are 1°C centred on integers (so [k-0.5, k+0.5)) and
        2°F (so [k-1, k+1)). This method uses the unit's bucket width.
        """
        half_width = 0.5 if unit == "C" else 1.0
        return self.prob_in_range(
            center - half_width, center + half_width, unit, method
        )

    def bucket_pmf(
        self,
        low_threshold: int,
        high_threshold: int,
        unit: Unit = "C",
        method: str = "empirical",
    ) -> list[tuple[str, float]]:
        """Probability mass function over Polymarket buckets.

        Polymarket uses 1°C buckets for °C markets and 2°F buckets for °F
        markets. Bucket label `k` represents the range [k - half, k + half)
        where half is 0.5 °C or 1 °F. For °F markets the bucket *centres* are
        spaced 2 apart (so they tile [..., 60, 62, 64, ...]).

        Returns a list of (label, prob) ordered low → high, with the first
        and last buckets being the "or below" / "or higher" tails:

            [("≤10", p), ("11", p), …, ("19", p), ("≥20", p)]   # °C
            [("≤60", p), ("62", p), ("64", p), …, ("≥80", p)]   # °F
        """
        step = 1 if unit == "C" else 2
        half = step / 2.0
        out: list[tuple[str, float]] = []

        # Low tail: observation ≤ low_threshold (i.e. below low_threshold + half)
        out.append(
            (f"≤{low_threshold}", self.prob_below(low_threshold + half, unit, method))
        )

        # Middle buckets, stepping by `step`
        for k in range(low_threshold + step, high_threshold, step):
            out.append((str(k), self.prob_in_bucket(k, unit, method)))

        # High tail: observation ≥ high_threshold
        out.append(
            (f"≥{high_threshold}", self.prob_above(high_threshold - half, unit, method))
        )

        return out

    def summary(self, unit: Unit = "C") -> str:
        u = "°C" if unit == "C" else "°F"
        m = from_celsius(self.mean, unit)
        # std is a delta, so convert by scaling only (no +32 offset for °F)
        sigma_scale = 1.0 if unit == "C" else 9.0 / 5.0
        s = self.std * sigma_scale
        q10 = from_celsius(self.quantile(0.1), unit)
        q50 = from_celsius(self.quantile(0.5), unit)
        q90 = from_celsius(self.quantile(0.9), unit)
        return (
            f"n={self.n_members:3d}  "
            f"mean={m:5.1f}{u}  std={s:4.2f}  "
            f"p10={q10:5.1f}  p50={q50:5.1f}  p90={q90:5.1f}"
        )


def bucket_prob(
    dist: TempDistribution,
    kind: str,
    threshold: int,
    unit: Unit = "C",
    method: str = "empirical",
) -> float:
    """Probability for a Polymarket bucket given its kind and threshold.

    Dispatches to the right TempDistribution method:
      "low_tail"  → prob_below(threshold + half_bucket)
      "high_tail" → prob_above(threshold - half_bucket)
      "mid"       → prob_in_bucket(threshold)
    """
    half = 0.5 if unit == "C" else 1.0
    if kind == "low_tail":
        return dist.prob_below(threshold + half, unit, method)
    if kind == "high_tail":
        return dist.prob_above(threshold - half, unit, method)
    return dist.prob_in_bucket(threshold, unit, method)


def distribution_from_forecast(
    forecast: EnsembleForecast,
    target: date,
) -> TempDistribution:
    return TempDistribution(
        location_name=forecast.location.name,
        target_date=target,
        members=forecast.daily_max(target),
    )


def blend_distributions(
    dists: list[TempDistribution],
    weights: list[float] | None = None,
    seed: int = 0,
) -> TempDistribution:
    """Pool ensemble members across models into a single distribution.

    Equal-weight pooling (weights=None) is the simplest defensible blend and
    treats every member as one independent draw. Weighted blending resamples
    each model's members so the final pool reflects model-skill weights.
    """
    if not dists:
        raise ValueError("Need at least one distribution to blend")

    if weights is None:
        members = np.concatenate([d.members for d in dists])
    else:
        if len(weights) != len(dists):
            raise ValueError("weights length must match dists length")
        if any(w < 0 for w in weights):
            raise ValueError("weights must be non-negative")
        total = sum(weights)
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        target_n = max(d.n_members for d in dists) * len(dists)
        rng = np.random.default_rng(seed)
        parts = []
        for d, w in zip(dists, weights):
            n = max(1, int(round(target_n * w / total)))
            clean = d.members[~np.isnan(d.members)]
            parts.append(rng.choice(clean, size=n, replace=True))
        members = np.concatenate(parts)

    return TempDistribution(
        location_name=dists[0].location_name,
        target_date=dists[0].target_date,
        members=members,
    )
