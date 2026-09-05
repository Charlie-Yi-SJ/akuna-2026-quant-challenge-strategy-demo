"""Synthetic illustration of an uncertainty-aware market maker.

This module is an educational reconstruction. It is not a competition entry,
does not reproduce a competition interface, and is not intended for live trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, isfinite, sqrt
from statistics import NormalDist
from typing import Literal


MakerSide = Literal["buy", "sell"]
_STANDARD_NORMAL = NormalDist()
_EPSILON = 1e-12


def _finite(name: str, value: float) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        finite = isfinite(value)
    except (TypeError, OverflowError):
        finite = False
    if not finite:
        raise ValueError(f"{name} must be finite")


def _unit_interval(name: str, value: float) -> None:
    _finite(name, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


@dataclass(frozen=True)
class Belief:
    """Estimated event probability and remaining relative uncertainty."""

    probability: float
    uncertainty: float

    def __post_init__(self) -> None:
        _unit_interval("probability", self.probability)
        _unit_interval("uncertainty", self.uncertainty)


@dataclass(frozen=True)
class MarketState:
    """Small set of state variables used by the illustrative quote policy."""

    inventory: int
    cash_available: float
    equity: float
    peak_equity: float
    toxicity: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.inventory, int) or isinstance(self.inventory, bool):
            raise ValueError("inventory must be an integer")
        for name in ("cash_available", "equity", "peak_equity"):
            value = getattr(self, name)
            _finite(name, value)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.peak_equity <= 0.0:
            raise ValueError("peak_equity must be positive")
        if self.equity > self.peak_equity:
            raise ValueError("peak_equity must be at least equity")
        _unit_interval("toxicity", self.toxicity)


@dataclass(frozen=True)
class QuotePolicy:
    """Illustrative parameters for this synthetic demo."""

    tick_size: float = 0.01
    position_limit: int = 12
    base_size: int = 3
    base_half_spread: float = 0.03
    uncertainty_weight: float = 0.04
    inventory_shift: float = 0.03
    toxicity_weight: float = 0.06
    drawdown_weight: float = 0.08
    drawdown_limit: float = 0.30
    capital_fraction: float = 0.10

    def __post_init__(self) -> None:
        for name in ("position_limit", "base_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        positive_values = {
            "tick_size": self.tick_size,
            "base_half_spread": self.base_half_spread,
            "drawdown_limit": self.drawdown_limit,
            "capital_fraction": self.capital_fraction,
        }
        for name, value in positive_values.items():
            _finite(name, value)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "uncertainty_weight",
            "inventory_shift",
            "toxicity_weight",
            "drawdown_weight",
        ):
            value = getattr(self, name)
            _finite(name, value)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.tick_size >= 1.0:
            raise ValueError("tick_size must be smaller than 1")
        if self.tick_size < 1e-6:
            raise ValueError("tick_size must be at least 0.000001")
        price_levels = round(1.0 / self.tick_size)
        if (
            price_levels > 1_000_000
            or self.tick_size != 1.0 / price_levels
        ):
            raise ValueError(
                "tick_size must be an exact reciprocal that creates at most "
                "1,000,000 price levels"
            )
        if not 0.0 < self.drawdown_limit <= 1.0:
            raise ValueError("drawdown_limit must be between 0 and 1")
        if not 0.0 < self.capital_fraction <= 1.0:
            raise ValueError("capital_fraction must be between 0 and 1")


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float
    bid_size: int
    ask_size: int

    def __post_init__(self) -> None:
        _unit_interval("bid", self.bid)
        _unit_interval("ask", self.ask)
        if self.bid >= self.ask:
            raise ValueError("bid must be below ask")
        for name in ("bid_size", "ask_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


def posterior_binary_belief(
    *,
    prior_mean: float,
    prior_variance: float,
    signals: tuple[float, float],
    noise_variances: tuple[float, float],
    correlation: float,
) -> Belief:
    """Update a latent Gaussian event score from two correlated observations.

    The binary contract pays one when the latent score is positive. Uncertainty
    is the posterior standard deviation divided by the prior standard deviation.
    """

    _finite("prior_mean", prior_mean)
    if len(signals) != 2 or len(noise_variances) != 2:
        raise ValueError("exactly two signals and two noise variances are required")
    for index, signal in enumerate(signals):
        _finite(f"signals[{index}]", signal)
    for index, variance in enumerate(noise_variances):
        _finite(f"noise_variances[{index}]", variance)
        if variance <= 0.0:
            raise ValueError("noise variances must be positive")
    _finite("prior_variance", prior_variance)
    if prior_variance <= 0.0:
        raise ValueError("prior_variance must be positive")
    _finite("correlation", correlation)
    if abs(correlation) >= 1.0:
        raise ValueError("correlation must be strictly between -1 and 1")

    first_variance, second_variance = noise_variances
    inverse_first = 1.0 / first_variance
    inverse_second = 1.0 / second_variance
    inverse_cross_scale = 1.0 / (
        sqrt(first_variance) * sqrt(second_variance)
    )
    correlation_scale = 1.0 - correlation * correlation
    signal_information = (
        inverse_first
        + inverse_second
        - 2.0 * correlation * inverse_cross_scale
    ) / correlation_scale
    weighted_signals = (
        signals[0] * inverse_first
        + signals[1] * inverse_second
        - correlation
        * (signals[0] * inverse_cross_scale + signals[1] * inverse_cross_scale)
    ) / correlation_scale
    if not isfinite(signal_information) or signal_information <= 0.0:
        raise ValueError("signal covariance is numerically ill-conditioned")
    if not isfinite(weighted_signals):
        raise ValueError("signals are numerically ill-conditioned")

    posterior_precision = 1.0 / prior_variance + signal_information
    posterior_variance = 1.0 / posterior_precision
    posterior_mean = posterior_variance * (
        prior_mean / prior_variance + weighted_signals
    )
    if (
        not isfinite(posterior_variance)
        or posterior_variance <= 0.0
        or not isfinite(posterior_mean)
    ):
        raise ValueError("posterior is numerically ill-conditioned")

    probability = _STANDARD_NORMAL.cdf(
        posterior_mean / sqrt(posterior_variance)
    )
    relative_uncertainty = sqrt(posterior_variance / prior_variance)
    return Belief(
        probability=_clamp(probability, 0.0, 1.0),
        uncertainty=_clamp(relative_uncertainty, 0.0, 1.0),
    )


def _floor_to_tick(value: float, tick: float) -> float:
    price_levels = round(1.0 / tick)
    return floor(value * price_levels + _EPSILON) / price_levels


def _ceil_to_tick(value: float, tick: float) -> float:
    price_levels = round(1.0 / tick)
    return ceil(value * price_levels - _EPSILON) / price_levels


def _loss_capacity(budget: float, loss_per_contract: float, desired: int) -> int:
    if desired <= 0:
        return 0
    if loss_per_contract <= _EPSILON:
        return desired
    return min(desired, max(0, floor(budget / loss_per_contract)))


def quote_market(
    belief: Belief,
    state: MarketState,
    policy: QuotePolicy = QuotePolicy(),
) -> Quote:
    """Create a bounded two-sided quote with conservative risk-aware sizes."""

    drawdown = _clamp(1.0 - state.equity / state.peak_equity, 0.0, 1.0)
    inventory_ratio = _clamp(
        state.inventory / policy.position_limit,
        -1.0,
        1.0,
    )
    center = belief.probability - policy.inventory_shift * inventory_ratio
    half_spread = (
        policy.base_half_spread
        + policy.uncertainty_weight * belief.uncertainty
        + policy.toxicity_weight * state.toxicity
        + policy.drawdown_weight
        * min(1.0, drawdown / policy.drawdown_limit)
    )

    bid = _floor_to_tick(
        _clamp(center - half_spread, 0.0, 1.0),
        policy.tick_size,
    )
    ask = _ceil_to_tick(
        _clamp(center + half_spread, 0.0, 1.0),
        policy.tick_size,
    )
    if bid >= ask:
        bid = _floor_to_tick(
            _clamp(center - policy.tick_size, 0.0, 1.0 - policy.tick_size),
            policy.tick_size,
        )
        ask = _ceil_to_tick(
            min(1.0, bid + policy.tick_size),
            policy.tick_size,
        )

    severity = min(1.0, drawdown / policy.drawdown_limit)
    at_drawdown_limit = drawdown >= policy.drawdown_limit
    size_multiplier = max(0.0, (1.0 - 0.65 * state.toxicity) * (1.0 - severity))
    desired_size = policy.base_size if at_drawdown_limit else floor(
        policy.base_size * size_multiplier
    )
    if not at_drawdown_limit and size_multiplier > 0.0:
        desired_size = max(1, desired_size)

    risk_budget = state.cash_available * policy.capital_fraction
    # Split incremental-risk capacity because both displayed sides may fill
    # before state is refreshed. Existing inventory can still be reduced below.
    side_budget = risk_budget if at_drawdown_limit else 0.5 * risk_budget
    bid_capacity = _loss_capacity(side_budget, bid, desired_size)
    ask_capacity = _loss_capacity(side_budget, 1.0 - ask, desired_size)
    bid_size = min(
        bid_capacity,
        max(0, policy.position_limit - state.inventory),
    )
    ask_size = min(
        ask_capacity,
        max(0, policy.position_limit + state.inventory),
    )

    if state.inventory > 0:
        bid_size = min(
            bid_size,
            floor(desired_size * (1.0 - min(1.0, inventory_ratio))),
        )
        ask_size = max(ask_size, min(desired_size, state.inventory))
    elif state.inventory < 0:
        ask_size = min(
            ask_size,
            floor(desired_size * (1.0 - min(1.0, -inventory_ratio))),
        )
        bid_size = max(bid_size, min(desired_size, -state.inventory))

    if at_drawdown_limit:
        if state.inventory > 0:
            bid_size = 0
            ask_size = min(policy.base_size, state.inventory)
        elif state.inventory < 0:
            ask_size = 0
            bid_size = min(policy.base_size, -state.inventory)
        else:
            bid_size = ask_size = 0

    return Quote(bid, ask, bid_size, ask_size)


@dataclass
class ToxicityTracker:
    """Bounded EWMA of adverse fair-value moves observed after fills."""

    alpha: float = 0.15
    scale: float = 0.08
    level: float = 0.0

    def __post_init__(self) -> None:
        _finite("alpha", self.alpha)
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        _finite("scale", self.scale)
        if self.scale <= 0.0:
            raise ValueError("scale must be positive and finite")
        _unit_interval("level", self.level)

    def observe(
        self,
        *,
        maker_side: MakerSide,
        fair_at_fill: float,
        later_fair: float,
    ) -> float:
        """Record a later mark; adverse movement raises the toxicity estimate."""

        if maker_side not in ("buy", "sell"):
            raise ValueError("maker_side must be 'buy' or 'sell'")
        _unit_interval("fair_at_fill", fair_at_fill)
        _unit_interval("later_fair", later_fair)
        adverse_move = (
            fair_at_fill - later_fair
            if maker_side == "buy"
            else later_fair - fair_at_fill
        )
        observation = _clamp(adverse_move / self.scale, 0.0, 1.0)
        self.level = _clamp(
            (1.0 - self.alpha) * self.level + self.alpha * observation,
            0.0,
            1.0,
        )
        return self.level


def main() -> None:
    belief = posterior_binary_belief(
        prior_mean=0.0,
        prior_variance=1.0,
        signals=(0.35, 0.10),
        noise_variances=(0.80, 1.20),
        correlation=0.25,
    )
    state = MarketState(
        inventory=2,
        cash_available=100.0,
        equity=96.0,
        peak_equity=100.0,
        toxicity=0.10,
    )
    quote = quote_market(belief, state)
    print(f"fair value:  {belief.probability:.3f}")
    print(f"uncertainty: {belief.uncertainty:.3f}")
    print(
        "quote:       "
        f"{quote.bid:.2f} x {quote.bid_size} / "
        f"{quote.ask:.2f} x {quote.ask_size}"
    )


if __name__ == "__main__":
    main()
