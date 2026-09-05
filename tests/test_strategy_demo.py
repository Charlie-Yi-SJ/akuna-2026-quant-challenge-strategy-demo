import math
import unittest

from strategy_demo import (
    Belief,
    MarketState,
    QuotePolicy,
    ToxicityTracker,
    posterior_binary_belief,
    quote_market,
)


class BeliefTests(unittest.TestCase):
    def test_neutral_signals_produce_even_probability(self) -> None:
        belief = posterior_binary_belief(
            prior_mean=0.0,
            prior_variance=1.0,
            signals=(0.0, 0.0),
            noise_variances=(1.0, 1.0),
            correlation=0.0,
        )
        self.assertAlmostEqual(belief.probability, 0.5)
        self.assertLess(belief.uncertainty, 1.0)

    def test_signal_direction_moves_probability(self) -> None:
        def belief_for(signal: float) -> Belief:
            return posterior_binary_belief(
                prior_mean=0.0,
                prior_variance=1.0,
                signals=(signal, signal),
                noise_variances=(1.0, 1.0),
                correlation=0.2,
            )

        self.assertGreater(belief_for(1.0).probability, 0.5)
        self.assertLess(belief_for(-1.0).probability, 0.5)

    def test_positive_correlation_preserves_more_uncertainty(self) -> None:
        independent = posterior_binary_belief(
            prior_mean=0.0,
            prior_variance=1.0,
            signals=(0.0, 0.0),
            noise_variances=(1.0, 1.0),
            correlation=0.0,
        )
        correlated = posterior_binary_belief(
            prior_mean=0.0,
            prior_variance=1.0,
            signals=(0.0, 0.0),
            noise_variances=(1.0, 1.0),
            correlation=0.7,
        )
        self.assertGreater(correlated.uncertainty, independent.uncertainty)

    def test_small_valid_noise_variances_remain_stable(self) -> None:
        belief = posterior_binary_belief(
            prior_mean=0.0,
            prior_variance=1.0,
            signals=(1e-4, -1e-4),
            noise_variances=(1e-8, 1e-8),
            correlation=0.0,
        )
        self.assertAlmostEqual(belief.probability, 0.5)
        self.assertGreater(belief.uncertainty, 0.0)

    def test_invalid_or_nonfinite_inputs_are_rejected(self) -> None:
        invalid_cases = (
            {"prior_variance": 0.0},
            {"noise_variances": (1.0, 0.0)},
            {"correlation": 1.0},
            {"prior_mean": math.inf},
        )
        base = {
            "prior_mean": 0.0,
            "prior_variance": 1.0,
            "signals": (0.0, 0.0),
            "noise_variances": (1.0, 1.0),
            "correlation": 0.0,
        }
        for change in invalid_cases:
            with self.subTest(change=change), self.assertRaises(ValueError):
                posterior_binary_belief(**(base | change))


class QuoteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = QuotePolicy()
        self.neutral_state = MarketState(
            inventory=0,
            cash_available=100.0,
            equity=100.0,
            peak_equity=100.0,
        )

    def test_quotes_are_bounded_on_ticks_and_do_not_cross(self) -> None:
        for probability in (0.0, 0.01, 0.5, 0.99, 1.0):
            with self.subTest(probability=probability):
                quote = quote_market(
                    Belief(probability, 0.5),
                    self.neutral_state,
                    self.policy,
                )
                self.assertGreaterEqual(quote.bid, 0.0)
                self.assertLessEqual(quote.ask, 1.0)
                self.assertLess(quote.bid, quote.ask)
                self.assertAlmostEqual(
                    quote.bid / self.policy.tick_size,
                    round(quote.bid / self.policy.tick_size),
                )
                self.assertAlmostEqual(
                    quote.ask / self.policy.tick_size,
                    round(quote.ask / self.policy.tick_size),
                )

    def test_invalid_tick_grid_is_rejected(self) -> None:
        for tick_size in (0.03, 0.333333333, 1e-11, 1e-320):
            with self.subTest(tick_size=tick_size), self.assertRaises(ValueError):
                QuotePolicy(tick_size=tick_size)

    def test_supported_non_decimal_tick_stays_bounded(self) -> None:
        policy = QuotePolicy(tick_size=1.0 / 3.0)
        quote = quote_market(Belief(1.0, 0.5), self.neutral_state, policy)
        self.assertLess(quote.bid, quote.ask)
        self.assertLessEqual(quote.ask, 1.0)

    def test_uncertainty_and_toxicity_do_not_narrow_the_market(self) -> None:
        calm = quote_market(Belief(0.5, 0.0), self.neutral_state, self.policy)
        uncertain = quote_market(Belief(0.5, 1.0), self.neutral_state, self.policy)
        toxic = quote_market(
            Belief(0.5, 0.0),
            MarketState(0, 100.0, 100.0, 100.0, toxicity=1.0),
            self.policy,
        )
        calm_width = calm.ask - calm.bid
        self.assertGreaterEqual(uncertain.ask - uncertain.bid, calm_width)
        self.assertGreaterEqual(toxic.ask - toxic.bid, calm_width)
        self.assertLessEqual(toxic.bid_size, calm.bid_size)
        self.assertLessEqual(toxic.ask_size, calm.ask_size)

    def test_inventory_skews_prices_and_sizes_toward_reduction(self) -> None:
        belief = Belief(0.5, 0.2)
        neutral = quote_market(belief, self.neutral_state, self.policy)
        long = quote_market(
            belief,
            MarketState(8, 100.0, 100.0, 100.0),
            self.policy,
        )
        short = quote_market(
            belief,
            MarketState(-8, 100.0, 100.0, 100.0),
            self.policy,
        )
        self.assertLess(long.bid + long.ask, neutral.bid + neutral.ask)
        self.assertGreater(short.bid + short.ask, neutral.bid + neutral.ask)
        self.assertLess(long.bid_size, long.ask_size)
        self.assertLess(short.ask_size, short.bid_size)

    def test_position_limit_disables_exposure_increasing_side(self) -> None:
        belief = Belief(0.5, 0.2)
        at_long_limit = quote_market(
            belief,
            MarketState(self.policy.position_limit, 100.0, 100.0, 100.0),
            self.policy,
        )
        at_short_limit = quote_market(
            belief,
            MarketState(-self.policy.position_limit, 100.0, 100.0, 100.0),
            self.policy,
        )
        self.assertEqual(at_long_limit.bid_size, 0)
        self.assertEqual(at_short_limit.ask_size, 0)

    def test_drawdown_limit_allows_only_inventory_reduction(self) -> None:
        belief = Belief(0.5, 0.2)
        long = quote_market(
            belief,
            MarketState(5, 100.0, 70.0, 100.0),
            self.policy,
        )
        flat = quote_market(
            belief,
            MarketState(0, 100.0, 70.0, 100.0),
            self.policy,
        )
        self.assertEqual(long.bid_size, 0)
        self.assertGreater(long.ask_size, 0)
        self.assertEqual((flat.bid_size, flat.ask_size), (0, 0))

    def test_two_sides_share_one_cash_risk_budget(self) -> None:
        state = MarketState(0, 20.0, 20.0, 20.0)
        quote = quote_market(Belief(0.5, 0.0), state, self.policy)
        total_maximum_loss = (
            quote.bid_size * quote.bid
            + quote.ask_size * (1.0 - quote.ask)
        )
        self.assertLessEqual(
            total_maximum_loss,
            state.cash_available * self.policy.capital_fraction + 1e-12,
        )

    def test_risk_reducing_side_remains_available_without_free_cash(self) -> None:
        ordinary = quote_market(
            Belief(0.5, 0.2),
            MarketState(4, 0.0, 100.0, 100.0),
            self.policy,
        )
        hard_drawdown = quote_market(
            Belief(0.5, 0.2),
            MarketState(4, 0.0, 70.0, 100.0),
            self.policy,
        )
        self.assertEqual(ordinary.bid_size, 0)
        self.assertGreater(ordinary.ask_size, 0)
        self.assertEqual(hard_drawdown.bid_size, 0)
        self.assertGreater(hard_drawdown.ask_size, 0)


class ToxicityTests(unittest.TestCase):
    def test_adverse_moves_raise_a_bounded_estimate(self) -> None:
        tracker = ToxicityTracker(alpha=0.5, scale=0.10)
        adverse = tracker.observe(
            maker_side="sell",
            fair_at_fill=0.50,
            later_fair=0.60,
        )
        self.assertGreater(adverse, 0.0)
        for _ in range(20):
            tracker.observe(
                maker_side="sell",
                fair_at_fill=0.0,
                later_fair=1.0,
            )
        self.assertLessEqual(tracker.level, 1.0)

    def test_favorable_move_does_not_create_adverse_signal(self) -> None:
        tracker = ToxicityTracker(alpha=0.5, scale=0.10)
        tracker.observe(
            maker_side="buy",
            fair_at_fill=0.50,
            later_fair=0.60,
        )
        self.assertEqual(tracker.level, 0.0)


if __name__ == "__main__":
    unittest.main()
