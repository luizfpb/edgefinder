"""Testes das métricas de backtest (metrics.py)."""

import numpy as np
import pandas as pd
import pytest

from edgefinder.backtest.metrics import (
    hit_rate,
    max_drawdown,
    prob_metrics,
    roi,
    sharpe_ratio,
    significance,
    summary,
    yield_per_bet,
)


def _bets(
    results: list[str],
    odds: float | list[float] = 2.0,
    stake: float | list[float] = 1.0,
) -> pd.DataFrame:
    n = len(results)
    odds_list = [odds] * n if isinstance(odds, float) else odds
    stake_list = [stake] * n if isinstance(stake, float) else stake
    return pd.DataFrame({"stake": stake_list, "odds": odds_list, "result": results})


class TestRoiAndYield:
    def test_balanced_even_odds_is_zero(self) -> None:
        df = _bets(["win", "lose"], odds=2.0)
        assert roi(df) == pytest.approx(0.0)
        assert yield_per_bet(df) == pytest.approx(0.0)

    def test_known_values(self) -> None:
        # win: 2 * (3 - 1) = +4; lose: -2 => pnl 2 sobre volume 4
        df = _bets(["win", "lose"], odds=3.0, stake=2.0)
        assert roi(df) == pytest.approx(0.5)
        assert yield_per_bet(df) == pytest.approx(0.5)

    def test_push_is_neutral_but_counts_in_turnover(self) -> None:
        df = _bets(["win", "push", "lose"], odds=2.0)
        assert roi(df) == pytest.approx(0.0)
        assert hit_rate(df) == pytest.approx(0.5)

    def test_accepts_bet_odds_alias(self) -> None:
        df = pd.DataFrame({"stake": [1.0], "bet_odds": [3.0], "result": ["win"]})
        assert roi(df) == pytest.approx(2.0)

    def test_empty_is_zero(self) -> None:
        empty = pd.DataFrame(columns=["stake", "odds", "result"])
        assert roi(empty) == 0.0
        assert yield_per_bet(empty) == 0.0
        assert hit_rate(empty) == 0.0

    def test_fair_bets_roi_near_zero(self) -> None:
        """Apostas justas (p = 1/odds) têm EV zero, logo ROI ~ 0."""
        rng = np.random.default_rng(11)
        n = 20_000
        odds = 2.0
        results = np.where(rng.random(n) < 1.0 / odds, "win", "lose")
        df = _bets(list(results), odds=odds)
        assert abs(roi(df)) < 0.03

    def test_yield_differs_from_roi_with_variable_stakes(self) -> None:
        # aposta grande perdida domina o ROI mas não o yield por aposta
        df = _bets(["win", "lose"], odds=2.0, stake=[1.0, 9.0])
        assert roi(df) == pytest.approx((1.0 - 9.0) / 10.0)
        assert yield_per_bet(df) == pytest.approx(0.0)


class TestMaxDrawdown:
    def test_known_series(self) -> None:
        equity = [1.0, 1.2, 0.9, 1.1, 0.8]
        assert max_drawdown(equity) == pytest.approx(1.0 - 0.8 / 1.2)

    def test_monotone_increasing_is_zero(self) -> None:
        assert max_drawdown([1.0, 1.1, 1.2, 1.5]) == pytest.approx(0.0)

    def test_bounds_and_empty(self) -> None:
        assert max_drawdown([]) == 0.0
        rng = np.random.default_rng(5)
        equity = np.cumprod(1.0 + rng.normal(0.0, 0.02, 500))
        dd = max_drawdown(equity)
        assert 0.0 <= dd < 1.0


class TestSharpe:
    def test_known_value(self) -> None:
        # mean = 2, std amostral = 1
        assert sharpe_ratio([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_zero_mean_is_zero(self) -> None:
        assert sharpe_ratio([0.1, -0.1]) == pytest.approx(0.0)

    def test_degenerate_is_zero(self) -> None:
        assert sharpe_ratio([0.5]) == 0.0
        assert sharpe_ratio([0.5, 0.5, 0.5]) == 0.0
        assert sharpe_ratio([]) == 0.0


class TestProbMetrics:
    def test_perfect_predictions(self) -> None:
        out = prob_metrics([0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0])
        assert out["brier"] == pytest.approx(0.0)
        assert out["log_loss"] == pytest.approx(0.0, abs=1e-10)

    def test_uninformative_half(self) -> None:
        out = prob_metrics([0.0, 1.0], [0.5, 0.5])
        assert out["brier"] == pytest.approx(0.25)
        assert out["log_loss"] == pytest.approx(np.log(2.0))

    def test_confident_wrong_is_punished(self) -> None:
        good = prob_metrics([1.0], [0.9])
        bad = prob_metrics([1.0], [0.1])
        assert bad["log_loss"] > good["log_loss"]
        assert bad["brier"] > good["brier"]

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="0 e 1"):
            prob_metrics([0.0, 2.0], [0.5, 0.5])
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            prob_metrics([0.0, 1.0], [0.5, 1.2])
        with pytest.raises(ValueError, match="shapes"):
            prob_metrics([0.0, 1.0], [0.5])
        with pytest.raises(ValueError, match="vazio"):
            prob_metrics([], [])


class TestSignificance:
    def test_two_percent_on_200_bets_is_not_significant(self) -> None:
        """O exemplo do briefing: +2% em 200 apostas é ruído, não edge."""
        yields = np.array([1.0] * 102 + [-1.0] * 98)
        out = significance(yields)
        assert out["mean"] == pytest.approx(0.02)
        assert out["p_value"] > 0.05
        assert out["p_value_bootstrap"] > 0.05
        assert out["ci95_low"] < 0.0 < out["ci95_high"]

    def test_strong_edge_is_significant(self) -> None:
        rng = np.random.default_rng(21)
        yields = rng.normal(0.5, 1.0, 400)
        out = significance(yields)
        assert out["p_value"] < 1e-6
        assert out["p_value_bootstrap"] < 0.01
        assert out["ci95_low"] > 0.0

    def test_negative_yield_has_high_p_value(self) -> None:
        rng = np.random.default_rng(22)
        yields = rng.normal(-0.3, 1.0, 300)
        out = significance(yields)
        assert out["p_value"] > 0.5

    def test_ci_brackets_the_mean(self) -> None:
        rng = np.random.default_rng(23)
        yields = rng.normal(0.1, 1.0, 250)
        out = significance(yields)
        assert out["ci95_low"] <= out["mean"] <= out["ci95_high"]

    def test_deterministic_given_seed(self) -> None:
        yields = np.array([1.0, -1.0, 1.0, 1.0, -1.0])
        assert significance(yields, seed=7) == significance(yields, seed=7)

    def test_degenerate_cases_are_conservative(self) -> None:
        assert significance(np.array([]))["p_value"] == 1.0
        assert significance(np.array([0.5]))["p_value"] == 1.0
        assert significance(np.array([0.5, 0.5, 0.5]))["p_value"] == 1.0


class TestSummary:
    def test_core_keys_and_values(self) -> None:
        df = _bets(["win", "lose", "win", "push"], odds=2.0)
        out = summary(df)
        assert out["n_bets"] == 4
        assert out["roi"] == pytest.approx(0.25)
        assert out["hit_rate"] == pytest.approx(2.0 / 3.0)
        assert out["total_pnl"] == pytest.approx(1.0)
        assert 0.0 <= out["max_drawdown"] < 1.0
        assert "avg_ev" not in out
        assert "avg_clv" not in out

    def test_optional_columns(self) -> None:
        df = _bets(["win", "lose"], odds=2.0)
        df["ev"] = [0.05, 0.03]
        df["closing_fair"] = [0.55, 0.50]
        out = summary(df)
        assert out["avg_ev"] == pytest.approx(0.04)
        # clv = odds * p_fair - 1: (2*0.55-1 + 2*0.50-1)/2 = 0.05
        assert out["avg_clv"] == pytest.approx(0.05)

    def test_bet_ev_alias_and_explicit_clv(self) -> None:
        df = _bets(["win"], odds=2.0)
        df["bet_ev"] = [0.08]
        df["clv"] = [0.02]
        out = summary(df)
        assert out["avg_ev"] == pytest.approx(0.08)
        assert out["avg_clv"] == pytest.approx(0.02)

    def test_empty(self) -> None:
        out = summary(pd.DataFrame())
        assert out["n_bets"] == 0
        assert out["roi"] == 0.0
        assert out["max_drawdown"] == 0.0

    def test_drawdown_uses_initial_bankroll_of_one(self) -> None:
        # primeira aposta perdida: equity 1.0 -> 0.5, drawdown de 50%
        df = _bets(["lose", "win"], odds=2.0, stake=0.5)
        out = summary(df)
        assert out["max_drawdown"] == pytest.approx(0.5)
