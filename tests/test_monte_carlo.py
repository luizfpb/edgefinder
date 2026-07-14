"""Testes da simulação Monte Carlo de banca (monte_carlo.py)."""

import numpy as np
import pytest

from edgefinder.backtest.monte_carlo import simulate_bankroll

QUANTILE_KEYS = ["final_q05", "final_q25", "final_q50", "final_q75", "final_q95"]


def _flat(n: int, p: float, odds: float, frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.full(n, p),
        np.full(n, odds),
        np.full(n, frac),
    )


def test_positive_ev_median_grows() -> None:
    """EV+ claro (p=0.6 a odds 2.0): crescimento logarítmico esperado
    E[ln m] = 0.6 ln(1.05) + 0.4 ln(0.95) > 0, então a mediana final
    fica acima da banca inicial."""
    p, o, f = _flat(200, 0.6, 2.0, 0.05)
    out = simulate_bankroll(p, o, f, n_paths=4000, seed=1)
    assert out["final_q50"] > 1.0
    assert out["final_mean"] > 1.0


def test_negative_ev_median_shrinks() -> None:
    p, o, f = _flat(200, 0.4, 2.0, 0.05)
    out = simulate_bankroll(p, o, f, n_paths=4000, seed=2)
    assert out["final_q50"] < 1.0


def test_quantiles_are_ordered() -> None:
    p, o, f = _flat(100, 0.5, 2.0, 0.1)
    out = simulate_bankroll(p, o, f, n_paths=2000, seed=3)
    values = [out[k] for k in QUANTILE_KEYS]
    assert values == sorted(values)
    assert values[0] < values[-1]


def test_zero_stakes_change_nothing() -> None:
    p, o, f = _flat(50, 0.5, 2.0, 0.0)
    out = simulate_bankroll(p, o, f, n_paths=500, seed=4)
    for key in QUANTILE_KEYS:
        assert out[key] == pytest.approx(1.0)
    assert out["prob_ruin"] == 0.0
    assert out["expected_max_drawdown"] == pytest.approx(0.0)


def test_overbetting_ruins_almost_surely() -> None:
    """Arriscar 95% da banca por aposta: uma única derrota derruba a banca
    para 5% (< limiar de 10%), e P(nenhuma derrota em 50) = 0.5^50 ~ 0."""
    p, o, f = _flat(50, 0.5, 2.0, 0.95)
    out = simulate_bankroll(p, o, f, n_paths=2000, seed=5)
    assert out["prob_ruin"] > 0.99
    assert out["expected_max_drawdown"] > 0.9


def test_expected_max_drawdown_bounds() -> None:
    p, o, f = _flat(150, 0.55, 2.0, 0.05)
    out = simulate_bankroll(p, o, f, n_paths=1000, seed=6)
    assert 0.0 < out["expected_max_drawdown"] < 1.0


def test_reproducible_with_seed() -> None:
    p, o, f = _flat(80, 0.55, 1.9, 0.03)
    a = simulate_bankroll(p, o, f, n_paths=1000, seed=42)
    b = simulate_bankroll(p, o, f, n_paths=1000, seed=42)
    c = simulate_bankroll(p, o, f, n_paths=1000, seed=43)
    assert a == b
    assert a != c


def test_initial_scales_output() -> None:
    p, o, f = _flat(50, 0.6, 2.0, 0.05)
    small = simulate_bankroll(p, o, f, n_paths=1000, initial=1.0, seed=7)
    big = simulate_bankroll(p, o, f, n_paths=1000, initial=100.0, seed=7)
    assert big["final_q50"] == pytest.approx(100.0 * small["final_q50"])
    # ruína e drawdown são relativos à banca inicial: invariantes de escala
    assert big["prob_ruin"] == small["prob_ruin"]
    assert big["expected_max_drawdown"] == pytest.approx(small["expected_max_drawdown"])


def test_empty_bets_is_degenerate_identity() -> None:
    empty = np.array([])
    out = simulate_bankroll(empty, empty, empty, n_paths=100, seed=8)
    assert out["n_bets"] == 0
    for key in QUANTILE_KEYS:
        assert out[key] == 1.0
    assert out["prob_ruin"] == 0.0
    assert out["expected_max_drawdown"] == 0.0


def test_validation_errors() -> None:
    p, o, f = _flat(10, 0.5, 2.0, 0.05)
    with pytest.raises(ValueError, match="mesmo tamanho"):
        simulate_bankroll(p[:5], o, f)
    with pytest.raises(ValueError, match=r"p_win"):
        simulate_bankroll(np.full(10, 1.5), o, f)
    with pytest.raises(ValueError, match="odds"):
        simulate_bankroll(p, np.full(10, 1.0), f)
    with pytest.raises(ValueError, match="stakes_frac"):
        simulate_bankroll(p, o, np.full(10, 1.5))
    with pytest.raises(ValueError, match="n_paths"):
        simulate_bankroll(p, o, f, n_paths=0)
    with pytest.raises(ValueError, match="initial"):
        simulate_bankroll(p, o, f, initial=0.0)
    with pytest.raises(ValueError, match="ruin_threshold"):
        simulate_bankroll(p, o, f, ruin_threshold=1.0)
