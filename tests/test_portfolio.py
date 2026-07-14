"""Testes do módulo de portfólio: banca imutável e controles de risco."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from edgefinder.portfolio.bankroll import BankrollState, apply_bet_result, equity_curve
from edgefinder.portfolio.risk import (
    NovaAposta,
    check_exposure,
    drawdown_guard,
    stake_com_limites,
)

TOL = 1e-9


def _estado(current: float = 100.0, initial: float = 100.0) -> BankrollState:
    return BankrollState(current=current, initial=initial, history=[])


class TestApplyBetResult:
    def test_win_credita_lucro(self) -> None:
        novo = apply_bet_result(_estado(), stake=10.0, odds=2.5, result="win")
        assert novo.current == pytest.approx(115.0, abs=TOL)
        assert novo.history == [pytest.approx(115.0)]

    def test_lose_debita_stake(self) -> None:
        novo = apply_bet_result(_estado(), stake=10.0, odds=2.5, result="lose")
        assert novo.current == pytest.approx(90.0, abs=TOL)

    def test_push_nao_altera_saldo_mas_registra(self) -> None:
        novo = apply_bet_result(_estado(), stake=10.0, odds=2.5, result="push")
        assert novo.current == pytest.approx(100.0, abs=TOL)
        assert len(novo.history) == 1

    def test_estado_original_nao_e_mutado(self) -> None:
        original = _estado()
        apply_bet_result(original, stake=10.0, odds=2.0, result="win")
        assert original.current == pytest.approx(100.0)
        assert original.history == []

    def test_dataclass_e_congelada(self) -> None:
        estado = _estado()
        with pytest.raises(dataclasses.FrozenInstanceError):
            estado.current = 0.0  # type: ignore[misc]

    def test_initial_e_preservado_atraves_das_transicoes(self) -> None:
        estado = _estado(current=100.0, initial=100.0)
        for _ in range(3):
            estado = apply_bet_result(estado, stake=5.0, odds=2.0, result="lose")
        assert estado.initial == pytest.approx(100.0)
        assert estado.history == [pytest.approx(v) for v in (95.0, 90.0, 85.0)]

    def test_resultado_invalido_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="resultado invalido"):
            apply_bet_result(_estado(), 10.0, 2.0, "void")  # type: ignore[arg-type]

    def test_stake_negativo_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="stake"):
            apply_bet_result(_estado(), -1.0, 2.0, "win")

    def test_odds_abaixo_de_um_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="odds"):
            apply_bet_result(_estado(), 10.0, 0.9, "win")

    def test_win_com_odds_um_e_neutro(self) -> None:
        novo = apply_bet_result(_estado(), stake=10.0, odds=1.0, result="win")
        assert novo.current == pytest.approx(100.0, abs=TOL)


class TestEquityCurve:
    def test_caminho_feliz(self) -> None:
        df = pd.DataFrame(
            {
                "stake": [10.0, 10.0, 20.0],
                "odds": [2.0, 3.0, 1.5],
                "result": ["win", "lose", "push"],
            }
        )
        curva = equity_curve(df, initial=100.0)
        assert list(curva) == [
            pytest.approx(110.0),
            pytest.approx(100.0),
            pytest.approx(100.0),
        ]

    def test_default_initial_zero_da_pnl_acumulado(self) -> None:
        df = pd.DataFrame({"stake": [10.0], "odds": [2.0], "result": ["win"]})
        assert equity_curve(df).iloc[-1] == pytest.approx(10.0)

    def test_preserva_indice_do_dataframe(self) -> None:
        df = pd.DataFrame(
            {"stake": [5.0, 5.0], "odds": [2.0, 2.0], "result": ["win", "win"]},
            index=[7, 42],
        )
        assert list(equity_curve(df).index) == [7, 42]

    def test_df_vazio_retorna_serie_vazia(self) -> None:
        df = pd.DataFrame(columns=["stake", "odds", "result"])
        curva = equity_curve(df)
        assert curva.empty
        assert curva.dtype == float

    def test_coluna_faltante_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="colunas ausentes"):
            equity_curve(pd.DataFrame({"stake": [1.0], "odds": [2.0]}))

    def test_resultado_invalido_levanta_erro(self) -> None:
        df = pd.DataFrame({"stake": [1.0], "odds": [2.0], "result": ["cancel"]})
        with pytest.raises(ValueError, match="resultados invalidos"):
            equity_curve(df)

    def test_propriedade_final_igual_soma_dos_pnls(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        stakes = rng.uniform(1.0, 20.0, n)
        odds = rng.uniform(1.1, 5.0, n)
        results = rng.choice(["win", "lose", "push"], n)
        df = pd.DataFrame({"stake": stakes, "odds": odds, "result": results})
        pnl_esperado = sum(
            s * (o - 1.0) if r == "win" else (-s if r == "lose" else 0.0)
            for s, o, r in zip(stakes, odds, results, strict=True)
        )
        curva = equity_curve(df, initial=1000.0)
        assert curva.iloc[-1] == pytest.approx(1000.0 + pnl_esperado, rel=1e-12)

    def test_so_derrotas_e_monotona_decrescente(self) -> None:
        df = pd.DataFrame({"stake": [5.0] * 4, "odds": [2.0] * 4, "result": ["lose"] * 4})
        curva = equity_curve(df, initial=100.0)
        assert curva.is_monotonic_decreasing

    def test_consistencia_com_apply_bet_result(self) -> None:
        df = pd.DataFrame(
            {
                "stake": [10.0, 15.0, 5.0],
                "odds": [1.8, 2.2, 3.0],
                "result": ["win", "lose", "win"],
            }
        )
        estado = _estado(current=100.0, initial=100.0)
        for linha in df.itertuples(index=False):
            estado = apply_bet_result(
                estado,
                float(linha.stake),
                float(linha.odds),
                str(linha.result),  # type: ignore[arg-type]
            )
        curva = equity_curve(df, initial=100.0)
        assert list(curva) == [pytest.approx(v) for v in estado.history]


def _dia(match_ids: list[str], stakes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"match_id": match_ids, "stake": stakes})


class TestCheckExposure:
    def test_aprova_dentro_dos_limites(self) -> None:
        dia = _dia(["m1"], [10.0])
        nova: NovaAposta = {"match_id": "m2", "stake": 10.0}
        ok, motivo = check_exposure(dia, nova, 1000.0, 0.05, 0.10)
        assert ok
        assert "aprovada" in motivo

    def test_rejeita_por_limite_do_jogo(self) -> None:
        dia = _dia(["m1"], [40.0])
        nova: NovaAposta = {"match_id": "m1", "stake": 20.0}
        ok, motivo = check_exposure(dia, nova, 1000.0, 0.05, 0.50)
        assert not ok
        assert "jogo" in motivo

    def test_rejeita_por_limite_do_dia(self) -> None:
        dia = _dia(["m1", "m2"], [40.0, 40.0])
        nova: NovaAposta = {"match_id": "m3", "stake": 30.0}
        ok, motivo = check_exposure(dia, nova, 1000.0, 0.05, 0.10)
        assert not ok
        assert "diario" in motivo

    def test_limite_do_jogo_ignora_outros_jogos(self) -> None:
        dia = _dia(["m1", "m2"], [45.0, 45.0])
        nova: NovaAposta = {"match_id": "m3", "stake": 50.0}
        ok, _ = check_exposure(dia, nova, 1000.0, 0.05, 0.50)
        assert ok

    def test_exatamente_no_limite_e_aprovada(self) -> None:
        dia = _dia(["m1"], [30.0])
        nova: NovaAposta = {"match_id": "m1", "stake": 20.0}
        ok, _ = check_exposure(dia, nova, 1000.0, 0.05, 1.0)
        assert ok

    def test_dia_vazio_compara_so_a_nova(self) -> None:
        vazio = pd.DataFrame(columns=["match_id", "stake"])
        nova: NovaAposta = {"match_id": "m1", "stake": 60.0}
        ok, motivo = check_exposure(vazio, nova, 1000.0, 0.05, 0.10)
        assert not ok
        assert "jogo" in motivo

    def test_bankroll_nao_positivo_levanta_erro(self) -> None:
        nova: NovaAposta = {"match_id": "m1", "stake": 1.0}
        with pytest.raises(ValueError, match="bankroll"):
            check_exposure(_dia([], []), nova, 0.0, 0.05, 0.10)

    def test_stake_negativo_levanta_erro(self) -> None:
        nova: NovaAposta = {"match_id": "m1", "stake": -1.0}
        with pytest.raises(ValueError, match="stake"):
            check_exposure(_dia([], []), nova, 1000.0, 0.05, 0.10)

    def test_coluna_faltante_levanta_erro(self) -> None:
        dia = pd.DataFrame({"match_id": ["m1"]})
        nova: NovaAposta = {"match_id": "m1", "stake": 1.0}
        with pytest.raises(ValueError, match="colunas ausentes"):
            check_exposure(dia, nova, 1000.0, 0.05, 0.10)


class TestDrawdownGuard:
    def test_abaixo_do_limite_nao_para(self) -> None:
        equity = pd.Series([100.0, 110.0, 100.0])
        assert drawdown_guard(equity, max_dd=0.25) is False

    def test_acima_do_limite_para(self) -> None:
        equity = pd.Series([100.0, 120.0, 80.0])
        assert drawdown_guard(equity, max_dd=0.25) is True

    def test_exatamente_no_limite_para(self) -> None:
        equity = pd.Series([100.0, 75.0])
        assert drawdown_guard(equity, max_dd=0.25) is True

    def test_curva_recuperada_nao_para(self) -> None:
        equity = pd.Series([100.0, 60.0, 105.0])
        assert drawdown_guard(equity, max_dd=0.25) is False

    def test_curva_monotona_crescente_nunca_para(self) -> None:
        equity = pd.Series(np.linspace(100.0, 200.0, 50))
        assert drawdown_guard(equity, max_dd=0.01) is False

    def test_serie_vazia_nao_para(self) -> None:
        assert drawdown_guard(pd.Series(dtype=float)) is False

    def test_banca_zerada_para(self) -> None:
        equity = pd.Series([0.0, 0.0])
        assert drawdown_guard(equity) is True

    def test_max_dd_invalido_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="max_dd"):
            drawdown_guard(pd.Series([100.0]), max_dd=0.0)

    def test_monotonicidade_em_max_dd(self) -> None:
        equity = pd.Series([100.0, 85.0])
        assert drawdown_guard(equity, max_dd=0.10) is True
        assert drawdown_guard(equity, max_dd=0.20) is False


class TestStakeComLimites:
    def test_sem_truncamento_quando_ha_folga(self) -> None:
        stake = stake_com_limites(20.0, 0.0, 0.0, 1000.0, 0.05, 0.10)
        assert stake == pytest.approx(20.0)

    def test_trunca_pela_folga_do_jogo(self) -> None:
        stake = stake_com_limites(40.0, 30.0, 0.0, 1000.0, 0.05, 0.50)
        assert stake == pytest.approx(20.0)

    def test_trunca_pela_folga_do_dia(self) -> None:
        stake = stake_com_limites(40.0, 0.0, 90.0, 1000.0, 0.05, 0.10)
        assert stake == pytest.approx(10.0)

    def test_zero_quando_limites_esgotados(self) -> None:
        stake = stake_com_limites(40.0, 50.0, 100.0, 1000.0, 0.05, 0.10)
        assert stake == 0.0

    def test_zero_quando_exposicao_excede_limite(self) -> None:
        stake = stake_com_limites(40.0, 60.0, 0.0, 1000.0, 0.05, 0.50)
        assert stake == 0.0

    def test_kelly_negativo_vira_zero(self) -> None:
        assert stake_com_limites(-5.0, 0.0, 0.0, 1000.0, 0.05, 0.10) == 0.0

    def test_bankroll_nao_positivo_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="bankroll"):
            stake_com_limites(10.0, 0.0, 0.0, -1.0, 0.05, 0.10)

    def test_propriedade_nunca_excede_kelly_nem_folgas(self) -> None:
        rng = np.random.default_rng(7)
        for _ in range(200):
            kelly = float(rng.uniform(-10.0, 100.0))
            exp_jogo = float(rng.uniform(0.0, 80.0))
            exp_dia = float(rng.uniform(0.0, 150.0))
            stake = stake_com_limites(kelly, exp_jogo, exp_dia, 1000.0, 0.05, 0.10)
            assert stake >= 0.0
            assert stake <= max(kelly, 0.0) + TOL
            assert exp_jogo + stake <= max(0.05 * 1000.0, exp_jogo) + TOL
            assert exp_dia + stake <= max(0.10 * 1000.0, exp_dia) + TOL

    def test_consistencia_com_check_exposure(self) -> None:
        """Um stake truncado por stake_com_limites deve passar em check_exposure."""
        dia = _dia(["m1", "m2"], [30.0, 40.0])
        stake = stake_com_limites(50.0, 30.0, 70.0, 1000.0, 0.05, 0.10)
        nova: NovaAposta = {"match_id": "m1", "stake": stake}
        ok, _ = check_exposure(dia, nova, 1000.0, 0.05, 0.10)
        assert ok
