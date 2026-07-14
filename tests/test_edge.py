"""Testes de EV, Kelly e correlação entre pernas de múltiplas."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from edgefinder.edge.correlation import joint_prob_gaussian_copula, parlay_ev
from edgefinder.edge.ev import ev_table, expected_value
from edgefinder.edge.kelly import kelly_fraction, kelly_stake

TOL = 1e-9
TOL_COPULA = 1e-5


class TestExpectedValue:
    def test_preco_justo_tem_ev_zero(self) -> None:
        assert float(expected_value(0.5, 2.0)) == pytest.approx(0.0, abs=TOL)

    def test_edge_positivo(self) -> None:
        assert float(expected_value(0.55, 2.0)) == pytest.approx(0.10, abs=TOL)

    def test_equivale_a_p_vezes_odds_menos_um(self) -> None:
        rng = np.random.default_rng(0)
        p = rng.uniform(0.05, 0.95, size=50)
        odds = rng.uniform(1.1, 12.0, size=50)
        assert expected_value(p, odds) == pytest.approx(p * odds - 1.0, abs=TOL)

    def test_vetorizado(self) -> None:
        ev = expected_value(np.array([0.5, 0.6]), np.array([2.0, 2.0]))
        assert ev == pytest.approx([0.0, 0.2], abs=TOL)

    def test_entradas_invalidas_levantam_erro(self) -> None:
        with pytest.raises(ValueError, match="p_model"):
            expected_value(1.2, 2.0)
        with pytest.raises(ValueError, match="odds"):
            expected_value(0.5, 1.0)


class TestEvTable:
    def test_adiciona_coluna_ev_sem_mutar_original(self) -> None:
        df = pd.DataFrame({"p_model": [0.5, 0.6], "odds": [2.0, 2.0], "id": [1, 2]})
        resultado = ev_table(df)
        assert list(resultado["ev"]) == pytest.approx([0.0, 0.2], abs=TOL)
        assert "ev" not in df.columns
        assert list(resultado["id"]) == [1, 2]

    def test_coluna_faltando_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="p_model"):
            ev_table(pd.DataFrame({"odds": [2.0]}))


class TestKellyFraction:
    def test_valor_conhecido(self) -> None:
        assert float(kelly_fraction(0.5, 2.2)) == pytest.approx(0.1 / 1.2, abs=TOL)

    def test_zero_quando_ev_nao_positivo(self) -> None:
        assert float(kelly_fraction(0.5, 2.0)) == pytest.approx(0.0, abs=TOL)
        assert float(kelly_fraction(0.4, 2.0)) == pytest.approx(0.0, abs=TOL)

    def test_positivo_sse_ev_positivo(self) -> None:
        rng = np.random.default_rng(1)
        p = rng.uniform(0.05, 0.95, size=200)
        odds = rng.uniform(1.1, 12.0, size=200)
        f = kelly_fraction(p, odds)
        ev = expected_value(p, odds)
        assert np.all((f > 0.0) == (ev > 0.0))
        assert np.all(f >= 0.0)

    def test_monotono_em_p(self) -> None:
        p = np.linspace(0.55, 0.95, 9)
        f = kelly_fraction(p, np.full_like(p, 2.0))
        assert np.all(np.diff(f) > 0.0)

    def test_entradas_invalidas_levantam_erro(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            kelly_fraction(-0.1, 2.0)
        with pytest.raises(ValueError, match="odds"):
            kelly_fraction(0.5, 0.9)


class TestKellyStake:
    def test_fracao_limita_quando_edge_pequeno(self) -> None:
        stake = kelly_stake(0.52, 2.0, bankroll=1000.0)
        assert stake == pytest.approx(1000.0 * 0.25 * 0.04, abs=TOL)

    def test_cap_limita_quando_edge_grande(self) -> None:
        stake = kelly_stake(0.9, 2.0, bankroll=1000.0)
        assert stake == pytest.approx(1000.0 * 0.05, abs=TOL)

    def test_zero_sem_edge(self) -> None:
        assert kelly_stake(0.5, 2.0, bankroll=1000.0) == pytest.approx(0.0, abs=TOL)

    def test_linear_na_banca(self) -> None:
        s1 = kelly_stake(0.55, 2.0, bankroll=100.0)
        s2 = kelly_stake(0.55, 2.0, bankroll=200.0)
        assert s2 == pytest.approx(2.0 * s1, abs=TOL)

    def test_parametros_invalidos_levantam_erro(self) -> None:
        with pytest.raises(ValueError, match="bankroll"):
            kelly_stake(0.55, 2.0, bankroll=-1.0)
        with pytest.raises(ValueError, match="fraction"):
            kelly_stake(0.55, 2.0, bankroll=100.0, fraction=0.0)
        with pytest.raises(ValueError, match="fraction"):
            kelly_stake(0.55, 2.0, bankroll=100.0, fraction=1.5)
        with pytest.raises(ValueError, match="cap"):
            kelly_stake(0.55, 2.0, bankroll=100.0, cap=0.0)


class TestJointProbGaussianCopula:
    def test_rho_zero_e_o_produto(self) -> None:
        casos = [(0.5, 0.5), (0.7, 0.2), (0.9, 0.9), (0.1, 0.3)]
        for p_a, p_b in casos:
            conjunta = joint_prob_gaussian_copula(p_a, p_b, 0.0)
            assert conjunta == pytest.approx(p_a * p_b, abs=TOL_COPULA)

    def test_rho_um_e_o_minimo(self) -> None:
        assert joint_prob_gaussian_copula(0.6, 0.4, 1.0) == pytest.approx(0.4, abs=TOL)

    def test_rho_menos_um_e_a_cota_inferior_de_frechet(self) -> None:
        assert joint_prob_gaussian_copula(0.6, 0.7, -1.0) == pytest.approx(0.3, abs=TOL)
        assert joint_prob_gaussian_copula(0.2, 0.3, -1.0) == pytest.approx(0.0, abs=TOL)

    def test_probabilidades_de_borda(self) -> None:
        assert joint_prob_gaussian_copula(0.0, 0.7, 0.5) == pytest.approx(0.0, abs=TOL)
        assert joint_prob_gaussian_copula(1.0, 0.7, 0.5) == pytest.approx(0.7, abs=TOL)
        assert joint_prob_gaussian_copula(0.7, 1.0, -0.5) == pytest.approx(0.7, abs=TOL)

    def test_monotona_em_rho(self) -> None:
        rhos = [-0.8, -0.4, 0.0, 0.4, 0.8]
        conjuntas = [joint_prob_gaussian_copula(0.6, 0.5, r) for r in rhos]
        assert all(b > a for a, b in itertools.pairwise(conjuntas))

    def test_dentro_das_cotas_de_frechet(self) -> None:
        rng = np.random.default_rng(2)
        for _ in range(30):
            p_a = float(rng.uniform(0.05, 0.95))
            p_b = float(rng.uniform(0.05, 0.95))
            rho = float(rng.uniform(-0.95, 0.95))
            conjunta = joint_prob_gaussian_copula(p_a, p_b, rho)
            assert conjunta >= max(0.0, p_a + p_b - 1.0) - 1e-6
            assert conjunta <= min(p_a, p_b) + 1e-6

    def test_simetrica_nos_argumentos(self) -> None:
        assert joint_prob_gaussian_copula(0.7, 0.2, 0.3) == pytest.approx(
            joint_prob_gaussian_copula(0.2, 0.7, 0.3), abs=1e-7
        )

    def test_parametros_invalidos_levantam_erro(self) -> None:
        with pytest.raises(ValueError, match="rho"):
            joint_prob_gaussian_copula(0.5, 0.5, 1.5)
        with pytest.raises(ValueError, match="p_a"):
            joint_prob_gaussian_copula(-0.1, 0.5, 0.0)
        with pytest.raises(ValueError, match="p_b"):
            joint_prob_gaussian_copula(0.5, 1.1, 0.0)


class TestParlayEv:
    def test_independente_usa_o_produto(self) -> None:
        assert parlay_ev([0.5, 0.5], 4.0) == pytest.approx(0.0, abs=TOL)
        assert parlay_ev([0.5, 0.5, 0.5], 8.0) == pytest.approx(0.0, abs=TOL)

    def test_rho_zero_equivale_a_independencia(self) -> None:
        rho0 = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert parlay_ev([0.6, 0.5], 3.5, rho_matrix=rho0) == pytest.approx(
            parlay_ev([0.6, 0.5], 3.5), abs=1e-4
        )

    def test_correlacao_positiva_aumenta_o_ev(self) -> None:
        pos = np.array([[1.0, 0.5], [0.5, 1.0]])
        neg = np.array([[1.0, -0.5], [-0.5, 1.0]])
        ev_ind = parlay_ev([0.6, 0.5], 3.4)
        assert parlay_ev([0.6, 0.5], 3.4, rho_matrix=pos) > ev_ind
        assert parlay_ev([0.6, 0.5], 3.4, rho_matrix=neg) < ev_ind

    def test_rho_matrix_com_tres_pernas_levanta_erro(self) -> None:
        rho0 = np.eye(2)
        with pytest.raises(ValueError, match="2 pernas"):
            parlay_ev([0.5, 0.5, 0.5], 8.0, rho_matrix=rho0)

    def test_rho_matrix_invalida_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="2x2"):
            parlay_ev([0.5, 0.5], 4.0, rho_matrix=np.eye(3))
        with pytest.raises(ValueError, match="simetrica"):
            parlay_ev([0.5, 0.5], 4.0, rho_matrix=np.array([[1.0, 0.3], [0.1, 1.0]]))
        with pytest.raises(ValueError, match="simetrica"):
            parlay_ev([0.5, 0.5], 4.0, rho_matrix=np.array([[0.9, 0.3], [0.3, 0.9]]))

    def test_entradas_invalidas_levantam_erro(self) -> None:
        with pytest.raises(ValueError, match="2 pernas"):
            parlay_ev([0.5], 2.0)
        with pytest.raises(ValueError, match="odds_parlay"):
            parlay_ev([0.5, 0.5], 1.0)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            parlay_ev([0.5, 1.2], 4.0)
