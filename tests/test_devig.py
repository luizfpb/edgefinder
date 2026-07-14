"""Testes do devig: probabilidades implícitas e remoção do overround."""

from __future__ import annotations

import numpy as np
import pytest

from edgefinder.market.devig import (
    devig,
    devig_additive,
    devig_proportional,
    devig_shin,
    implied_probs,
    shin_z,
)

TOL = 1e-9


def _odds_shin_sinteticas(p_true: np.ndarray, z: float) -> np.ndarray:
    """Gera odds pelo modelo de Shin invertido, com z conhecido.

    Da fórmula de devig, q_i = sqrt(B * p_i * (z + (1 - z) * p_i)) com
    B = (sum_i sqrt(p_i * (z + (1 - z) * p_i)))^2.
    """
    s = np.sqrt(p_true * (z + (1.0 - z) * p_true))
    b = float(s.sum()) ** 2
    q = np.sqrt(b) * s
    resultado: np.ndarray = 1.0 / q
    return resultado


class TestImpliedProbs:
    def test_odds_pares_dao_meio_a_meio(self) -> None:
        q = implied_probs(np.array([2.0, 2.0]))
        assert q == pytest.approx([0.5, 0.5], abs=TOL)

    def test_odds_com_margem_somam_mais_que_um(self) -> None:
        q = implied_probs(np.array([1.9, 1.9]))
        assert float(q.sum()) > 1.0

    @pytest.mark.parametrize("ruim", [[1.0, 2.0], [0.5, 2.0], [-2.0, 2.0]])
    def test_odds_menor_igual_a_um_levanta_erro(self, ruim: list[float]) -> None:
        with pytest.raises(ValueError, match=r"> 1\.0"):
            implied_probs(np.array(ruim))

    def test_nan_e_infinito_levantam_erro(self) -> None:
        with pytest.raises(ValueError, match="NaN ou infinito"):
            implied_probs(np.array([2.0, np.nan]))
        with pytest.raises(ValueError, match="NaN ou infinito"):
            implied_probs(np.array([2.0, np.inf]))

    def test_vazio_e_2d_levantam_erro(self) -> None:
        with pytest.raises(ValueError, match="vazio"):
            implied_probs(np.array([]))
        with pytest.raises(ValueError, match="1-D"):
            implied_probs(np.array([[2.0, 3.0]]))


class TestDevigProportional:
    def test_recupera_p_verdadeiro_de_margem_multiplicativa(self) -> None:
        p_true = np.array([0.55, 0.25, 0.20])
        odds = 1.0 / (p_true * 1.05)
        assert devig_proportional(odds) == pytest.approx(p_true, abs=1e-12)

    def test_soma_um(self) -> None:
        p = devig_proportional(np.array([1.85, 3.6, 4.5]))
        assert float(p.sum()) == pytest.approx(1.0, abs=TOL)

    def test_preserva_ordenacao(self) -> None:
        odds = np.array([1.5, 3.0, 8.0])
        p = devig_proportional(odds)
        assert p[0] > p[1] > p[2]


class TestDevigAdditive:
    def test_recupera_p_verdadeiro_de_margem_aditiva(self) -> None:
        p_true = np.array([0.55, 0.25, 0.20])
        odds = 1.0 / (p_true + 0.06 / 3)
        assert devig_additive(odds) == pytest.approx(p_true, abs=1e-12)

    def test_soma_um(self) -> None:
        p = devig_additive(np.array([1.85, 3.6, 4.5]))
        assert float(p.sum()) == pytest.approx(1.0, abs=TOL)

    def test_azarao_extremo_levanta_erro(self) -> None:
        odds = np.array([1.3, 6.0, 8.0, 60.0])
        with pytest.raises(ValueError, match="probabilidade <= 0"):
            devig_additive(odds)
        p_fallback = devig_proportional(odds)
        assert np.all(p_fallback > 0.0)


class TestDevigShin:
    def test_dois_resultados_equilibrados_igual_proporcional(self) -> None:
        odds = np.array([1.9, 1.9])
        assert devig_shin(odds) == pytest.approx([0.5, 0.5], abs=TOL)
        assert devig_shin(odds) == pytest.approx(devig_proportional(odds), abs=TOL)

    def test_dois_resultados_quase_equilibrados_proximo_do_proporcional(self) -> None:
        odds = np.array([1.87, 1.95])
        assert devig_shin(odds) == pytest.approx(devig_proportional(odds), abs=5e-3)

    def test_recupera_p_e_z_de_odds_sinteticas(self) -> None:
        p_true = np.array([0.5, 0.3, 0.2])
        z_true = 0.05
        odds = _odds_shin_sinteticas(p_true, z_true)
        assert devig_shin(odds) == pytest.approx(p_true, abs=1e-8)
        assert shin_z(odds) == pytest.approx(z_true, abs=1e-8)

    def test_z_nao_negativo_e_menor_que_um(self) -> None:
        casos = [
            np.array([1.9, 1.9]),
            np.array([1.5, 4.2, 7.0]),
            np.array([1.05, 15.0]),
            np.array([2.05, 2.05]),
        ]
        for odds in casos:
            z = shin_z(odds)
            assert 0.0 <= z < 1.0

    def test_sem_margem_devolve_probs_implicitas_e_z_zero(self) -> None:
        odds = np.array([2.0, 4.0, 4.0])
        assert shin_z(odds) == pytest.approx(0.0, abs=TOL)
        assert devig_shin(odds) == pytest.approx([0.5, 0.25, 0.25], abs=TOL)

    def test_soma_menor_que_um_normaliza_sem_quebrar(self) -> None:
        p = devig_shin(np.array([2.2, 2.2]))
        assert p == pytest.approx([0.5, 0.5], abs=TOL)

    def test_corrige_vies_favorito_azarao(self) -> None:
        odds = np.array([1.05, 15.0])
        p_shin = devig_shin(odds)
        p_prop = devig_proportional(odds)
        assert p_shin[1] < p_prop[1]
        assert p_shin[0] > p_prop[0]

    def test_soma_um_e_preserva_ordenacao(self) -> None:
        odds = np.array([1.45, 4.4, 7.5])
        p = devig_shin(odds)
        assert float(p.sum()) == pytest.approx(1.0, abs=TOL)
        assert p[0] > p[1] > p[2]
        assert np.all(p > 0.0)


class TestDevigDispatcher:
    def test_default_e_shin(self) -> None:
        odds = np.array([1.5, 4.0, 8.0])
        assert devig(odds) == pytest.approx(devig_shin(odds), abs=TOL)

    def test_roteia_para_cada_metodo(self) -> None:
        odds = np.array([1.85, 3.6, 4.5])
        assert devig(odds, method="proportional") == pytest.approx(
            devig_proportional(odds), abs=TOL
        )
        assert devig(odds, method="additive") == pytest.approx(devig_additive(odds), abs=TOL)
        assert devig(odds, method="shin") == pytest.approx(devig_shin(odds), abs=TOL)

    def test_metodo_desconhecido_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="desconhecido"):
            devig(np.array([1.9, 1.9]), method="power")
