"""Testes de consenso entre casas e de CLV (Closing Line Value)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edgefinder.market.clv import ClvReport, clv, clv_report
from edgefinder.market.consensus import consensus_prob

TOL = 1e-9


class TestConsensusProb:
    def test_uma_casa_devolve_probs_renormalizadas(self) -> None:
        p = consensus_prob({"pinnacle": np.array([0.6, 0.4])})
        assert p == pytest.approx([0.6, 0.4], abs=TOL)

    def test_casas_identicas_devolvem_o_mesmo(self) -> None:
        probs = np.array([0.5, 0.3, 0.2])
        p = consensus_prob({"a": probs, "b": probs.copy(), "c": probs.copy()})
        assert p == pytest.approx(probs, abs=TOL)

    def test_peso_default_puxa_para_pinnacle(self) -> None:
        p = consensus_prob({"pinnacle": np.array([0.6, 0.4]), "soft": np.array([0.5, 0.5])})
        assert p == pytest.approx([0.575, 0.425], abs=TOL)

    def test_market_avg_tem_peso_intermediario(self) -> None:
        p = consensus_prob({"market_avg": np.array([0.6, 0.4]), "soft": np.array([0.5, 0.5])})
        assert p == pytest.approx([1.4 / 2.5, 1.1 / 2.5], abs=TOL)

    def test_nome_com_maiuscula_recebe_peso_default(self) -> None:
        p = consensus_prob({"Pinnacle": np.array([0.6, 0.4]), "soft": np.array([0.5, 0.5])})
        assert p == pytest.approx([0.575, 0.425], abs=TOL)

    def test_pesos_explicitos_substituem_default(self) -> None:
        p = consensus_prob(
            {"pinnacle": np.array([0.6, 0.4]), "soft": np.array([0.5, 0.5])},
            weights={"pinnacle": 1.0, "soft": 1.0},
        )
        assert p == pytest.approx([0.55, 0.45], abs=TOL)

    def test_resultado_soma_um_mesmo_com_entradas_nao_normalizadas(self) -> None:
        p = consensus_prob({"a": np.array([0.62, 0.42]), "b": np.array([0.55, 0.51])})
        assert float(p.sum()) == pytest.approx(1.0, abs=TOL)

    def test_dict_vazio_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="vazio"):
            consensus_prob({})

    def test_tamanhos_diferentes_levantam_erro(self) -> None:
        with pytest.raises(ValueError, match="tamanho"):
            consensus_prob({"a": np.array([0.6, 0.4]), "b": np.array([0.3, 0.3, 0.4])})

    def test_prob_invalida_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="invalido"):
            consensus_prob({"a": np.array([0.6, np.nan])})
        with pytest.raises(ValueError, match="invalido"):
            consensus_prob({"a": np.array([-0.1, 1.1])})

    def test_peso_nao_positivo_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="> 0"):
            consensus_prob({"a": np.array([0.6, 0.4])}, weights={"a": 0.0})


class TestClv:
    def test_formula_escalar(self) -> None:
        assert float(clv(2.10, 2.00)) == pytest.approx(0.05, abs=TOL)

    def test_vetorizado(self) -> None:
        resultado = clv(np.array([2.1, 2.0, 1.9]), np.array([2.0, 2.0, 2.0]))
        assert resultado == pytest.approx([0.05, 0.0, -0.05], abs=TOL)

    def test_positivo_quando_vence_o_fechamento(self) -> None:
        assert float(clv(2.2, 2.0)) > 0.0
        assert float(clv(1.9, 2.0)) < 0.0

    def test_odds_invalidas_levantam_erro(self) -> None:
        with pytest.raises(ValueError, match="odds_taken"):
            clv(1.0, 2.0)
        with pytest.raises(ValueError, match="closing_odds_fair"):
            clv(2.0, 0.9)
        with pytest.raises(ValueError, match="NaN ou infinito"):
            clv(np.nan, 2.0)


class TestClvReport:
    def _df(self) -> pd.DataFrame:
        return pd.DataFrame({"odds_taken": [2.1, 2.0, 1.9], "closing_fair": [2.0, 2.0, 2.0]})

    def test_estatisticas_conhecidas(self) -> None:
        rel = clv_report(self._df())
        assert rel.mean == pytest.approx(0.0, abs=TOL)
        assert rel.median == pytest.approx(0.0, abs=TOL)
        assert rel.pct_positive == pytest.approx(1.0 / 3.0, abs=TOL)
        assert rel.n == 3

    def test_ic_bootstrap_contem_a_media(self) -> None:
        rel = clv_report(self._df())
        assert rel.ci_low <= rel.mean <= rel.ci_high

    def test_clv_constante_colapsa_o_ic(self) -> None:
        df = pd.DataFrame({"odds_taken": [2.1, 2.1], "closing_fair": [2.0, 2.0]})
        rel = clv_report(df)
        assert rel.ci_low == pytest.approx(0.05, abs=TOL)
        assert rel.ci_high == pytest.approx(0.05, abs=TOL)
        assert rel.pct_positive == pytest.approx(1.0, abs=TOL)

    def test_deterministico_com_mesma_semente(self) -> None:
        assert clv_report(self._df(), seed=7) == clv_report(self._df(), seed=7)

    def test_devolve_dataclass_tipada(self) -> None:
        assert isinstance(clv_report(self._df()), ClvReport)

    def test_coluna_faltando_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="closing_fair"):
            clv_report(pd.DataFrame({"odds_taken": [2.0]}))

    def test_df_vazio_levanta_erro(self) -> None:
        df = pd.DataFrame({"odds_taken": [], "closing_fair": []})
        with pytest.raises(ValueError, match="vazio"):
            clv_report(df)

    def test_n_boot_invalido_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="n_boot"):
            clv_report(self._df(), n_boot=0)
