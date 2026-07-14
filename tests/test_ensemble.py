"""Testes do ensemble: pesos softmax por log-loss e blending em log-odds."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit, logit

from edgefinder.models.ensemble import blend_probs, ensemble_weights

TOL = 1e-9


class TestEnsembleWeights:
    def test_pesos_somam_1(self) -> None:
        w = ensemble_weights({"poisson": 0.62, "elo": 0.65, "mercado": 0.60})
        assert sum(w.values()) == pytest.approx(1.0, abs=TOL)

    def test_pesos_positivos(self) -> None:
        w = ensemble_weights({"a": 0.5, "b": 5.0})
        assert all(wi > 0.0 for wi in w.values())

    def test_menor_logloss_recebe_maior_peso(self) -> None:
        w = ensemble_weights({"bom": 0.55, "medio": 0.65, "ruim": 0.80})
        assert w["bom"] > w["medio"] > w["ruim"]

    def test_loglosses_iguais_dao_pesos_iguais(self) -> None:
        w = ensemble_weights({"a": 0.6, "b": 0.6, "c": 0.6})
        for wi in w.values():
            assert wi == pytest.approx(1.0 / 3.0, abs=TOL)

    def test_modelo_unico_recebe_peso_1(self) -> None:
        assert ensemble_weights({"solo": 0.7}) == {"solo": pytest.approx(1.0)}

    def test_temperatura_alta_aproxima_pesos_iguais(self) -> None:
        w = ensemble_weights({"a": 0.5, "b": 0.9}, temperature=1000.0)
        assert w["a"] == pytest.approx(0.5, abs=1e-3)
        assert w["b"] == pytest.approx(0.5, abs=1e-3)

    def test_temperatura_baixa_concentra_no_melhor(self) -> None:
        w = ensemble_weights({"a": 0.5, "b": 0.9}, temperature=0.01)
        assert w["a"] > 0.999

    def test_temperatura_menor_concentra_mais(self) -> None:
        ll = {"a": 0.5, "b": 0.7}
        w_quente = ensemble_weights(ll, temperature=2.0)
        w_frio = ensemble_weights(ll, temperature=0.5)
        assert w_frio["a"] > w_quente["a"]

    def test_estavel_com_loglosses_enormes(self) -> None:
        # Sem o truque de subtrair o máximo, exp(600) estouraria float64.
        w = ensemble_weights({"a": 600.0, "b": 601.0})
        assert sum(w.values()) == pytest.approx(1.0, abs=TOL)
        assert w["a"] > w["b"]

    def test_dict_vazio_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="vazio"):
            ensemble_weights({})

    def test_temperatura_invalida_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            ensemble_weights({"a": 0.5}, temperature=0.0)
        with pytest.raises(ValueError, match="temperature"):
            ensemble_weights({"a": 0.5}, temperature=-1.0)

    def test_logloss_nao_finito_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="finito"):
            ensemble_weights({"a": float("nan"), "b": 0.5})
        with pytest.raises(ValueError, match="finito"):
            ensemble_weights({"a": float("inf"), "b": 0.5})


class TestBlendProbs:
    def test_modelos_identicos_devolvem_o_proprio_modelo(self) -> None:
        p = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
        blended = blend_probs({"a": p, "b": p.copy()}, {"a": 0.5, "b": 0.5})
        np.testing.assert_allclose(blended, p, atol=1e-9)

    def test_peso_total_em_um_modelo_devolve_esse_modelo(self) -> None:
        pa = np.array([0.2, 0.6, 0.85])
        pb = np.array([0.5, 0.5, 0.5])
        blended = blend_probs({"a": pa, "b": pb}, {"a": 1.0, "b": 0.0})
        np.testing.assert_allclose(blended, pa, atol=1e-9)

    def test_blend_e_media_ponderada_de_logits(self) -> None:
        pa = np.array([0.3, 0.7])
        pb = np.array([0.6, 0.9])
        blended = blend_probs({"a": pa, "b": pb}, {"a": 0.25, "b": 0.75})
        esperado = expit(0.25 * logit(pa) + 0.75 * logit(pb))
        np.testing.assert_allclose(blended, esperado, atol=1e-9)

    def test_blend_fica_entre_o_minimo_e_o_maximo(self) -> None:
        rng = np.random.default_rng(31)
        pa = rng.uniform(0.01, 0.99, size=200)
        pb = rng.uniform(0.01, 0.99, size=200)
        blended = blend_probs({"a": pa, "b": pb}, {"a": 0.4, "b": 0.6})
        assert np.all(blended >= np.minimum(pa, pb) - TOL)
        assert np.all(blended <= np.maximum(pa, pb) + TOL)

    def test_pesos_nao_normalizados_dao_o_mesmo_resultado(self) -> None:
        pa = np.array([0.3, 0.7])
        pb = np.array([0.6, 0.2])
        b1 = blend_probs({"a": pa, "b": pb}, {"a": 0.25, "b": 0.75})
        b2 = blend_probs({"a": pa, "b": pb}, {"a": 1.0, "b": 3.0})
        np.testing.assert_allclose(b1, b2, atol=1e-9)

    def test_fallback_linear_para_prob_degenerada(self) -> None:
        # Em log-odds, p=0 dominaria com certeza infinita; a média linear
        # pondera a certeza degenerada pelo peso do modelo: 0.5*0 + 0.5*0.5.
        pa = np.array([0.0])
        pb = np.array([0.5])
        blended = blend_probs({"a": pa, "b": pb}, {"a": 0.5, "b": 0.5})
        assert blended[0] == pytest.approx(0.25, abs=TOL)

    def test_fallback_aplicado_somente_nas_posicoes_degeneradas(self) -> None:
        pa = np.array([1.0, 0.4])
        pb = np.array([0.6, 0.8])
        blended = blend_probs({"a": pa, "b": pb}, {"a": 0.5, "b": 0.5})
        assert blended[0] == pytest.approx(0.8, abs=TOL)
        esperado = float(expit(0.5 * logit(0.4) + 0.5 * logit(0.8)))
        assert blended[1] == pytest.approx(esperado, abs=TOL)

    def test_saida_sempre_em_01(self) -> None:
        rng = np.random.default_rng(37)
        probs = {f"m{i}": rng.uniform(size=100) for i in range(4)}
        pesos = {f"m{i}": float(rng.uniform(0.1, 1.0)) for i in range(4)}
        blended = blend_probs(probs, pesos)
        assert np.all(blended >= 0.0)
        assert np.all(blended <= 1.0)

    def test_aumentar_peso_do_modelo_mais_alto_sobe_o_blend(self) -> None:
        pa = np.array([0.3])
        pb = np.array([0.8])
        baixo = blend_probs({"a": pa, "b": pb}, {"a": 0.7, "b": 0.3})
        alto = blend_probs({"a": pa, "b": pb}, {"a": 0.3, "b": 0.7})
        assert alto[0] > baixo[0]

    def test_2d_renormaliza_linhas_para_somar_1(self) -> None:
        pa = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
        pb = np.array([[0.6, 0.25, 0.15], [0.1, 0.4, 0.5]])
        blended = blend_probs({"a": pa, "b": pb}, {"a": 0.5, "b": 0.5})
        assert blended.shape == (2, 3)
        np.testing.assert_allclose(blended.sum(axis=1), np.ones(2), atol=1e-9)

    def test_2d_modelos_identicos_preservam_a_distribuicao(self) -> None:
        p = np.array([[0.5, 0.3, 0.2], [0.25, 0.25, 0.5]])
        blended = blend_probs({"a": p, "b": p.copy()}, {"a": 0.3, "b": 0.7})
        np.testing.assert_allclose(blended, p, atol=1e-9)

    def test_integracao_com_ensemble_weights(self) -> None:
        pa = np.array([0.55, 0.30])
        pb = np.array([0.45, 0.40])
        w = ensemble_weights({"a": 0.60, "b": 0.66})
        blended = blend_probs({"a": pa, "b": pb}, w)
        assert blended.shape == (2,)
        assert np.all((blended > 0.0) & (blended < 1.0))
        # O modelo "a" tem log-loss menor, logo o blend fica mais perto dele.
        assert abs(blended[0] - pa[0]) < abs(blended[0] - pb[0])

    def test_probs_vazio_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="vazio"):
            blend_probs({}, {})

    def test_chaves_divergentes_levantam_erro(self) -> None:
        with pytest.raises(ValueError, match="chaves"):
            blend_probs({"a": np.array([0.5])}, {"b": 1.0})

    def test_shapes_incompativeis_levantam_erro(self) -> None:
        with pytest.raises(ValueError, match="shapes"):
            blend_probs({"a": np.array([0.5, 0.6]), "b": np.array([0.5])}, {"a": 0.5, "b": 0.5})

    def test_prob_fora_de_01_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            blend_probs({"a": np.array([1.2])}, {"a": 1.0})

    def test_prob_nan_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            blend_probs({"a": np.array([np.nan])}, {"a": 1.0})

    def test_peso_negativo_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            blend_probs({"a": np.array([0.5]), "b": np.array([0.5])}, {"a": -0.5, "b": 1.5})

    def test_pesos_todos_zero_levantam_erro(self) -> None:
        with pytest.raises(ValueError, match="> 0"):
            blend_probs({"a": np.array([0.5])}, {"a": 0.0})

    def test_array_3d_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="1-D ou 2-D"):
            blend_probs({"a": np.full((2, 2, 2), 0.5)}, {"a": 1.0})

    def test_array_vazio_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="vazio"):
            blend_probs({"a": np.array([])}, {"a": 1.0})
