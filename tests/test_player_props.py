"""Testes de player_props: shrinkage, convergência, preditiva NB e comparação de modelos."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from scipy.stats import nbinom, poisson

from edgefinder.models.player_props import (
    adjusted_rate,
    fit_population,
    model_comparison,
    posterior_rate,
    prob_over,
    prob_over_poisson,
)

FloatArray = npt.NDArray[np.float64]


def _cell(df: pd.DataFrame, row: str, col: str) -> float:
    """Acessa uma célula como float (o .at dos stubs devolve um Scalar amplo)."""
    value = df.at[row, col]
    assert isinstance(value, float)
    return float(value)


def _simulate_players(
    rng: np.random.Generator,
    alpha: float,
    beta: float,
    n_players: int,
    exposure_range: tuple[float, float],
) -> tuple[FloatArray, FloatArray]:
    """Gera (contagem total, exposição total) por jogador sob a hierarquia."""
    theta = rng.gamma(shape=alpha, scale=1.0 / beta, size=n_players)
    e = rng.uniform(*exposure_range, size=n_players)
    x = rng.poisson(theta * e).astype(np.float64)
    return x, e.astype(np.float64)


class TestPosteriorRate:
    def test_formula_conjugada_exata(self) -> None:
        a, b = posterior_rate(6.0, 2.0, alpha=2.0, beta=2.0)
        assert a == pytest.approx(8.0)
        assert b == pytest.approx(4.0)
        assert a / b == pytest.approx(2.0)

    def test_sem_dados_devolve_o_prior(self) -> None:
        a, b = posterior_rate(0.0, 0.0, alpha=3.0, beta=1.5)
        assert (a, b) == (3.0, 1.5)

    def test_shrinkage_amostra_pequena_encolhe_mais(self) -> None:
        # Mesma taxa bruta (3.0 por 90 min), prior com média 1.0: o jogador de
        # 2 jogos deve ser puxado para a média populacional muito mais que o
        # de 50 jogos.
        alpha, beta = 2.0, 2.0
        prior_mean = alpha / beta
        raw_rate = 3.0
        a2, b2 = posterior_rate(raw_rate * 2.0, 2.0, alpha, beta)
        a50, b50 = posterior_rate(raw_rate * 50.0, 50.0, alpha, beta)
        mean2 = a2 / b2
        mean50 = a50 / b50
        assert abs(mean2 - prior_mean) < abs(mean50 - prior_mean)
        shrink2 = (raw_rate - mean2) / (raw_rate - prior_mean)
        shrink50 = (raw_rate - mean50) / (raw_rate - prior_mean)
        assert shrink2 > 5.0 * shrink50
        assert prior_mean < mean2 < mean50 < raw_rate

    def test_convergencia_para_taxa_verdadeira(self) -> None:
        rng = np.random.default_rng(42)
        theta_true = 1.7
        e = rng.uniform(0.5, 1.0, size=4000)
        x = rng.poisson(theta_true * e)
        a, b = posterior_rate(float(x.sum()), float(e.sum()), alpha=1.0, beta=1.0)
        assert a / b == pytest.approx(theta_true, rel=0.05)

    def test_validacoes(self) -> None:
        with pytest.raises(ValueError):
            posterior_rate(-1.0, 1.0, 1.0, 1.0)
        with pytest.raises(ValueError):
            posterior_rate(1.0, -1.0, 1.0, 1.0)
        with pytest.raises(ValueError):
            posterior_rate(1.0, 1.0, 0.0, 1.0)
        with pytest.raises(ValueError):
            posterior_rate(1.0, 1.0, 1.0, -2.0)
        with pytest.raises(ValueError):
            posterior_rate(float("nan"), 1.0, 1.0, 1.0)


class TestFitPopulation:
    def test_recupera_momentos_em_dados_sinteticos(self) -> None:
        rng = np.random.default_rng(7)
        alpha_true, beta_true = 4.0, 2.0
        x, e = _simulate_players(rng, alpha_true, beta_true, n_players=3000, exposure_range=(5, 40))
        alpha, beta = fit_population(x, e)
        assert alpha / beta == pytest.approx(alpha_true / beta_true, rel=0.05)
        assert alpha / beta**2 == pytest.approx(alpha_true / beta_true**2, rel=0.25)

    def test_sem_heterogeneidade_gera_prior_concentrado(self) -> None:
        # Taxas idênticas (2.0 exato) => variância entre jogadores nula: o piso
        # numérico entra e o prior fica quase degenerado na média (shrinkage
        # quase total), preservando alpha/beta = m.
        e = np.array([2.0, 5.0, 10.0, 20.0])
        x = 2.0 * e
        alpha, beta = fit_population(x, e)
        assert alpha / beta == pytest.approx(2.0)
        assert alpha > 1e5

    def test_exposicao_pequena_nao_infla_heterogeneidade(self) -> None:
        # Todos os jogadores com o MESMO theta, mas metade com exposição minúscula
        # (taxas brutas muito ruidosas): a correção -m/e_barra deve impedir que o
        # ruído de Poisson vire heterogeneidade, mantendo Var(theta) perto de 0
        # (alpha grande) em vez da variância ingênua das taxas brutas.
        rng = np.random.default_rng(11)
        theta = 1.5
        e = np.concatenate([rng.uniform(0.3, 1.0, 400), rng.uniform(20, 40, 400)])
        x = rng.poisson(theta * e).astype(np.float64)
        alpha, beta = fit_population(x, e)
        var_hat = alpha / beta**2
        naive_var = float(np.var(x / e))
        assert var_hat < 0.1 * naive_var

    def test_groups_ajusta_cada_populacao_separada(self) -> None:
        rng = np.random.default_rng(3)
        x_a, e_a = _simulate_players(rng, 8.0, 4.0, n_players=800, exposure_range=(10, 30))
        x_d, e_d = _simulate_players(rng, 2.0, 8.0, n_players=800, exposure_range=(10, 30))
        x = np.concatenate([x_a, x_d])
        e = np.concatenate([e_a, e_d])
        g = np.array(["ATA"] * 800 + ["DEF"] * 800)
        por_grupo = fit_population(x, e, groups=g)
        assert set(por_grupo) == {"ATA", "DEF"}
        assert por_grupo["ATA"] == fit_population(x_a, e_a)
        assert por_grupo["DEF"] == fit_population(x_d, e_d)
        mean_ata = por_grupo["ATA"][0] / por_grupo["ATA"][1]
        mean_def = por_grupo["DEF"][0] / por_grupo["DEF"][1]
        assert mean_ata == pytest.approx(2.0, rel=0.1)
        assert mean_def == pytest.approx(0.25, rel=0.1)
        assert mean_ata > mean_def

    def test_aceita_somas_ponderadas_fracionarias(self) -> None:
        alpha, beta = fit_population([1.3, 4.7, 0.4], [2.1, 6.3, 1.9])
        assert alpha > 0 and beta > 0

    def test_validacoes(self) -> None:
        with pytest.raises(ValueError):
            fit_population([1.0, 2.0], [1.0, 0.0])
        with pytest.raises(ValueError):
            fit_population([1.0, 2.0], [1.0, -1.0])
        with pytest.raises(ValueError):
            fit_population([0.0, 0.0], [1.0, 2.0])
        with pytest.raises(ValueError):
            fit_population([5.0], [3.0])
        with pytest.raises(ValueError):
            fit_population([1.0, 2.0], [1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            fit_population([], [])
        with pytest.raises(ValueError):
            fit_population([-1.0, 2.0], [1.0, 2.0])
        with pytest.raises(ValueError):
            fit_population([1.0, 2.0], [1.0, 2.0], groups=["A"])


class TestPredictive:
    def test_nb_tem_variancia_maior_que_poisson_com_mesma_media(self) -> None:
        a, b, e = 3.0, 2.0, 1.0
        p = b / (b + e)
        nb_mean, nb_var = (float(v) for v in nbinom.stats(a, p, moments="mv"))
        mu = a * e / b
        assert nb_mean == pytest.approx(mu)
        assert nb_var == pytest.approx(mu * (b + e) / b)
        assert nb_var > nb_mean  # Var da Poisson com a mesma média é mu

    def test_probs_em_0_1_e_decrescentes_na_linha(self) -> None:
        a, b, e = 4.0, 3.0, 1.0
        lines = np.arange(0.5, 15.0, 1.0)
        probs_nb = [prob_over(ln, a, b, e) for ln in lines]
        probs_po = [prob_over_poisson(ln, a, b, e) for ln in lines]
        for probs in (probs_nb, probs_po):
            assert all(0.0 <= p <= 1.0 for p in probs)
            diffs = np.diff(np.asarray(probs))
            assert np.all(diffs <= 1e-12)
        assert probs_nb[0] > probs_nb[5]

    def test_linha_negativa_e_certeza(self) -> None:
        assert prob_over(-0.5, 2.0, 2.0, 1.0) == 1.0
        assert prob_over_poisson(-0.5, 2.0, 2.0, 1.0) == 1.0

    def test_prob_over_bate_com_nbinom_manual(self) -> None:
        a, b, e = 5.0, 4.0, 0.9
        p = b / (b + e)
        assert prob_over(2.5, a, b, e) == pytest.approx(float(nbinom.sf(2, a, p)))
        # Linha inteira: over 2.0 paga só com X >= 3 (push fora), mesmo corte de 2.5.
        assert prob_over(2.0, a, b, e) == pytest.approx(prob_over(2.5, a, b, e))

    def test_prob_over_poisson_bate_com_poisson_manual(self) -> None:
        a, b, e = 5.0, 4.0, 0.9
        assert prob_over_poisson(2.5, a, b, e) == pytest.approx(float(poisson.sf(2, a / b * e)))

    def test_cauda_da_nb_e_mais_gorda_que_a_da_poisson(self) -> None:
        # Mesma média (a*e/b = 1.0); bem acima da média a NB deve dar mais
        # probabilidade, porque a incerteza de theta engrossa a cauda.
        a, b, e = 2.0, 2.0, 1.0
        assert prob_over(4.5, a, b, e) > prob_over_poisson(4.5, a, b, e)

    def test_multiplier_equivale_a_escalar_exposicao(self) -> None:
        a, b, e, m = 3.0, 2.5, 1.0, 1.3
        assert prob_over(1.5, a, b, e, multiplier=m) == pytest.approx(prob_over(1.5, a, b, e * m))
        assert prob_over_poisson(1.5, a, b, e, multiplier=m) == pytest.approx(
            prob_over_poisson(1.5, a, b, e * m)
        )

    def test_multiplier_maior_aumenta_prob_over(self) -> None:
        a, b, e = 3.0, 2.5, 1.0
        assert prob_over(1.5, a, b, e, multiplier=1.4) > prob_over(1.5, a, b, e, multiplier=1.0)

    def test_adjusted_rate(self) -> None:
        assert adjusted_rate(6.0, 4.0) == pytest.approx(1.5)
        assert adjusted_rate(6.0, 4.0, multiplier=1.2) == pytest.approx(1.8)
        with pytest.raises(ValueError):
            adjusted_rate(6.0, 4.0, multiplier=0.0)
        with pytest.raises(ValueError):
            adjusted_rate(0.0, 4.0)

    def test_validacoes(self) -> None:
        with pytest.raises(ValueError):
            prob_over(1.5, 0.0, 1.0, 1.0)
        with pytest.raises(ValueError):
            prob_over(1.5, 1.0, -1.0, 1.0)
        with pytest.raises(ValueError):
            prob_over(1.5, 1.0, 1.0, 0.0)
        with pytest.raises(ValueError):
            prob_over(1.5, 1.0, 1.0, 1.0, multiplier=-0.5)
        with pytest.raises(ValueError):
            prob_over(float("inf"), 1.0, 1.0, 1.0)
        with pytest.raises(ValueError):
            prob_over_poisson(1.5, 1.0, 1.0, 0.0)


class TestModelComparison:
    @staticmethod
    def _marginal_nb_sample(
        rng: np.random.Generator,
        alpha: float,
        beta: float,
        n: int,
    ) -> tuple[FloatArray, FloatArray]:
        e = rng.uniform(0.5, 1.0, size=n)
        theta = rng.gamma(shape=alpha, scale=1.0 / beta, size=n)
        x = rng.poisson(theta * e).astype(np.float64)
        return x, e.astype(np.float64)

    def test_estrutura_do_dataframe(self) -> None:
        rng = np.random.default_rng(0)
        x, e = self._marginal_nb_sample(rng, 2.0, 1.0, 200)
        df = model_comparison(x[:150], e[:150], x[150:], e[150:])
        assert list(df.index) == ["poisson", "gamma_poisson"]
        assert list(df.columns) == [
            "n_params",
            "loglik_train",
            "aic_train",
            "bic_train",
            "log_loss_test",
        ]
        assert df["n_params"].tolist() == [1, 2]
        assert np.all(np.isfinite(df.to_numpy(dtype=np.float64)))

    def test_nb_vence_em_dados_sobredispersos(self) -> None:
        # theta_i ~ Gamma => a marginal é NB de verdade: a NB tem de ganhar em
        # log-loss preditivo e em AIC, comprovando empiricamente (não por
        # suposição) que o parâmetro extra paga o próprio custo.
        rng = np.random.default_rng(123)
        x_tr, e_tr = self._marginal_nb_sample(rng, 2.0, 1.0, 1500)
        x_te, e_te = self._marginal_nb_sample(rng, 2.0, 1.0, 800)
        df = model_comparison(x_tr, e_tr, x_te, e_te)
        assert _cell(df, "gamma_poisson", "log_loss_test") < _cell(df, "poisson", "log_loss_test")
        assert _cell(df, "gamma_poisson", "aic_train") < _cell(df, "poisson", "aic_train")
        assert _cell(df, "gamma_poisson", "loglik_train") > _cell(df, "poisson", "loglik_train")

    def test_poisson_compete_em_dados_homogeneos(self) -> None:
        # theta fixo => não há sobredispersão: a NB no máximo empata (colapsa na
        # Poisson via alpha grande) e o BIC deve preferir o modelo de 1 parâmetro.
        rng = np.random.default_rng(321)
        theta = 1.5
        e_tr = rng.uniform(0.5, 1.0, size=1000)
        x_tr = rng.poisson(theta * e_tr).astype(np.float64)
        e_te = rng.uniform(0.5, 1.0, size=500)
        x_te = rng.poisson(theta * e_te).astype(np.float64)
        df = model_comparison(x_tr, e_tr, x_te, e_te)
        diff = _cell(df, "gamma_poisson", "log_loss_test") - _cell(df, "poisson", "log_loss_test")
        assert abs(diff) < 0.02
        assert _cell(df, "poisson", "bic_train") <= _cell(df, "gamma_poisson", "bic_train")

    def test_validacoes(self) -> None:
        with pytest.raises(ValueError):
            model_comparison([1.5, 2.0], [1.0, 1.0], [1.0], [1.0])  # contagem fracionária
        with pytest.raises(ValueError):
            model_comparison([1.0, 2.0], [1.0, 0.0], [1.0], [1.0])
        with pytest.raises(ValueError):
            model_comparison([0.0, 0.0], [1.0, 1.0], [1.0], [1.0])
        with pytest.raises(ValueError):
            model_comparison([2.0], [1.0], [1.0], [1.0])
