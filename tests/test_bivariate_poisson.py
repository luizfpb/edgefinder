"""Testes do Poisson bivariado (Karlis & Ntzoufras, 2003).

Cobrem quatro frentes: correção numérica da pmf em log-espaço (soma ~1,
redução ao produto de Poissons quando lambda3 = 0, momentos analíticos),
propriedades estruturais do modelo (X - Y independe do choque comum, logo o
1x2 não depende de lambda3), recuperação de parâmetros em dados sintéticos e
casos degenerados de entrada.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from scipy.stats import poisson

from edgefinder.models.bivariate_poisson import BivariatePoisson, log_pmf, time_decay_weights

FloatArray = npt.NDArray[np.float64]


def _pmf_direct(x: int, y: int, l1: float, l2: float, l3: float) -> float:
    """Fórmula de Karlis & Ntzoufras avaliada de forma ingênua (referência).

    P(X=x, Y=y) = exp(-(l1+l2+l3)) * (l1^x/x!) * (l2^y/y!)
                  * sum_{k=0}^{min(x,y)} C(x,k) C(y,k) k! (l3/(l1 l2))^k

    Só é confiável para placares pequenos, exatamente o regime em que serve
    de oráculo para a versão em log-espaço.
    """
    s = sum(
        math.comb(x, k) * math.comb(y, k) * math.factorial(k) * (l3 / (l1 * l2)) ** k
        for k in range(min(x, y) + 1)
    )
    base = math.exp(-(l1 + l2 + l3))
    return base * l1**x / math.factorial(x) * l2**y / math.factorial(y) * s


def _pmf_grid(l1: float, l2: float, l3: float, max_goals: int) -> FloatArray:
    goals = np.arange(max_goals + 1, dtype=np.float64)
    return np.asarray(np.exp(log_pmf(goals[:, None], goals[None, :], l1, l2, l3)))


def _simulate_matches(
    rng: np.random.Generator,
    teams: list[str],
    attack: dict[str, float],
    defence: dict[str, float],
    mu: float,
    gamma: float,
    lambda3: float,
    n_replicates: int,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Gera partidas pela construção geradora X = W1 + W3, Y = W2 + W3."""
    rows: list[dict[str, object]] = []
    day = 0
    for _ in range(n_replicates):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                l1 = math.exp(mu + gamma + attack[home] + defence[away])
                l2 = math.exp(mu + attack[away] + defence[home])
                w3 = int(rng.poisson(lambda3))
                rows.append(
                    {
                        "date": end_date - pd.Timedelta(days=day % 300),
                        "home_team": home,
                        "away_team": away,
                        "home_goals": int(rng.poisson(l1)) + w3,
                        "away_goals": int(rng.poisson(l2)) + w3,
                    }
                )
                day += 1
    return pd.DataFrame(rows)


def _manual_model(lambda3: float = 0.2) -> BivariatePoisson:
    """Modelo com parâmetros fixados à mão para testes determinísticos."""
    model = BivariatePoisson()
    model.teams_ = ["A", "B"]
    model.attack_ = pd.Series({"A": 0.3, "B": -0.3}, dtype=np.float64)
    model.defence_ = pd.Series({"A": -0.2, "B": 0.2}, dtype=np.float64)
    model.intercept_ = 0.1
    model.home_advantage_ = 0.25
    model.lambda3_ = lambda3
    model._fitted = True
    return model


TRUE_MU = 0.10
TRUE_GAMMA = 0.30
TRUE_LAMBDA3 = 0.25
TRUE_ATTACK = {
    "ARS": 0.35,
    "BOU": -0.25,
    "CHE": 0.10,
    "EVE": -0.30,
    "LIV": 0.30,
    "MCI": 0.40,
    "NEW": -0.20,
    "TOT": -0.40,
}
TRUE_DEFENCE = {
    "ARS": -0.30,
    "BOU": 0.25,
    "CHE": -0.10,
    "EVE": 0.30,
    "LIV": -0.25,
    "MCI": -0.35,
    "NEW": 0.20,
    "TOT": 0.25,
}
REF_DATE = pd.Timestamp("2025-06-01")


@pytest.fixture(scope="module")
def synthetic_matches() -> pd.DataFrame:
    rng = np.random.default_rng(20250601)
    teams = sorted(TRUE_ATTACK)
    return _simulate_matches(
        rng, teams, TRUE_ATTACK, TRUE_DEFENCE, TRUE_MU, TRUE_GAMMA, TRUE_LAMBDA3, 12, REF_DATE
    )


@pytest.fixture(scope="module")
def fitted(synthetic_matches: pd.DataFrame) -> BivariatePoisson:
    return BivariatePoisson().fit(synthetic_matches, ref_date=REF_DATE, half_life_days=1e5)


class TestLogPmf:
    def test_soma_um_na_grade(self) -> None:
        for l1, l2, l3 in [(1.4, 1.1, 0.3), (0.6, 2.3, 0.05), (2.0, 2.0, 1.0)]:
            total = float(_pmf_grid(l1, l2, l3, 40).sum())
            assert total == pytest.approx(1.0, abs=1e-9)

    def test_lambda3_zero_reduz_a_produto_de_poissons(self) -> None:
        goals = np.arange(15, dtype=np.float64)
        gx, gy = np.meshgrid(goals, goals, indexing="ij")
        got = log_pmf(gx, gy, 1.7, 0.9, 0.0)
        expected = poisson.logpmf(gx, 1.7) + poisson.logpmf(gy, 0.9)
        np.testing.assert_allclose(got, expected, atol=1e-10)

    def test_coincide_com_formula_direta(self) -> None:
        l1, l2, l3 = 1.3, 0.8, 0.45
        for x in range(7):
            for y in range(7):
                got = float(np.exp(log_pmf(x, y, l1, l2, l3)))
                assert got == pytest.approx(_pmf_direct(x, y, l1, l2, l3), rel=1e-10)

    def test_marginais_sao_poisson_com_taxa_somada(self) -> None:
        """X = W1 + W3 é Poisson(l1 + l3): as marginais da grade devem bater."""
        l1, l2, l3 = 1.2, 0.9, 0.4
        m = _pmf_grid(l1, l2, l3, 35)
        goals = np.arange(36, dtype=np.float64)
        np.testing.assert_allclose(m.sum(axis=1), poisson.pmf(goals, l1 + l3), atol=1e-9)
        np.testing.assert_allclose(m.sum(axis=0), poisson.pmf(goals, l2 + l3), atol=1e-9)

    def test_covariancia_analitica_igual_lambda3(self) -> None:
        """Cov(X, Y) = lambda3 por construção; a grade deve reproduzir isso."""
        l1, l2, l3 = 1.3, 0.9, 0.4
        m = _pmf_grid(l1, l2, l3, 35)
        goals = np.arange(36, dtype=np.float64)
        ex = float((goals[:, None] * m).sum())
        ey = float((goals[None, :] * m).sum())
        exy = float((goals[:, None] * goals[None, :] * m).sum())
        assert exy - ex * ey == pytest.approx(l3, abs=1e-6)

    def test_placares_altos_sao_estaveis(self) -> None:
        val = float(log_pmf(30, 28, 1.5, 1.2, 0.6))
        assert math.isfinite(val)
        assert val < -20.0

    def test_broadcasting_preserva_forma(self) -> None:
        goals = np.arange(5, dtype=np.float64)
        out = log_pmf(goals[:, None], goals[None, :], 1.0, 1.0, 0.2)
        assert out.shape == (5, 5)

    def test_entradas_invalidas(self) -> None:
        with pytest.raises(ValueError, match="inteiras"):
            log_pmf(-1, 0, 1.0, 1.0, 0.1)
        with pytest.raises(ValueError, match="inteiras"):
            log_pmf(1.5, 0, 1.0, 1.0, 0.1)
        with pytest.raises(ValueError, match="positivos"):
            log_pmf(1, 1, 0.0, 1.0, 0.1)
        with pytest.raises(ValueError, match="não negativo"):
            log_pmf(1, 1, 1.0, 1.0, -0.1)


class TestTimeDecayWeights:
    def test_valores_de_referencia(self) -> None:
        """w = exp(-xi * dt): dt = 0 pesa 1 e dt = meia-vida pesa 0.5."""
        dates = pd.Series(pd.to_datetime(["2025-06-01", "2025-05-02", "2025-04-02"]))
        w = time_decay_weights(dates, "2025-06-01", half_life_days=30.0)
        np.testing.assert_allclose(w, [1.0, 0.5, 0.25], atol=1e-12)

    def test_monotonicidade_no_tempo(self) -> None:
        dates = pd.Series(pd.to_datetime(["2025-06-01", "2025-03-01", "2024-06-01"]))
        w = time_decay_weights(dates, "2025-06-01", half_life_days=90.0)
        assert w[0] > w[1] > w[2] > 0.0

    def test_data_futura_recusada(self) -> None:
        dates = pd.Series(pd.to_datetime(["2025-06-02"]))
        with pytest.raises(ValueError, match="posteriores"):
            time_decay_weights(dates, "2025-06-01", half_life_days=30.0)

    def test_meia_vida_invalida(self) -> None:
        dates = pd.Series(pd.to_datetime(["2025-06-01"]))
        for bad in (0.0, -10.0, float("nan")):
            with pytest.raises(ValueError, match="positivo"):
                time_decay_weights(dates, "2025-06-01", half_life_days=bad)


class TestFit:
    def test_recupera_parametros_sinteticos(self, fitted: BivariatePoisson) -> None:
        assert fitted.converged_
        assert fitted.home_advantage_ == pytest.approx(TRUE_GAMMA, abs=0.12)
        assert fitted.intercept_ == pytest.approx(TRUE_MU, abs=0.15)
        assert fitted.lambda3_ == pytest.approx(TRUE_LAMBDA3, abs=0.15)
        teams = sorted(TRUE_ATTACK)
        atk_true = np.array([TRUE_ATTACK[t] for t in teams])
        def_true = np.array([TRUE_DEFENCE[t] for t in teams])
        atk_est = fitted.attack_.loc[teams].to_numpy(dtype=np.float64)
        def_est = fitted.defence_.loc[teams].to_numpy(dtype=np.float64)
        assert float(np.corrcoef(atk_true, atk_est)[0, 1]) > 0.9
        assert float(np.corrcoef(def_true, def_est)[0, 1]) > 0.9
        assert float(np.max(np.abs(atk_est - atk_true))) < 0.35
        assert float(np.max(np.abs(def_est - def_true))) < 0.35

    def test_identificacao_soma_zero(self, fitted: BivariatePoisson) -> None:
        assert float(fitted.attack_.sum()) == pytest.approx(0.0, abs=1e-8)
        assert float(fitted.defence_.sum()) == pytest.approx(0.0, abs=1e-8)

    def test_fit_retorna_self_e_metadados(
        self, fitted: BivariatePoisson, synthetic_matches: pd.DataFrame
    ) -> None:
        assert isinstance(fitted, BivariatePoisson)
        assert fitted.n_matches_ == len(synthetic_matches)
        assert math.isfinite(fitted.loglik_)
        assert fitted.lambda3_ >= 0.0

    def test_partidas_futuras_sao_ignoradas(self, synthetic_matches: pd.DataFrame) -> None:
        """Incluir um jogo posterior a ref_date não pode mudar nada (anti-vazamento)."""
        base = BivariatePoisson().fit(synthetic_matches, ref_date=REF_DATE, half_life_days=1e5)
        future = pd.DataFrame(
            [
                {
                    "date": REF_DATE + pd.Timedelta(days=30),
                    "home_team": "ARS",
                    "away_team": "TOT",
                    "home_goals": 25,
                    "away_goals": 0,
                }
            ]
        )
        polluted = pd.concat([synthetic_matches, future], ignore_index=True)
        other = BivariatePoisson().fit(polluted, ref_date=REF_DATE, half_life_days=1e5)
        assert other.n_matches_ == base.n_matches_
        assert other.lambda3_ == pytest.approx(base.lambda3_, abs=1e-12)
        pd.testing.assert_series_equal(other.attack_, base.attack_)

    def test_partidas_sem_resultado_sao_descartadas(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-01", "2025-01-08", "2025-01-15"]),
                "home_team": ["A", "B", "A"],
                "away_team": ["B", "A", "B"],
                "home_goals": [1.0, np.nan, 2.0],
                "away_goals": [0.0, np.nan, 1.0],
            }
        )
        model = BivariatePoisson().fit(df, ref_date="2025-02-01")
        assert model.n_matches_ == 2

    def test_colunas_ausentes(self) -> None:
        with pytest.raises(ValueError, match="colunas ausentes"):
            BivariatePoisson().fit(pd.DataFrame({"date": []}), ref_date="2025-01-01")

    def test_sem_partidas_ate_ref_date(self, synthetic_matches: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="nenhuma partida"):
            BivariatePoisson().fit(synthetic_matches, ref_date="2000-01-01")

    def test_gols_nao_inteiros_recusados(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-01", "2025-01-08"]),
                "home_team": ["A", "B"],
                "away_team": ["B", "A"],
                "home_goals": [1.5, 1.0],
                "away_goals": [0.0, 2.0],
            }
        )
        with pytest.raises(ValueError, match="contagens inteiras"):
            BivariatePoisson().fit(df, ref_date="2025-02-01")

    def test_menos_de_dois_times(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-01"]),
                "home_team": ["A"],
                "away_team": ["A"],
                "home_goals": [1],
                "away_goals": [1],
            }
        )
        with pytest.raises(ValueError, match="2 times"):
            BivariatePoisson().fit(df, ref_date="2025-02-01")


class TestDerivados:
    def test_score_matrix_soma_aproximadamente_um(self) -> None:
        m = _manual_model().score_matrix("A", "B", max_goals=12)
        assert m.shape == (13, 13)
        assert np.all(m >= 0.0)
        assert float(m.sum()) == pytest.approx(1.0, abs=1e-4)

    def test_score_matrix_max_goals_invalido(self) -> None:
        with pytest.raises(ValueError, match="max_goals"):
            _manual_model().score_matrix("A", "B", max_goals=0)

    def test_probs_1x2_somam_um(self) -> None:
        p = _manual_model().probs_1x2("A", "B")
        assert p["home"] + p["draw"] + p["away"] == pytest.approx(1.0, abs=1e-12)
        assert all(0.0 <= v <= 1.0 for v in p.values())

    def test_1x2_independe_de_lambda3(self) -> None:
        """X - Y = W1 - W2 não envolve o choque comum W3, então o 1x2 não pode
        depender de lambda3 (mantidos lambda1 e lambda2 fixos)."""
        p0 = _manual_model(lambda3=0.0).probs_1x2("A", "B", max_goals=25)
        p1 = _manual_model(lambda3=0.6).probs_1x2("A", "B", max_goals=25)
        for key in ("home", "draw", "away"):
            assert p0[key] == pytest.approx(p1[key], abs=1e-6)

    def test_time_mais_forte_e_favorito(self) -> None:
        model = _manual_model()
        em_casa = model.probs_1x2("A", "B")
        fora = model.probs_1x2("B", "A")
        assert em_casa["home"] > em_casa["away"]
        assert fora["away"] > fora["home"]

    def test_over_under_linha_fracionaria_soma_um(self) -> None:
        p = _manual_model().probs_over_under("A", "B", line=2.5)
        assert p["over"] + p["under"] == pytest.approx(1.0, abs=1e-12)

    def test_over_under_linha_inteira_deixa_push_de_fora(self) -> None:
        p = _manual_model().probs_over_under("A", "B", line=2.0)
        assert p["over"] + p["under"] < 1.0

    def test_over_monotono_na_linha(self) -> None:
        model = _manual_model()
        overs = [model.probs_over_under("A", "B", line=ln)["over"] for ln in (0.5, 1.5, 2.5, 3.5)]
        assert overs == sorted(overs, reverse=True)
        unders = [model.probs_over_under("A", "B", line=ln)["under"] for ln in (0.5, 1.5, 2.5, 3.5)]
        assert unders == sorted(unders)

    def test_match_rates_consistentes_com_parametros(self) -> None:
        model = _manual_model(lambda3=0.3)
        l1, l2, l3 = model.match_rates("A", "B")
        assert l1 == pytest.approx(math.exp(0.1 + 0.25 + 0.3 + 0.2), rel=1e-12)
        assert l2 == pytest.approx(math.exp(0.1 + (-0.3) + (-0.2)), rel=1e-12)
        assert l3 == pytest.approx(0.3, rel=1e-12)

    def test_time_desconhecido(self) -> None:
        with pytest.raises(ValueError, match="desconhecido"):
            _manual_model().score_matrix("A", "ZZZ")

    def test_modelo_nao_ajustado(self) -> None:
        with pytest.raises(RuntimeError, match="não ajustado"):
            BivariatePoisson().score_matrix("A", "B")


class TestSimulacao:
    def test_covariancia_empirica_proxima_de_lambda3(self) -> None:
        """Na construção geradora, Cov(X, Y) = Var(W3) = lambda3."""
        rng = np.random.default_rng(7)
        n = 30_000
        l1, l2, l3 = 1.2, 0.9, 0.5
        w3 = rng.poisson(l3, n)
        x = rng.poisson(l1, n) + w3
        y = rng.poisson(l2, n) + w3
        cov = float(np.cov(x, y)[0, 1])
        assert cov == pytest.approx(l3, abs=0.06)
        assert cov > 0.0

    def test_pmf_bate_com_frequencias_simuladas(self) -> None:
        rng = np.random.default_rng(11)
        n = 50_000
        l1, l2, l3 = 1.1, 0.8, 0.35
        w3 = rng.poisson(l3, n)
        x = rng.poisson(l1, n) + w3
        y = rng.poisson(l2, n) + w3
        for xi, yi in [(0, 0), (1, 1), (2, 1), (0, 2)]:
            freq = float(np.mean((x == xi) & (y == yi)))
            prob = float(np.exp(log_pmf(xi, yi, l1, l2, l3)))
            assert freq == pytest.approx(prob, abs=0.01)
