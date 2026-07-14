"""Testes do modelo Dixon-Coles: recuperação de parâmetros e propriedades das probabilidades."""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from edgefinder.models.dixon_coles import DixonColes, tau_correction

FloatArray = npt.NDArray[np.float64]

REF_DATE = pd.Timestamp("2025-06-01")
N_TEAMS = 12
N_MATCHES = 2000
TRUE_HOME_ADV = 0.30
TRUE_RHO = -0.10
MAX_GOALS_SIM = 12


def _true_params(rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """Forças verdadeiras com média zero, como a parametrização do modelo exige."""
    atk = rng.normal(0.0, 0.35, N_TEAMS)
    dfn = rng.normal(0.0, 0.30, N_TEAMS)
    return atk - atk.mean(), dfn - dfn.mean()


def _true_score_matrix(lam: float, mu: float, rho: float) -> FloatArray:
    """Pmf conjunta do DC verdadeiro, reimplementada de forma independente do módulo."""
    goals = np.arange(MAX_GOALS_SIM + 1, dtype=np.float64)
    px = np.exp(-lam) * lam**goals / np.array([math.factorial(int(g)) for g in goals])
    py = np.exp(-mu) * mu**goals / np.array([math.factorial(int(g)) for g in goals])
    m = px[:, None] * py[None, :]
    m[0, 0] *= 1.0 - lam * mu * rho
    m[0, 1] *= 1.0 + lam * rho
    m[1, 0] *= 1.0 + mu * rho
    m[1, 1] *= 1.0 - rho
    m = np.maximum(m, 0.0)
    return np.asarray(m / m.sum(), dtype=np.float64)


def _simulate_matches(
    atk: FloatArray,
    dfn: FloatArray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    homes = rng.integers(0, N_TEAMS, N_MATCHES)
    aways = (homes + rng.integers(1, N_TEAMS, N_MATCHES)) % N_TEAMS
    days_ago = rng.integers(0, 300, N_MATCHES)
    rows: list[dict[str, object]] = []
    for h, a, d in zip(homes, aways, days_ago, strict=True):
        lam = math.exp(atk[h] - dfn[a] + TRUE_HOME_ADV)
        mu = math.exp(atk[a] - dfn[h])
        m = _true_score_matrix(lam, mu, TRUE_RHO)
        idx = rng.choice(m.size, p=m.ravel())
        rows.append(
            {
                "home_team": f"T{h:02d}",
                "away_team": f"T{a:02d}",
                "home_goals": idx // (MAX_GOALS_SIM + 1),
                "away_goals": idx % (MAX_GOALS_SIM + 1),
                "match_date": REF_DATE - pd.Timedelta(days=int(d)),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic() -> tuple[pd.DataFrame, FloatArray, FloatArray]:
    rng = np.random.default_rng(42)
    atk, dfn = _true_params(rng)
    return _simulate_matches(atk, dfn, rng), atk, dfn


@pytest.fixture(scope="module")
def fitted(synthetic: tuple[pd.DataFrame, FloatArray, FloatArray]) -> DixonColes:
    df, _, _ = synthetic
    # Half-life enorme = ajuste efetivamente não ponderado, para comparar com
    # os parâmetros verdadeiros (que não variam no tempo na simulação).
    return DixonColes().fit(df, ref_date=REF_DATE, half_life_days=1e6)


def _manual_model(rho: float) -> DixonColes:
    """Modelo montado à mão (white-box) para testar efeitos de rho isoladamente."""
    m = DixonColes()
    teams = ["A", "B"]
    m.teams_ = teams
    m.attack_ = pd.Series([0.1, -0.1], index=teams, dtype=np.float64)
    m.defence_ = pd.Series([0.05, -0.05], index=teams, dtype=np.float64)
    m.home_advantage_ = 0.25
    m.rho_ = rho
    m._fitted = True
    return m


class TestParameterRecovery:
    def test_attack_correlation(
        self, fitted: DixonColes, synthetic: tuple[pd.DataFrame, FloatArray, FloatArray]
    ) -> None:
        _, atk_true, _ = synthetic
        est = fitted.attack_.loc[[f"T{i:02d}" for i in range(N_TEAMS)]].to_numpy()
        corr = float(np.corrcoef(atk_true, est)[0, 1])
        assert corr > 0.9

    def test_defence_correlation(
        self, fitted: DixonColes, synthetic: tuple[pd.DataFrame, FloatArray, FloatArray]
    ) -> None:
        _, _, dfn_true = synthetic
        est = fitted.defence_.loc[[f"T{i:02d}" for i in range(N_TEAMS)]].to_numpy()
        corr = float(np.corrcoef(dfn_true, est)[0, 1])
        assert corr > 0.9

    def test_home_advantage(self, fitted: DixonColes) -> None:
        assert abs(fitted.home_advantage_ - TRUE_HOME_ADV) < 0.05

    def test_rho(self, fitted: DixonColes) -> None:
        assert fitted.rho_ < 0.0
        assert abs(fitted.rho_ - TRUE_RHO) < 0.08

    def test_zero_mean_constraints(self, fitted: DixonColes) -> None:
        assert abs(float(fitted.attack_.mean())) < 1e-10
        assert abs(float(fitted.defence_.mean())) < 1e-10

    def test_metadata(self, fitted: DixonColes) -> None:
        assert fitted.n_matches_ == N_MATCHES
        assert fitted.converged_
        assert math.isfinite(fitted.loglik_)
        assert fitted.params_.shape == (2 * (N_TEAMS - 1) + 2,)


class TestScoreMatrix:
    def test_sums_to_one_and_nonnegative(self, fitted: DixonColes) -> None:
        m = fitted.score_matrix("T00", "T01")
        assert m.shape == (11, 11)
        assert np.all(m >= 0.0)
        assert m.sum() == pytest.approx(1.0, abs=1e-12)

    def test_max_goals_controls_shape(self, fitted: DixonColes) -> None:
        m = fitted.score_matrix("T00", "T01", max_goals=6)
        assert m.shape == (7, 7)
        assert m.sum() == pytest.approx(1.0, abs=1e-12)

    def test_invalid_max_goals(self, fitted: DixonColes) -> None:
        with pytest.raises(ValueError, match="max_goals"):
            fitted.score_matrix("T00", "T01", max_goals=0)

    def test_negative_rho_shifts_mass_to_draws(self) -> None:
        # Comparação relativa a uma célula não afetada pelo tau (2, 2): rho < 0
        # deve inflar 0-0 e 1-1 e esvaziar 1-0 e 0-1 em relação ao independente.
        m_ind = _manual_model(0.0).score_matrix("A", "B")
        m_neg = _manual_model(-0.1).score_matrix("A", "B")
        assert m_neg[0, 0] / m_neg[2, 2] > m_ind[0, 0] / m_ind[2, 2]
        assert m_neg[1, 1] / m_neg[2, 2] > m_ind[1, 1] / m_ind[2, 2]
        assert m_neg[1, 0] / m_neg[2, 2] < m_ind[1, 0] / m_ind[2, 2]
        assert m_neg[0, 1] / m_neg[2, 2] < m_ind[0, 1] / m_ind[2, 2]


class TestTauCorrection:
    def test_table_values(self) -> None:
        lam, mu, rho = 1.5, 1.2, -0.1
        assert tau_correction(0, 0, lam, mu, rho) == pytest.approx(1.0 - lam * mu * rho)
        assert tau_correction(0, 1, lam, mu, rho) == pytest.approx(1.0 + lam * rho)
        assert tau_correction(1, 0, lam, mu, rho) == pytest.approx(1.0 + mu * rho)
        assert tau_correction(1, 1, lam, mu, rho) == pytest.approx(1.0 - rho)
        assert tau_correction(2, 3, lam, mu, rho) == pytest.approx(1.0)

    def test_rho_zero_is_identity(self) -> None:
        goals = np.arange(5, dtype=np.float64)
        t = tau_correction(goals[:, None], goals[None, :], 1.4, 1.1, 0.0)
        np.testing.assert_allclose(t, np.ones((5, 5)))


class TestDerivedProbabilities:
    def test_1x2_sums_to_one(self, fitted: DixonColes) -> None:
        ph, px, pa = fitted.probs_1x2("T03", "T07")
        assert ph + px + pa == pytest.approx(1.0, abs=1e-12)
        assert min(ph, px, pa) > 0.0

    def test_over_under_complement_half_line(self, fitted: DixonColes) -> None:
        over = fitted.prob_over("T03", "T07", line=2.5)
        under = fitted.prob_under("T03", "T07", line=2.5)
        assert over + under == pytest.approx(1.0, abs=1e-12)
        assert 0.0 < over < 1.0

    def test_over_under_integer_line_leaves_push_out(self, fitted: DixonColes) -> None:
        over = fitted.prob_over("T03", "T07", line=2.0)
        under = fitted.prob_under("T03", "T07", line=2.0)
        m = fitted.score_matrix("T03", "T07")
        goals = np.arange(11, dtype=np.float64)
        p_exact = float(m[goals[:, None] + goals[None, :] == 2.0].sum())
        assert p_exact > 0.0
        assert over + under + p_exact == pytest.approx(1.0, abs=1e-12)

    def test_over_monotonic_in_line(self, fitted: DixonColes) -> None:
        overs = [fitted.prob_over("T03", "T07", line=line) for line in (0.5, 1.5, 2.5, 3.5)]
        assert all(a > b for a, b in pairwise(overs))


class TestAsianHandicap:
    def test_half_line_has_no_push(self, fitted: DixonColes) -> None:
        p_home, p_push, p_away = fitted.prob_ah("T03", "T07", handicap=-0.5)
        assert p_push == 0.0
        assert p_home + p_away == pytest.approx(1.0, abs=1e-12)

    def test_integer_line_has_push(self, fitted: DixonColes) -> None:
        p_home, p_push, p_away = fitted.prob_ah("T03", "T07", handicap=-1.0)
        assert p_push > 0.0
        assert p_home + p_push + p_away == pytest.approx(1.0, abs=1e-12)

    def test_handicap_zero_matches_1x2(self, fitted: DixonColes) -> None:
        ph, px, pa = fitted.probs_1x2("T03", "T07")
        p_home, p_push, p_away = fitted.prob_ah("T03", "T07", handicap=0.0)
        assert p_home == pytest.approx(ph, abs=1e-12)
        assert p_push == pytest.approx(px, abs=1e-12)
        assert p_away == pytest.approx(pa, abs=1e-12)

    def test_monotonic_in_handicap(self, fitted: DixonColes) -> None:
        lines = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
        p_homes = [fitted.prob_ah("T03", "T07", handicap=h)[0] for h in lines]
        assert all(a <= b + 1e-12 for a, b in pairwise(p_homes))
        assert p_homes[0] < p_homes[-1]

    def test_quarter_line_rejected(self, fitted: DixonColes) -> None:
        with pytest.raises(ValueError, match="quarto"):
            fitted.prob_ah("T03", "T07", handicap=-0.25)


class TestValidationAndErrors:
    def test_unknown_team_raises_keyerror(self, fitted: DixonColes) -> None:
        with pytest.raises(KeyError, match="Chelsea"):
            fitted.score_matrix("Chelsea", "T00")
        with pytest.raises(KeyError, match="Chelsea"):
            fitted.probs_1x2("T00", "Chelsea")

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            DixonColes().score_matrix("A", "B")

    def test_missing_columns_raises(self) -> None:
        df = pd.DataFrame({"home_team": ["A"], "away_team": ["B"]})
        with pytest.raises(ValueError, match="colunas ausentes"):
            DixonColes().fit(df, ref_date=REF_DATE)

    def test_no_matches_before_ref_date_raises(
        self, synthetic: tuple[pd.DataFrame, FloatArray, FloatArray]
    ) -> None:
        df, _, _ = synthetic
        with pytest.raises(ValueError, match="ref_date"):
            DixonColes().fit(df, ref_date="2000-01-01")

    def test_future_matches_are_excluded(
        self, synthetic: tuple[pd.DataFrame, FloatArray, FloatArray]
    ) -> None:
        df, _, _ = synthetic
        future = pd.DataFrame(
            {
                "home_team": ["T00"],
                "away_team": ["T01"],
                "home_goals": [9],
                "away_goals": [9],
                "match_date": [REF_DATE + pd.Timedelta(days=30)],
            }
        )
        model = DixonColes().fit(pd.concat([df, future]), ref_date=REF_DATE, half_life_days=1e6)
        assert model.n_matches_ == N_MATCHES

    def test_invalid_half_life_raises(
        self, synthetic: tuple[pd.DataFrame, FloatArray, FloatArray]
    ) -> None:
        df, _, _ = synthetic
        with pytest.raises(ValueError, match="half_life"):
            DixonColes().fit(df, ref_date=REF_DATE, half_life_days=0.0)

    def test_non_integer_goals_raises(self) -> None:
        df = pd.DataFrame(
            {
                "home_team": ["A", "B"],
                "away_team": ["B", "A"],
                "home_goals": [1.5, 2],
                "away_goals": [0, 1],
                "match_date": [REF_DATE, REF_DATE],
            }
        )
        with pytest.raises(ValueError, match="contagens inteiras"):
            DixonColes().fit(df, ref_date=REF_DATE)

    def test_single_team_raises(self) -> None:
        df = pd.DataFrame(
            {
                "home_team": ["A"],
                "away_team": ["A"],
                "home_goals": [1],
                "away_goals": [1],
                "match_date": [REF_DATE],
            }
        )
        with pytest.raises(ValueError, match="2 times"):
            DixonColes().fit(df, ref_date=REF_DATE)


class TestWarmStart:
    def test_warm_start_reaches_same_optimum(
        self, fitted: DixonColes, synthetic: tuple[pd.DataFrame, FloatArray, FloatArray]
    ) -> None:
        df, _, _ = synthetic
        warm = DixonColes().fit(
            df, ref_date=REF_DATE, half_life_days=1e6, init_params=fitted.params_
        )
        assert warm.loglik_ == pytest.approx(fitted.loglik_, abs=1e-3)
        np.testing.assert_allclose(warm.attack_.to_numpy(), fitted.attack_.to_numpy(), atol=1e-3)

    def test_wrong_length_falls_back_to_cold_start(
        self, synthetic: tuple[pd.DataFrame, FloatArray, FloatArray]
    ) -> None:
        df, _, _ = synthetic
        model = DixonColes().fit(
            df.head(200), ref_date=REF_DATE, half_life_days=1e6, init_params=np.zeros(3)
        )
        assert model._fitted


class TestTimeDecay:
    def test_recent_form_dominates_with_short_half_life(self) -> None:
        # O time A perde tudo de 3-0 no passado distante e ganha tudo de 3-0
        # na janela recente: com half-life curto o modelo deve prever mais
        # gols de A do que com half-life longo (que dilui a fase recente).
        teams = [f"T{i}" for i in range(6)]
        rows: list[dict[str, object]] = []
        for days_ago, (gf, ga) in ((400, (0, 3)), (5, (3, 0))):
            for rep in range(10):
                for opp in teams[1:]:
                    rows.append(
                        {
                            "home_team": "T0",
                            "away_team": opp,
                            "home_goals": gf,
                            "away_goals": ga,
                            "match_date": REF_DATE - pd.Timedelta(days=days_ago + rep),
                        }
                    )
                    rows.append(
                        {
                            "home_team": opp,
                            "away_team": "T0",
                            "home_goals": ga,
                            "away_goals": gf,
                            "match_date": REF_DATE - pd.Timedelta(days=days_ago + rep),
                        }
                    )
        df = pd.DataFrame(rows)
        short = DixonColes().fit(df, ref_date=REF_DATE, half_life_days=30.0)
        long = DixonColes().fit(df, ref_date=REF_DATE, half_life_days=100_000.0)
        lam_short, _ = short.match_rates("T0", "T1")
        lam_long, _ = long.match_rates("T0", "T1")
        assert lam_short > lam_long
