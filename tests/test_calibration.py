"""Testes de calibração: métricas (Brier, log-loss, ECE) e correções (isotônica, Platt)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from scipy.special import expit, logit

from edgefinder.models.calibration import (
    brier_score,
    expected_calibration_error,
    fit_isotonic,
    fit_platt,
    log_loss_safe,
    reliability_table,
)

TOL = 1e-9


def _amostra_distorcida(
    n: int, seed: int, a: float = 0.5, b: float = 0.0
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Gera (y, p_raw) onde p_raw distorce a probabilidade real de forma afim no logit.

    Verdade: p_true = sigma(z), z ~ N(0, 1.5^2). Distorção: p_raw = sigma(a*z + b),
    ou seja, o modelo cru é subconfiante (a < 1) e/ou enviesado (b != 0). A
    correção exata é p_cal = sigma((logit(p_raw) - b) / a), que é afim no logit —
    o caso que o Platt cobre por construção e a isotônica cobre por ser monótona.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.5, size=n)
    p_true = expit(z)
    y = (rng.uniform(size=n) < p_true).astype(np.float64)
    p_raw = np.asarray(expit(a * z + b), dtype=np.float64)
    return y, p_raw


class TestReliabilityTable:
    def test_colunas_e_contagem_total(self) -> None:
        rng = np.random.default_rng(1)
        p = rng.uniform(0.0, 1.0, size=500)
        y = (rng.uniform(size=500) < p).astype(float)
        table = reliability_table(y, p, n_bins=10)
        assert list(table.columns) == ["bin_low", "bin_high", "prob_mean", "freq_observed", "n"]
        assert int(table["n"].sum()) == 500

    def test_quantile_gera_bins_equipopulados(self) -> None:
        rng = np.random.default_rng(2)
        p = rng.beta(8.0, 20.0, size=1000)
        y = (rng.uniform(size=1000) < p).astype(float)
        table = reliability_table(y, p, n_bins=10, strategy="quantile")
        assert (table["n"] >= 50).all()

    def test_uniform_respeita_bordas(self) -> None:
        p = np.array([0.05, 0.15, 0.15, 0.95])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        table = reliability_table(y, p, n_bins=10, strategy="uniform")
        assert len(table) == 3
        primeira = table.iloc[0]
        assert primeira["bin_low"] == pytest.approx(0.0, abs=TOL)
        assert primeira["bin_high"] == pytest.approx(0.1, abs=TOL)
        assert primeira["n"] == 1

    def test_freq_e_prob_dentro_de_01(self) -> None:
        rng = np.random.default_rng(3)
        p = rng.uniform(size=200)
        y = rng.integers(0, 2, size=200).astype(float)
        table = reliability_table(y, p)
        assert ((table["prob_mean"] >= 0.0) & (table["prob_mean"] <= 1.0)).all()
        assert ((table["freq_observed"] >= 0.0) & (table["freq_observed"] <= 1.0)).all()

    def test_previsao_constante_colapsa_em_um_bin(self) -> None:
        p = np.full(50, 0.3)
        y = np.array([1.0] * 15 + [0.0] * 35)
        table = reliability_table(y, p, n_bins=10, strategy="quantile")
        assert len(table) == 1
        assert table.iloc[0]["prob_mean"] == pytest.approx(0.3, abs=TOL)
        assert table.iloc[0]["freq_observed"] == pytest.approx(0.3, abs=TOL)

    def test_strategy_invalida_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="strategy"):
            reliability_table([0.0, 1.0], [0.2, 0.8], strategy="kmeans")  # type: ignore[arg-type]

    def test_n_bins_invalido_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="n_bins"):
            reliability_table([0.0, 1.0], [0.2, 0.8], n_bins=0)

    def test_tamanhos_diferentes_levantam_erro(self) -> None:
        with pytest.raises(ValueError, match="tamanhos"):
            reliability_table([0.0, 1.0], [0.5])

    def test_entrada_vazia_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="vazias"):
            reliability_table([], [])

    def test_rotulo_nao_binario_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="apenas 0 e 1"):
            reliability_table([0.0, 2.0], [0.5, 0.5])

    def test_prob_fora_de_01_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            reliability_table([0.0, 1.0], [0.5, 1.5])


class TestBrierScore:
    def test_previsao_perfeita_da_zero(self) -> None:
        y = np.array([0.0, 1.0, 1.0, 0.0])
        assert brier_score(y, y) == pytest.approx(0.0, abs=TOL)

    def test_chute_meio_a_meio_da_um_quarto(self) -> None:
        y = np.array([0.0, 1.0, 0.0, 1.0])
        assert brier_score(y, np.full(4, 0.5)) == pytest.approx(0.25, abs=TOL)

    def test_valor_conhecido(self) -> None:
        assert brier_score([1.0, 0.0], [0.8, 0.3]) == pytest.approx((0.04 + 0.09) / 2.0, abs=TOL)

    def test_previsao_invertida_da_um(self) -> None:
        y = np.array([0.0, 1.0])
        assert brier_score(y, 1.0 - y) == pytest.approx(1.0, abs=TOL)


class TestLogLossSafe:
    def test_previsao_perfeita_fica_perto_de_zero(self) -> None:
        y = np.array([0.0, 1.0, 1.0])
        assert log_loss_safe(y, y) == pytest.approx(0.0, abs=1e-10)

    def test_clipping_torna_erro_extremo_finito(self) -> None:
        valor = log_loss_safe([1.0], [0.0], eps=1e-12)
        assert np.isfinite(valor)
        assert valor == pytest.approx(-np.log(1e-12), rel=1e-9)

    def test_valor_conhecido(self) -> None:
        esperado = -(np.log(0.8) + np.log(1.0 - 0.3)) / 2.0
        assert log_loss_safe([1.0, 0.0], [0.8, 0.3]) == pytest.approx(esperado, abs=TOL)

    def test_eps_invalido_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="eps"):
            log_loss_safe([1.0], [0.5], eps=0.0)
        with pytest.raises(ValueError, match="eps"):
            log_loss_safe([1.0], [0.5], eps=0.5)

    def test_chute_meio_a_meio_da_ln2(self) -> None:
        y = np.array([0.0, 1.0])
        assert log_loss_safe(y, np.full(2, 0.5)) == pytest.approx(np.log(2.0), abs=TOL)


class TestFitIsotonic:
    def test_melhora_log_loss_de_probs_distorcidas(self) -> None:
        y, p_raw = _amostra_distorcida(n=5000, seed=42, a=0.4, b=0.5)
        cal = fit_isotonic(y, p_raw)
        p_cal = cal.transform(p_raw)
        assert log_loss_safe(y, p_cal) < log_loss_safe(y, p_raw)

    def test_reduz_ece(self) -> None:
        y, p_raw = _amostra_distorcida(n=5000, seed=7, a=0.4, b=0.5)
        cal = fit_isotonic(y, p_raw)
        assert expected_calibration_error(y, cal.transform(p_raw)) < expected_calibration_error(
            y, p_raw
        )

    def test_transform_e_nao_decrescente(self) -> None:
        y, p_raw = _amostra_distorcida(n=2000, seed=3)
        cal = fit_isotonic(y, p_raw)
        grade = np.linspace(0.0, 1.0, 201)
        saida = cal.transform(grade)
        assert np.all(np.diff(saida) >= -TOL)

    def test_saida_em_01_mesmo_fora_do_intervalo_de_treino(self) -> None:
        y = np.array([0.0, 0.0, 1.0, 1.0])
        p = np.array([0.3, 0.4, 0.6, 0.7])
        cal = fit_isotonic(y, p)
        saida = cal.transform(np.array([0.0, 0.05, 0.95, 1.0]))
        assert np.all(saida >= 0.0)
        assert np.all(saida <= 1.0)

    def test_entrada_invalida_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="apenas 0 e 1"):
            fit_isotonic([0.0, 3.0], [0.2, 0.8])


class TestFitPlatt:
    def test_recupera_distorcao_afim_no_logit(self) -> None:
        # p_raw = sigma(0.5*z + 0.4) com verdade sigma(z): a correção exata é
        # a = 1/0.5 = 2 e b = -0.4/0.5 = -0.8 em p_cal = sigma(a*logit(p_raw) + b).
        y, p_raw = _amostra_distorcida(n=40000, seed=11, a=0.5, b=0.4)
        cal = fit_platt(y, p_raw)
        assert cal.a == pytest.approx(2.0, abs=0.15)
        assert cal.b == pytest.approx(-0.8, abs=0.15)

    def test_melhora_log_loss_de_probs_distorcidas(self) -> None:
        y, p_raw = _amostra_distorcida(n=5000, seed=42, a=0.4, b=0.5)
        cal = fit_platt(y, p_raw)
        assert log_loss_safe(y, cal.transform(p_raw)) < log_loss_safe(y, p_raw)

    def test_transform_em_01_e_monotono(self) -> None:
        y, p_raw = _amostra_distorcida(n=2000, seed=5)
        cal = fit_platt(y, p_raw)
        grade = np.linspace(0.0, 1.0, 101)
        saida = cal.transform(grade)
        assert np.all(saida >= 0.0)
        assert np.all(saida <= 1.0)
        assert np.all(np.diff(saida) >= -TOL) if cal.a >= 0 else True

    def test_classe_unica_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="duas classes"):
            fit_platt([1.0, 1.0, 1.0], [0.2, 0.5, 0.8])

    def test_probs_ja_calibradas_dao_a_perto_de_1_e_b_perto_de_0(self) -> None:
        y, p_raw = _amostra_distorcida(n=40000, seed=13, a=1.0, b=0.0)
        cal = fit_platt(y, p_raw)
        assert cal.a == pytest.approx(1.0, abs=0.1)
        assert cal.b == pytest.approx(0.0, abs=0.1)


class TestExpectedCalibrationError:
    def test_ece_zero_para_probs_perfeitas(self) -> None:
        # Frequências empíricas construídas exatamente iguais à confiança de cada bin.
        p = np.concatenate([np.full(100, 0.25), np.full(100, 0.75)])
        y = np.concatenate([np.repeat([1.0, 0.0], [25, 75]), np.repeat([1.0, 0.0], [75, 25])])
        assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.0, abs=TOL)

    def test_ece_captura_descalibracao_grosseira(self) -> None:
        p = np.full(200, 0.9)
        y = np.repeat([1.0, 0.0], [100, 100])
        assert expected_calibration_error(y, p) == pytest.approx(0.4, abs=TOL)

    def test_ece_em_0_1(self) -> None:
        rng = np.random.default_rng(17)
        p = rng.uniform(size=300)
        y = rng.integers(0, 2, size=300).astype(float)
        valor = expected_calibration_error(y, p)
        assert 0.0 <= valor <= 1.0

    def test_ece_pequeno_para_amostra_calibrada_grande(self) -> None:
        rng = np.random.default_rng(19)
        p = rng.uniform(0.05, 0.95, size=50000)
        y = (rng.uniform(size=50000) < p).astype(float)
        assert expected_calibration_error(y, p, n_bins=10) < 0.02

    def test_isotonic_aproxima_ece_de_zero(self) -> None:
        y, p_raw = _amostra_distorcida(n=20000, seed=23, a=0.4, b=0.5)
        cal = fit_isotonic(y, p_raw)
        assert expected_calibration_error(y, cal.transform(p_raw)) < 0.02


class TestCoerenciaEntreMetricas:
    def test_platt_de_logit_e_consistente_com_formula(self) -> None:
        y, p_raw = _amostra_distorcida(n=1000, seed=29)
        cal = fit_platt(y, p_raw)
        p = np.array([0.2, 0.5, 0.8])
        esperado = expit(cal.a * logit(p) + cal.b)
        np.testing.assert_allclose(cal.transform(p), esperado, atol=1e-9)

    def test_metricas_aceitam_listas_python(self) -> None:
        assert brier_score([0.0, 1.0], [0.1, 0.9]) == pytest.approx(0.01, abs=TOL)
        assert np.isfinite(log_loss_safe([0.0, 1.0], [0.1, 0.9]))
