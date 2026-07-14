"""Testes do módulo de decaimento exponencial por recência."""

from __future__ import annotations

import itertools
import math
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from edgefinder.features.decay import exponential_weights, half_life_grid, weighted_rate

REF = pd.Timestamp("2025-06-01")
TOL = 1e-12


class TestExponentialWeights:
    def test_peso_um_na_data_de_referencia(self) -> None:
        w = exponential_weights(pd.Series([REF]), REF, half_life_days=30.0)
        assert w[0] == pytest.approx(1.0, abs=TOL)

    def test_peso_meio_a_uma_meia_vida(self) -> None:
        datas = pd.Series([REF - pd.Timedelta(days=30)])
        w = exponential_weights(datas, REF, half_life_days=30.0)
        assert w[0] == pytest.approx(0.5, abs=TOL)

    def test_peso_um_quarto_a_duas_meias_vidas(self) -> None:
        datas = pd.Series([REF - pd.Timedelta(days=60)])
        w = exponential_weights(datas, REF, half_life_days=30.0)
        assert w[0] == pytest.approx(0.25, abs=TOL)

    def test_fracao_de_dia_conta(self) -> None:
        datas = pd.Series([REF - pd.Timedelta(hours=36)])
        w = exponential_weights(datas, REF, half_life_days=3.0)
        assert w[0] == pytest.approx(2.0 ** (-0.5), abs=TOL)

    def test_decaimento_monotonico(self) -> None:
        datas = pd.Series([REF - pd.Timedelta(days=d) for d in range(0, 400, 7)])
        w = exponential_weights(datas, REF, half_life_days=90.0)
        assert np.all(np.diff(w) < 0)

    def test_pesos_sempre_em_zero_um(self) -> None:
        rng = np.random.default_rng(42)
        dias = rng.uniform(0.0, 2000.0, size=200)
        datas = pd.Series([REF - pd.Timedelta(days=float(d)) for d in dias])
        w = exponential_weights(datas, REF, half_life_days=60.0)
        assert np.all(w > 0.0)
        assert np.all(w <= 1.0)

    def test_propriedade_multiplicativa(self) -> None:
        h = 45.0
        datas = pd.Series(
            [REF - pd.Timedelta(days=10), REF - pd.Timedelta(days=25), REF - pd.Timedelta(days=35)]
        )
        w = exponential_weights(datas, REF, half_life_days=h)
        assert w[2] == pytest.approx(w[0] * w[1], rel=1e-12)

    def test_meia_vida_menor_decai_mais_rapido(self) -> None:
        datas = pd.Series([REF - pd.Timedelta(days=50)])
        w_curta = exponential_weights(datas, REF, half_life_days=30.0)
        w_longa = exponential_weights(datas, REF, half_life_days=180.0)
        assert w_curta[0] < w_longa[0]

    def test_data_futura_levanta_valueerror(self) -> None:
        datas = pd.Series([REF - pd.Timedelta(days=5), REF + pd.Timedelta(days=1)])
        with pytest.raises(ValueError, match="vazamento"):
            exponential_weights(datas, REF, half_life_days=30.0)

    def test_ref_date_como_datetime_puro(self) -> None:
        ref = datetime(2025, 6, 1)
        datas = pd.Series([pd.Timestamp("2025-05-02")])
        w = exponential_weights(datas, ref, half_life_days=30.0)
        assert w[0] == pytest.approx(0.5, abs=TOL)

    @pytest.mark.parametrize("h", [0.0, -1.0, math.nan, math.inf])
    def test_meia_vida_invalida_levanta(self, h: float) -> None:
        datas = pd.Series([REF - pd.Timedelta(days=1)])
        with pytest.raises(ValueError, match="half_life_days"):
            exponential_weights(datas, REF, half_life_days=h)

    def test_nat_levanta_valueerror(self) -> None:
        datas = pd.Series([REF - pd.Timedelta(days=1), pd.NaT])
        with pytest.raises(ValueError, match="NaT"):
            exponential_weights(datas, REF, half_life_days=30.0)

    def test_serie_vazia_retorna_array_vazio(self) -> None:
        datas = pd.Series(pd.DatetimeIndex([]))
        w = exponential_weights(datas, REF, half_life_days=30.0)
        assert w.shape == (0,)
        assert w.dtype == np.float64

    def test_ordem_preservada(self) -> None:
        datas = pd.Series([REF - pd.Timedelta(days=100), REF, REF - pd.Timedelta(days=10)])
        w = exponential_weights(datas, REF, half_life_days=50.0)
        assert w[1] == pytest.approx(1.0, abs=TOL)
        assert w[0] < w[2] < w[1]


class TestWeightedRate:
    def test_taxa_conhecida_pesos_uniformes(self) -> None:
        values = np.array([3.0, 0.0])
        exposures = np.array([1.0, 1.0])
        weights = np.array([1.0, 1.0])
        assert weighted_rate(values, exposures, weights) == pytest.approx(1.5, abs=TOL)

    def test_pesos_uniformes_igual_taxa_bruta(self) -> None:
        rng = np.random.default_rng(7)
        values = rng.poisson(1.3, size=50).astype(np.float64)
        exposures = rng.uniform(0.1, 1.0, size=50)
        weights = np.full(50, 0.37)
        esperado = float(values.sum() / exposures.sum())
        assert weighted_rate(values, exposures, weights) == pytest.approx(esperado, rel=1e-12)

    def test_invariancia_a_escala_dos_pesos(self) -> None:
        values = np.array([2.0, 1.0, 0.0])
        exposures = np.array([1.0, 0.7, 0.5])
        weights = np.array([1.0, 0.5, 0.25])
        r1 = weighted_rate(values, exposures, weights)
        r2 = weighted_rate(values, exposures, weights * 123.4)
        assert r2 == pytest.approx(r1, rel=1e-12)

    def test_exposicao_em_unidades_de_90min(self) -> None:
        # 1 gol em meio jogo (45 min = 0.5 unidade) e taxa de 2 por 90.
        values = np.array([1.0])
        exposures = np.array([0.5])
        weights = np.array([1.0])
        assert weighted_rate(values, exposures, weights) == pytest.approx(2.0, abs=TOL)

    def test_peso_maior_puxa_para_o_jogo_recente(self) -> None:
        values = np.array([0.0, 3.0])
        exposures = np.array([1.0, 1.0])
        uniforme = weighted_rate(values, exposures, np.array([1.0, 1.0]))
        recente = weighted_rate(values, exposures, np.array([0.1, 1.0]))
        assert recente > uniforme

    def test_exposicao_total_zero_retorna_nan(self) -> None:
        values = np.array([1.0, 2.0])
        exposures = np.array([0.0, 0.0])
        weights = np.array([1.0, 1.0])
        assert math.isnan(weighted_rate(values, exposures, weights))

    def test_pesos_todos_zero_retorna_nan(self) -> None:
        values = np.array([1.0])
        exposures = np.array([1.0])
        weights = np.array([0.0])
        assert math.isnan(weighted_rate(values, exposures, weights))

    def test_arrays_vazios_retornam_nan(self) -> None:
        vazio = np.array([], dtype=np.float64)
        assert math.isnan(weighted_rate(vazio, vazio, vazio))

    def test_shapes_diferentes_levantam(self) -> None:
        with pytest.raises(ValueError, match="shapes"):
            weighted_rate(np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0, 1.0]))

    def test_exposicao_negativa_levanta(self) -> None:
        with pytest.raises(ValueError, match="negativos"):
            weighted_rate(np.array([1.0]), np.array([-0.5]), np.array([1.0]))

    def test_peso_negativo_levanta(self) -> None:
        with pytest.raises(ValueError, match="negativos"):
            weighted_rate(np.array([1.0]), np.array([1.0]), np.array([-1.0]))

    def test_valor_nao_finito_levanta(self) -> None:
        with pytest.raises(ValueError, match="finitos"):
            weighted_rate(np.array([math.nan]), np.array([1.0]), np.array([1.0]))


class TestHalfLifeGrid:
    def test_valores_documentados(self) -> None:
        assert half_life_grid() == [30.0, 60.0, 90.0, 120.0, 180.0, 365.0, 730.0]

    def test_todos_floats_positivos(self) -> None:
        grid = half_life_grid()
        assert all(isinstance(h, float) for h in grid)
        assert all(h > 0 for h in grid)

    def test_estritamente_crescente(self) -> None:
        grid = half_life_grid()
        assert all(a < b for a, b in itertools.pairwise(grid))

    def test_chamadas_independentes(self) -> None:
        grid = half_life_grid()
        grid.append(-1.0)
        assert half_life_grid() == [30.0, 60.0, 90.0, 120.0, 180.0, 365.0, 730.0]
