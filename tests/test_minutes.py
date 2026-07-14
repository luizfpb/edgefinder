"""Testes do modelo de minutos esperados por jogador.

Cobre a decomposição E[min] = P(joga) * E[min | joga], a ponderação
exponencial por recência (w = 2^(-dt/h)), o calendário do time explícito e
o aproximado (fallback), as conversões por 90 minutos e as barreiras
anti-vazamento temporal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edgefinder.features.minutes import expected_minutes, rescale_to_expected, to_per90

TOL = 1e-9
AS_OF = pd.Timestamp("2026-06-01")


def _hist(
    player_id: str,
    dates: list[pd.Timestamp],
    minutes: list[float],
    started: list[object] | None = None,
) -> pd.DataFrame:
    """Monta um histórico mínimo de um jogador com as colunas obrigatórias."""
    if started is None:
        started = [m > 0 for m in minutes]
    return pd.DataFrame(
        {
            "player_id": [player_id] * len(dates),
            "match_date": dates,
            "minutes": minutes,
            "started": pd.Series(started, dtype=object),
        }
    )


def _weekly_dates(last_offset_days: int, n: int = 10) -> list[pd.Timestamp]:
    """N datas semanais terminando ``last_offset_days`` dias antes de AS_OF."""
    last = AS_OF - pd.Timedelta(days=last_offset_days)
    return [last - pd.Timedelta(days=7 * i) for i in range(n - 1, -1, -1)]


class TestTitularFixo:
    def test_titular_de_90_minutos_projeta_90(self) -> None:
        hist = _hist("a", _weekly_dates(7), [90.0] * 10)
        out = expected_minutes(hist, AS_OF)
        row = out.iloc[0]
        assert row["p_play"] == pytest.approx(1.0, abs=TOL)
        assert row["exp_minutes_if_plays"] == pytest.approx(90.0, abs=TOL)
        assert row["exp_minutes"] == pytest.approx(90.0, abs=TOL)

    def test_decomposicao_exp_igual_p_vezes_condicional(self) -> None:
        dates = _weekly_dates(7)
        minutes = [90.0, 0.0, 85.0, 0.0, 90.0, 70.0, 0.0, 90.0, 88.0, 90.0]
        out = expected_minutes(_hist("a", dates, minutes), AS_OF)
        row = out.iloc[0]
        assert row["exp_minutes"] == pytest.approx(
            row["p_play"] * row["exp_minutes_if_plays"], abs=TOL
        )

    def test_saida_tem_colunas_e_ordenada_por_player_id(self) -> None:
        hist = pd.concat(
            [
                _hist("zeta", _weekly_dates(7), [90.0] * 10),
                _hist("alfa", _weekly_dates(7), [45.0] * 10),
            ],
            ignore_index=True,
        )
        out = expected_minutes(hist, AS_OF)
        assert list(out.columns) == [
            "player_id",
            "exp_minutes",
            "p_play",
            "exp_minutes_if_plays",
        ]
        assert list(out["player_id"]) == ["alfa", "zeta"]
        assert len(out) == 2


class TestPPlay:
    def test_jogador_parado_ha_meses_tem_p_play_baixo(self) -> None:
        hist = _hist("a", _weekly_dates(200), [90.0] * 10)
        out = expected_minutes(hist, AS_OF)
        row = out.iloc[0]
        assert row["p_play"] < 0.05
        assert row["exp_minutes"] < 5.0
        assert row["exp_minutes_if_plays"] == pytest.approx(90.0, abs=TOL)

    def test_afastamento_parcial_reduz_p_play_sem_zerar(self) -> None:
        ativo = expected_minutes(_hist("a", _weekly_dates(7), [90.0] * 10), AS_OF)
        parcial = expected_minutes(_hist("a", _weekly_dates(30), [90.0] * 10), AS_OF)
        p_ativo = float(ativo.iloc[0]["p_play"])
        p_parcial = float(parcial.iloc[0]["p_play"])
        assert 0.0 < p_parcial < p_ativo

    def test_team_dates_presenca_total_da_p_play_1(self) -> None:
        dates = _weekly_dates(7)
        team = pd.Series(dates)
        out = expected_minutes(_hist("a", dates, [90.0] * 10), AS_OF, team_dates=team)
        assert out.iloc[0]["p_play"] == pytest.approx(1.0, abs=TOL)

    def test_team_dates_presenca_parcial_com_pesos_uniformes(self) -> None:
        team_dates = _weekly_dates(7)
        played = team_dates[::2]
        team = pd.Series(team_dates)
        out = expected_minutes(
            _hist("a", played, [90.0] * len(played)),
            AS_OF,
            half_life_days=1e9,
            team_dates=team,
        )
        assert out.iloc[0]["p_play"] == pytest.approx(0.5, abs=1e-6)

    def test_presencas_recentes_pesam_mais_que_antigas(self) -> None:
        team_dates = _weekly_dates(7)
        team = pd.Series(team_dates)
        recente = _hist("a", team_dates[5:], [90.0] * 5)
        antigo = _hist("a", team_dates[:5], [90.0] * 5)
        p_recente = float(expected_minutes(recente, AS_OF, team_dates=team).iloc[0]["p_play"])
        p_antigo = float(expected_minutes(antigo, AS_OF, team_dates=team).iloc[0]["p_play"])
        assert p_recente > 0.5 > p_antigo

    def test_team_dates_futuras_sao_ignoradas(self) -> None:
        dates = _weekly_dates(7)
        team_passado = pd.Series(dates)
        team_com_fixture = pd.Series([*dates, AS_OF + pd.Timedelta(days=3)])
        hist = _hist("a", dates[::2], [90.0] * 5)
        out_a = expected_minutes(hist, AS_OF, team_dates=team_passado)
        out_b = expected_minutes(hist, AS_OF, team_dates=team_com_fixture)
        pd.testing.assert_frame_equal(out_a, out_b)

    def test_team_dates_comparadas_por_dia_ignorando_horario(self) -> None:
        match_dt = pd.Timestamp("2026-05-25 16:00")
        team = pd.Series([pd.Timestamp("2026-05-25 20:00")])
        out = expected_minutes(_hist("a", [match_dt], [90.0]), AS_OF, team_dates=team)
        assert out.iloc[0]["p_play"] == pytest.approx(1.0, abs=TOL)

    def test_minutos_zero_conta_como_ausencia(self) -> None:
        dates = _weekly_dates(7)
        minutes = [90.0] * 9 + [0.0]
        team = pd.Series(dates)
        out = expected_minutes(_hist("a", dates, minutes), AS_OF, team_dates=team)
        row = out.iloc[0]
        assert row["p_play"] < 1.0
        assert row["exp_minutes_if_plays"] == pytest.approx(90.0, abs=TOL)

    def test_p_play_sempre_em_0_1(self) -> None:
        rng = np.random.default_rng(42)
        team_dates = _weekly_dates(3)
        frames: list[pd.DataFrame] = []
        for pid in ("a", "b", "c", "d"):
            mask = rng.random(10) < 0.6
            played = [d for d, m in zip(team_dates, mask, strict=True) if m]
            if played:
                mins = list(rng.uniform(1.0, 90.0, size=len(played)))
                frames.append(_hist(pid, played, mins))
        out = expected_minutes(
            pd.concat(frames, ignore_index=True), AS_OF, team_dates=pd.Series(team_dates)
        )
        assert bool((out["p_play"] >= 0.0).all())
        assert bool((out["p_play"] <= 1.0).all())
        assert bool((out["exp_minutes"] <= out["exp_minutes_if_plays"] + TOL).all())


class TestExpMinutesIfPlays:
    def test_media_ponderada_verificada_na_mao(self) -> None:
        dates = [AS_OF - pd.Timedelta(days=61), AS_OF - pd.Timedelta(days=1)]
        hist = _hist("a", dates, [90.0, 60.0])
        out = expected_minutes(hist, AS_OF, half_life_days=60.0)
        w_recente = 2.0 ** (-1.0 / 60.0)
        w_antigo = 2.0 ** (-61.0 / 60.0)
        esperado = (60.0 * w_recente + 90.0 * w_antigo) / (w_recente + w_antigo)
        assert out.iloc[0]["exp_minutes_if_plays"] == pytest.approx(esperado, abs=1e-9)
        assert w_antigo / w_recente == pytest.approx(0.5, abs=TOL)

    def test_meia_vida_menor_puxa_para_o_recente(self) -> None:
        dates = [AS_OF - pd.Timedelta(days=61), AS_OF - pd.Timedelta(days=1)]
        hist = _hist("a", dates, [90.0, 60.0])
        curto = float(
            expected_minutes(hist, AS_OF, half_life_days=5.0).iloc[0]["exp_minutes_if_plays"]
        )
        longo = float(
            expected_minutes(hist, AS_OF, half_life_days=500.0).iloc[0]["exp_minutes_if_plays"]
        )
        assert 60.0 < curto < longo < 90.0

    def test_condicional_fica_entre_min_e_max_observados(self) -> None:
        minutes = [12.0, 90.0, 45.0, 77.0, 30.0]
        hist = _hist("a", _weekly_dates(7, n=5), minutes)
        val = float(expected_minutes(hist, AS_OF).iloc[0]["exp_minutes_if_plays"])
        assert min(minutes) <= val <= max(minutes)

    def test_jogador_sem_minutos_positivos(self) -> None:
        dates = _weekly_dates(7, n=3)
        out = expected_minutes(_hist("a", dates, [0.0, 0.0, 0.0]), AS_OF)
        row = out.iloc[0]
        assert np.isnan(row["exp_minutes_if_plays"])
        assert row["p_play"] == pytest.approx(0.0, abs=TOL)
        assert row["exp_minutes"] == pytest.approx(0.0, abs=TOL)


class TestAntiLeakage:
    def test_linha_futura_levanta_value_error(self) -> None:
        dates = [*_weekly_dates(7, n=3), AS_OF + pd.Timedelta(days=1)]
        hist = _hist("a", dates, [90.0] * 4)
        with pytest.raises(ValueError, match="as_of"):
            expected_minutes(hist, AS_OF)

    def test_linha_na_propria_data_as_of_levanta_value_error(self) -> None:
        hist = _hist("a", [AS_OF], [90.0])
        with pytest.raises(ValueError, match="as_of"):
            expected_minutes(hist, AS_OF)


class TestValidacao:
    def test_coluna_obrigatoria_ausente_levanta_erro(self) -> None:
        hist = _hist("a", _weekly_dates(7, n=2), [90.0, 90.0]).drop(columns=["minutes"])
        with pytest.raises(ValueError, match="minutes"):
            expected_minutes(hist, AS_OF)

    def test_minutos_negativos_levantam_erro(self) -> None:
        hist = _hist("a", _weekly_dates(7, n=2), [90.0, -5.0])
        with pytest.raises(ValueError, match="negativo"):
            expected_minutes(hist, AS_OF)

    @pytest.mark.parametrize("h", [0.0, -10.0, float("nan"), float("inf")])
    def test_meia_vida_invalida_levanta_erro(self, h: float) -> None:
        hist = _hist("a", _weekly_dates(7, n=2), [90.0, 90.0])
        with pytest.raises(ValueError, match="half_life_days"):
            expected_minutes(hist, AS_OF, half_life_days=h)

    def test_coluna_started_com_bool_e_nan_e_aceita(self) -> None:
        dates = _weekly_dates(7, n=3)
        hist = _hist("a", dates, [90.0, 60.0, 0.0], started=[True, float("nan"), False])
        out = expected_minutes(hist, AS_OF)
        assert len(out) == 1


class TestToPer90:
    def test_escalar_simples(self) -> None:
        assert to_per90(1.0, 45.0) == pytest.approx(2.0, abs=TOL)

    def test_minutos_zero_devolve_nan(self) -> None:
        assert np.isnan(float(np.asarray(to_per90(3.0, 0.0))))

    def test_minutos_negativos_devolvem_nan(self) -> None:
        assert np.isnan(float(np.asarray(to_per90(3.0, -10.0))))

    def test_array_elemento_a_elemento(self) -> None:
        value = np.array([1.0, 2.0, 3.0])
        minutes = np.array([90.0, 45.0, 0.0])
        out = np.asarray(to_per90(value, minutes))
        assert out[0] == pytest.approx(1.0, abs=TOL)
        assert out[1] == pytest.approx(4.0, abs=TOL)
        assert np.isnan(out[2])

    def test_series_preserva_indice(self) -> None:
        value = pd.Series([1.0, 2.0], index=["x", "y"])
        minutes = pd.Series([90.0, 90.0], index=["x", "y"])
        out = to_per90(value, minutes)
        assert isinstance(out, pd.Series)
        assert list(out.index) == ["x", "y"]
        assert out["y"] == pytest.approx(2.0, abs=TOL)


class TestRescaleToExpected:
    def test_escalar_simples(self) -> None:
        assert rescale_to_expected(2.0, 45.0) == pytest.approx(1.0, abs=TOL)

    def test_exp_minutes_zero_devolve_zero(self) -> None:
        assert rescale_to_expected(2.0, 0.0) == pytest.approx(0.0, abs=TOL)

    def test_nan_propaga(self) -> None:
        assert np.isnan(float(np.asarray(rescale_to_expected(float("nan"), 90.0))))
        assert np.isnan(float(np.asarray(rescale_to_expected(2.0, float("nan")))))

    def test_ida_e_volta_com_to_per90(self) -> None:
        value = np.array([0.0, 1.0, 4.5])
        minutes = np.array([30.0, 60.0, 90.0])
        out = np.asarray(rescale_to_expected(to_per90(value, minutes), minutes))
        np.testing.assert_allclose(out, value, atol=1e-12)

    def test_pipeline_com_expected_minutes(self) -> None:
        hist = _hist("a", _weekly_dates(7), [90.0] * 10)
        exp_min = float(expected_minutes(hist, AS_OF).iloc[0]["exp_minutes"])
        rate90 = float(np.asarray(to_per90(3.0, 90.0)))
        projecao = float(np.asarray(rescale_to_expected(rate90, exp_min)))
        assert projecao == pytest.approx(3.0, abs=1e-9)
