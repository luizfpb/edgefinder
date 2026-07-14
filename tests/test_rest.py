"""Testes das features de descanso e congestão de calendário."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edgefinder.features.rest import rest_features

COLUNAS = ["team", "match_date", "days_since_last", "matches_last_14d"]


def _calendario(team: str, datas: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"team": team, "match_date": pd.to_datetime(datas)})


class TestCaminhoFeliz:
    def test_um_time_sequencia_simples(self) -> None:
        df = _calendario("gremio", ["2025-01-01", "2025-01-05", "2025-01-10", "2025-01-30"])
        out = rest_features(df)
        assert pd.isna(out.loc[0, "days_since_last"])
        assert out.loc[1, "days_since_last"] == pytest.approx(4.0)
        assert out.loc[2, "days_since_last"] == pytest.approx(5.0)
        assert out.loc[3, "days_since_last"] == pytest.approx(20.0)
        assert out["matches_last_14d"].tolist() == [0, 1, 2, 0]

    def test_colunas_e_dtypes(self) -> None:
        df = _calendario("gremio", ["2025-01-01", "2025-01-05"])
        out = rest_features(df)
        assert list(out.columns) == COLUNAS
        assert out["days_since_last"].dtype == np.float64
        assert out["matches_last_14d"].dtype == np.int64

    def test_horario_gera_descanso_fracionario(self) -> None:
        df = pd.DataFrame(
            {
                "team": ["inter", "inter"],
                "match_date": pd.to_datetime(["2025-03-01 16:00", "2025-03-03 04:00"]),
            }
        )
        out = rest_features(df)
        assert out.loc[1, "days_since_last"] == pytest.approx(1.5)

    def test_colunas_extras_sao_ignoradas(self) -> None:
        df = _calendario("gremio", ["2025-01-01", "2025-01-05"])
        df["placar"] = [1, 2]
        out = rest_features(df)
        assert list(out.columns) == COLUNAS


class TestJanela14Dias:
    def test_limite_inferior_inclusivo_14_dias(self) -> None:
        df = _calendario("bahia", ["2025-01-01", "2025-01-15"])
        out = rest_features(df)
        assert out.loc[1, "matches_last_14d"] == 1

    def test_15_dias_fica_fora_da_janela(self) -> None:
        df = _calendario("bahia", ["2025-01-01", "2025-01-16"])
        out = rest_features(df)
        assert out.loc[1, "matches_last_14d"] == 0

    def test_propria_partida_excluida(self) -> None:
        df = _calendario("bahia", ["2025-01-10"])
        out = rest_features(df)
        assert out.loc[0, "matches_last_14d"] == 0

    def test_jogo_futuro_nao_conta_nem_altera_o_passado(self) -> None:
        passado = ["2025-02-01", "2025-02-05", "2025-02-09"]
        so_passado = rest_features(_calendario("santos", passado))
        com_futuro = rest_features(_calendario("santos", [*passado, "2025-02-12"]))
        pd.testing.assert_frame_equal(com_futuro.iloc[:3], so_passado)

    def test_congestao_acumula(self) -> None:
        datas = [f"2025-04-{d:02d}" for d in range(1, 15, 3)]  # jogos a cada 3 dias
        out = rest_features(_calendario("flamengo", datas))
        assert out["matches_last_14d"].tolist() == [0, 1, 2, 3, 4]


class TestTimesIndependentes:
    def test_um_time_nao_contamina_outro(self) -> None:
        df = pd.DataFrame(
            {
                "team": ["a", "b", "a", "b"],
                "match_date": pd.to_datetime(
                    ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]
                ),
            }
        )
        out = rest_features(df)
        assert pd.isna(out.loc[0, "days_since_last"])
        assert pd.isna(out.loc[1, "days_since_last"])
        assert out.loc[2, "days_since_last"] == pytest.approx(2.0)
        assert out.loc[3, "days_since_last"] == pytest.approx(2.0)
        assert out["matches_last_14d"].tolist() == [0, 0, 1, 1]

    def test_equivale_a_calcular_cada_time_isolado(self) -> None:
        rng = np.random.default_rng(11)
        linhas: list[dict[str, object]] = []
        base = pd.Timestamp("2025-01-01")
        for team in ("a", "b", "c"):
            dias = np.cumsum(rng.integers(1, 10, size=12))
            linhas.extend(
                {"team": team, "match_date": base + pd.Timedelta(days=int(d))} for d in dias
            )
        df = pd.DataFrame(linhas).sample(frac=1.0, random_state=3).reset_index(drop=True)
        out = rest_features(df)
        for team in ("a", "b", "c"):
            isolado = rest_features(df[df["team"] == team])
            pd.testing.assert_frame_equal(out[out["team"] == team], isolado)


class TestAlinhamento:
    def test_entrada_desordenada_preserva_ordem_e_indice(self) -> None:
        df = pd.DataFrame(
            {
                "team": ["x", "x", "x"],
                "match_date": pd.to_datetime(["2025-01-10", "2025-01-01", "2025-01-05"]),
            },
            index=[7, 3, 5],
        )
        out = rest_features(df)
        assert list(out.index) == [7, 3, 5]
        assert out.loc[7, "days_since_last"] == pytest.approx(5.0)
        assert pd.isna(out.loc[3, "days_since_last"])
        assert out.loc[5, "days_since_last"] == pytest.approx(4.0)
        assert out.loc[7, "matches_last_14d"] == 2

    def test_indice_duplicado_nao_quebra(self) -> None:
        df = pd.DataFrame(
            {
                "team": ["x", "x"],
                "match_date": pd.to_datetime(["2025-01-01", "2025-01-04"]),
            },
            index=[0, 0],
        )
        out = rest_features(df)
        assert len(out) == 2
        assert out["days_since_last"].iloc[1] == pytest.approx(3.0)


class TestDegenerados:
    def test_dataframe_vazio(self) -> None:
        df = pd.DataFrame({"team": pd.Series(dtype=object), "match_date": pd.Series(dtype=object)})
        out = rest_features(df)
        assert list(out.columns) == COLUNAS
        assert len(out) == 0

    def test_coluna_faltando_levanta(self) -> None:
        with pytest.raises(ValueError, match="match_date"):
            rest_features(pd.DataFrame({"team": ["a"]}))

    def test_nat_levanta(self) -> None:
        df = pd.DataFrame({"team": ["a"], "match_date": [pd.NaT]})
        with pytest.raises(ValueError, match="NaT"):
            rest_features(df)

    def test_mesmo_dia_duplicado(self) -> None:
        df = _calendario("y", ["2025-01-01", "2025-01-01"])
        out = rest_features(df)
        assert pd.isna(out.loc[0, "days_since_last"])
        assert out.loc[1, "days_since_last"] == pytest.approx(0.0)
        # Jogo na mesma data não está nos dias ANTERIORES: nenhum conta.
        assert out["matches_last_14d"].tolist() == [0, 0]

    def test_entrada_nao_e_mutada(self) -> None:
        df = _calendario("z", ["2025-01-01", "2025-01-05"])
        copia = df.copy(deep=True)
        rest_features(df)
        pd.testing.assert_frame_equal(df, copia)


class TestPropriedades:
    def test_descanso_nao_negativo_e_carga_limitada(self) -> None:
        rng = np.random.default_rng(99)
        base = pd.Timestamp("2024-08-01")
        dias = np.cumsum(rng.integers(1, 8, size=60))
        df = pd.DataFrame(
            {"team": "w", "match_date": [base + pd.Timedelta(days=int(d)) for d in dias]}
        )
        out = rest_features(df)
        validos = out["days_since_last"].dropna()
        assert (validos >= 0).all()
        assert (out["matches_last_14d"] >= 0).all()
        # A carga na janela nunca excede o total de jogos anteriores.
        assert (out["matches_last_14d"].to_numpy() <= np.arange(len(out))).all()

    def test_carga_conta_somente_a_janela(self) -> None:
        # Jogos diários: aos i-ésimos, os anteriores dentro de 14 dias são min(i, 14).
        datas = pd.date_range("2025-05-01", periods=20, freq="D")
        df = pd.DataFrame({"team": "v", "match_date": datas})
        out = rest_features(df)
        esperado = [min(i, 14) for i in range(20)]
        assert out["matches_last_14d"].tolist() == esperado
