"""Testes de edgefinder.features.stability.

Cobre caminho feliz, degenerados e propriedades: r_sb = 1 para sinal puro,
r_sb ~ 0 para ruido puro, autocorrelacao exata (+1 tendencia, -1 alternancia),
invariancia a embaralhamento de linhas (ordenacao interna por data) e
consistencia do relatorio com as funcoes individuais.
"""

import math

import numpy as np
import pandas as pd
import pytest

from edgefinder.features.stability import (
    autocorrelation_lag1,
    split_half_reliability,
    stability_report,
)


def _daily_dates(n: int, start: str = "2026-01-01") -> list[pd.Timestamp]:
    base = pd.Timestamp(start)
    return [base + pd.Timedelta(days=i) for i in range(n)]


def _player_frame(series_by_player: dict[int, list[float]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pid, values in series_by_player.items():
        for date, value in zip(_daily_dates(len(values)), values, strict=True):
            rows.append({"player_id": pid, "match_date": date, "value": value})
    return pd.DataFrame(rows)


class TestSplitHalfReliability:
    def test_sinal_puro_da_um(self) -> None:
        # Valor constante por jogador: as duas metades coincidem exatamente,
        # r = 1 e a correcao de Spearman-Brown preserva 1.
        df = _player_frame({1: [1.0] * 10, 2: [2.0] * 10, 3: [3.0] * 10})
        assert split_half_reliability(df[["player_id", "value"]]) == pytest.approx(1.0)

    def test_sinal_forte_fica_perto_de_um(self) -> None:
        rng = np.random.default_rng(7)
        data = {
            pid: (float(rng.normal(10.0, 3.0)) + rng.normal(0.0, 0.5, 12)).tolist()
            for pid in range(30)
        }
        r_sb = split_half_reliability(_player_frame(data)[["player_id", "value"]])
        assert r_sb > 0.9

    def test_ruido_puro_fica_perto_de_zero(self) -> None:
        rng = np.random.default_rng(123)
        data = {pid: rng.normal(5.0, 1.0, 12).tolist() for pid in range(60)}
        r_sb = split_half_reliability(_player_frame(data)[["player_id", "value"]])
        assert abs(r_sb) < 0.35

    def test_r_zero_da_r_sb_zero(self) -> None:
        # medias pares [0, 1, 2] e impares [0, 1, 0]: produto interno das
        # dispersoes e zero, logo r = 0 e r_sb = 2*0/(1+0) = 0.
        df = pd.DataFrame(
            {"player_id": [1, 1, 2, 2, 3, 3], "value": [0.0, 0.0, 1.0, 1.0, 2.0, 0.0]}
        )
        assert split_half_reliability(df, min_games=2) == pytest.approx(0.0, abs=1e-12)

    def test_r_menos_um_da_nan(self) -> None:
        # r = -1 anula o denominador de 2r/(1+r): indefinido -> NaN.
        df = pd.DataFrame({"player_id": [1, 1, 2, 2], "value": [0.0, 1.0, 1.0, 0.0]})
        assert math.isnan(split_half_reliability(df, min_games=2))

    def test_filtra_jogadores_com_poucos_jogos(self) -> None:
        base = _player_frame({1: [1.0] * 10, 2: [2.0] * 10, 3: [3.0] * 10})
        curto = _player_frame({99: [50.0, -50.0, 100.0]})
        com_ruido = pd.concat([base, curto], ignore_index=True)
        r_base = split_half_reliability(base[["player_id", "value"]], min_games=10)
        r_full = split_half_reliability(com_ruido[["player_id", "value"]], min_games=10)
        assert r_full == pytest.approx(r_base)

    def test_menos_de_dois_elegiveis_da_nan(self) -> None:
        df = _player_frame({1: [1.0] * 10, 2: [2.0] * 3})
        assert math.isnan(split_half_reliability(df[["player_id", "value"]], min_games=10))

    def test_entrada_vazia_da_nan(self) -> None:
        df = pd.DataFrame({"player_id": pd.Series(dtype=object), "value": []})
        assert math.isnan(split_half_reliability(df))

    def test_variancia_zero_entre_jogadores_da_nan(self) -> None:
        df = _player_frame({1: [4.0] * 10, 2: [4.0] * 10})
        assert math.isnan(split_half_reliability(df[["player_id", "value"]]))

    @pytest.mark.parametrize("bad", [1, 0, -3])
    def test_min_games_invalido(self, bad: int) -> None:
        df = pd.DataFrame({"player_id": [1], "value": [1.0]})
        with pytest.raises(ValueError, match="min_games"):
            split_half_reliability(df, min_games=bad)

    def test_valor_nao_finito(self) -> None:
        df = pd.DataFrame({"player_id": [1, 1], "value": [1.0, float("nan")]})
        with pytest.raises(ValueError, match="value"):
            split_half_reliability(df, min_games=2)

    def test_coluna_ausente(self) -> None:
        with pytest.raises(ValueError, match="colunas"):
            split_half_reliability(pd.DataFrame({"player_id": [1]}))


class TestAutocorrelationLag1:
    def test_tendencia_monotona_da_um(self) -> None:
        df = _player_frame({1: [float(i) for i in range(10)]})
        assert autocorrelation_lag1(df) == pytest.approx(1.0)

    def test_alternancia_da_menos_um(self) -> None:
        df = _player_frame({1: [0.0, 1.0] * 5})
        assert autocorrelation_lag1(df) == pytest.approx(-1.0)

    def test_media_entre_jogadores(self) -> None:
        df = _player_frame({1: [float(i) for i in range(10)], 2: [0.0, 1.0] * 5})
        assert autocorrelation_lag1(df) == pytest.approx(0.0, abs=1e-12)

    def test_ordena_por_data_internamente(self) -> None:
        df = _player_frame({1: [float(i) for i in range(12)]})
        shuffled = df.sample(frac=1.0, random_state=5).reset_index(drop=True)
        assert autocorrelation_lag1(shuffled) == pytest.approx(1.0)

    def test_ruido_iid_fica_perto_de_zero(self) -> None:
        rng = np.random.default_rng(99)
        df = _player_frame({1: rng.normal(0.0, 1.0, 200).tolist()})
        assert abs(autocorrelation_lag1(df)) < 0.25

    def test_jogador_constante_fica_fora_da_media(self) -> None:
        df = _player_frame({1: [5.0] * 10, 2: [float(i) for i in range(10)]})
        assert autocorrelation_lag1(df) == pytest.approx(1.0)

    def test_filtra_jogadores_com_poucos_jogos(self) -> None:
        df = _player_frame({1: [float(i) for i in range(10)], 2: [9.0, 0.0, 9.0]})
        assert autocorrelation_lag1(df, min_games=10) == pytest.approx(1.0)

    def test_sem_elegiveis_da_nan(self) -> None:
        df = _player_frame({1: [1.0, 2.0]})
        assert math.isnan(autocorrelation_lag1(df, min_games=10))

    def test_entrada_vazia_da_nan(self) -> None:
        df = pd.DataFrame(
            {
                "player_id": pd.Series(dtype=object),
                "match_date": pd.Series(dtype="datetime64[ns]"),
                "value": pd.Series(dtype=np.float64),
            }
        )
        assert math.isnan(autocorrelation_lag1(df))

    def test_data_nula(self) -> None:
        df = pd.DataFrame(
            {
                "player_id": [1, 1],
                "match_date": [pd.Timestamp("2026-01-01"), pd.NaT],
                "value": [1.0, 2.0],
            }
        )
        with pytest.raises(ValueError, match="match_date"):
            autocorrelation_lag1(df, min_games=2)

    def test_min_games_invalido(self) -> None:
        df = _player_frame({1: [1.0]})
        with pytest.raises(ValueError, match="min_games"):
            autocorrelation_lag1(df, min_games=1)

    def test_coluna_ausente(self) -> None:
        with pytest.raises(ValueError, match="colunas"):
            autocorrelation_lag1(pd.DataFrame({"player_id": [1], "value": [1.0]}))


class TestStabilityReport:
    @staticmethod
    def _long_frame() -> pd.DataFrame:
        rng = np.random.default_rng(21)
        rows: list[dict[str, object]] = []
        for pid in range(15):
            skill = float(rng.normal(10.0, 3.0))
            dates = _daily_dates(12)
            for i, date in enumerate(dates):
                rows.append(
                    {
                        "player_id": pid,
                        "match_date": date,
                        "market": "chutes",
                        "value": skill + float(rng.normal(0.0, 0.4)),
                    }
                )
                rows.append(
                    {
                        "player_id": pid,
                        "match_date": date,
                        "market": "tendencia",
                        "value": float(i),
                    }
                )
        # jogador com poucos jogos: nao deve contar em n_players
        for date in _daily_dates(3):
            rows.append({"player_id": 999, "match_date": date, "market": "chutes", "value": 1.0})
        return pd.DataFrame(rows)

    def test_estrutura_e_contagens(self) -> None:
        rep = stability_report(self._long_frame(), min_games=10)
        assert list(rep.columns) == ["market", "split_half", "autocorr", "n_players"]
        assert list(rep["market"]) == ["chutes", "tendencia"]
        assert rep["n_players"].tolist() == [15, 15]
        assert str(rep["n_players"].dtype) == "int64"

    def test_valores_refletem_a_natureza_de_cada_mercado(self) -> None:
        rep = stability_report(self._long_frame(), min_games=10).set_index("market")
        # chutes: habilidade estavel + ruido iid -> split-half alto,
        # autocorrelacao baixa (o ruido em torno da habilidade nao persiste).
        assert float(rep.loc["chutes", "split_half"]) > 0.9
        assert abs(float(rep.loc["chutes", "autocorr"])) < 0.4
        # tendencia: todo jogador cresce linearmente -> autocorr = 1, mas as
        # medias das metades sao identicas entre jogadores -> split_half NaN.
        assert float(rep.loc["tendencia", "autocorr"]) == pytest.approx(1.0)
        assert math.isnan(float(rep.loc["tendencia", "split_half"]))

    def test_consistente_com_funcoes_individuais(self) -> None:
        df = self._long_frame()
        rep = stability_report(df, min_games=10).set_index("market")
        for market in ["chutes", "tendencia"]:
            sub = df[df["market"] == market].sort_values(["player_id", "match_date"], kind="stable")
            expected_sh = split_half_reliability(sub[["player_id", "value"]], min_games=10)
            expected_ac = autocorrelation_lag1(
                sub[["player_id", "match_date", "value"]], min_games=10
            )
            got_sh = float(rep.loc[market, "split_half"])
            got_ac = float(rep.loc[market, "autocorr"])
            assert (math.isnan(got_sh) and math.isnan(expected_sh)) or got_sh == pytest.approx(
                expected_sh
            )
            assert got_ac == pytest.approx(expected_ac)

    def test_linhas_embaralhadas_dao_o_mesmo_relatorio(self) -> None:
        df = self._long_frame()
        rep_a = stability_report(df, min_games=10)
        rep_b = stability_report(df.sample(frac=1.0, random_state=1), min_games=10)
        pd.testing.assert_frame_equal(rep_a.reset_index(drop=True), rep_b.reset_index(drop=True))

    def test_entrada_vazia(self) -> None:
        df = pd.DataFrame(
            {
                "player_id": pd.Series(dtype=object),
                "match_date": pd.Series(dtype="datetime64[ns]"),
                "market": pd.Series(dtype=object),
                "value": pd.Series(dtype=np.float64),
            }
        )
        rep = stability_report(df)
        assert list(rep.columns) == ["market", "split_half", "autocorr", "n_players"]
        assert rep.empty

    def test_coluna_ausente(self) -> None:
        with pytest.raises(ValueError, match="colunas"):
            stability_report(pd.DataFrame({"player_id": [1], "value": [1.0]}))

    def test_min_games_invalido(self) -> None:
        with pytest.raises(ValueError, match="min_games"):
            stability_report(self._long_frame(), min_games=1)
