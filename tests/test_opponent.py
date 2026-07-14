"""Testes de edgefinder.features.opponent.

Cobre caminho feliz, casos degenerados e as propriedades matemáticas:
média ponderada dos fatores igual a 1, monotonicidade de adjust_counts em k
e, crucialmente, a equivalência exata entre rolling_concession_factors e
concession_factors aplicado ao passado estrito de cada linha (anti-leakage).
"""

import numpy as np
import pandas as pd
import pytest

from edgefinder.features.opponent import (
    adjust_counts,
    concession_factors,
    rolling_concession_factors,
)


def _frame(teams: list[str], dates: list[str | pd.Timestamp], values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"team": teams, "match_date": pd.to_datetime(dates), "stat_conceded": values}
    )


class TestConcessionFactors:
    def test_caminho_feliz_pesos_uniformes(self) -> None:
        df = _frame(
            ["A", "A", "B", "B"],
            ["2026-01-01"] * 4,
            [10.0, 10.0, 20.0, 20.0],
        )
        out = concession_factors(df, half_life_days=30.0)
        assert list(out.columns) == ["team", "factor"]
        assert list(out["team"]) == ["A", "B"]
        assert out.loc[out["team"] == "A", "factor"].iloc[0] == pytest.approx(10.0 / 15.0)
        assert out.loc[out["team"] == "B", "factor"].iloc[0] == pytest.approx(20.0 / 15.0)

    def test_ponderacao_por_recencia(self) -> None:
        # half_life = 10 e jogos a 10 dias de distância: peso antigo = 0.5.
        # A: dia 0 -> 0, dia 10 -> 10; B: dia 0 -> 10, dia 10 -> 0.
        # taxa_A = (1*10 + 0.5*0)/1.5 = 20/3; taxa_B = 10/3; liga = 5.
        df = _frame(
            ["A", "A", "B", "B"],
            ["2026-01-01", "2026-01-11", "2026-01-01", "2026-01-11"],
            [0.0, 10.0, 10.0, 0.0],
        )
        out = concession_factors(df, half_life_days=10.0)
        factor = dict(zip(out["team"], out["factor"], strict=True))
        assert factor["A"] == pytest.approx((20.0 / 3.0) / 5.0)
        assert factor["B"] == pytest.approx((10.0 / 3.0) / 5.0)

    def test_media_dos_fatores_e_um_com_amostra_balanceada(self) -> None:
        # Mesmas datas e mesmo numero de jogos por time => pesos totais iguais
        # por time, logo a media simples dos fatores deve ser exatamente 1.
        rng = np.random.default_rng(11)
        teams = [t for t in "ABCDE" for _ in range(8)]
        dates = ["2026-03-01"] * len(teams)
        values = rng.poisson(9.0, size=len(teams)).astype(np.float64).tolist()
        out = concession_factors(_frame(teams, dates, values), half_life_days=60.0)
        assert float(out["factor"].mean()) == pytest.approx(1.0, abs=1e-12)

    def test_entrada_vazia(self) -> None:
        out = concession_factors(_frame([], [], []), half_life_days=30.0)
        assert list(out.columns) == ["team", "factor"]
        assert out.empty

    def test_liga_sem_concessoes_devolve_fator_neutro(self) -> None:
        df = _frame(["A", "B"], ["2026-01-01", "2026-01-02"], [0.0, 0.0])
        out = concession_factors(df, half_life_days=30.0)
        assert np.allclose(out["factor"].to_numpy(), 1.0)

    def test_coluna_ausente(self) -> None:
        with pytest.raises(ValueError, match="colunas"):
            concession_factors(pd.DataFrame({"team": ["A"]}), half_life_days=30.0)

    def test_valor_negativo_ou_nan(self) -> None:
        with pytest.raises(ValueError, match="stat_conceded"):
            concession_factors(_frame(["A"], ["2026-01-01"], [-1.0]), 30.0)
        with pytest.raises(ValueError, match="stat_conceded"):
            concession_factors(_frame(["A"], ["2026-01-01"], [float("nan")]), 30.0)

    def test_data_nula(self) -> None:
        df = pd.DataFrame({"team": ["A"], "match_date": [pd.NaT], "stat_conceded": [1.0]})
        with pytest.raises(ValueError, match="match_date"):
            concession_factors(df, half_life_days=30.0)

    @pytest.mark.parametrize("bad", [0.0, -5.0, float("inf"), float("nan")])
    def test_half_life_invalida(self, bad: float) -> None:
        df = _frame(["A"], ["2026-01-01"], [1.0])
        with pytest.raises(ValueError, match="half_life_days"):
            concession_factors(df, half_life_days=bad)


class TestAdjustCounts:
    def test_k_um_divide_pelo_fator(self) -> None:
        out = adjust_counts([10.0, 20.0], [2.0, 0.5], k=1.0)
        assert out == pytest.approx([5.0, 40.0])

    def test_k_zero_e_identidade(self) -> None:
        counts = np.array([3.0, 7.0, 0.0])
        out = adjust_counts(counts, np.array([1.7, 0.4, 2.0]), k=0.0)
        assert out == pytest.approx(counts)

    def test_fator_neutro_nao_altera(self) -> None:
        counts = np.array([1.0, 2.0, 3.0])
        out = adjust_counts(counts, np.ones(3), k=0.8)
        assert out == pytest.approx(counts)

    def test_k_fracionario(self) -> None:
        out = adjust_counts([8.0], [4.0], k=0.5)
        assert out == pytest.approx([8.0 / 2.0])

    def test_monotonicidade_em_k(self) -> None:
        # factor > 1: quanto maior k, menor a contagem ajustada.
        # factor < 1: quanto maior k, maior a contagem ajustada.
        ks = [0.0, 0.5, 1.0, 2.0]
        permissivo = [float(adjust_counts([10.0], [1.5], k=k)[0]) for k in ks]
        solido = [float(adjust_counts([10.0], [0.7], k=k)[0]) for k in ks]
        assert permissivo == sorted(permissivo, reverse=True)
        assert solido == sorted(solido)

    def test_shapes_incompativeis(self) -> None:
        with pytest.raises(ValueError, match="shapes"):
            adjust_counts([1.0, 2.0], [1.0], k=1.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_fator_invalido(self, bad: float) -> None:
        with pytest.raises(ValueError, match="opponents_factors"):
            adjust_counts([1.0], [bad], k=1.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_k_invalido(self, bad: float) -> None:
        with pytest.raises(ValueError, match="k deve"):
            adjust_counts([1.0], [1.0], k=bad)

    def test_entrada_vazia(self) -> None:
        out = adjust_counts(np.array([]), np.array([]), k=1.0)
        assert out.shape == (0,)


class TestRollingConcessionFactors:
    def test_equivale_a_concession_factors_no_passado_estrito(self) -> None:
        # Propriedade central anti-leakage: o fator de cada linha deve ser
        # identico ao de concession_factors calculado SO com jogos anteriores.
        # A referencia temporal difere (max do passado vs. data da linha), mas
        # o fator e razao de taxas com os mesmos pesos, entao a constante de
        # decaimento extra cancela e a igualdade e exata.
        rng = np.random.default_rng(42)
        teams = np.array(["A", "B", "C", "D"])
        rows: list[dict[str, object]] = []
        base = pd.Timestamp("2026-01-01")
        for _ in range(30):
            date = base + pd.Timedelta(days=int(rng.integers(0, 180)))
            for team in rng.choice(teams, size=2, replace=False):
                rows.append(
                    {
                        "team": str(team),
                        "match_date": date,
                        "stat_conceded": float(rng.poisson(12.0)),
                    }
                )
        df = pd.DataFrame(rows)
        half_life = 30.0
        out = rolling_concession_factors(df, half_life)
        for i in range(len(df)):
            past = df[df["match_date"] < df.loc[i, "match_date"]]
            expected = 1.0
            if not past.empty:
                cf = concession_factors(past, half_life)
                match = cf.loc[cf["team"] == df.loc[i, "team"], "factor"]
                if not match.empty:
                    expected = float(match.iloc[0])
            assert float(out.loc[i, "factor"]) == pytest.approx(expected, rel=1e-9), f"linha {i}"

    def test_primeiro_jogo_recebe_prior_neutro(self) -> None:
        df = _frame(
            ["A", "B", "A"],
            ["2026-01-01", "2026-01-05", "2026-01-10"],
            [10.0, 20.0, 5.0],
        )
        out = rolling_concession_factors(df, half_life_days=1000.0)
        assert float(out.loc[0, "factor"]) == 1.0
        # B na segunda linha: tem passado da liga (jogo de A), mas B mesmo
        # nunca jogou -> prior neutro.
        assert float(out.loc[1, "factor"]) == 1.0

    def test_exclui_jogos_da_mesma_data(self) -> None:
        d1, d2 = "2026-01-01", "2026-01-08"
        df = _frame(
            ["A", "B", "A", "B"],
            [d1, d1, d2, d2],
            [10.0, 20.0, 100.0, 200.0],
        )
        out = rolling_concession_factors(df, half_life_days=50.0)
        # Nas linhas de d2 so contam os jogos de d1 (o proprio d2 fica fora):
        # taxa_A = 10, taxa_B = 20, liga = 15 (razoes cancelam o decaimento).
        assert float(out.loc[2, "factor"]) == pytest.approx(10.0 / 15.0, rel=1e-12)
        assert float(out.loc[3, "factor"]) == pytest.approx(20.0 / 15.0, rel=1e-12)
        assert float(out.loc[0, "factor"]) == 1.0
        assert float(out.loc[1, "factor"]) == 1.0

    def test_preserva_ordem_e_indice_da_entrada(self) -> None:
        rng = np.random.default_rng(7)
        n = 40
        df = _frame(
            [str(t) for t in rng.choice(["A", "B", "C"], size=n)],
            list(pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 90, n), unit="D")),
            rng.poisson(8.0, n).astype(np.float64).tolist(),
        )
        shuffled = df.sample(frac=1.0, random_state=3)
        out = rolling_concession_factors(shuffled, half_life_days=20.0)
        assert list(out.index) == list(shuffled.index)
        base = rolling_concession_factors(df, half_life_days=20.0)
        for idx in df.index:
            assert float(out.loc[idx, "factor"]) == pytest.approx(
                float(base.loc[idx, "factor"]), rel=1e-12
            )

    def test_liga_com_passado_zerado_devolve_neutro(self) -> None:
        df = _frame(
            ["A", "B", "A"],
            ["2026-01-01", "2026-01-02", "2026-01-03"],
            [0.0, 0.0, 4.0],
        )
        out = rolling_concession_factors(df, half_life_days=30.0)
        assert np.allclose(out["factor"].to_numpy(), 1.0)

    def test_estavel_em_horizontes_longos(self) -> None:
        # 500 jogos ao longo de ~55 anos com meia-vida curta: a recorrencia
        # multiplicativa nao pode estourar nem virar NaN (o truque ingenuo de
        # cumsum com exp(+xi*t) estouraria em float64 muito antes disso).
        rng = np.random.default_rng(0)
        n = 500
        days = np.sort(rng.integers(0, 20_000, n))
        df = _frame(
            [str(t) for t in rng.choice(["A", "B", "C"], size=n)],
            list(pd.Timestamp("1970-01-01") + pd.to_timedelta(days, unit="D")),
            rng.poisson(10.0, n).astype(np.float64).tolist(),
        )
        out = rolling_concession_factors(df, half_life_days=5.0)
        factors = out["factor"].to_numpy(dtype=np.float64)
        assert bool(np.all(np.isfinite(factors)))
        assert bool(np.all(factors >= 0.0))

    def test_entrada_vazia(self) -> None:
        out = rolling_concession_factors(_frame([], [], []), half_life_days=30.0)
        assert "factor" in out.columns
        assert out.empty

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("inf")])
    def test_half_life_invalida(self, bad: float) -> None:
        df = _frame(["A"], ["2026-01-01"], [1.0])
        with pytest.raises(ValueError, match="half_life_days"):
            rolling_concession_factors(df, half_life_days=bad)
