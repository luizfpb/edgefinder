"""Testes do módulo de sequências ("N dos últimos N")."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edgefinder.edge.streaks import (
    combined_streaks,
    hits_over_line,
    market_streak_text,
    player_summary,
    players_view_from_frame,
    streak_table,
    team_stats_averages,
    team_stats_streaks,
    team_streaks,
    team_view,
)


def _matches() -> pd.DataFrame:
    """Seis jogos do 'alfa' (3 casa, 3 fora), do mais antigo ao mais recente.

    Cronologia do alfa (gf x ga): 2x0, 0x0, 3x2, 1x1, 0x1, 4x0.
    """
    return pd.DataFrame(
        {
            "match_date": pd.to_datetime(
                ["2026-01-01", "2026-01-08", "2026-01-15", "2026-01-22", "2026-01-29", "2026-02-05"]
            ),
            "home_team": ["alfa", "beta", "alfa", "gama", "alfa", "delta"],
            "away_team": ["beta", "alfa", "gama", "alfa", "delta", "alfa"],
            "home_goals": [2, 0, 3, 1, 0, 0],
            "away_goals": [0, 0, 2, 1, 1, 4],
        }
    )


class TestTeamView:
    def test_perspectiva_do_time_gf_ga(self) -> None:
        view = team_view(_matches(), "alfa")
        assert len(view) == 6
        # mais recente primeiro: delta 0x4 (alfa fora, gf=4)
        assert view.iloc[0]["gf"] == 4
        assert view.iloc[0]["ga"] == 0
        assert view.iloc[0]["venue"] == "fora"
        assert view.iloc[0]["opponent"] == "delta"
        # mais antigo por ultimo: alfa 2x0 beta em casa
        assert view.iloc[-1]["gf"] == 2
        assert view.iloc[-1]["venue"] == "casa"

    def test_time_sem_jogos_devolve_vazio(self) -> None:
        assert team_view(_matches(), "omega").empty


class TestTeamStreaks:
    def test_contagens_ultimos_5(self) -> None:
        view = team_view(_matches(), "alfa")
        streaks = {s.key: s for s in team_streaks(view, 5)}
        # ultimos 5 do alfa (recente->antigo): 4x0, 0x1, 1x1, 3x2, 0x0
        assert streaks["total_over_0.5"].hits == 4  # so o 0x0 falha
        assert streaks["total_over_0.5"].total == 5
        assert streaks["total_over_2.5"].hits == 2  # 4x0 e 3x2
        assert streaks["team_scored"].hits == 3  # 4, 1, 3
        assert streaks["win"].hits == 2  # 4x0 e 3x2
        assert streaks["draw"].hits == 2  # 1x1 e 0x0
        assert streaks["loss"].hits == 1  # 0x1
        assert streaks["btts"].hits == 2  # 1x1 e 3x2

    def test_amostra_menor_que_n_aparece_no_denominador(self) -> None:
        view = team_view(_matches(), "beta")  # beta tem 2 jogos
        streaks = team_streaks(view, 10)
        assert all(s.total == 2 for s in streaks)

    def test_n_invalido_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="n deve ser"):
            team_streaks(team_view(_matches(), "alfa"), 0)


class TestCombined:
    def test_combinado_soma_hits_e_totais(self) -> None:
        va = team_view(_matches(), "alfa")
        vb = team_view(_matches(), "beta")
        combined = {s.key: s for s in combined_streaks(va, vb, 2)}
        # alfa ultimos 2: 4x0 e 0x1 -> over0.5 = 2; beta: 0x0 e 0x2 -> over0.5 = 1
        assert combined["total_over_0.5"].hits == 3
        assert combined["total_over_0.5"].total == 4


class TestStreakTable:
    def test_tabela_lado_a_lado(self) -> None:
        va = team_view(_matches(), "alfa")
        vb = team_view(_matches(), "beta")
        df = streak_table(va, vb, "alfa", "beta", 2)
        assert list(df.columns) == ["condicao", "alfa", "beta", "combinado"]
        over05 = df[df["condicao"] == "mais de 0.5 gols no jogo"].iloc[0]
        assert over05["alfa"] == "2/2"
        assert over05["beta"] == "1/2"
        assert over05["combinado"] == "3/4"


class TestAdHoc:
    def test_linha_arbitraria(self) -> None:
        view = team_view(_matches(), "alfa")
        s = hits_over_line(view, 5, 4.5)
        assert s.hits == 1  # so o 3x2 tem 5 gols
        assert s.total == 5
        assert "4.5" in s.label


def _stats_view() -> pd.DataFrame:
    """Quatro jogos com stats; escanteios ausentes no jogo mais antigo (NaN)."""
    return pd.DataFrame(
        {
            "match_date": pd.to_datetime(["2026-02-05", "2026-01-29", "2026-01-22", "2026-01-15"]),
            "gf": [2, 0, 1, 3],
            "ga": [0, 1, 1, 2],
            "shots": [15, 8, 11, 18],
            "shots_on_target": [7, 2, 4, 9],
            "corners": [6.0, 3.0, 5.0, np.nan],
            "fouls": [10.0, 14.0, 12.0, 11.0],
            "cards": [1.0, 3.0, 2.0, 1.0],
            "shots_opp": [7, 13, 10, 9],
            "shots_on_target_opp": [2, 6, 3, 4],
            "corners_opp": [4.0, 6.0, 4.0, np.nan],
            "fouls_opp": [13.0, 9.0, 15.0, 12.0],
            "cards_opp": [2.0, 2.0, 3.0, 1.0],
        }
    )


class TestTeamStatsStreaks:
    def test_denominador_conta_so_jogos_com_dado(self) -> None:
        lines = {s.key: s for s in team_stats_streaks(_stats_view(), 4)}
        # escanteios: só 3 jogos têm dado; totais 10, 9, 9 -> over 8.5 em 3/3
        assert lines["corners_over_8.5"].hits == 3
        assert lines["corners_over_8.5"].total == 3
        assert lines["corners_over_9.5"].hits == 1  # só o 6+4
        # cartões existem nos 4 jogos: totais 3, 5, 5, 2 -> over 3.5 em 2/4
        assert lines["cards_over_3.5"].hits == 2
        assert lines["cards_over_3.5"].total == 4
        # chutes no gol do time: 7, 2, 4, 9 -> over 4.5 em 2/4
        assert lines["team_sot_over_4.5"].hits == 2

    def test_condicao_sem_dado_algum_fica_de_fora(self) -> None:
        view = _stats_view().assign(corners=np.nan, corners_opp=np.nan)
        keys = {s.key for s in team_stats_streaks(view, 4)}
        assert "corners_over_8.5" not in keys
        assert "cards_over_3.5" in keys

    def test_view_vazia_devolve_lista_vazia(self) -> None:
        assert team_stats_streaks(pd.DataFrame(), 5) == []


class TestTeamStatsAverages:
    def test_medias_feitas_e_sofridas(self) -> None:
        df = team_stats_averages(_stats_view(), 4)
        chutes = df[df["stat"] == "chutes"].iloc[0]
        assert chutes["media"] == pytest.approx((15 + 8 + 11 + 18) / 4)
        assert chutes["media sofrida"] == pytest.approx((7 + 13 + 10 + 9) / 4)
        escanteios = df[df["stat"] == "escanteios"].iloc[0]
        assert escanteios["jogos"] == 3  # NaN no mais antigo


def _players() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": [1, 1, 2, 2, 2],
            "jogador": ["ana", "bia", "ana", "bia", "clara"],
            "minutes": [90, 90, 88, 0, 45],
            "goals": [1, 0, 2, 0, 0],
            "assists": [0, 1, 0, 0, 0],
            "shots": [3, 1, 5, 0, 1],
            "shots_on_target": [2, 0, 3, 0, 0],
            "yellow_cards": [0, 1, 1, 0, 0],
            "red_cards": [0, 0, 0, 0, 0],
            "fouls_committed": [2, 3, 1, 0, 0],
        }
    )


class TestPlayerSummary:
    def test_agrega_por_jogador_e_ordena_por_gols(self) -> None:
        out = player_summary(_players())
        assert out.iloc[0]["jogador"] == "ana"  # 3 gols
        ana = out.iloc[0]
        assert ana["jogos"] == 2
        assert ana["gols"] == 3
        assert ana["marcou em"] == "2/2"
        assert ana["chutes"] == 8
        assert ana["cartoes"] == 1

    def test_quem_nao_entrou_em_campo_fica_de_fora(self) -> None:
        out = player_summary(_players())
        bia = out[out["jogador"] == "bia"].iloc[0]
        assert bia["jogos"] == 1  # o jogo com 0 minutos nao conta
        assert bia["marcou em"] == "0/1"

    def test_vazio_devolve_vazio(self) -> None:
        assert player_summary(pd.DataFrame()).empty


class TestPlayersViewFromFrame:
    def test_prefere_fbref_e_limita_aos_ultimos_n_jogos(self) -> None:
        base = _players().assign(
            team="alfa",
            source="fbref",
            match_date=pd.to_datetime(["2026-01-01"] * 2 + ["2026-01-08"] * 3),
        )
        understat = base.assign(source="understat")
        frame = pd.concat([base, understat])
        view = players_view_from_frame(frame, "alfa", 1)
        assert (view["source"] == "fbref").all()
        assert set(view["match_id"]) == {2}  # so o jogo mais recente

    def test_time_ausente_devolve_vazio(self) -> None:
        assert players_view_from_frame(_players().assign(team="alfa", source="fbref"), "x", 5).empty


class TestMarketStreakText:
    def test_over(self) -> None:
        va = team_view(_matches(), "alfa")
        vb = team_view(_matches(), "beta")
        # totais do beta: 0 (0x0) e 2 (0x2) -> nenhum over 2.5
        txt = market_streak_text(va, vb, "ou", "over", 2.5, 5, "alfa", "beta")
        assert txt == "over 2.5: alfa 2/5, beta 0/2 (2/7)"

    def test_under_e_o_complemento(self) -> None:
        va = team_view(_matches(), "alfa")
        vb = team_view(_matches(), "beta")
        txt = market_streak_text(va, vb, "ou", "under", 2.5, 5, "alfa", "beta")
        assert txt == "under 2.5: alfa 3/5, beta 2/2 (5/7)"

    def test_1x2_home(self) -> None:
        va = team_view(_matches(), "alfa")
        vb = team_view(_matches(), "beta")
        txt = market_streak_text(va, vb, "1x2", "home", 0.0, 5, "alfa", "beta")
        assert txt == "alfa venceu 2/5"

    def test_mercado_desconhecido_devolve_vazio(self) -> None:
        va = team_view(_matches(), "alfa")
        assert market_streak_text(va, va, "ah", "home", 0.0, 5, "a", "b") == ""

    def test_sem_historico_devolve_vazio(self) -> None:
        vazio = team_view(_matches(), "omega")
        assert market_streak_text(vazio, vazio, "ou", "over", 2.5, 5, "a", "b") == ""
