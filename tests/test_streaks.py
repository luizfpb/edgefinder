"""Testes do módulo de sequências ("N dos últimos N")."""

from __future__ import annotations

import pandas as pd
import pytest

from edgefinder.edge.streaks import (
    combined_streaks,
    hits_over_line,
    market_streak_text,
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


class TestAdHoc:
    def test_linha_arbitraria(self) -> None:
        view = team_view(_matches(), "alfa")
        s = hits_over_line(view, 5, 4.5)
        assert s.hits == 1  # so o 3x2 tem 5 gols
        assert s.total == 5
        assert "4.5" in s.label


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
