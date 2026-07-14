"""Testes do loop de CLV: registro (com dedup), liquidação e closing line.

Banco SQLite em memória com fixtures sintéticas — nenhuma rede, nenhum banco
real (regra do projeto).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import Engine, create_engine

from edgefinder.edge.tracking import (
    record_defensible_paper_bets,
    record_paper_bet,
    settle_bets,
    update_clv,
)
from edgefinder.storage import schema
from edgefinder.storage.repository import read_df, upsert
from edgefinder.timeutil import utcnow_naive


@pytest.fixture()
def engine() -> Engine:
    eng = create_engine("sqlite:///:memory:")
    schema.metadata.create_all(eng)
    upsert(
        eng,
        schema.competitions,
        [{"id": "ENG-Premier League", "name": "Premier League", "tier": 1, "kind": "league"}],
        conflict_cols=["id"],
    )
    upsert(
        eng,
        schema.teams,
        [{"id": 1, "name": "alfa"}, {"id": 2, "name": "beta"}],
        conflict_cols=["id"],
    )
    upsert(
        eng,
        schema.matches,
        [
            {
                "id": 10,
                "competition_id": "ENG-Premier League",
                "season": "2526",
                "match_date": utcnow_naive() - timedelta(days=1),
                "home_team_id": 1,
                "away_team_id": 2,
                "home_goals": 2,
                "away_goals": 1,
                "status": "played",
            },
            {
                "id": 11,
                "competition_id": "ENG-Premier League",
                "season": "2526",
                "match_date": utcnow_naive() + timedelta(days=3),
                "home_team_id": 2,
                "away_team_id": 1,
                "home_goals": None,
                "away_goals": None,
                "status": "scheduled",
            },
        ],
        conflict_cols=["id"],
    )
    return eng


class TestRecordPaperBet:
    def test_registra_e_deduplica(self, engine: Engine) -> None:
        assert record_paper_bet(engine, 10, "1x2", "home", 2.10, 1.0)
        assert not record_paper_bet(engine, 10, "1x2", "home", 2.30, 1.0)  # dedup
        bets = read_df(engine, "SELECT * FROM bets")
        assert len(bets) == 1
        assert bets.iloc[0]["odds_taken"] == 2.10  # a primeira odd fica

    def test_selecoes_diferentes_nao_colidem(self, engine: Engine) -> None:
        assert record_paper_bet(engine, 10, "1x2", "home", 2.10, 1.0)
        assert record_paper_bet(engine, 10, "1x2", "away", 3.40, 1.0)
        assert record_paper_bet(engine, 10, "ou", "over", 1.90, 1.0, line=2.5)
        assert record_paper_bet(engine, 10, "ou", "over", 2.00, 1.0, line=3.5)
        assert len(read_df(engine, "SELECT * FROM bets")) == 4


class TestRecordDefensible:
    def test_registra_so_defensaveis(self, engine: Engine) -> None:
        analysis = pd.DataFrame(
            [
                {
                    "match_id": 11,
                    "mercado": "1x2",
                    "selecao": "home",
                    "melhor_odd": 2.05,
                    "linha": 0.0,
                    "veredicto": "defensavel",
                },
                {
                    "match_id": 11,
                    "mercado": "ou",
                    "selecao": "over",
                    "melhor_odd": 1.95,
                    "linha": 2.5,
                    "veredicto": "evitar",
                },
            ]
        )
        assert record_defensible_paper_bets(engine, analysis) == 1
        # rodar de novo nao duplica
        assert record_defensible_paper_bets(engine, analysis) == 0
        bets = read_df(engine, "SELECT * FROM bets")
        assert len(bets) == 1
        assert bets.iloc[0]["stake"] == 1.0

    def test_df_vazio_e_no_op(self, engine: Engine) -> None:
        assert record_defensible_paper_bets(engine, pd.DataFrame()) == 0


class TestSettleBets:
    def test_1x2_win_lose(self, engine: Engine) -> None:
        record_paper_bet(engine, 10, "1x2", "home", 2.0, 1.0)  # placar 2x1: ganha
        record_paper_bet(engine, 10, "1x2", "away", 3.0, 1.0)  # perde
        assert settle_bets(engine) == 2
        bets = read_df(engine, "SELECT selection, result, pnl FROM bets ORDER BY selection")
        away, home = bets.iloc[0], bets.iloc[1]
        assert home["result"] == "win" and home["pnl"] == pytest.approx(1.0)
        assert away["result"] == "lose" and away["pnl"] == pytest.approx(-1.0)

    def test_ou_win_lose_push(self, engine: Engine) -> None:
        record_paper_bet(engine, 10, "ou", "over", 1.9, 1.0, line=2.5)  # 3 gols: ganha
        record_paper_bet(engine, 10, "ou", "under", 1.9, 1.0, line=3.5)  # ganha
        record_paper_bet(engine, 10, "ou", "over", 1.9, 1.0, line=3.0)  # push (3 == 3.0)
        assert settle_bets(engine) == 3
        bets = read_df(engine, "SELECT line, selection, result, pnl FROM bets")
        by_key = {(r["line"], r["selection"]): r for _, r in bets.iterrows()}
        assert by_key[(2.5, "over")]["result"] == "win"
        assert by_key[(3.5, "under")]["result"] == "win"
        assert by_key[(3.0, "over")]["result"] == "push"
        assert by_key[(3.0, "over")]["pnl"] == pytest.approx(0.0)

    def test_jogo_sem_placar_fica_aberto(self, engine: Engine) -> None:
        record_paper_bet(engine, 11, "1x2", "home", 2.0, 1.0)
        assert settle_bets(engine) == 0


class TestUpdateClv:
    def _snapshot(
        self, engine: Engine, collected_at: datetime, odds: float, commence: datetime
    ) -> None:
        upsert(
            engine,
            schema.odds_snapshots,
            [
                {
                    "match_id": 10,
                    "home_team_raw": "alfa",
                    "away_team_raw": "beta",
                    "commence_time": commence,
                    "source": "theoddsapi",
                    "bookmaker": "book",
                    "market": "1x2",
                    "selection": "home",
                    "line": 0.0,
                    "odds_decimal": odds,
                    "collected_at": collected_at,
                    "is_closing": False,
                }
            ],
            conflict_cols=[
                "match_id",
                "source",
                "bookmaker",
                "market",
                "selection",
                "line",
                "collected_at",
            ],
        )

    def test_clv_usa_o_ultimo_snapshot_pre_kickoff(self, engine: Engine) -> None:
        kickoff = utcnow_naive() - timedelta(hours=2)
        record_paper_bet(engine, 10, "1x2", "home", 2.20, 1.0)
        self._snapshot(engine, kickoff - timedelta(days=1), 2.30, kickoff)
        self._snapshot(engine, kickoff - timedelta(hours=1), 2.00, kickoff)  # closing
        assert update_clv(engine) == 1
        bet = read_df(engine, "SELECT closing_odds, clv FROM bets").iloc[0]
        assert bet["closing_odds"] == pytest.approx(2.00)
        assert bet["clv"] == pytest.approx(2.20 / 2.00 - 1.0)  # +10%: bateu o fechamento

    def test_jogo_futuro_nao_recebe_clv(self, engine: Engine) -> None:
        kickoff = utcnow_naive() + timedelta(days=2)
        record_paper_bet(engine, 11, "1x2", "home", 2.20, 1.0)
        upsert(
            engine,
            schema.odds_snapshots,
            [
                {
                    "match_id": 11,
                    "commence_time": kickoff,
                    "source": "theoddsapi",
                    "bookmaker": "book",
                    "market": "1x2",
                    "selection": "home",
                    "line": 0.0,
                    "odds_decimal": 2.0,
                    "collected_at": utcnow_naive(),
                    "is_closing": False,
                }
            ],
            conflict_cols=[
                "match_id",
                "source",
                "bookmaker",
                "market",
                "selection",
                "line",
                "collected_at",
            ],
        )
        assert update_clv(engine) == 0
