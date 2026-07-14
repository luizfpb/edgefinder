"""Extração de frames do banco para o backtest.

O walk-forward consome DataFrames puros (sem tocar o banco durante a
simulação): um frame de jogos com resultados e um de odds de fechamento
de-vigáveis. A ordem de preferência de casa para o benchmark é pinnacle >
market_avg > bet365 > market_max — a Pinnacle é a referência sharp clássica,
mas desde 23/07/2025 o football-data.co.uk avisa que o fechamento dela ficou
não confiável, então a média de mercado cobre os buracos (hipótese H5 no
research-log).
"""

import pandas as pd
from sqlalchemy import Engine

from edgefinder.storage.repository import read_df

BOOKMAKER_PREFERENCE = ["pinnacle", "market_avg", "bet365", "market_max"]


def matches_frame(
    engine: Engine,
    competitions: list[str],
    min_season: str | None = None,
) -> pd.DataFrame:
    """Jogos disputados com nomes canônicos e placar, ordenados por data."""
    placeholders = ",".join(f":c{i}" for i in range(len(competitions)))
    params: dict[str, object] = {f"c{i}": c for i, c in enumerate(competitions)}
    sql = f"""
        SELECT m.id AS match_id, m.competition_id, m.season, m.match_date,
               th.name AS home_team, ta.name AS away_team,
               m.home_goals, m.away_goals
        FROM matches m
        JOIN teams th ON th.id = m.home_team_id
        JOIN teams ta ON ta.id = m.away_team_id
        WHERE m.competition_id IN ({placeholders})
          AND m.status = 'played'
          AND m.home_goals IS NOT NULL
    """
    df = read_df(engine, sql, params)
    df["match_date"] = pd.to_datetime(df["match_date"])
    if min_season is not None:
        df = df[df["season"] >= min_season]
    return df.sort_values("match_date").reset_index(drop=True)


def closing_odds_1x2(engine: Engine, competitions: list[str]) -> pd.DataFrame:
    """Odds de fechamento 1X2, uma linha por jogo, com a melhor casa disponível.

    Colunas: match_id, bookmaker, odds_home, odds_draw, odds_away.
    """
    placeholders = ",".join(f":c{i}" for i in range(len(competitions)))
    params: dict[str, object] = {f"c{i}": c for i, c in enumerate(competitions)}
    sql = f"""
        SELECT o.match_id, o.bookmaker, o.selection, o.odds_decimal
        FROM odds_snapshots o
        JOIN matches m ON m.id = o.match_id
        WHERE m.competition_id IN ({placeholders})
          AND o.market = '1x2' AND o.is_closing = 1
    """
    df = read_df(engine, sql, params)
    if df.empty:
        return pd.DataFrame(
            columns=["match_id", "bookmaker", "odds_home", "odds_draw", "odds_away"]
        )
    wide = df.pivot_table(
        index=["match_id", "bookmaker"],
        columns="selection",
        values="odds_decimal",
        aggfunc="last",
    ).reset_index()
    wide = wide.rename(
        columns={"home": "odds_home", "draw": "odds_draw", "away": "odds_away"}
    ).dropna(subset=["odds_home", "odds_draw", "odds_away"])
    wide["bookmaker"] = pd.Categorical(
        wide["bookmaker"], categories=BOOKMAKER_PREFERENCE, ordered=True
    )
    wide = (
        wide.dropna(subset=["bookmaker"])
        .sort_values(["match_id", "bookmaker"])
        .drop_duplicates("match_id", keep="first")
    )
    return wide.reset_index(drop=True)


def closing_odds_ou25(engine: Engine, competitions: list[str]) -> pd.DataFrame:
    """Odds de fechamento Over/Under 2.5: match_id, bookmaker, odds_over, odds_under."""
    placeholders = ",".join(f":c{i}" for i in range(len(competitions)))
    params: dict[str, object] = {f"c{i}": c for i, c in enumerate(competitions)}
    sql = f"""
        SELECT o.match_id, o.bookmaker, o.selection, o.odds_decimal
        FROM odds_snapshots o
        JOIN matches m ON m.id = o.match_id
        WHERE m.competition_id IN ({placeholders})
          AND o.market = 'ou' AND o.line = 2.5 AND o.is_closing = 1
    """
    df = read_df(engine, sql, params)
    if df.empty:
        return pd.DataFrame(columns=["match_id", "bookmaker", "odds_over", "odds_under"])
    wide = df.pivot_table(
        index=["match_id", "bookmaker"], columns="selection", values="odds_decimal", aggfunc="last"
    ).reset_index()
    wide = wide.rename(columns={"over": "odds_over", "under": "odds_under"}).dropna(
        subset=["odds_over", "odds_under"]
    )
    wide["bookmaker"] = pd.Categorical(
        wide["bookmaker"], categories=BOOKMAKER_PREFERENCE, ordered=True
    )
    wide = (
        wide.dropna(subset=["bookmaker"])
        .sort_values(["match_id", "bookmaker"])
        .drop_duplicates("match_id", keep="first")
    )
    return wide.reset_index(drop=True)


def player_stats_frame(engine: Engine, competitions: list[str], source: str) -> pd.DataFrame:
    """Stats de jogador por partida (fonte única) com data e adversário."""
    placeholders = ",".join(f":c{i}" for i in range(len(competitions)))
    params: dict[str, object] = {f"c{i}": c for i, c in enumerate(competitions)}
    params["src"] = source
    sql = f"""
        SELECT p.match_id, p.player_id, pl.name AS player, p.team_id,
               t.name AS team, m.competition_id, m.season, m.match_date,
               m.home_team_id, m.away_team_id,
               p.minutes, p.position, p.shots, p.shots_on_target, p.goals,
               p.assists, p.yellow_cards, p.red_cards, p.fouls_committed,
               p.xg, p.xa, p.key_passes
        FROM player_match_stats p
        JOIN matches m ON m.id = p.match_id
        JOIN players pl ON pl.id = p.player_id
        LEFT JOIN teams t ON t.id = p.team_id
        WHERE m.competition_id IN ({placeholders}) AND p.source = :src
    """
    df = read_df(engine, sql, params)
    df["match_date"] = pd.to_datetime(df["match_date"])
    if not df.empty:
        df["opponent_team_id"] = df["home_team_id"].where(
            df["team_id"] == df["away_team_id"], df["away_team_id"]
        )
        df["is_home"] = df["team_id"] == df["home_team_id"]
    return df.sort_values("match_date").reset_index(drop=True)
