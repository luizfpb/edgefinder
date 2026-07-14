"""Understat via soccerdata: xG por time por partida e stats de jogador por partida.

Cobertura: apenas as 5 grandes ligas europeias (limitação da fonte, Fase 0).
Custo de rede: 1 request por liga-temporada (schedule + xG de time) e 1 por
partida (player stats). Com o cache aquecido, tudo sai do disco.
"""

from typing import Any

import pandas as pd
import structlog

from edgefinder.ingest._sd import sd

log = structlog.get_logger()

UNDERSTAT_LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
]


def read_team_xg(league: str, season: str) -> pd.DataFrame:
    """xG por time por partida, direto do schedule (1 request por temporada).

    Colunas de saída: game_id, date, home_team, away_team, home/away_goals,
    home/away_xg, home/away_np_xg, ppda, deep_completions (quando presentes).
    """
    us = sd.Understat(leagues=league, seasons=season)
    sched: pd.DataFrame = us.read_schedule().reset_index()
    try:
        tms: pd.DataFrame = us.read_team_match_stats().reset_index()
    except Exception as exc:
        log.warning("understat.team_match_stats_failed", league=league, error=str(exc))
        tms = pd.DataFrame()

    out = sched[
        [
            "game",
            "game_id",
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "home_xg",
            "away_xg",
            "is_result",
        ]
    ].copy()
    if not tms.empty:
        extra_cols = [
            c
            for c in [
                "home_np_xg",
                "away_np_xg",
                "home_ppda",
                "away_ppda",
                "home_deep_completions",
                "away_deep_completions",
            ]
            if c in tms.columns
        ]
        out = out.merge(tms[["game_id", *extra_cols]], on="game_id", how="left")
    out["league"] = league
    out["season"] = season
    return out


def read_player_match(league: str, season: str) -> pd.DataFrame:
    """Stats de jogador por partida para todos os jogos já disputados.

    Uma linha por (jogo, jogador): minutes, shots, goals, xg, xa, key_passes,
    assists, cartões. Lê jogo a jogo (cache local após o aquecimento).
    """
    us = sd.Understat(leagues=league, seasons=season)
    sched = us.read_schedule().reset_index()
    played = sched[sched["is_result"].fillna(False)]
    frames: list[pd.DataFrame] = []
    for gid in played["game_id"].dropna():
        try:
            df = us.read_player_match_stats(match_id=int(gid)).reset_index()
            df["game_id"] = int(gid)
            frames.append(df)
        except Exception as exc:
            log.warning("understat.player_match_failed", game_id=gid, error=str(exc))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["league"] = league
    out["season"] = season
    return out


def player_rows_for_db(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Converte o frame do Understat para linhas de player_match_stats."""
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        rows.append(
            {
                "source": "understat",
                "game_id": str(r["game_id"]),
                "team": str(r.get("team", "")),
                "player": str(r.get("player", "")),
                "player_source_id": str(r.get("player_id", r.get("player", ""))),
                "minutes": int(r["minutes"]) if pd.notna(r.get("minutes")) else None,
                "position": str(r.get("position", "")) or None,
                "shots": int(r["shots"]) if pd.notna(r.get("shots")) else None,
                "goals": int(r["goals"]) if pd.notna(r.get("goals")) else None,
                "assists": int(r["assists"]) if pd.notna(r.get("assists")) else None,
                "yellow_cards": int(r["yellow_cards"]) if pd.notna(r.get("yellow_cards")) else None,
                "red_cards": int(r["red_cards"]) if pd.notna(r.get("red_cards")) else None,
                "xg": float(r["xg"]) if pd.notna(r.get("xg")) else None,
                "xa": float(r["xa"]) if pd.notna(r.get("xa")) else None,
                "key_passes": int(r["key_passes"]) if pd.notna(r.get("key_passes")) else None,
            }
        )
    return rows
