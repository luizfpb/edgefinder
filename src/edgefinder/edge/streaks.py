"""Sequências: "aconteceu em N dos últimos N jogos" de cada time.

A pergunta que este módulo responde é descritiva, não preditiva: em quantos
dos últimos N jogos do time X o jogo teve mais de 1.5 gols? E do adversário?
E somando os dois? É a "tabelinha" clássica de tendências, com o tamanho da
amostra sempre na cara.

Aviso estatístico (impresso junto com as tabelas): sequência NÃO é
probabilidade. "5 dos últimos 5" com N=5 é uma amostra minúscula — um time
mediano bate over 0.5 em 5 seguidos com frequência por puro acaso, e o
mercado já precifica tendências óbvias. A tabela serve para dar contexto
rápido e apontar onde olhar, nunca como prova de valor.

Núcleo puro (DataFrames -> contagens), testável sem banco; a camada de I/O
fica em funções separadas que recebem o engine.
"""

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import Engine

from edgefinder.storage.repository import read_df


@dataclass(frozen=True)
class StreakLine:
    key: str  # ex.: "total_over_1.5"
    label: str  # ex.: "mais de 1.5 gols no jogo"
    hits: int
    total: int

    def __str__(self) -> str:
        return f"{self.label}: {self.hits} de {self.total}"


# Cada condição é um predicado sobre a visão do time (colunas gf/ga/total/result).
_Condition = Callable[[pd.DataFrame], "pd.Series[bool]"]

CONDITIONS: list[tuple[str, str, _Condition]] = [
    ("total_over_0.5", "mais de 0.5 gols no jogo", lambda v: v["total"] > 0.5),
    ("total_over_1.5", "mais de 1.5 gols no jogo", lambda v: v["total"] > 1.5),
    ("total_over_2.5", "mais de 2.5 gols no jogo", lambda v: v["total"] > 2.5),
    ("total_over_3.5", "mais de 3.5 gols no jogo", lambda v: v["total"] > 3.5),
    ("team_scored", "o time marcou", lambda v: v["gf"] > 0.5),
    ("team_over_1.5", "o time marcou mais de 1.5", lambda v: v["gf"] > 1.5),
    ("team_conceded", "o time sofreu gol", lambda v: v["ga"] > 0.5),
    ("btts", "ambos marcaram", lambda v: (v["gf"] > 0.5) & (v["ga"] > 0.5)),
    ("win", "vitoria", lambda v: v["gf"] > v["ga"]),
    ("draw", "empate", lambda v: v["gf"] == v["ga"]),
    ("loss", "derrota", lambda v: v["gf"] < v["ga"]),
]


def team_view(matches: pd.DataFrame, team: str) -> pd.DataFrame:
    """Converte jogos (home/away) para a perspectiva do time: gf, ga, total.

    Espera colunas home_team, away_team, home_goals, away_goals, match_date.
    Retorna ordenado do jogo mais recente para o mais antigo.
    """
    mine = matches[(matches["home_team"] == team) | (matches["away_team"] == team)].copy()
    if mine.empty:
        return pd.DataFrame(columns=["match_date", "opponent", "venue", "gf", "ga", "total"])
    is_home = mine["home_team"] == team
    mine["gf"] = mine["home_goals"].where(is_home, mine["away_goals"])
    mine["ga"] = mine["away_goals"].where(is_home, mine["home_goals"])
    mine["total"] = mine["home_goals"] + mine["away_goals"]
    mine["opponent"] = mine["away_team"].where(is_home, mine["home_team"])
    mine["venue"] = pd.Series("casa", index=mine.index).where(is_home, "fora")
    cols = ["match_date", "opponent", "venue", "gf", "ga", "total"]
    return mine.sort_values("match_date", ascending=False)[cols].reset_index(drop=True)


def team_streaks(view: pd.DataFrame, n: int) -> list[StreakLine]:
    """Contagens de cada condição nos últimos n jogos da visão do time.

    Se o time tem menos de n jogos, o total reflete o que existe — a amostra
    real aparece no denominador, nunca inflada.
    """
    if n < 1:
        raise ValueError(f"n deve ser >= 1, recebido {n}")
    last = view.head(n)
    return [
        StreakLine(key=key, label=label, hits=int(cond(last).sum()), total=len(last))
        for key, label, cond in CONDITIONS
    ]


def combined_streaks(view_home: pd.DataFrame, view_away: pd.DataFrame, n: int) -> list[StreakLine]:
    """Soma as contagens dos dois times: "N de 2N jogos dos dois times"."""
    home = team_streaks(view_home, n)
    away = team_streaks(view_away, n)
    return [
        StreakLine(key=h.key, label=h.label, hits=h.hits + a.hits, total=h.total + a.total)
        for h, a in zip(home, away, strict=True)
    ]


def hits_over_line(view: pd.DataFrame, n: int, line: float) -> StreakLine:
    """Contagem ad-hoc para uma linha qualquer de gols totais (ex.: over 4.5)."""
    last = view.head(n)
    return StreakLine(
        key=f"total_over_{line:g}",
        label=f"mais de {line:g} gols no jogo",
        hits=int((last["total"] > line).sum()),
        total=len(last),
    )


def market_streak_text(
    view_home: pd.DataFrame,
    view_away: pd.DataFrame,
    market: str,
    selection: str,
    line: float,
    n: int,
    home: str,
    away: str,
) -> str:
    """Sequência relevante para UMA seleção de mercado, em texto curto.

    Ex.: para ou/over 2.5 -> "over 2.5: Arsenal 4/5, Chelsea 3/5 (7/10)".
    String vazia quando não há sequência que faça sentido para a seleção.
    """
    vh, va = view_home.head(n), view_away.head(n)
    if len(vh) == 0 and len(va) == 0:
        return ""
    if market == "ou":
        hits_h = int((vh["total"] > line).sum())
        hits_a = int((va["total"] > line).sum())
        if selection == "under":
            hits_h, hits_a = len(vh) - hits_h, len(va) - hits_a
        word = "over" if selection == "over" else "under"
        return (
            f"{word} {line:g}: {home} {hits_h}/{len(vh)}, {away} {hits_a}/{len(va)} "
            f"({hits_h + hits_a}/{len(vh) + len(va)})"
        )
    if market == "1x2":
        if selection == "home":
            return f"{home} venceu {int((vh['gf'] > vh['ga']).sum())}/{len(vh)}"
        if selection == "away":
            return f"{away} venceu {int((va['gf'] > va['ga']).sum())}/{len(va)}"
        return (
            f"empates: {home} {int((vh['gf'] == vh['ga']).sum())}/{len(vh)}, "
            f"{away} {int((va['gf'] == va['ga']).sum())}/{len(va)}"
        )
    return ""


# --- I/O ---------------------------------------------------------------------


def last_matches(engine: Engine, team: str, n: int, competition: str | None = None) -> pd.DataFrame:
    """Últimos n jogos disputados do time (visão do time), direto do banco."""
    comp_filter = "AND m.competition_id = :comp" if competition else ""
    sql = f"""
        SELECT th.name AS home_team, ta.name AS away_team,
               m.home_goals, m.away_goals, m.match_date, m.competition_id
        FROM matches m
        JOIN teams th ON th.id = m.home_team_id
        JOIN teams ta ON ta.id = m.away_team_id
        WHERE (th.name = :t OR ta.name = :t) AND m.home_goals IS NOT NULL
        {comp_filter}
        ORDER BY m.match_date DESC LIMIT :n
    """
    params: dict[str, object] = {"t": team, "n": n}
    if competition:
        params["comp"] = competition
    df = read_df(engine, sql, params)
    if not df.empty:
        df["match_date"] = pd.to_datetime(df["match_date"])
    return team_view(df, team)


def match_streak_table(engine: Engine, home: str, away: str, n: int) -> pd.DataFrame:
    """Tabela lado a lado: condição x (casa, fora, combinado), com denominadores.

    Colunas: condicao, home, away, combinado — valores no formato "hits/total".
    """
    vh = last_matches(engine, home, n)
    va = last_matches(engine, away, n)
    rows = []
    for h, a, c in zip(
        team_streaks(vh, n), team_streaks(va, n), combined_streaks(vh, va, n), strict=True
    ):
        rows.append(
            {
                "condicao": h.label,
                home: f"{h.hits}/{h.total}",
                away: f"{a.hits}/{a.total}",
                "combinado": f"{c.hits}/{c.total}",
            }
        )
    return pd.DataFrame(rows)
