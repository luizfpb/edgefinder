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


def streak_table(
    view_home: pd.DataFrame, view_away: pd.DataFrame, home: str, away: str, n: int
) -> pd.DataFrame:
    """Tabela lado a lado: condição x (casa, fora, combinado), com denominadores.

    Colunas: condicao, home, away, combinado — valores no formato "hits/total".
    Pura (recebe as visões dos times), para servir tanto o banco local quanto
    o snapshot parquet do dashboard publicado.
    """
    rows = []
    for h, a, c in zip(
        team_streaks(view_home, n),
        team_streaks(view_away, n),
        combined_streaks(view_home, view_away, n),
        strict=True,
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


def match_streak_table(engine: Engine, home: str, away: str, n: int) -> pd.DataFrame:
    """`streak_table` alimentada pelo banco local."""
    return streak_table(last_matches(engine, home, n), last_matches(engine, away, n), home, away, n)


# --- estatísticas de time (escanteios, cartões, faltas, chutes) --------------

# Condições sobre a visão de stats do time. Cada uma declara as colunas de que
# depende: o denominador conta só os jogos em que o dado existe (escanteios e
# cartões vêm do football-data.co.uk — Europa; o Brasileirão não tem).
STAT_CONDITIONS: list[tuple[str, str, list[str], _Condition]] = [
    (
        "corners_over_8.5",
        "mais de 8.5 escanteios no jogo",
        ["corners", "corners_opp"],
        lambda v: (v["corners"] + v["corners_opp"]) > 8.5,
    ),
    (
        "corners_over_9.5",
        "mais de 9.5 escanteios no jogo",
        ["corners", "corners_opp"],
        lambda v: (v["corners"] + v["corners_opp"]) > 9.5,
    ),
    (
        "corners_over_10.5",
        "mais de 10.5 escanteios no jogo",
        ["corners", "corners_opp"],
        lambda v: (v["corners"] + v["corners_opp"]) > 10.5,
    ),
    (
        "team_corners_over_4.5",
        "o time bateu mais de 4.5 escanteios",
        ["corners"],
        lambda v: v["corners"] > 4.5,
    ),
    (
        "cards_over_3.5",
        "mais de 3.5 cartoes no jogo",
        ["cards", "cards_opp"],
        lambda v: (v["cards"] + v["cards_opp"]) > 3.5,
    ),
    (
        "cards_over_4.5",
        "mais de 4.5 cartoes no jogo",
        ["cards", "cards_opp"],
        lambda v: (v["cards"] + v["cards_opp"]) > 4.5,
    ),
    (
        "team_cards_over_1.5",
        "o time levou mais de 1.5 cartoes",
        ["cards"],
        lambda v: v["cards"] > 1.5,
    ),
    (
        "team_sot_over_4.5",
        "o time deu mais de 4.5 chutes no gol",
        ["shots_on_target"],
        lambda v: v["shots_on_target"] > 4.5,
    ),
]

# (rótulo, coluna feita, coluna sofrida) para a tabela de médias por jogo.
_AVG_STATS: list[tuple[str, str, str]] = [
    ("gols", "gf", "ga"),
    ("chutes", "shots", "shots_opp"),
    ("chutes no gol", "shots_on_target", "shots_on_target_opp"),
    ("escanteios", "corners", "corners_opp"),
    ("faltas", "fouls", "fouls_opp"),
    ("cartoes", "cards", "cards_opp"),
]


def team_stats_streaks(view: pd.DataFrame, n: int) -> list[StreakLine]:
    """Contagens de escanteios/cartões/chutes nos últimos n jogos COM dado.

    Condição sem nenhum jogo com dado fica de fora (em vez de "0/0" enganoso).
    """
    last = view.head(n)
    out: list[StreakLine] = []
    for key, label, needed, cond in STAT_CONDITIONS:
        if any(col not in last.columns for col in needed):
            continue
        with_data = last.dropna(subset=needed)
        if with_data.empty:
            continue
        out.append(
            StreakLine(key=key, label=label, hits=int(cond(with_data).sum()), total=len(with_data))
        )
    return out


def team_stats_averages(view: pd.DataFrame, n: int) -> pd.DataFrame:
    """Médias por jogo (feitas e sofridas) nos últimos n jogos com dado.

    Colunas: stat, media, media sofrida, jogos com dado.
    """
    last = view.head(n)
    rows = []
    for label, col_for, col_against in _AVG_STATS:
        if col_for not in last.columns:
            continue
        with_data = last.dropna(subset=[col_for])
        if with_data.empty:
            continue
        against = (
            float(with_data[col_against].mean())
            if col_against in with_data.columns
            else float("nan")
        )
        rows.append(
            {
                "stat": label,
                "media": round(float(with_data[col_for].mean()), 2),
                "media sofrida": round(against, 2),
                "jogos": len(with_data),
            }
        )
    return pd.DataFrame(rows)


def player_summary(players: pd.DataFrame) -> pd.DataFrame:
    """Agrega as linhas jogador-jogo dos últimos N jogos do time em uma tabela.

    Colunas: jogador, jogos, min, gols, marcou em, assist, chutes, no gol,
    cartoes, faltas. Só entra quem esteve em campo (minutos > 0). Soma de stat
    inteiramente ausente na fonte (ex.: cartões no Understat) vira NaN, não 0.
    """
    if players.empty:
        return pd.DataFrame()
    on_pitch = players[players["minutes"].fillna(0) > 0].copy()
    if on_pitch.empty:
        return pd.DataFrame()
    on_pitch["cards"] = on_pitch["yellow_cards"].astype(float) + on_pitch["red_cards"].astype(float)
    grouped = on_pitch.groupby("jogador")
    out = pd.DataFrame(
        {
            "jogos": grouped.size(),
            "min": grouped["minutes"].sum(),
            "gols": grouped["goals"].sum(min_count=1),
            "marcou_em": grouped["goals"].apply(lambda g: int((g.fillna(0) > 0).sum())),
            "assist": grouped["assists"].sum(min_count=1),
            "chutes": grouped["shots"].sum(min_count=1),
            "no_gol": grouped["shots_on_target"].sum(min_count=1),
            "cartoes": grouped["cards"].sum(min_count=1),
            "faltas": grouped["fouls_committed"].sum(min_count=1),
        }
    ).reset_index()
    out["marcou em"] = out["marcou_em"].astype(str) + "/" + out["jogos"].astype(str)
    out = out.drop(columns=["marcou_em"])
    return out.sort_values(
        ["gols", "chutes", "min"], ascending=[False, False, False], na_position="last"
    ).reset_index(drop=True)


def team_stats_last(engine: Engine, team: str, n: int) -> pd.DataFrame:
    """Últimos n jogos do time com stats próprias e do adversário (self-join)."""
    sql = """
        SELECT m.match_date, o.name AS opponent, s.is_home,
               s.goals AS gf, so.goals AS ga,
               s.shots, s.shots_on_target, s.corners, s.fouls,
               s.yellow_cards + COALESCE(s.red_cards, 0) AS cards,
               so.shots AS shots_opp, so.shots_on_target AS shots_on_target_opp,
               so.corners AS corners_opp, so.fouls AS fouls_opp,
               so.yellow_cards + COALESCE(so.red_cards, 0) AS cards_opp
        FROM team_match_stats s
        JOIN matches m ON m.id = s.match_id
        JOIN teams t ON t.id = s.team_id
        JOIN team_match_stats so ON so.match_id = s.match_id AND so.team_id != s.team_id
        JOIN teams o ON o.id = so.team_id
        WHERE t.name = :t AND m.home_goals IS NOT NULL
        ORDER BY m.match_date DESC LIMIT :n
    """
    df = read_df(engine, sql, {"t": team, "n": n})
    if not df.empty:
        df["match_date"] = pd.to_datetime(df["match_date"])
    return df


def team_players_last(engine: Engine, team: str, n: int) -> pd.DataFrame:
    """Linhas jogador-jogo dos últimos n jogos do time (FBref; senão Understat)."""
    sql = """
        SELECT pl.name AS jogador, m.match_date, p.source, p.minutes, p.goals,
               p.assists, p.shots, p.shots_on_target, p.yellow_cards,
               p.red_cards, p.fouls_committed
        FROM player_match_stats p
        JOIN matches m ON m.id = p.match_id
        JOIN players pl ON pl.id = p.player_id
        JOIN teams t ON t.id = p.team_id
        WHERE t.name = :t AND p.source = :src AND m.id IN (
            SELECT m2.id FROM matches m2
            JOIN teams th ON th.id = m2.home_team_id
            JOIN teams ta ON ta.id = m2.away_team_id
            WHERE (th.name = :t OR ta.name = :t) AND m2.home_goals IS NOT NULL
            ORDER BY m2.match_date DESC LIMIT :n
        )
    """
    for source in ("fbref", "understat"):
        df = read_df(engine, sql, {"t": team, "n": n, "src": source})
        if not df.empty:
            return df
    return pd.DataFrame()


def matches_snapshot(engine: Engine) -> pd.DataFrame:
    """Todos os jogos disputados, compactos (placar + competição + id)."""
    return read_df(
        engine,
        """
        SELECT m.id AS match_id, m.match_date, th.name AS home_team,
               ta.name AS away_team, m.home_goals, m.away_goals, m.competition_id
        FROM matches m
        JOIN teams th ON th.id = m.home_team_id
        JOIN teams ta ON ta.id = m.away_team_id
        WHERE m.home_goals IS NOT NULL
        ORDER BY m.match_date
        """,
    )


# Janela dos snapshots por time: precisa cobrir o maior N selecionável na UI.
SNAPSHOT_WINDOW = 30


def export_streak_snapshots(engine: Engine) -> dict[str, int]:
    """Grava em data/reports os artefatos que a aba de sequências do site lê.

    O Streamlit Cloud não tem o banco (gitignored, e o FBref bloqueia IPs de
    datacenter), então o `daily` exporta: todos os placares
    (matches_streaks.parquet) e, para os últimos SNAPSHOT_WINDOW jogos de cada
    time, as stats de time (team_stats_streaks.parquet) e as linhas
    jogador-jogo (player_stats_streaks.parquet).
    """
    from edgefinder.config import settings

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, int] = {}

    matches = matches_snapshot(engine)
    if matches.empty:
        return {"matches": 0, "team_stats": 0, "players": 0}
    matches.to_parquet(settings.reports_dir / "matches_streaks.parquet")
    out["matches"] = len(matches)

    # pares (team, match_id) dos últimos SNAPSHOT_WINDOW jogos de cada time
    long = pd.concat(
        [
            matches[["match_id", "match_date", "home_team"]].rename(columns={"home_team": "team"}),
            matches[["match_id", "match_date", "away_team"]].rename(columns={"away_team": "team"}),
        ]
    ).sort_values("match_date", ascending=False)
    long["rank"] = long.groupby("team").cumcount()
    kept = long[long["rank"] < SNAPSHOT_WINDOW][["team", "match_id"]]

    team_stats = read_df(
        engine,
        """
        SELECT s.match_id, m.match_date, t.name AS team, o.name AS opponent, s.is_home,
               s.goals AS gf, so.goals AS ga,
               s.shots, s.shots_on_target, s.corners, s.fouls,
               s.yellow_cards + COALESCE(s.red_cards, 0) AS cards,
               so.shots AS shots_opp, so.shots_on_target AS shots_on_target_opp,
               so.corners AS corners_opp, so.fouls AS fouls_opp,
               so.yellow_cards + COALESCE(so.red_cards, 0) AS cards_opp
        FROM team_match_stats s
        JOIN matches m ON m.id = s.match_id
        JOIN teams t ON t.id = s.team_id
        JOIN team_match_stats so ON so.match_id = s.match_id AND so.team_id != s.team_id
        JOIN teams o ON o.id = so.team_id
        WHERE m.home_goals IS NOT NULL
        """,
    )
    team_stats = team_stats.merge(kept, on=["team", "match_id"])
    team_stats.to_parquet(settings.reports_dir / "team_stats_streaks.parquet")
    out["team_stats"] = len(team_stats)

    players = read_df(
        engine,
        """
        SELECT p.match_id, m.match_date, t.name AS team, pl.name AS jogador,
               p.source, p.minutes, p.goals, p.assists, p.shots,
               p.shots_on_target, p.yellow_cards, p.red_cards, p.fouls_committed
        FROM player_match_stats p
        JOIN matches m ON m.id = p.match_id
        JOIN players pl ON pl.id = p.player_id
        JOIN teams t ON t.id = p.team_id
        WHERE m.home_goals IS NOT NULL
        """,
    )
    players = players.merge(kept, on=["team", "match_id"])
    players.to_parquet(settings.reports_dir / "player_stats_streaks.parquet")
    out["players"] = len(players)
    return out


def stats_view_from_frame(team_stats: pd.DataFrame, team: str, n: int) -> pd.DataFrame:
    """Visão de stats do time a partir do snapshot parquet (site publicado)."""
    if team_stats.empty:
        return pd.DataFrame()
    view = team_stats[team_stats["team"] == team].copy()
    view["match_date"] = pd.to_datetime(view["match_date"])
    return view.sort_values("match_date", ascending=False).head(n).reset_index(drop=True)


def players_view_from_frame(players: pd.DataFrame, team: str, n: int) -> pd.DataFrame:
    """Linhas jogador-jogo dos últimos n jogos do time a partir do snapshot.

    Prefere FBref (tem cartões e faltas); cai para Understat quando é a única
    fonte do time.
    """
    if players.empty:
        return pd.DataFrame()
    mine = players[players["team"] == team].copy()
    if mine.empty:
        return mine
    source = "fbref" if (mine["source"] == "fbref").any() else "understat"
    mine = mine[mine["source"] == source]
    mine["match_date"] = pd.to_datetime(mine["match_date"])
    last_ids = (
        mine[["match_id", "match_date"]]
        .drop_duplicates("match_id")
        .sort_values("match_date", ascending=False)
        .head(n)["match_id"]
    )
    return mine[mine["match_id"].isin(last_ids)].reset_index(drop=True)
