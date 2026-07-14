"""Carga do banco a partir do cache bruto (idempotente por construção).

Ordem de carga e papel de cada fonte:

1. football-data.co.uk  -> espinha dorsal: matches, team_match_stats, odds
   históricas de abertura/fechamento (Europa) e fechamento 1X2 (Brasil).
2. Understat            -> enriquecimento: xG de time (UPDATE, nunca upsert
   cego — não pode apagar chutes/escanteios já carregados) e stats de
   jogador por partida (Big 5).
3. FBref                -> Brasileirão: stats de jogador por partida; cria os
   jogos que não existem em outra fonte (ex.: Série B). Lê SÓ o que está no
   cache — nunca dispara rede durante a carga (12,5 s/página é proibitivo).

A vinculação entre fontes usa (home_id, away_id) dentro de liga-temporada —
cada par mandante/visitante ocorre uma única vez por temporada de pontos
corridos, o que dispensa casar datas (que divergem por fuso entre fontes).
"""

import re
from typing import Any

import pandas as pd
import structlog
from sqlalchemy import Engine, select, update

from edgefinder.config import COMPETITION_TIERS, settings
from edgefinder.data.teamnames import AmbiguousTeamMatch, match_team_sets, normalize_team_name
from edgefinder.ingest import fbref as fbref_ingest
from edgefinder.ingest import matchhistory as mh
from edgefinder.ingest import understat as us_ingest
from edgefinder.storage import schema
from edgefinder.storage.repository import upsert

log = structlog.get_logger()


# --------------------------------------------------------------------------
# helpers de resolução
# --------------------------------------------------------------------------


def resolve_teams(engine: Engine, source: str, names: list[str]) -> dict[str, int]:
    """Resolve nomes crus de UMA fonte para ids canônicos, criando o que faltar."""
    names = sorted(set(names))
    result: dict[str, int] = {}
    with engine.begin() as conn:
        rows = conn.execute(
            select(schema.team_aliases.c.alias, schema.team_aliases.c.team_id).where(
                schema.team_aliases.c.source == source,
                schema.team_aliases.c.alias.in_(names),
            )
        ).fetchall()
        result.update({str(alias): int(team_id) for alias, team_id in rows})

        canon_rows = conn.execute(select(schema.teams.c.name, schema.teams.c.id)).fetchall()
        canon_map = {str(name): int(team_id) for name, team_id in canon_rows}

        for name in names:
            if name in result:
                continue
            canonical = normalize_team_name(name)
            team_id = canon_map.get(canonical)
            if team_id is None:
                pk = conn.execute(schema.teams.insert().values(name=canonical)).inserted_primary_key
                assert pk is not None
                team_id = int(pk[0])
                canon_map[canonical] = team_id
            conn.execute(
                schema.team_aliases.insert().values(team_id=team_id, source=source, alias=name)
            )
            result[name] = team_id
    return result


def link_source_teams(
    engine: Engine, source: str, competition_id: str, season: str, source_names: list[str]
) -> dict[str, int]:
    """Casa os times de uma fonte com os já existentes na liga-temporada.

    Diferente de `resolve_teams`, aqui NÃO se cria time novo: o conjunto da
    fonte e o do banco são a mesma liga-temporada, então todo time da fonte
    tem de casar com um existente (via match_team_sets, com erro explícito em
    ambiguidade). Usado para enriquecimento (Understat/FBref sobre a espinha
    do football-data.co.uk).
    """
    source_names = sorted(set(source_names))
    with engine.begin() as conn:
        alias_rows_db = conn.execute(
            select(schema.team_aliases.c.alias, schema.team_aliases.c.team_id).where(
                schema.team_aliases.c.source == source,
                schema.team_aliases.c.alias.in_(source_names),
            )
        ).fetchall()
        existing: dict[str, int] = {str(a): int(t) for a, t in alias_rows_db}
        missing = [n for n in source_names if n not in existing]
        if not missing:
            return {str(k): int(v) for k, v in existing.items()}

        id_rows = conn.execute(
            select(schema.matches.c.home_team_id, schema.matches.c.away_team_id).where(
                schema.matches.c.competition_id == competition_id,
                schema.matches.c.season == season,
            )
        ).fetchall()
        league_team_ids = {int(h) for h, _ in id_rows} | {int(a) for _, a in id_rows}
        team_rows = conn.execute(
            select(schema.teams.c.id, schema.teams.c.name).where(
                schema.teams.c.id.in_(league_team_ids)
            )
        ).fetchall()
    canonical_by_name = {str(name): int(tid) for tid, name in team_rows}
    if not canonical_by_name:
        raise ValueError(f"Nenhum time no banco para {competition_id} {season}")

    mapping = match_team_sets(missing, list(canonical_by_name))
    alias_rows = [
        {"team_id": canonical_by_name[canon], "source": source, "alias": alias}
        for alias, canon in mapping.items()
    ]
    upsert(engine, schema.team_aliases, alias_rows, conflict_cols=["source", "alias"])
    return existing | {alias: canonical_by_name[canon] for alias, canon in mapping.items()}


def match_id_map(engine: Engine, competition_id: str, season: str) -> dict[tuple[int, int], int]:
    """{(home_id, away_id): match_id} de uma liga-temporada."""
    with engine.connect() as conn:
        rows = conn.execute(
            select(
                schema.matches.c.home_team_id,
                schema.matches.c.away_team_id,
                schema.matches.c.id,
            ).where(
                schema.matches.c.competition_id == competition_id,
                schema.matches.c.season == season,
            )
        ).fetchall()
    return {(int(h), int(a)): int(m) for h, a, m in rows}


def resolve_players(engine: Engine, source: str, players: list[tuple[str, str]]) -> dict[str, int]:
    """{source_player_id: player_id}; cria os que faltarem."""
    wanted = {pid: name for pid, name in players if pid}
    with engine.begin() as conn:
        rows = conn.execute(
            select(schema.players.c.source_player_id, schema.players.c.id).where(
                schema.players.c.source == source,
                schema.players.c.source_player_id.in_(list(wanted)),
            )
        ).fetchall()
        result = {str(pid): int(dbid) for pid, dbid in rows}
        for pid, name in wanted.items():
            if pid not in result:
                pk = conn.execute(
                    schema.players.insert().values(name=name, source=source, source_player_id=pid)
                ).inserted_primary_key
                assert pk is not None
                result[pid] = int(pk[0])
    return result


# --------------------------------------------------------------------------
# 1. football-data.co.uk (espinha dorsal)
# --------------------------------------------------------------------------


def load_matchhistory(engine: Engine) -> dict[str, int]:
    totals = {"matches": 0, "team_stats": 0, "odds": 0}
    files = sorted((settings.raw_dir / "matchhistory").glob("*.csv"))
    for path in files:
        stem = path.stem
        if "_" in stem and stem.split("_")[0] in mh.DIV_TO_COMPETITION:
            div, season = stem.split("_")
            parsed = mh.parse_classic(mh.read_classic_csv(path), div, season)
        elif stem in mh.NEW_FORMAT_COMPETITION:
            if mh.NEW_FORMAT_COMPETITION[stem] not in COMPETITION_TIERS:
                continue  # ex.: ARG baixado por completude, fora do escopo
            parsed = mh.parse_new_format(path, stem)
        else:
            continue
        counts = _load_parsed_matchhistory(engine, parsed)
        for key, val in counts.items():
            totals[key] += val
        log.info("etl.matchhistory.file", file=stem, **counts)
    return totals


def _load_parsed_matchhistory(
    engine: Engine, parsed: dict[str, list[dict[str, Any]]]
) -> dict[str, int]:
    if not parsed["matches"]:
        return {"matches": 0, "team_stats": 0, "odds": 0}

    names = [m["home"] for m in parsed["matches"]] + [m["away"] for m in parsed["matches"]]
    team_ids = resolve_teams(engine, "football-data.co.uk", names)

    match_rows = [
        {
            "competition_id": m["competition_id"],
            "season": m["season"],
            "match_date": m["match_date"],
            "home_team_id": team_ids[m["home"]],
            "away_team_id": team_ids[m["away"]],
            "home_goals": m["home_goals"],
            "away_goals": m["away_goals"],
            "status": m["status"],
        }
        for m in parsed["matches"]
    ]
    n_matches = upsert(
        engine,
        schema.matches,
        match_rows,
        conflict_cols=["competition_id", "season", "match_date", "home_team_id", "away_team_id"],
    )

    keys = {(m["competition_id"], m["season"]) for m in parsed["matches"]}
    id_map: dict[tuple[str, str, int, int], int] = {}
    for comp, season in keys:
        for (h, a), mapped_id in match_id_map(engine, comp, season).items():
            id_map[(comp, season, h, a)] = mapped_id

    def mid_of(row: dict[str, Any]) -> int | None:
        return id_map.get(
            (row["competition_id"], row["season"], team_ids[row["home"]], team_ids[row["away"]])
        )

    stat_rows = []
    for s in parsed["team_stats"]:
        mid = mid_of(s)
        if mid is None:
            continue
        stat_rows.append(
            {
                "match_id": mid,
                "team_id": team_ids[s["team"]],
                "is_home": s["is_home"],
                "goals": s["goals"],
                "shots": s["shots"],
                "shots_on_target": s["shots_on_target"],
                "corners": s["corners"],
                "fouls": s["fouls"],
                "yellow_cards": s["yellow_cards"],
                "red_cards": s["red_cards"],
                "ht_goals": s["ht_goals"],
            }
        )
    n_stats = upsert(
        engine, schema.team_match_stats, stat_rows, conflict_cols=["match_id", "team_id"]
    )

    odds_rows = []
    for o in parsed["odds"]:
        mid = mid_of(o)
        if mid is None:
            continue
        odds_rows.append(
            {
                "match_id": mid,
                "source": o["source"],
                "bookmaker": o["bookmaker"],
                "market": o["market"],
                "selection": o["selection"],
                "line": o["line"] if o["line"] is not None else 0.0,
                "odds_decimal": o["odds_decimal"],
                "collected_at": o["match_date"],
                "is_closing": o["is_closing"],
            }
        )
    n_odds = upsert(
        engine,
        schema.odds_snapshots,
        odds_rows,
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
    return {"matches": n_matches, "team_stats": n_stats, "odds": n_odds}


# --------------------------------------------------------------------------
# 2. Understat (enriquecimento xG + jogadores, Big 5)
# --------------------------------------------------------------------------


def load_understat_team_xg(engine: Engine, league: str, season: str) -> int:
    frame = us_ingest.read_team_xg(league, season)
    if frame.empty:
        return 0
    names = list(frame["home_team"]) + list(frame["away_team"])
    alias_map = link_source_teams(engine, "understat", league, season, names)
    mid_map = match_id_map(engine, league, season)

    updated = 0
    with engine.begin() as conn:
        for _, row in frame.iterrows():
            home_id = alias_map.get(str(row["home_team"]))
            away_id = alias_map.get(str(row["away_team"]))
            if home_id is None or away_id is None:
                continue
            mid = mid_map.get((home_id, away_id))
            if mid is None:
                continue
            conn.execute(
                update(schema.matches)
                .where(schema.matches.c.id == mid)
                .values(understat_id=str(row["game_id"]))
            )
            for team_id, xg_col, npxg_col in (
                (home_id, "home_xg", "home_np_xg"),
                (away_id, "away_xg", "away_np_xg"),
            ):
                values: dict[str, Any] = {}
                if xg_col in row.index and pd.notna(row[xg_col]):
                    values["xg"] = float(row[xg_col])
                if npxg_col in row.index and pd.notna(row[npxg_col]):
                    values["np_xg"] = float(row[npxg_col])
                if values:
                    result = conn.execute(
                        update(schema.team_match_stats)
                        .where(
                            schema.team_match_stats.c.match_id == mid,
                            schema.team_match_stats.c.team_id == team_id,
                        )
                        .values(**values)
                    )
                    if result.rowcount == 0:
                        conn.execute(
                            schema.team_match_stats.insert().values(
                                match_id=mid,
                                team_id=team_id,
                                is_home=(team_id == home_id),
                                **values,
                            )
                        )
                    updated += 1
    return updated


def load_understat_players(engine: Engine, league: str, season: str) -> int:
    frame = us_ingest.read_player_match(league, season)
    if frame.empty:
        return 0
    rows = us_ingest.player_rows_for_db(frame)

    with engine.connect() as conn:
        us_rows = conn.execute(
            select(schema.matches.c.understat_id, schema.matches.c.id).where(
                schema.matches.c.competition_id == league,
                schema.matches.c.season == season,
                schema.matches.c.understat_id.is_not(None),
            )
        ).fetchall()
    us_map: dict[str, int] = {str(u): int(m) for u, m in us_rows}
    alias_map = link_source_teams(
        engine, "understat", league, season, sorted({r["team"] for r in rows if r["team"]})
    )
    player_map = resolve_players(
        engine, "understat", [(r["player_source_id"], r["player"]) for r in rows]
    )

    db_rows = []
    for r in rows:
        mid = us_map.get(r["game_id"])
        if mid is None:
            continue
        db_rows.append(
            {
                "match_id": int(mid),
                "team_id": alias_map.get(r["team"]),
                "player_id": player_map[r["player_source_id"]],
                "source": "understat",
                "minutes": r["minutes"],
                "position": r["position"],
                "shots": r["shots"],
                "goals": r["goals"],
                "assists": r["assists"],
                "yellow_cards": r["yellow_cards"],
                "red_cards": r["red_cards"],
                "xg": r["xg"],
                "xa": r["xa"],
                "key_passes": r["key_passes"],
            }
        )
    return upsert(
        engine,
        schema.player_match_stats,
        db_rows,
        conflict_cols=["match_id", "player_id", "source"],
    )


# --------------------------------------------------------------------------
# 3. FBref (Brasileirão: jogos + jogadores; lê só o cache)
# --------------------------------------------------------------------------

_SCORE_RE = re.compile(r"(\d+)\D+(\d+)")


def cached_fbref_game_ids(game_ids: list[str]) -> list[str]:
    """Filtra para os jogos cujo match report já está no cache em disco."""
    cache_dir = settings.raw_dir / "soccerdata" / "data" / "FBref"
    if not cache_dir.exists():
        return []
    cached = "\n".join(p.name for p in cache_dir.iterdir())
    return [g for g in game_ids if g in cached]


def load_fbref_matches(engine: Engine, league: str, season: str) -> int:
    """Garante os jogos da liga-temporada no banco e grava fbref_id em todos."""
    sched = fbref_ingest.read_schedule(league, season)
    if sched.empty:
        return 0
    names = list(sched["home_team"]) + list(sched["away_team"])
    existing = match_id_map(engine, league, season)
    if existing:
        alias_map = link_source_teams(engine, "fbref", league, season, names)
    else:
        alias_map = resolve_teams(engine, "fbref", names)

    match_rows: list[dict[str, Any]] = []
    for _, row in sched.iterrows():
        date = pd.to_datetime(str(row.get("date", "")), errors="coerce")
        if pd.isna(date):
            continue
        score_raw = row.get("score")
        score = "" if score_raw is None or pd.isna(score_raw) else str(score_raw)
        parsed_score = _SCORE_RE.search(score)
        match_rows.append(
            {
                "competition_id": league,
                "season": season,
                "match_date": date.to_pydatetime(),
                "home_team_id": alias_map[str(row["home_team"])],
                "away_team_id": alias_map[str(row["away_team"])],
                "home_goals": int(parsed_score.group(1)) if parsed_score else None,
                "away_goals": int(parsed_score.group(2)) if parsed_score else None,
                "status": "played" if parsed_score else "scheduled",
                "fbref_id": str(row["game_id"]),
            }
        )

    if existing:
        # jogos já existem (espinha do football-data.co.uk): só vincula o fbref_id
        updated = 0
        with engine.begin() as conn:
            for mrow in match_rows:
                mid = existing.get((int(mrow["home_team_id"]), int(mrow["away_team_id"])))
                if mid is not None:
                    conn.execute(
                        update(schema.matches)
                        .where(schema.matches.c.id == mid)
                        .values(fbref_id=mrow["fbref_id"])
                    )
                    updated += 1
        return updated
    return upsert(
        engine,
        schema.matches,
        match_rows,
        conflict_cols=["competition_id", "season", "match_date", "home_team_id", "away_team_id"],
    )


def load_fbref_players(engine: Engine, league: str, season: str) -> int:
    """Carrega stats de jogador dos match reports JÁ CACHEADOS."""
    sched = fbref_ingest.read_schedule(league, season)
    all_ids = fbref_ingest.played_game_ids(sched)
    game_ids = cached_fbref_game_ids(all_ids)
    if not game_ids:
        log.warning("etl.fbref.sem_cache", league=league, season=season)
        return 0
    if len(game_ids) < len(all_ids):
        log.info(
            "etl.fbref.cache_parcial",
            league=league,
            season=season,
            cached=len(game_ids),
            played=len(all_ids),
        )
    frame = fbref_ingest.read_player_match(league, season, game_ids)
    if frame.empty:
        return 0
    rows = fbref_ingest.player_rows_for_db(frame)

    with engine.connect() as conn:
        fb_rows = conn.execute(
            select(schema.matches.c.fbref_id, schema.matches.c.id).where(
                schema.matches.c.competition_id == league,
                schema.matches.c.season == season,
                schema.matches.c.fbref_id.is_not(None),
            )
        ).fetchall()
    fb_map: dict[str, int] = {str(f): int(m) for f, m in fb_rows}
    alias_map = link_source_teams(
        engine, "fbref", league, season, sorted({r["team"] for r in rows if r["team"]})
    )
    player_map = resolve_players(
        engine, "fbref", [(r["player_source_id"], r["player"]) for r in rows]
    )

    db_rows = []
    for r in rows:
        mid = fb_map.get(r["game_id"])
        if mid is None:
            continue
        db_rows.append(
            {
                "match_id": int(mid),
                "team_id": alias_map.get(r["team"]),
                "player_id": player_map[r["player_source_id"]],
                "source": "fbref",
                "minutes": r["minutes"],
                "position": r["position"],
                "shots": r["shots"],
                "shots_on_target": r["shots_on_target"],
                "goals": r["goals"],
                "assists": r["assists"],
                "yellow_cards": r["yellow_cards"],
                "red_cards": r["red_cards"],
                "fouls_committed": r["fouls_committed"],
                "fouls_drawn": r["fouls_drawn"],
                "tackles_won": r["tackles_won"],
                "interceptions": r["interceptions"],
                "crosses": r["crosses"],
            }
        )
    return upsert(
        engine,
        schema.player_match_stats,
        db_rows,
        conflict_cols=["match_id", "player_id", "source"],
    )


class LeagueLinkError(AmbiguousTeamMatch):
    pass
