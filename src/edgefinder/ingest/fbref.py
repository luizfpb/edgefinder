"""FBref via soccerdata: calendários e stats básicas de jogador por partida.

Pós-Opta (jan/2026), o FBref entrega apenas o stat_type "summary" por jogador:
minutos, posição, chutes, chutes no gol, gols, assistências, cartões, faltas,
pênaltis, desarmes, interceptações. É o suficiente para props de contagem, e
é a ÚNICA fonte gratuita de dados de jogador para o Brasileirão.

Acesso: seleniumbase UC + Chrome headless (contorno de Cloudflare), rate limit
de 7s+jitter imposto pela lib (política FBref: máx. 10 req/min). Com o cache
aquecido, as leituras saem do disco sem tocar a rede.
"""

from typing import Any

import pandas as pd
import structlog

from edgefinder.ingest._sd import sd

log = structlog.get_logger()


def read_schedule(league: str, season: str) -> pd.DataFrame:
    fb = sd.FBref(leagues=league, seasons=season)
    sched: pd.DataFrame = fb.read_schedule().reset_index()
    sched["league"] = league
    sched["season"] = season
    return sched


def played_game_ids(schedule: pd.DataFrame) -> list[str]:
    played = schedule[schedule["score"].notna() & (schedule["score"].astype(str).str.strip() != "")]
    return [str(g) for g in played["game_id"].dropna()]


def read_player_match(league: str, season: str, game_ids: list[str] | None = None) -> pd.DataFrame:
    """Stats summary de jogador por partida, jogo a jogo (cache-friendly)."""
    fb = sd.FBref(leagues=league, seasons=season)
    if game_ids is None:
        game_ids = played_game_ids(fb.read_schedule().reset_index())
    frames: list[pd.DataFrame] = []
    for gid in game_ids:
        try:
            df = fb.read_player_match_stats(stat_type="summary", match_id=gid)
            df = df.reset_index()
            # colunas MultiIndex ("Performance", "Sh") -> "Sh"
            df.columns = [
                c[1]
                if isinstance(c, tuple) and c[1]
                else str(c[0])
                if isinstance(c, tuple)
                else str(c)
                for c in df.columns
            ]
            df["game_id"] = gid
            frames.append(df)
        except Exception as exc:
            log.warning("fbref.player_match_failed", game_id=gid, error=str(exc))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["league"] = league
    out["season"] = season
    return out


def _first(row: pd.Series, *names: str) -> Any:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    return None


def player_rows_for_db(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Converte o frame do FBref (summary) para linhas de player_match_stats."""
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        minutes = _first(r, "min")
        rows.append(
            {
                "source": "fbref",
                "game_id": str(r["game_id"]),
                "team": str(_first(r, "team") or ""),
                "player": str(_first(r, "player") or ""),
                # FBref não expõe id de jogador nesta tabela: o nome (dentro da
                # fonte) é a chave natural disponível.
                "player_source_id": str(_first(r, "player") or ""),
                "minutes": int(minutes) if minutes is not None else None,
                "position": (str(_first(r, "pos")) or None),
                "shots": _as_int(_first(r, "Sh")),
                "shots_on_target": _as_int(_first(r, "SoT")),
                "goals": _as_int(_first(r, "Gls")),
                "assists": _as_int(_first(r, "Ast")),
                "yellow_cards": _as_int(_first(r, "CrdY")),
                "red_cards": _as_int(_first(r, "CrdR")),
                "fouls_committed": _as_int(_first(r, "Fls")),
                "fouls_drawn": _as_int(_first(r, "Fld")),
                "tackles_won": _as_int(_first(r, "TklW")),
                "interceptions": _as_int(_first(r, "Int")),
                "crosses": _as_int(_first(r, "Crs")),
            }
        )
    return rows


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
