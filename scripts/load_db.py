"""Carga completa do banco a partir do cache aquecido (sem tocar a rede).

Ordem: matchhistory (espinha) -> Understat (xG + jogadores, Big 5) ->
FBref (Brasileirao A/B + EPL: jogos e jogadores). Cada bloco em try/except:
uma liga problematica nao derruba a carga.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from edgefinder.ingest.etl import (
    load_fbref_matches,
    load_fbref_players,
    load_matchhistory,
    load_understat_players,
    load_understat_team_xg,
)
from edgefinder.storage.repository import get_engine, init_db, set_coverage

BIG5 = ["ENG-Premier League", "ESP-La Liga", "ITA-Serie A", "GER-Bundesliga", "FRA-Ligue 1"]
UNDERSTAT_XG_SEASONS = [
    "1516",
    "1617",
    "1718",
    "1819",
    "1920",
    "2021",
    "2122",
    "2223",
    "2324",
    "2425",
]
UNDERSTAT_PLAYER_TARGETS = [
    ("ENG-Premier League", "2425"),
    ("ENG-Premier League", "2324"),
    ("ENG-Premier League", "2223"),
    ("ESP-La Liga", "2425"),
    ("ITA-Serie A", "2425"),
    ("GER-Bundesliga", "2425"),
    ("FRA-Ligue 1", "2425"),
    ("ESP-La Liga", "2324"),
    ("ITA-Serie A", "2324"),
]
FBREF_TARGETS = [
    ("BRA-Serie A", ["2023", "2024", "2025", "2026"]),
    ("BRA-Serie B", ["2024", "2025"]),
    ("ENG-Premier League", ["2425"]),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    engine = get_engine()
    init_db(engine)

    log("1/3 matchhistory (idempotente, atualiza temporada corrente)")
    totals = load_matchhistory(engine)
    log(f"matchhistory: {totals}")

    log("2/3 Understat: xG de time")
    for league in BIG5:
        for season in UNDERSTAT_XG_SEASONS:
            try:
                n = load_understat_team_xg(engine, league, season)
                log(f"understat xg {league} {season}: {n} updates")
            except Exception as exc:
                log(f"understat xg {league} {season}: ERR {type(exc).__name__}: {exc}")
    log("2/3 Understat: jogadores")
    for league, season in UNDERSTAT_PLAYER_TARGETS:
        try:
            n = load_understat_players(engine, league, season)
            log(f"understat players {league} {season}: {n} linhas")
        except Exception as exc:
            log(f"understat players {league} {season}: ERR {type(exc).__name__}: {exc}")
    set_coverage(engine, "ENG-Premier League", "xg", "ok", "Understat 2015-2025")

    log("3/3 FBref: jogos + jogadores")
    for league, seasons in FBREF_TARGETS:
        for season in seasons:
            try:
                nm = load_fbref_matches(engine, league, season)
                np_ = load_fbref_players(engine, league, season)
                log(f"fbref {league} {season}: matches={nm} players={np_}")
            except Exception as exc:
                log(f"fbref {league} {season}: ERR {type(exc).__name__}: {exc}")
    set_coverage(engine, "BRA-Serie A", "player_stats", "ok", "FBref summary 2023-2026")
    set_coverage(engine, "BRA-Serie B", "player_stats", "ok", "FBref summary 2024-2025")

    log("load_db: DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
