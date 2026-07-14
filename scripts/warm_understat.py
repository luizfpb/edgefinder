"""Aquecimento de cache do Understat (xG por time/jogador/chute, Big 5).

Custo de rede: 1 request por liga-temporada (schedule + xG de time das 380
partidas de uma vez) e 1 request por partida (player stats + shot events saem
da mesma pagina). O cache do soccerdata em data/raw/soccerdata torna qualquer
re-execucao gratuita, entao o script e retomavel: se cair, roda de novo.
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["SOCCERDATA_DIR"] = str(ROOT / "data" / "raw" / "soccerdata")

import soccerdata as sd  # noqa: E402 - o env var precisa vir antes do import

BIG5 = ["ENG-Premier League", "ESP-La Liga", "ITA-Serie A", "GER-Bundesliga", "FRA-Ligue 1"]
# Temporadas com xG de TIME (barato: 1 request cada). Understat comeca em 2014/15.
TEAM_SEASONS = [
    "1415",
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
# (liga, temporada) com dados POR JOGADOR POR PARTIDA (caro: ~380 requests cada),
# em ordem de prioridade para o backtest de props.
PLAYER_TARGETS = [
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


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    for league in BIG5:
        for season in TEAM_SEASONS:
            try:
                us = sd.Understat(leagues=league, seasons=season)
                sched = us.read_schedule()
                log(f"schedule {league} {season}: {len(sched)} jogos")
            except Exception as exc:
                log(f"schedule {league} {season}: ERR {type(exc).__name__}: {exc}")

    for league, season in PLAYER_TARGETS:
        try:
            us = sd.Understat(leagues=league, seasons=season)
            sched = us.read_schedule()
        except Exception as exc:
            log(f"player-warm {league} {season}: schedule ERR {exc}")
            continue
        played = sched[sched["is_result"].fillna(False)]
        game_ids = [int(g) for g in played["game_id"].dropna().tolist()]
        log(f"player-warm {league} {season}: {len(game_ids)} jogos a aquecer")
        ok = err = 0
        for i, gid in enumerate(game_ids):
            try:
                us.read_player_match_stats(match_id=gid)
                us.read_shot_events(match_id=gid)  # mesma pagina, sem request extra
                ok += 1
            except Exception as exc:
                err += 1
                log(f"  game {gid}: ERR {type(exc).__name__}: {exc}")
            if (i + 1) % 25 == 0:
                log(f"  {league} {season}: {i + 1}/{len(game_ids)} (ok={ok} err={err})")
        log(f"player-warm {league} {season}: DONE ok={ok} err={err}")

    log("warm_understat: DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
