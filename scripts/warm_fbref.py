"""Aquecimento de cache do FBref (o gargalo do projeto: ~12.5s por pagina).

Ordem: calendarios de todas as competicoes-alvo primeiro (baratos, ~40s cada),
depois stats de jogador por partida do Brasileirao (1 request por jogo, ~80min
por temporada de 380 jogos), por fim bonus (Serie B, EPL) se a noite render.

Retomavel: o cache em data/raw/soccerdata torna re-execucoes gratuitas, entao
rodar de novo continua de onde parou. Rate limit de 7s+jitter e do proprio
soccerdata (politica do FBref: max 10 req/min; violar = 'jail' de ate 1 dia).
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SDDIR = ROOT / "data" / "raw" / "soccerdata"
os.environ["SOCCERDATA_DIR"] = str(SDDIR)

LEAGUE_DICT = {
    "BRA-Serie A": {"FBref": "Campeonato Brasileiro Série A", "season_code": "single-year"},
    "BRA-Serie B": {"FBref": "Campeonato Brasileiro Série B", "season_code": "single-year"},
    "EUR-Champions League": {
        "FBref": "UEFA Champions League",
        "season_start": "Sep",
        "season_end": "May",
    },
    "SAM-Copa Libertadores": {
        "FBref": "Copa Libertadores de América",
        "season_code": "single-year",
    },
}

config_path = SDDIR / "config" / "league_dict.json"
config_path.parent.mkdir(parents=True, exist_ok=True)
existing = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
existing.update(LEAGUE_DICT)
config_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

import soccerdata as sd  # noqa: E402 - env var e config precisam vir antes

SCHEDULE_TARGETS = [
    ("ENG-Premier League", ["2223", "2324", "2425"]),
    ("ESP-La Liga", ["2223", "2324", "2425"]),
    ("ITA-Serie A", ["2223", "2324", "2425"]),
    ("GER-Bundesliga", ["2223", "2324", "2425"]),
    ("FRA-Ligue 1", ["2223", "2324", "2425"]),
    ("BRA-Serie A", ["2023", "2024", "2025", "2026"]),
    ("BRA-Serie B", ["2024", "2025"]),
    ("EUR-Champions League", ["2324", "2425"]),
    ("SAM-Copa Libertadores", ["2024", "2025"]),
    ("INT-World Cup", ["2022"]),
    ("INT-European Championship", ["2024"]),
]

# Stats de jogador por partida, em ordem de prioridade. O Brasileirao vem
# primeiro: e a unica fonte de dados de jogador para o BR (Understat nao cobre).
PLAYER_TARGETS = [
    ("BRA-Serie A", "2025"),
    ("BRA-Serie A", "2024"),
    ("BRA-Serie A", "2026"),
    ("BRA-Serie A", "2023"),
    ("BRA-Serie B", "2025"),
    ("ENG-Premier League", "2425"),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    for league, seasons in SCHEDULE_TARGETS:
        for season in seasons:
            try:
                fb = sd.FBref(leagues=league, seasons=season)
                sched = fb.read_schedule()
                log(f"schedule {league} {season}: {len(sched)} jogos")
            except Exception as exc:
                log(f"schedule {league} {season}: ERR {type(exc).__name__}: {exc}")

    for league, season in PLAYER_TARGETS:
        try:
            fb = sd.FBref(leagues=league, seasons=season)
            sched = fb.read_schedule()
        except Exception as exc:
            log(f"player-warm {league} {season}: schedule ERR {exc}")
            continue
        played = sched[sched["score"].notna() & (sched["score"].astype(str).str.strip() != "")]
        game_ids = [g for g in played["game_id"].dropna().tolist()]
        log(f"player-warm {league} {season}: {len(game_ids)} jogos a aquecer")
        ok = err = consecutive_err = 0
        for i, gid in enumerate(game_ids):
            try:
                fb.read_player_match_stats(stat_type="summary", match_id=gid)
                ok += 1
                consecutive_err = 0
            except Exception as exc:
                err += 1
                consecutive_err += 1
                log(f"  game {gid}: ERR {type(exc).__name__}: {exc}")
                if consecutive_err >= 3:
                    # driver do Chrome pode ter morrido: recria a instancia
                    log("  3 erros consecutivos; recriando a instancia FBref")
                    try:
                        fb = sd.FBref(leagues=league, seasons=season)
                    except Exception as exc2:
                        log(f"  recriacao falhou ({exc2}); aguardando 120s")
                        time.sleep(120)
                    consecutive_err = 0
            if (i + 1) % 20 == 0:
                log(f"  {league} {season}: {i + 1}/{len(game_ids)} (ok={ok} err={err})")
        log(f"player-warm {league} {season}: DONE ok={ok} err={err}")

    log("warm_fbref: DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
