"""Aquecimento rapido de cache: football-data.co.uk, eloratings.net, ClubElo.

Roda em minutos. Salva tudo em data/raw/ (imutavel: nunca sobrescreve um
arquivo ja baixado, exceto os da temporada corrente, que mudam semana a semana).
"""

import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CURRENT_SEASON = "2526"  # temporada europeia em curso: sempre re-baixar
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch(client: httpx.Client, url: str, dest: Path, overwrite: bool = False) -> str:
    if dest.exists() and not overwrite:
        return "cached"
    try:
        r = client.get(url, timeout=30)
        if r.status_code != 200:
            return f"HTTP {r.status_code}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return f"ok ({len(r.content)} bytes)"
    except Exception as exc:
        return f"ERR {type(exc).__name__}: {exc}"


def main() -> int:
    client = httpx.Client(headers=HEADERS, follow_redirects=True)

    # football-data.co.uk formato classico: odds abertura+fechamento+stats
    divs = ["E0", "E1", "D1", "I1", "SP1", "F1", "N1", "P1"]
    seasons = [
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
        "2526",
    ]
    for season in seasons:
        for div in divs:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
            dest = RAW / "matchhistory" / f"{div}_{season}.csv"
            res = fetch(client, url, dest, overwrite=(season == CURRENT_SEASON))
            log(f"matchhistory {div} {season}: {res}")
            if res.startswith("ok"):
                time.sleep(1.5)

    # formato 'new': so 1X2 de fechamento, mas cobre Brasil/Argentina
    for country in ["BRA", "ARG"]:
        url = f"https://www.football-data.co.uk/new/{country}.csv"
        dest = RAW / "matchhistory" / f"{country}.csv"
        log(f"matchhistory {country}: {fetch(client, url, dest, overwrite=True)}")
        time.sleep(1.5)

    # Elo de selecoes (Tier 3): snapshot do dia
    url = "https://www.eloratings.net/World.tsv"
    dest = RAW / "eloratings" / "World_2026-07-13.tsv"
    log(f"eloratings: {fetch(client, url, dest)}")

    # ClubElo: snapshot no inicio de cada temporada europeia (prior de forca)
    for year in range(2015, 2027):
        url = f"http://api.clubelo.com/{year}-08-01"
        dest = RAW / "clubelo" / f"{year}-08-01.csv"
        res = fetch(client, url, dest)
        log(f"clubelo {year}-08-01: {res}")
        if res.startswith("ok"):
            time.sleep(1.5)

    log("warm_fast: DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
