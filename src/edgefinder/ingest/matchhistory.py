"""football-data.co.uk: resultados + stats de partida + odds de abertura/fechamento.

Cliente próprio com httpx — o reader MatchHistory do soccerdata usa tls-client,
cujo fingerprint o WAF do site bloqueia com 503 (verificado na Fase 0),
enquanto httpx puro recebe 200. URLs previsíveis:

- Formato clássico: /mmz4281/{season}/{div}.csv (odds abertura+fechamento de
  1X2/OU2.5/AH + chutes/escanteios/faltas/cartões). Encoding: utf-8-sig nas
  temporadas novas, latin-1 nas antigas.
- Formato "new": /new/{PAIS}.csv (Brasil etc.: só 1X2 de FECHAMENTO, sem stats).

Colunas de fechamento (sufixo C) existem a partir de ~2019-20; antes disso os
agregados de mercado usam prefixo Bb (BbMxH, BbAvH...). Tudo tratado aqui.
"""

import io
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import structlog

from edgefinder.config import settings

log = structlog.get_logger()

BASE = "https://www.football-data.co.uk"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

DIV_TO_COMPETITION = {
    "E0": "ENG-Premier League",
    "E1": "ENG-Championship",
    "D1": "GER-Bundesliga",
    "I1": "ITA-Serie A",
    "SP1": "ESP-La Liga",
    "F1": "FRA-Ligue 1",
    "N1": "NED-Eredivisie",
    "P1": "POR-Liga Portugal",
}
NEW_FORMAT_COMPETITION = {"BRA": "BRA-Serie A", "ARG": "ARG-Primera Division"}

# (bookmaker canônico, prefixo abertura, prefixo fechamento) para 1X2.
_1X2_BOOKS = [
    ("pinnacle", "PS", "PSC"),
    ("bet365", "B365", "B365C"),
    ("market_max", "Max", "MaxC"),
    ("market_avg", "Avg", "AvgC"),
    # temporadas antigas: agregados Betbrain no lugar de Max/Avg
    ("market_max", "BbMx", None),
    ("market_avg", "BbAv", None),
]


def _cache_path(filename: str) -> Path:
    return settings.raw_dir / "matchhistory" / filename


def download_csv(url_path: str, filename: str, overwrite: bool = False) -> Path:
    """Baixa um CSV para data/raw/matchhistory (imutável, salvo overwrite)."""
    dest = _cache_path(filename)
    if dest.exists() and not overwrite:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers=HEADERS, follow_redirects=True) as client:
        resp = client.get(f"{BASE}/{url_path}", timeout=30)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    log.info("matchhistory.downloaded", url=url_path, bytes=len(resp.content))
    time.sleep(1.5)  # cortesia com o servidor; só ocorre em cache miss
    return dest


def read_classic_csv(path: Path) -> pd.DataFrame:
    """Lê um CSV do formato clássico, tolerante a encoding e colunas ausentes."""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.StringIO(raw.decode(encoding)), on_bad_lines="skip")
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - latin-1 nunca falha em decode
        raise ValueError(f"encoding ilegível: {path}")
    df = df.dropna(subset=["HomeTeam", "AwayTeam"], how="any")
    df = df[df["HomeTeam"].astype(str).str.strip() != ""]
    return df


def _num(row: pd.Series, col: str) -> float | None:
    if col not in row.index:
        return None
    value = row[col]
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(row: pd.Series, col: str) -> int | None:
    value = _num(row, col)
    return int(value) if value is not None else None


def parse_classic(df: pd.DataFrame, div: str, season: str) -> dict[str, list[dict[str, Any]]]:
    """Transforma o CSV clássico em linhas para matches/team_stats/odds.

    Retorna dicts prontos para o ETL (nomes de time ainda crus — a resolução
    para id canônico acontece na carga, onde há acesso ao banco).
    """
    competition = DIV_TO_COMPETITION[div]
    matches: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    odds: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        date = pd.to_datetime(str(row["Date"]), dayfirst=True, errors="coerce")
        if pd.isna(date):
            continue
        home, away = str(row["HomeTeam"]).strip(), str(row["AwayTeam"]).strip()
        base = {
            "competition_id": competition,
            "season": season,
            "match_date": date.to_pydatetime(),
            "home": home,
            "away": away,
        }
        matches.append(
            base
            | {
                "home_goals": _int(row, "FTHG"),
                "away_goals": _int(row, "FTAG"),
                "status": "played" if _int(row, "FTHG") is not None else "scheduled",
            }
        )
        for is_home, prefix, team in ((True, "H", home), (False, "A", away)):
            stats.append(
                base
                | {
                    "team": team,
                    "is_home": is_home,
                    "goals": _int(row, f"FT{prefix}G"),
                    "shots": _int(row, f"{prefix}S"),
                    "shots_on_target": _int(row, f"{prefix}ST"),
                    "corners": _int(row, f"{prefix}C"),
                    "fouls": _int(row, f"{prefix}F"),
                    "yellow_cards": _int(row, f"{prefix}Y"),
                    "red_cards": _int(row, f"{prefix}R"),
                    "ht_goals": _int(row, f"HT{prefix}G"),
                }
            )
        odds.extend(_parse_odds_row(row, base))

    return {"matches": matches, "team_stats": stats, "odds": odds}


def _parse_odds_row(row: pd.Series, base: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def emit(
        bookmaker: str,
        market: str,
        selection: str,
        price: float | None,
        line: float | None,
        is_closing: bool,
    ) -> None:
        if price is not None and price > 1.0:
            out.append(
                base
                | {
                    "source": "football-data.co.uk",
                    "bookmaker": bookmaker,
                    "market": market,
                    "selection": selection,
                    "line": line,
                    "odds_decimal": price,
                    "is_closing": is_closing,
                }
            )

    # 1X2, abertura e fechamento
    for bookmaker, open_prefix, close_prefix in _1X2_BOOKS:
        for suffix, selection in (("H", "home"), ("D", "draw"), ("A", "away")):
            emit(bookmaker, "1x2", selection, _num(row, f"{open_prefix}{suffix}"), None, False)
            if close_prefix:
                emit(bookmaker, "1x2", selection, _num(row, f"{close_prefix}{suffix}"), None, True)

    # Over/Under 2.5
    for bookmaker, open_prefix, close_prefix in (
        ("pinnacle", "P", "PC"),
        ("bet365", "B365", "B365C"),
        ("market_max", "Max", "MaxC"),
        ("market_avg", "Avg", "AvgC"),
        ("market_max", "BbMx", None),
        ("market_avg", "BbAv", None),
    ):
        for direction, selection in ((">", "over"), ("<", "under")):
            emit(bookmaker, "ou", selection, _num(row, f"{open_prefix}{direction}2.5"), 2.5, False)
            if close_prefix:
                emit(
                    bookmaker,
                    "ou",
                    selection,
                    _num(row, f"{close_prefix}{direction}2.5"),
                    2.5,
                    True,
                )

    # Handicap asiático (linha do mandante)
    ah_open = _num(row, "AHh") if "AHh" in row.index else _num(row, "BbAHh")
    ah_close = _num(row, "AHCh")
    for bookmaker, open_prefix, close_prefix in (
        ("pinnacle", "P", "PC"),
        ("bet365", "B365", "B365C"),
        ("market_max", "Max", "MaxC"),
        ("market_avg", "Avg", "AvgC"),
        ("market_max", "BbMx", None),
        ("market_avg", "BbAv", None),
    ):
        for side, selection in (("AHH", "home"), ("AHA", "away")):
            if ah_open is not None:
                emit(bookmaker, "ah", selection, _num(row, f"{open_prefix}{side}"), ah_open, False)
            if close_prefix and ah_close is not None:
                emit(bookmaker, "ah", selection, _num(row, f"{close_prefix}{side}"), ah_close, True)

    return out


def parse_new_format(path: Path, country: str) -> dict[str, list[dict[str, Any]]]:
    """Formato 'new' (Brasil etc.): resultados + 1X2 de fechamento apenas."""
    df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    competition = NEW_FORMAT_COMPETITION[country]
    matches: list[dict[str, Any]] = []
    odds: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        date = pd.to_datetime(str(row["Date"]), dayfirst=True, errors="coerce")
        if pd.isna(date):
            continue
        season = str(row["Season"]).strip().replace("/", "")
        # "20122013" -> "1213"; "2024" fica "2024" (single-year)
        if len(season) == 8:
            season = season[2:4] + season[6:8]
        base = {
            "competition_id": competition,
            "season": season,
            "match_date": date.to_pydatetime(),
            "home": str(row["Home"]).strip(),
            "away": str(row["Away"]).strip(),
        }
        home_goals = _int(row, "HG")
        matches.append(
            base
            | {
                "home_goals": home_goals,
                "away_goals": _int(row, "AG"),
                "status": "played" if home_goals is not None else "scheduled",
            }
        )
        for bookmaker, prefix in (
            ("pinnacle", "PSC"),
            ("bet365", "B365C"),
            ("market_max", "MaxC"),
            ("market_avg", "AvgC"),
        ):
            for suffix, selection in (("H", "home"), ("D", "draw"), ("A", "away")):
                price = _num(row, f"{prefix}{suffix}")
                if price is not None and price > 1.0:
                    odds.append(
                        base
                        | {
                            "source": "football-data.co.uk",
                            "bookmaker": bookmaker,
                            "market": "1x2",
                            "selection": selection,
                            "line": None,
                            "odds_decimal": price,
                            "is_closing": True,
                        }
                    )

    return {"matches": matches, "team_stats": [], "odds": odds}
