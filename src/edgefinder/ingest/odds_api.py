"""The Odds API: snapshots de odds correntes (free tier: 500 créditos/mês).

Regras do free tier (verificadas na Fase 0):
- custo de uma chamada /odds = n_mercados x n_regiões créditos;
- odds HISTÓRICAS são plano pago — o histórico de closing line é construído
  por NÓS, acumulando snapshots diários (por isso o coletor roda desde a
  Fase 1: cada dia sem coletar é dado de backtest perdido);
- player props de futebol: só Big 5 + MLS, via casas dos EUA.

Todo consumo é registrado em credit_ledger; o coletor se recusa a estourar o
orçamento mensal configurado.
"""

from datetime import UTC, datetime
from typing import Any

import httpx
import structlog
from sqlalchemy import Engine, func, select

from edgefinder.config import settings
from edgefinder.storage import schema
from edgefinder.storage.repository import upsert

log = structlog.get_logger()

BASE = "https://api.the-odds-api.com/v4"

SPORT_KEYS = {
    "ENG-Premier League": "soccer_epl",
    "ESP-La Liga": "soccer_spain_la_liga",
    "ITA-Serie A": "soccer_italy_serie_a",
    "GER-Bundesliga": "soccer_germany_bundesliga",
    "FRA-Ligue 1": "soccer_france_ligue_one",
    "BRA-Serie A": "soccer_brazil_campeonato",
    "BRA-Serie B": "soccer_brazil_serie_b",
    "EUR-Champions League": "soccer_uefa_champs_league",
    "SAM-Copa Libertadores": "soccer_conmebol_copa_libertadores",
    "INT-World Cup": "soccer_fifa_world_cup",
}


class OddsApiError(RuntimeError):
    pass


def credits_used_this_month(engine: Engine) -> int:
    now = datetime.now(UTC)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    with engine.connect() as conn:
        total = conn.execute(
            select(func.coalesce(func.sum(schema.credit_ledger.c.credits), 0)).where(
                schema.credit_ledger.c.api == "theoddsapi",
                schema.credit_ledger.c.used_at >= month_start.replace(tzinfo=None),
            )
        ).scalar_one()
    return int(total)


def collect_h2h_snapshot(
    engine: Engine,
    competitions: list[str],
    markets: str = "h2h,totals",
    regions: str = "eu",
) -> int:
    """Coleta um snapshot de odds para as competições dadas. Retorna n de linhas."""
    if not settings.odds_api_key:
        log.warning("odds_api.sem_chave", hint="defina ODDS_API_KEY no .env")
        return 0

    cost_per_call = len(markets.split(",")) * len(regions.split(","))
    used = credits_used_this_month(engine)
    budget = settings.odds_api_monthly_budget
    total_rows = 0

    with httpx.Client(timeout=30) as client:
        for comp in competitions:
            sport_key = SPORT_KEYS.get(comp)
            if sport_key is None:
                log.warning("odds_api.competicao_sem_sport_key", competition=comp)
                continue
            if used + cost_per_call > budget:
                log.error("odds_api.orcamento_estourado", used=used, budget=budget)
                break
            resp = client.get(
                f"{BASE}/sports/{sport_key}/odds",
                params={
                    "apiKey": settings.odds_api_key,
                    "regions": regions,
                    "markets": markets,
                    "oddsFormat": "decimal",
                },
            )
            used_header = resp.headers.get("x-requests-used")
            if resp.status_code != 200:
                log.error("odds_api.erro", status=resp.status_code, body=resp.text[:200])
                continue
            upsert(
                engine,
                schema.credit_ledger,
                [
                    {
                        "api": "theoddsapi",
                        "used_at": datetime.now(UTC).replace(tzinfo=None),
                        "credits": cost_per_call,
                        "endpoint": f"odds:{sport_key}:{markets}:{regions}",
                    }
                ],
                conflict_cols=["id"],
            )
            used += cost_per_call
            rows = _rows_from_response(resp.json(), comp)
            total_rows += upsert(
                engine,
                schema.odds_snapshots,
                rows,
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
            log.info(
                "odds_api.snapshot",
                competition=comp,
                rows=len(rows),
                credits_used_header=used_header,
            )
    return total_rows


def _rows_from_response(events: list[dict[str, Any]], competition: str) -> list[dict[str, Any]]:
    collected_at = datetime.now(UTC).replace(tzinfo=None)
    rows: list[dict[str, Any]] = []
    for event in events:
        home, away = event.get("home_team"), event.get("away_team")
        commence = event.get("commence_time")
        commence_dt = (
            datetime.fromisoformat(commence.replace("Z", "+00:00")).replace(tzinfo=None)
            if commence
            else None
        )
        for bookmaker in event.get("bookmakers", []):
            book_key = bookmaker["key"]
            for market in bookmaker.get("markets", []):
                market_key = {"h2h": "1x2", "totals": "ou"}.get(market["key"], market["key"])
                for outcome in market.get("outcomes", []):
                    name = outcome["name"]
                    if market_key == "1x2":
                        selection = "home" if name == home else "away" if name == away else "draw"
                    else:
                        selection = name.lower()
                    rows.append(
                        {
                            "match_id": None,
                            "home_team_raw": home,
                            "away_team_raw": away,
                            "commence_time": commence_dt,
                            "source": "theoddsapi",
                            "bookmaker": book_key,
                            "market": market_key,
                            "selection": selection,
                            "line": outcome.get("point"),
                            "odds_decimal": float(outcome["price"]),
                            "collected_at": collected_at,
                            "is_closing": False,
                        }
                    )
    return rows
