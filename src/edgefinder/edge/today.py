"""Edges sobre jogos futuros: snapshots de odds correntes vs. modelo.

Cada sugestão sai com explicação (drivers do modelo) — uma sugestão sem
explicação é inútil. E cada uma respeita o threshold de EV do tier da
competição (Tier 3 nem chega aqui: nasce desligado, PLAN.md seção 3).
"""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import structlog
from sqlalchemy import Engine

from edgefinder.backtest.data import matches_frame
from edgefinder.config import COMPETITION_TIERS, settings
from edgefinder.data.teamnames import normalize_team_name
from edgefinder.edge.ev import expected_value
from edgefinder.edge.kelly import kelly_stake
from edgefinder.market.devig import devig
from edgefinder.models.dixon_coles import DixonColes
from edgefinder.storage.repository import read_df

log = structlog.get_logger()

MIN_EV_BY_TIER = {1: "min_ev_tier1", 2: "min_ev_tier2", 3: "min_ev_tier3"}


def _latest_snapshots(engine: Engine) -> pd.DataFrame:
    sql = """
        SELECT home_team_raw, away_team_raw, commence_time, bookmaker, market,
               selection, line, odds_decimal, collected_at
        FROM odds_snapshots
        WHERE source = 'theoddsapi' AND commence_time > :now AND market = '1x2'
    """
    df = read_df(engine, sql, {"now": datetime.now().isoformat(sep=" ")})
    if df.empty:
        return df
    df["collected_at"] = pd.to_datetime(df["collected_at"])
    df = (
        df.sort_values("collected_at")
        .groupby(["home_team_raw", "away_team_raw", "bookmaker", "selection"], as_index=False)
        .last()
    )
    return df


def backtest_gate() -> tuple[bool, str]:
    """Trava de honestidade: sem backtest aprovado, não há sugestão de aposta.

    Regra do projeto: se o walk-forward mostra que o modelo não bate a closing
    line, o sistema DIZ isso e se recusa a sugerir — não é aviso, é bloqueio.
    """
    import json

    path = settings.reports_dir / "backtest_summary.json"
    if not path.exists():
        return False, "Nenhum backtest gravado ainda. Rode 'edgefinder backtest' primeiro."
    verdict = str(json.loads(path.read_text(encoding="utf-8")).get("verdict", ""))
    if "NAO BATE" in verdict or "INCONCLUSIVO" in verdict or not verdict:
        return False, verdict
    return True, verdict


def today_edges(
    engine: Engine,
    min_ev: float = 0.03,
    half_life_days: float = 180.0,
    ignore_gate: bool = False,
) -> pd.DataFrame:
    approved, verdict = backtest_gate()
    if not approved and not ignore_gate:
        log.warning("edges.bloqueado_pelo_backtest", verdict=verdict)
        return pd.DataFrame()
    snaps = _latest_snapshots(engine)
    if snaps.empty:
        return pd.DataFrame()

    all_matches = matches_frame(engine, list(COMPETITION_TIERS))
    team_comp = _team_competition_map(all_matches)

    models: dict[str, DixonColes] = {}
    rows: list[dict[str, Any]] = []
    for (home_raw, away_raw), event in snaps.groupby(["home_team_raw", "away_team_raw"]):
        home = normalize_team_name(str(home_raw))
        away = normalize_team_name(str(away_raw))
        comp = team_comp.get(home) or team_comp.get(away)
        if comp is None or COMPETITION_TIERS.get(comp, 3) == 3:
            continue  # sem historico ou Tier 3 (desligado por construcao)

        if comp not in models:
            train = all_matches[all_matches["competition_id"] == comp]
            if len(train) < 380:
                continue
            models[comp] = DixonColes().fit(
                train, ref_date=train["match_date"].max(), half_life_days=half_life_days
            )
        model = models[comp]
        try:
            p_home, p_draw, p_away = model.probs_1x2(home, away)
        except KeyError:
            continue

        pivot = event.pivot_table(
            index="bookmaker", columns="selection", values="odds_decimal", aggfunc="last"
        )
        if not {"home", "draw", "away"}.issubset(pivot.columns):
            continue
        best = pivot.max()
        p_market = np.mean(
            [
                devig(row[["home", "draw", "away"]].to_numpy(), method="shin")
                for _, row in pivot.dropna().iterrows()
            ],
            axis=0,
        )
        threshold = max(min_ev, getattr(settings, MIN_EV_BY_TIER[COMPETITION_TIERS.get(comp, 2)]))
        for i, sel in enumerate(["home", "draw", "away"]):
            p_model = (p_home, p_draw, p_away)[i]
            odds = float(best[sel])
            ev = expected_value(p_model, odds)
            if ev < threshold:
                continue
            rows.append(
                {
                    "home_team": home_raw,
                    "away_team": away_raw,
                    "competition": comp,
                    "commence_time": event["commence_time"].iloc[0],
                    "market": "1x2",
                    "selection": sel,
                    "p_model": p_model,
                    "p_market": float(p_market[i]),
                    "odds": odds,
                    "ev": ev,
                    "stake_frac": kelly_stake(
                        p_model,
                        odds,
                        bankroll=1.0,
                        fraction=settings.kelly_fraction,
                        cap=settings.max_exposure_per_match,
                    ),
                    "explanation": _explain(
                        model, home, away, p_model, float(p_market[i]), comp, half_life_days
                    ),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("ev", ascending=False).reset_index(drop=True)
        settings.reports_dir.mkdir(parents=True, exist_ok=True)
        out.to_parquet(settings.reports_dir / "edges_today.parquet")
    return out


def _team_competition_map(matches: pd.DataFrame) -> dict[str, str]:
    """Competição mais recente de cada time (para classificar eventos da API)."""
    recent = matches.sort_values("match_date").drop_duplicates("home_team", keep="last")
    mapping = dict(zip(recent["home_team"], recent["competition_id"], strict=False))
    recent_away = matches.sort_values("match_date").drop_duplicates("away_team", keep="last")
    for team, comp in zip(recent_away["away_team"], recent_away["competition_id"], strict=False):
        mapping.setdefault(team, comp)
    return mapping


def _explain(
    model: DixonColes,
    home: str,
    away: str,
    p_model: float,
    p_market: float,
    comp: str,
    half_life_days: float,
) -> str:
    parts = [f"modelo {p_model:.1%} vs mercado {p_market:.1%} (consenso de-vig Shin)"]
    try:
        lam, mu = model.match_rates(home, away)
        parts.append(
            f"gols esperados {lam:.2f} x {mu:.2f} "
            f"(ataque {home} {float(model.attack_[home]):+.2f}, "
            f"defesa {away} {float(model.defence_[away]):+.2f}, "
            f"mando {model.home_advantage_:+.2f})"
        )
    except (KeyError, AttributeError):
        pass
    parts.append(f"{comp}, decaimento meia-vida {half_life_days:.0f}d")
    return "; ".join(parts)
