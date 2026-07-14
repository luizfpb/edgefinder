"""H3: de-vig de Shin vs proporcional vs aditivo como benchmark de mercado.

Hipotese (research-log 2026-07-13): Shin produz probabilidade implicita mais
calibrada que o proporcional sobre odds de fechamento (corrige o vies
favorito-azarao), medido por log-loss multiclasse contra os resultados.

Teste: os tres metodos sobre o fechamento de cada casa disponivel (pinnacle,
market_avg, market_max), por liga e agregado. O vencedor vira o benchmark
oficial do backtest.

Uso: .venv/Scripts/python.exe scripts/exp_h3_devig.py
Artefato: data/reports/h3_devig_comparison.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from edgefinder.config import settings
from edgefinder.market.devig import devig
from edgefinder.storage.repository import get_engine, read_df

COMPETITIONS = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
    "ENG-Championship",
    "NED-Eredivisie",
    "POR-Liga Portugal",
    "BRA-Serie A",
]
METHODS = ["proportional", "additive", "shin"]
BOOKMAKERS = ["pinnacle", "market_avg", "market_max"]


def load_closing(engine, competitions: list[str]) -> pd.DataFrame:
    placeholders = ",".join(f":c{i}" for i in range(len(competitions)))
    params: dict[str, object] = {f"c{i}": c for i, c in enumerate(competitions)}
    sql = f"""
        SELECT o.match_id, o.bookmaker, o.selection, o.odds_decimal,
               m.competition_id, m.match_date, m.home_goals, m.away_goals
        FROM odds_snapshots o
        JOIN matches m ON m.id = o.match_id
        WHERE m.competition_id IN ({placeholders})
          AND o.market = '1x2' AND o.is_closing = 1
          AND m.home_goals IS NOT NULL
    """
    df = read_df(engine, sql, params)
    wide = df.pivot_table(
        index=["match_id", "bookmaker", "competition_id", "home_goals", "away_goals"],
        columns="selection",
        values="odds_decimal",
        aggfunc="last",
    ).reset_index()
    return wide.dropna(subset=["home", "draw", "away"])


def logloss_multiclass(probs: np.ndarray, outcome_idx: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(outcome_idx)), outcome_idx], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def main() -> int:
    engine = get_engine()
    wide = load_closing(engine, COMPETITIONS)
    outcome_idx = np.where(
        wide["home_goals"] > wide["away_goals"],
        0,
        np.where(wide["home_goals"] < wide["away_goals"], 2, 1),
    )
    wide = wide.assign(outcome_idx=outcome_idx)

    result: dict[str, object] = {"n_total": len(wide)}
    for book in BOOKMAKERS:
        sub = wide[wide["bookmaker"] == book]
        if sub.empty:
            continue
        odds = sub[["home", "draw", "away"]].to_numpy(dtype=float)
        oidx = sub["outcome_idx"].to_numpy()
        per_method: dict[str, object] = {"n": len(sub)}
        for method in METHODS:
            probs = np.vstack([devig(o, method=method) for o in odds])
            per_method[method] = {
                "logloss": round(logloss_multiclass(probs, oidx), 6),
                "brier_home": round(
                    float(np.mean((probs[:, 0] - (oidx == 0).astype(float)) ** 2)), 6
                ),
            }
        # por liga (so log-loss, para ver consistencia)
        by_league: dict[str, object] = {}
        for comp, grp in sub.groupby("competition_id"):
            go = grp[["home", "draw", "away"]].to_numpy(dtype=float)
            gi = grp["outcome_idx"].to_numpy()
            by_league[str(comp)] = {
                m: round(logloss_multiclass(np.vstack([devig(o, method=m) for o in go]), gi), 6)
                for m in METHODS
            }
        per_method["by_league"] = by_league
        result[book] = per_method
        print(f"{book}: n={len(sub)}")
        for m in METHODS:
            print(f"  {m}: logloss={per_method[m]['logloss']}")  # type: ignore[index]

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    out = settings.reports_dir / "h3_devig_comparison.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"artefato: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
