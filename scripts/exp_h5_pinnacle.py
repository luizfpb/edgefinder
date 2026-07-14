"""H5: o fechamento da Pinnacle pos-23/07/2025 segue confiavel como benchmark?

Contexto (research-log 2026-07-13): o proprio football-data.co.uk avisa que as
closing odds da Pinnacle ficaram nao confiaveis a partir de 23/07/2025.

Teste: comparar overround e log-loss (devig Shin) de pinnacle vs market_avg e
market_max antes/depois do corte. Se a Pinnacle divergir no periodo recente,
o benchmark do backtest passa a ser market_avg nesse periodo.

Uso: .venv/Scripts/python.exe scripts/exp_h5_pinnacle.py
Artefato: data/reports/h5_pinnacle_reliability.json
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

CUTOFF = "2025-07-23"
COMPETITIONS = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
    "ENG-Championship",
    "NED-Eredivisie",
    "POR-Liga Portugal",
]
BOOKMAKERS = ["pinnacle", "market_avg", "market_max"]


def logloss_multiclass(probs: np.ndarray, outcome_idx: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(outcome_idx)), outcome_idx], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def main() -> int:
    engine = get_engine()
    placeholders = ",".join(f":c{i}" for i in range(len(COMPETITIONS)))
    params: dict[str, object] = {f"c{i}": c for i, c in enumerate(COMPETITIONS)}
    sql = f"""
        SELECT o.match_id, o.bookmaker, o.selection, o.odds_decimal,
               m.match_date, m.home_goals, m.away_goals
        FROM odds_snapshots o
        JOIN matches m ON m.id = o.match_id
        WHERE m.competition_id IN ({placeholders})
          AND o.market = '1x2' AND o.is_closing = 1
          AND m.home_goals IS NOT NULL
    """
    df = read_df(engine, sql, params)
    wide = df.pivot_table(
        index=["match_id", "bookmaker", "match_date", "home_goals", "away_goals"],
        columns="selection",
        values="odds_decimal",
        aggfunc="last",
    ).reset_index()
    wide = wide.dropna(subset=["home", "draw", "away"])
    wide["match_date"] = pd.to_datetime(wide["match_date"])
    wide["period"] = np.where(wide["match_date"] < pd.Timestamp(CUTOFF), "pre", "pos")
    wide["outcome_idx"] = np.where(
        wide["home_goals"] > wide["away_goals"],
        0,
        np.where(wide["home_goals"] < wide["away_goals"], 2, 1),
    )
    odds_cols = ["home", "draw", "away"]
    wide["overround"] = (1.0 / wide[odds_cols]).sum(axis=1) - 1.0

    # so jogos em que TODAS as casas comparadas existem (comparacao pareada)
    counts = wide.groupby("match_id")["bookmaker"].nunique()
    paired_ids = counts[counts >= len(BOOKMAKERS)].index
    paired = wide[wide["match_id"].isin(paired_ids)]

    result: dict[str, object] = {"cutoff": CUTOFF, "n_paired_matches": len(paired_ids)}
    for period in ("pre", "pos"):
        block: dict[str, object] = {}
        sub_p = paired[paired["period"] == period]
        block["n_matches"] = int(sub_p["match_id"].nunique())
        for book in BOOKMAKERS:
            sub = sub_p[sub_p["bookmaker"] == book]
            if sub.empty:
                continue
            odds = sub[odds_cols].to_numpy(dtype=float)
            probs = np.vstack([devig(o, method="shin") for o in odds])
            block[book] = {
                "n": len(sub),
                "overround_mean": round(float(sub["overround"].mean()), 5),
                "overround_std": round(float(sub["overround"].std()), 5),
                "logloss_shin": round(logloss_multiclass(probs, sub["outcome_idx"].to_numpy()), 6),
            }
        result[period] = block
        print(f"periodo {period}: {json.dumps(block, indent=2)}")

    # divergencia media |p_pinnacle - p_avg| por periodo (no conjunto pareado)
    piv: dict[str, dict[int, np.ndarray]] = {}
    for book in ("pinnacle", "market_avg"):
        sub = paired[paired["bookmaker"] == book].set_index("match_id")
        piv[book] = {
            int(mid): devig(row[odds_cols].to_numpy(dtype=float), method="shin")
            for mid, row in sub.iterrows()
        }
    common = set(piv["pinnacle"]) & set(piv["market_avg"])
    per_dates = paired.drop_duplicates("match_id").set_index("match_id")["period"]
    outcome_by_id = (
        paired.drop_duplicates("match_id").set_index("match_id")["outcome_idx"].to_dict()
    )
    for period in ("pre", "pos"):
        ids = [m for m in common if per_dates.get(m) == period]
        if not ids:
            continue
        diffs = [np.abs(piv["pinnacle"][m] - piv["market_avg"][m]).mean() for m in ids]
        result[f"divergencia_media_{period}"] = round(float(np.mean(diffs)), 6)
        print(f"divergencia pinnacle vs avg ({period}): {np.mean(diffs):.6f} (n={len(ids)})")
        # log-loss PAREADO: mesmas partidas para as duas casas — sem isso a
        # comparacao pos-corte tem vies de selecao (a pinnacle some de metade
        # dos jogos e os que sobram nao sao amostra aleatoria)
        oidx = np.array([outcome_by_id[m] for m in ids])
        for book in ("pinnacle", "market_avg"):
            probs = np.vstack([piv[book][m] for m in ids])
            key = f"logloss_pareado_{period}_{book}"
            result[key] = round(logloss_multiclass(probs, oidx), 6)
            print(f"{key}: {result[key]} (n={len(ids)})")

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    out = settings.reports_dir / "h5_pinnacle_reliability.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"artefato: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
