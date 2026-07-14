"""Experimento H1 (research-log): sensibilidade à meia-vida do decaimento.

Roda o walk-forward 1X2 da Premier League para uma grade de meias-vidas e
compara o log-loss out-of-sample do modelo (e o do mercado, fixo, como
referência). Não escreve nos artefatos oficiais de data/reports.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from edgefinder.backtest.data import closing_odds_1x2, matches_frame
from edgefinder.backtest.metrics import prob_metrics
from edgefinder.backtest.runner import _make_fit_predict
from edgefinder.backtest.walkforward import walk_forward
from edgefinder.config import settings
from edgefinder.market.devig import devig
from edgefinder.storage.repository import get_engine

LEAGUE = "ENG-Premier League"
GRID = [60.0, 90.0, 180.0, 365.0, 730.0]
START_SEASON = "1920"


def main() -> int:
    engine = get_engine()
    matches = matches_frame(engine, [LEAGUE])
    odds = closing_odds_1x2(engine, [LEAGUE])
    results: dict[str, dict[str, float]] = {}

    market_ll: float | None = None
    for half_life in GRID:
        t0 = time.time()
        fit_fn, predict_fn = _make_fit_predict(half_life)
        preds = walk_forward(
            matches,
            fit_fn=fit_fn,
            predict_fn=predict_fn,
            freq="W",
            min_train=760,
            result_cols=["home_goals", "away_goals"],
        )
        df = preds.merge(matches, on="match_id").merge(odds, on="match_id", how="inner")
        df = df[df["season"] >= START_SEASON]
        y = np.asarray((df["home_goals"] > df["away_goals"]).to_numpy(), dtype=np.float64)
        model = prob_metrics(y, df["p_home"].to_numpy(dtype=np.float64))
        if market_ll is None:
            mkt = np.vstack(
                [devig(np.array([r.odds_home, r.odds_draw, r.odds_away])) for r in df.itertuples()]
            )
            market_ll = prob_metrics(y, mkt[:, 0])["log_loss"]
        results[str(half_life)] = {
            "model_logloss_home": model["log_loss"],
            "model_brier_home": model["brier"],
            "n": model["n"],
            "elapsed_s": round(time.time() - t0, 1),
        }
        print(
            f"half_life={half_life:>6.0f}d  logloss={model['log_loss']:.5f}  "
            f"brier={model['brier']:.5f}  n={model['n']:.0f}  ({time.time() - t0:.0f}s)",
            flush=True,
        )

    print(f"mercado (Shin sobre closing): logloss={market_ll:.5f}")
    out = settings.reports_dir / "h1_halflife_grid.json"
    out.write_text(
        json.dumps({"league": LEAGUE, "market_logloss_home": market_ll, "grid": results}, indent=2),
        encoding="utf-8",
    )
    print(f"salvo em {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
