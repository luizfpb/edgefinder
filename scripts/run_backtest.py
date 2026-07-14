"""Roda o backtest 1X2 completo e grava artefatos em data/reports/.

Uso: .venv/Scripts/python.exe scripts/run_backtest.py [half_life] [freq]
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from edgefinder.backtest.runner import BacktestConfig, run_1x2_backtest
from edgefinder.storage.repository import get_engine

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


def main() -> int:
    half_life = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0
    freq = sys.argv[2] if len(sys.argv) > 2 else "W"
    t0 = time.time()
    config = BacktestConfig(
        competitions=COMPETITIONS, half_life_days=half_life, freq=freq, start_season="1920"
    )
    report = run_1x2_backtest(get_engine(), config)
    print(f"[{time.strftime('%H:%M:%S')}] backtest DONE em {(time.time() - t0) / 60:.1f} min")
    print(f"VEREDITO: {report.get('verdict')}")
    for comp, s in (report.get("leagues") or {}).items():
        print(
            f"  {comp}: n={s['n_predictions']} ll_modelo={s['model_logloss_home']:.4f} "
            f"ll_mercado={s['market_logloss_home']:.4f} bate={s['model_beats_market_logloss']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
