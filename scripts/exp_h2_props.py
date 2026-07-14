"""Experimento H2 (research-log): Poisson vs Negativa Binomial em props.

Chutes e cartões por jogador-jogo, exposição = minutos/90. Treino nas
temporadas anteriores, teste na mais recente. O critério decisivo é o
log-loss preditivo fora da amostra (ver models/player_props.model_comparison).
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from edgefinder.backtest.data import player_stats_frame
from edgefinder.config import settings
from edgefinder.models.player_props import model_comparison
from edgefinder.storage.repository import get_engine

CASES = [
    ("ENG-Premier League", "understat", "shots", ["2223", "2324"], "2425"),
    ("BRA-Serie A", "fbref", "shots", ["2023", "2024"], "2025"),
    ("BRA-Serie A", "fbref", "yellow_cards", ["2023", "2024"], "2025"),
    ("ENG-Premier League", "fbref", "shots_on_target", ["2425"], "2425"),
]


def main() -> int:
    engine = get_engine()
    results: dict[str, dict[str, dict[str, float]]] = {}
    for league, source, stat, train_seasons, test_season in CASES:
        df = player_stats_frame(engine, [league], source)
        df = df.dropna(subset=[stat, "minutes"])
        df = df[df["minutes"] > 0]
        if df.empty:
            print(f"{league}/{source}/{stat}: sem dados")
            continue
        if train_seasons == [test_season]:
            # fallback quando so ha uma temporada: split temporal 70/30
            df = df.sort_values("match_date")
            cut = int(len(df) * 0.7)
            train, test = df.iloc[:cut], df.iloc[cut:]
            split_desc = "70/30 temporal dentro de " + test_season
        else:
            train = df[df["season"].isin(train_seasons)]
            test = df[df["season"] == test_season]
            split_desc = f"{'+'.join(train_seasons)} -> {test_season}"
        if len(train) < 100 or len(test) < 100:
            print(f"{league}/{source}/{stat}: amostra insuficiente ({len(train)}/{len(test)})")
            continue
        comp = model_comparison(
            train[stat].to_numpy(dtype=float),
            (train["minutes"] / 90.0).to_numpy(dtype=float),
            test[stat].to_numpy(dtype=float),
            (test["minutes"] / 90.0).to_numpy(dtype=float),
        )
        key = f"{league}|{source}|{stat}"
        results[key] = {
            "split": {"desc": split_desc, "n_train": len(train), "n_test": len(test)},  # type: ignore[dict-item]
            **{str(idx): row.to_dict() for idx, row in comp.iterrows()},
        }
        print(f"== {key} ({split_desc}, treino={len(train)}, teste={len(test)})")
        print(comp.to_string())

    out = settings.reports_dir / "h2_props_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] salvo em {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
