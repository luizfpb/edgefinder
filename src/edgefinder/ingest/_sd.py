"""Ponto único de import do soccerdata.

O soccerdata lê SOCCERDATA_DIR e o league_dict.json NO IMPORT, então este
módulo precisa ser o primeiro contato com a lib: configura o diretório de
cache dentro do projeto e garante as ligas custom (Brasileirão etc.) antes
de importar. Qualquer módulo que precise do soccerdata importa daqui.
"""

import json
import os
from pathlib import Path

from edgefinder.config import settings

SOCCERDATA_DIR = settings.raw_dir / "soccerdata"

CUSTOM_LEAGUES: dict[str, dict[str, str]] = {
    # O campo FBref exige o NOME EXATO da competição em fbref.com/en/comps/
    # (com acentos), não o id numérico. Validado na Fase 0.
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


def _ensure_config() -> None:
    os.environ.setdefault("SOCCERDATA_DIR", str(SOCCERDATA_DIR))
    config_path = Path(os.environ["SOCCERDATA_DIR"]) / "config" / "league_dict.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, str]] = {}
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
    merged = existing | CUSTOM_LEAGUES
    if merged != existing:
        config_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")


_ensure_config()

import soccerdata  # noqa: E402  (a configuração acima precisa preceder o import)

sd = soccerdata
