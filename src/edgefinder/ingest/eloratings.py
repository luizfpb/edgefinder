"""eloratings.net: Elo de seleções (prior fraco do Tier 3).

O site é um SPA que carrega TSVs abertos (sem chave). O snapshot diário fica
em data/raw/eloratings/World_<data>.tsv. O TSV não tem cabeçalho; as colunas
relevantes são posicionais: [0]=rank, [2]=código do país, [3]=rating Elo.
Tier 3 nasce desligado (sem odds históricas para validar CLV), então este
módulo existe para alimentar o prior quando essa fase for ativada.
"""

from pathlib import Path

import pandas as pd

from edgefinder.config import settings


def read_national_elo(path: Path | None = None) -> pd.DataFrame:
    """Lê o snapshot mais recente de Elo de seleções: [rank, country_code, elo]."""
    if path is None:
        candidates = sorted((settings.raw_dir / "eloratings").glob("World_*.tsv"))
        if not candidates:
            return pd.DataFrame(columns=["rank", "country_code", "elo"])
        path = candidates[-1]
    df = pd.read_csv(path, sep="\t", header=None, encoding="utf-8")
    out = df[[0, 2, 3]].copy()
    out.columns = ["rank", "country_code", "elo"]
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce")
    out["elo"] = pd.to_numeric(out["elo"], errors="coerce")
    return out.dropna(subset=["rank", "elo"]).reset_index(drop=True)
