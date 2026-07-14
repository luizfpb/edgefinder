"""CLV (Closing Line Value): valor capturado contra a linha de fechamento.

A linha de fechamento é o preço mais eficiente do mercado — agrega toda
a informação disponível até o início do jogo. Bater sistematicamente o
fechamento (CLV > 0) é o melhor preditor de lucratividade de longo
prazo de um apostador, muito antes de o resultado das apostas ter poder
estatístico. Por isso o CLV é a métrica primária de habilidade do
sistema, e o ROI realizado é apenas sua consequência ruidosa.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def clv(odds_taken: ArrayLike, closing_odds_fair: ArrayLike) -> FloatArray:
    """CLV de cada aposta: clv = odds_taken / closing_fair - 1.

    ``closing_odds_fair`` é a odd JUSTA de fechamento, isto é, já sem a
    margem da casa (pós-devig): closing_fair = 1 / p_close. Usar a odd
    bruta de fechamento inflaria o CLV pela margem. CLV > 0 significa
    que a aposta foi feita a preço melhor que o consenso final do
    mercado ("venceu o fechamento").
    """
    taken = np.asarray(odds_taken, dtype=np.float64)
    fechamento = np.asarray(closing_odds_fair, dtype=np.float64)
    for nome, o in (("odds_taken", taken), ("closing_odds_fair", fechamento)):
        if not np.all(np.isfinite(o)):
            raise ValueError(f"{nome} contem NaN ou infinito")
        if np.any(o <= 1.0):
            raise ValueError(f"{nome} deve ser odd decimal > 1.0")
    return np.asarray(taken / fechamento - 1.0, dtype=np.float64)


@dataclass(frozen=True)
class ClvReport:
    """Resumo distributivo do CLV de um conjunto de apostas."""

    mean: float
    median: float
    pct_positive: float
    ci_low: float
    ci_high: float
    n: int


def clv_report(df: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> ClvReport:
    """Relatório de CLV: média, mediana, % positivo e IC 95% bootstrap da média.

    Espera colunas ``odds_taken`` e ``closing_fair``. O intervalo de
    confiança usa bootstrap percentil (reamostragem com reposição,
    ``n_boot`` réplicas, semente fixa para reprodutibilidade):

        IC 95% = (quantil 2.5%, quantil 97.5%) de {mean(clv*_b)}_b,

    onde clv*_b é uma reamostra de tamanho n. Bootstrap em vez de IC
    normal porque a distribuição de CLV é tipicamente assimétrica e de
    cauda pesada em amostras pequenas, onde o TCL ainda não socorre.
    """
    faltantes = {"odds_taken", "closing_fair"} - set(df.columns)
    if faltantes:
        raise ValueError(f"df sem colunas obrigatorias: {sorted(faltantes)}")
    if len(df) == 0:
        raise ValueError("df vazio: nenhum CLV para resumir")
    if n_boot < 1:
        raise ValueError(f"n_boot deve ser >= 1, recebido {n_boot}")

    valores = clv(
        np.asarray(df["odds_taken"], dtype=np.float64),
        np.asarray(df["closing_fair"], dtype=np.float64),
    )
    n = valores.size

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_boot, n))
    medias_boot = valores[indices].mean(axis=1)

    return ClvReport(
        mean=float(valores.mean()),
        median=float(np.median(valores)),
        pct_positive=float(np.mean(valores > 0.0)),
        ci_low=float(np.quantile(medias_boot, 0.025)),
        ci_high=float(np.quantile(medias_boot, 0.975)),
        n=int(n),
    )
