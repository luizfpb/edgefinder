"""Valor esperado (EV) de apostas a odds decimais.

O EV por unidade apostada compara a probabilidade do modelo com o preço
oferecido: só há edge quando p_model excede a probabilidade implícita
justa do preço (p > 1/odds). É o filtro primário do sistema — stake e
gestão de risco vêm depois, em ``edgefinder.edge.kelly``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def expected_value(p_model: ArrayLike, odds: ArrayLike) -> FloatArray:
    """EV por unidade de stake: EV = p (odds - 1) - (1 - p).

    Derivação: com probabilidade p ganha-se o lucro líquido (odds - 1);
    com probabilidade (1 - p) perde-se a unidade apostada. Equivale a
    EV = p * odds - 1, zerando exatamente quando odds = 1/p (preço
    justo). Vetorizado: aceita escalares ou arrays com broadcast.
    """
    p = np.asarray(p_model, dtype=np.float64)
    o = np.asarray(odds, dtype=np.float64)
    if not np.all(np.isfinite(p)) or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("p_model deve estar em [0, 1] e ser finito")
    if not np.all(np.isfinite(o)) or np.any(o <= 1.0):
        raise ValueError("odds deve ser odd decimal > 1.0")
    return np.asarray(p * (o - 1.0) - (1.0 - p), dtype=np.float64)


def ev_table(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona coluna ``ev`` a uma cópia do DataFrame (não muta o original).

    Espera colunas ``p_model`` e ``odds``; aplica ``expected_value``
    vetorizado linha a linha. Retorna cópia para que o pipeline possa
    reprocessar o mesmo df de candidatos com modelos diferentes sem
    efeitos colaterais.
    """
    faltantes = {"p_model", "odds"} - set(df.columns)
    if faltantes:
        raise ValueError(f"df sem colunas obrigatorias: {sorted(faltantes)}")
    resultado = df.copy()
    resultado["ev"] = expected_value(
        np.asarray(df["p_model"], dtype=np.float64),
        np.asarray(df["odds"], dtype=np.float64),
    )
    return resultado
