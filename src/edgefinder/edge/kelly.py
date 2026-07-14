"""Critério de Kelly para dimensionamento de stakes.

Kelly maximiza a taxa de crescimento logarítmico esperada da banca,
E[log(1 + f X)]. Usamos Kelly FRACIONÁRIO por dois motivos:

1. Erro de estimação: f* depende de p, que vem de um modelo e carrega
   erro. Como a curva de crescimento é assimétrica em torno do ótimo —
   apostar 2 f* zera o crescimento e acima disso o crescimento é
   negativo, enquanto apostar f*/2 ainda captura ~75% do crescimento —
   superestimar p e apostar Kelly cheio é muito mais destrutivo do que
   subapostar. A fração é o seguro contra o próprio modelo.
2. Variância: a volatilidade da banca sob Kelly cheio (drawdowns de
   50%+ são esperados) é impraticável; fração 1/4 reduz a variância a
   ~1/16 do Kelly cheio.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def kelly_fraction(p: ArrayLike, odds: ArrayLike) -> FloatArray:
    """Fração de Kelly: f* = (p (odds - 1) - (1 - p)) / (odds - 1), clip em [0, inf).

    Com lucro líquido b = odds - 1, f* = (p b - (1 - p)) / b maximiza
    E[log(1 + f X)] para aposta binária. O numerador é o EV por unidade
    (ver ``edgefinder.edge.ev``): EV <= 0 implica f* <= 0, e como não
    há aposta "contra" neste mercado, o clip em zero significa
    simplesmente não apostar. Vetorizado com broadcast.
    """
    p_arr = np.asarray(p, dtype=np.float64)
    o = np.asarray(odds, dtype=np.float64)
    if not np.all(np.isfinite(p_arr)) or np.any(p_arr < 0.0) or np.any(p_arr > 1.0):
        raise ValueError("p deve estar em [0, 1] e ser finito")
    if not np.all(np.isfinite(o)) or np.any(o <= 1.0):
        raise ValueError("odds deve ser odd decimal > 1.0")
    b = o - 1.0
    f = (p_arr * b - (1.0 - p_arr)) / b
    return np.asarray(np.maximum(f, 0.0), dtype=np.float64)


def kelly_stake(
    p: float,
    odds: float,
    bankroll: float,
    fraction: float = 0.25,
    cap: float = 0.05,
) -> float:
    """Stake em unidades monetárias: stake = bankroll * min(fraction * f*, cap).

    ``fraction`` (default 1/4 de Kelly) protege contra erro de
    estimação de p — Kelly cheio com p superestimado sobre-aposta e o
    custo é assimétrico (ver docstring do módulo). ``cap`` (default 5%
    da banca) é um teto absoluto por aposta: mesmo quando o modelo
    declara edge enorme (cenário em que o mais provável é o modelo
    estar errado, não o mercado), nenhuma aposta isolada pode expor
    fração relevante da banca.
    """
    if bankroll < 0.0:
        raise ValueError(f"bankroll deve ser >= 0, recebido {bankroll}")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction deve estar em (0, 1], recebido {fraction}")
    if not 0.0 < cap <= 1.0:
        raise ValueError(f"cap deve estar em (0, 1], recebido {cap}")
    f_star = float(kelly_fraction(p, odds))
    return bankroll * min(fraction * f_star, cap)
