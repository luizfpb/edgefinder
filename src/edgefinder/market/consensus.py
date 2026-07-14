"""Consenso de probabilidades entre casas de aposta.

Casas diferentes têm qualidade informacional diferente. A linha de uma
casa "sharp" (Pinnacle: margem baixa, limites altos, aceita apostadores
vencedores) incorpora o dinheiro informado do mercado e é o melhor
estimador público disponível; casas "soft" recreativas movem a linha
com atraso e carregam viés de público. Um consenso ponderado extrai um
sinal melhor do que qualquer casa isolada ou uma média simples.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

DEFAULT_WEIGHTS: Final[dict[str, float]] = {
    "pinnacle": 3.0,
    "market_avg": 1.5,
}
_PESO_PADRAO: Final[float] = 1.0


def consensus_prob(
    probs_por_casa: Mapping[str, FloatArray],
    weights: Mapping[str, float] | None = None,
) -> FloatArray:
    """Média ponderada renormalizada das probabilidades (pós-devig) por casa.

    Fórmula: p_i = (sum_c w_c p_{c,i}) / (sum_c w_c), renormalizada em
    seguida por p_i / sum_j p_j para garantir soma 1 mesmo quando as
    entradas carregam resíduo numérico.

    Pesos default (busca pelo nome exato e, em seguida, pelo nome em
    minúsculas; casas ausentes do mapa recebem 1.0):

    - pinnacle = 3.0: casa sharp — margem baixa e limites altos fazem
      da sua linha o consenso do dinheiro informado, então ela vale
      mais que qualquer casa recreativa individual;
    - market_avg = 1.5: a média de mercado agrega muitas casas (reduz
      ruído idiossincrático) mas dilui o sinal sharp com casas soft,
      logo merece peso intermediário;
    - demais = 1.0.

    Passar ``weights`` substitui o mapa default por inteiro (casas fora
    do mapa passado continuam valendo 1.0).
    """
    if not probs_por_casa:
        raise ValueError("probs_por_casa vazio: nenhuma casa para agregar")

    mapa_pesos = DEFAULT_WEIGHTS if weights is None else weights

    shape: tuple[int, ...] | None = None
    acumulado: FloatArray | None = None
    soma_pesos = 0.0
    for casa, probs in probs_por_casa.items():
        p = np.asarray(probs, dtype=np.float64)
        if p.ndim != 1:
            raise ValueError(f"probs da casa {casa!r} deve ser vetor 1-D")
        if shape is None:
            shape = p.shape
        elif p.shape != shape:
            raise ValueError(f"probs da casa {casa!r} tem tamanho {p.shape}, esperado {shape}")
        if not np.all(np.isfinite(p)) or np.any(p < 0.0):
            raise ValueError(f"probs da casa {casa!r} contem valor invalido (NaN, inf ou < 0)")
        w = float(mapa_pesos.get(casa, mapa_pesos.get(casa.lower(), _PESO_PADRAO)))
        if w <= 0.0:
            raise ValueError(f"peso da casa {casa!r} deve ser > 0, recebido {w}")
        acumulado = w * p if acumulado is None else acumulado + w * p
        soma_pesos += w

    assert acumulado is not None
    media = acumulado / soma_pesos
    total = float(media.sum())
    if total <= 0.0:
        raise ValueError("probabilidades agregadas somam zero; entradas degeneradas")
    return media / total
