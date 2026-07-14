"""Probabilidade conjunta de pernas correlacionadas (múltiplas/parlays).

Pernas de uma MESMA partida não são independentes: "casa vence" e
"mais de 2.5 gols", por exemplo, compartilham o estado latente do jogo.
Multiplicar probabilidades marginais assume independência e portanto
SUBESTIMA a probabilidade conjunta (e o valor) de múltiplas com pernas
positivamente correlacionadas, e SUPERESTIMA a de pernas negativamente
correlacionadas. A cópula gaussiana acopla as marginais com um único
parâmetro rho, suficiente para o caso de duas pernas.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import multivariate_normal, norm

FloatArray = NDArray[np.float64]


def joint_prob_gaussian_copula(p_a: float, p_b: float, rho: float) -> float:
    """P(A e B) via cópula gaussiana bivariada.

    Fórmula: P(A e B) = Phi_2(Phi^{-1}(p_a), Phi^{-1}(p_b); rho),
    onde Phi^{-1} é o quantil da normal padrão (``norm.ppf``) e Phi_2 a
    CDF da normal bivariada com correlação rho
    (``multivariate_normal.cdf``). Intuição: cada evento é o corte de
    uma variável latente gaussiana; rho acopla as latentes.

    Casos-limite tratados analiticamente (a CDF numérica é instável ou
    singular neles):

    - p em {0, 1}: P = 0, ou P = probabilidade da outra perna;
    - rho = +1 (comonotonia): P = min(p_a, p_b);
    - rho = -1 (contramonotonia): P = max(p_a + p_b - 1, 0).

    Estes são exatamente os limites de Fréchet-Hoeffding, e o resultado
    para |rho| < 1 fica sempre entre eles. Com rho = 0 reduz-se ao
    produto p_a * p_b (independência).
    """
    for nome, p in (("p_a", p_a), ("p_b", p_b)):
        if not (math.isfinite(p) and 0.0 <= p <= 1.0):
            raise ValueError(f"{nome} deve estar em [0, 1], recebido {p}")
    if not (math.isfinite(rho) and -1.0 <= rho <= 1.0):
        raise ValueError(f"rho deve estar em [-1, 1], recebido {rho}")

    if p_a == 0.0 or p_b == 0.0:
        return 0.0
    if p_a == 1.0:
        return p_b
    if p_b == 1.0:
        return p_a
    if rho == 1.0:
        return min(p_a, p_b)
    if rho == -1.0:
        return max(p_a + p_b - 1.0, 0.0)

    z = np.array([float(norm.ppf(p_a)), float(norm.ppf(p_b))], dtype=np.float64)
    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=np.float64)
    conjunta = float(multivariate_normal.cdf(z, mean=np.zeros(2), cov=cov))
    return min(max(conjunta, 0.0), 1.0)


def parlay_ev(
    probs: ArrayLike,
    odds_parlay: float,
    rho_matrix: ArrayLike | None = None,
) -> float:
    """EV de uma múltipla: EV = p_joint (odds_parlay - 1) - (1 - p_joint).

    Sem ``rho_matrix`` assume independência entre as pernas:
    p_joint = prod_i p_i (adequado para pernas de partidas distintas).
    Com ``rho_matrix`` (matriz 2x2 simétrica com diagonal 1), acopla as
    DUAS pernas via cópula gaussiana com rho = rho_matrix[0, 1] —
    correlação só é suportada para o caso de 2 pernas, que cobre o uso
    prático (pernas da mesma partida); mais pernas correlacionadas
    exigiriam a CDF gaussiana n-variada e estimativa de toda a matriz.
    """
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 1 or p.size < 2:
        raise ValueError("probs deve ser vetor 1-D com pelo menos 2 pernas")
    if not np.all(np.isfinite(p)) or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("probs deve estar em [0, 1] e ser finito")
    if not (math.isfinite(odds_parlay) and odds_parlay > 1.0):
        raise ValueError(f"odds_parlay deve ser > 1.0, recebido {odds_parlay}")

    if rho_matrix is None:
        p_joint = float(np.prod(p))
    else:
        if p.size != 2:
            raise ValueError("rho_matrix so e suportada para multiplas de exatamente 2 pernas")
        m = np.asarray(rho_matrix, dtype=np.float64)
        if m.shape != (2, 2):
            raise ValueError(f"rho_matrix deve ser 2x2, recebido shape {m.shape}")
        if not np.allclose(m, m.T) or not np.allclose(np.diag(m), 1.0):
            raise ValueError("rho_matrix deve ser simetrica com diagonal 1")
        p_joint = joint_prob_gaussian_copula(float(p[0]), float(p[1]), float(m[0, 1]))

    return p_joint * (odds_parlay - 1.0) - (1.0 - p_joint)
