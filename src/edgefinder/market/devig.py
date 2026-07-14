"""Remoção do overround (vig) de odds decimais de mercado.

As casas de aposta publicam odds cuja soma das probabilidades implícitas
excede 1 (o overround, a margem da casa). Para comparar o modelo com o
mercado é preciso recuperar as probabilidades "justas" que o mercado
embute. Cada método faz uma hipótese distinta sobre COMO a margem foi
distribuída entre os resultados:

- proporcional: margem distribuída proporcionalmente à probabilidade;
- aditivo: margem distribuída igualmente entre os resultados;
- Shin: margem originada de proteção contra apostadores com informação
  privilegiada (insiders), o que concentra margem nos azarões e corrige
  o viés favorito-azarão (favourite-longshot bias).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

FloatArray = NDArray[np.float64]


def _validar_odds(odds: ArrayLike) -> FloatArray:
    """Converte e valida odds decimais: vetor 1-D, finito, com toda odd > 1.

    Odd decimal igual a 1 implicaria probabilidade 1 (evento certo) e
    odds menores não têm interpretação probabilística; ambas indicam
    dado corrompido, então falhamos cedo em vez de propagar lixo.
    """
    o = np.asarray(odds, dtype=np.float64)
    if o.ndim != 1:
        raise ValueError(f"odds deve ser vetor 1-D, recebido ndim={o.ndim}")
    if o.size == 0:
        raise ValueError("odds vazio")
    if not np.all(np.isfinite(o)):
        raise ValueError("odds contem NaN ou infinito")
    if np.any(o <= 1.0):
        raise ValueError("toda odd decimal deve ser > 1.0")
    return o


def implied_probs(odds: ArrayLike) -> FloatArray:
    """Probabilidades implícitas brutas: q_i = 1 / o_i.

    Com overround, sum(q) > 1: a diferença sum(q) - 1 é a margem da
    casa. Estas probabilidades NÃO somam 1 e não devem ser usadas
    diretamente como estimativa de probabilidade — aplique um dos
    métodos de devig antes.
    """
    return 1.0 / _validar_odds(odds)


def devig_proportional(odds: ArrayLike) -> FloatArray:
    """Devig proporcional (normalização): p_i = q_i / sum(q).

    Hipótese: a casa infla cada probabilidade pelo mesmo fator
    multiplicativo. É o método mais simples e o baseline padrão, mas
    ignora o viés favorito-azarão: em mercados com azarões extremos
    tende a superestimar a probabilidade do azarão.
    """
    q = implied_probs(odds)
    return q / q.sum()


def devig_additive(odds: ArrayLike) -> FloatArray:
    """Devig aditivo: p_i = q_i - (sum(q) - 1) / n.

    Hipótese: a casa adiciona a mesma margem absoluta a cada resultado.
    Falha estrutural: para azarões com q_i pequeno o desconto uniforme
    pode produzir p_i <= 0, o que não é uma probabilidade. Nesse caso
    levantamos ValueError e o chamador decide o fallback (tipicamente
    ``devig_proportional`` ou ``devig_shin``) — a escolha do método é
    decisão de modelagem, não deste módulo.
    """
    q = implied_probs(odds)
    p = q - (q.sum() - 1.0) / q.size
    if np.any(p <= 0.0):
        raise ValueError(
            "devig aditivo produziu probabilidade <= 0 (azarao com margem uniforme); "
            "use devig_proportional ou devig_shin"
        )
    return p


def _shin_probs(q: FloatArray, b: float, z: float) -> FloatArray:
    """Probabilidades de Shin para fração de insiders z fixada.

    p_i(z) = (sqrt(z^2 + 4 (1 - z) q_i^2 / B) - z) / (2 (1 - z)),
    com B = sum(q). Em z = 0 reduz-se a p_i = q_i / sqrt(B).
    """
    return (np.sqrt(z * z + 4.0 * (1.0 - z) * q * q / b) - z) / (2.0 * (1.0 - z))


def shin_z(odds: ArrayLike, tol: float = 1e-10) -> float:
    """Resolve a fração z de apostadores insiders do modelo de Shin.

    No modelo de Shin (1992, 1993) a casa enfrenta uma fração z de
    apostadores com informação perfeita e ajusta as odds para não perder
    dinheiro para eles; z é identificado pela restrição de que as
    probabilidades verdadeiras somam 1:

        f(z) = sum_i p_i(z) - 1 = 0,
        p_i(z) = (sqrt(z^2 + 4 (1 - z) q_i^2 / B) - z) / (2 (1 - z)).

    f é estritamente decrescente em z, com f(0) = sqrt(B) - 1 > 0
    quando há margem (B > 1) e limite negativo em z -> 1 (pois cada
    q_i < 1 implica sum q_i^2 / B < 1), logo a raiz em (0, 1) é única
    e ``brentq`` a encontra com garantia. Sem margem (B <= 1) não há
    insiders a explicar e retornamos z = 0.
    """
    q = implied_probs(odds)
    b = float(q.sum())
    if b <= 1.0 + tol:
        return 0.0

    def excesso(z: float) -> float:
        return float(_shin_probs(q, b, z).sum()) - 1.0

    return float(brentq(excesso, 0.0, 1.0 - 1e-9, xtol=tol))


def devig_shin(odds: ArrayLike, tol: float = 1e-10) -> FloatArray:
    """Devig pelo método de Shin (1992, 1993).

    Resolve z por ``shin_z`` e avalia

        p_i = (sqrt(z^2 + 4 (1 - z) q_i^2 / B) - z) / (2 (1 - z)),

    com q_i = 1/o_i e B = sum(q). Como a margem no modelo de Shin vem
    de proteção contra insiders, ela recai desproporcionalmente sobre
    os azarões: comparado ao devig proporcional, Shin atribui menos
    probabilidade ao azarão e mais ao favorito, corrigindo o viés
    favorito-azarão documentado empiricamente. Para mercados de dois
    resultados equilibrados os métodos praticamente coincidem.

    O resultado é renormalizado (divisão pela soma) para absorver o
    resíduo numérico da raiz e garantir sum(p) = 1 exato.
    """
    q = implied_probs(odds)
    b = float(q.sum())
    if b <= 1.0 + tol:
        return q / b
    p = _shin_probs(q, b, shin_z(odds, tol=tol))
    return p / p.sum()


def devig(odds: ArrayLike, method: str = "shin") -> FloatArray:
    """Despacha para o método de devig pedido: 'proportional', 'additive' ou 'shin'.

    Default 'shin' porque é o único dos três que modela a origem da
    margem (insiders) em vez de apenas redistribuí-la, e é o padrão em
    literatura de eficiência de mercados de aposta.
    """
    metodos = {
        "proportional": devig_proportional,
        "additive": devig_additive,
        "shin": devig_shin,
    }
    if method not in metodos:
        raise ValueError(f"metodo de devig desconhecido: {method!r}; use um de {sorted(metodos)}")
    return metodos[method](odds)
