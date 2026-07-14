"""Combinação de modelos: pesos por performance fora da amostra e blending de probabilidades.

Nenhum modelo isolado (Poisson bivariado, Elo, mercado, ...) domina em todos os
regimes; combiná-los reduz variância e protege contra o erro de especificação
de cada um. Este módulo resolve as duas metades do problema: quanto peso dar a
cada modelo (``ensemble_weights``, a partir do log-loss out-of-sample) e como
agregar as probabilidades dado o peso (``blend_probs``, pooling em log-odds).
"""

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
from scipy.special import expit, logit

# Mesmo limite do módulo de calibração: |logit(clip(p))| <= ~27.6, longe de
# overflow e sem distorcer probabilidades não degeneradas.
_CLIP_EPS: float = 1e-12


def ensemble_weights(
    logloss_por_modelo: Mapping[str, float], temperature: float = 1.0
) -> dict[str, float]:
    """Pesos softmax sobre o negativo do log-loss out-of-sample de cada modelo.

    w_m = exp(-LL_m / T) / sum_k exp(-LL_k / T)

    onde ``LL_m`` é o log-loss médio por observação do modelo ``m`` fora da
    amostra e ``T = temperature`` controla a concentração: T -> inf recupera
    pesos iguais, T -> 0 dá tudo ao melhor modelo (winner-takes-all).

    Por que não pesos iguais? Pesos iguais ignoram a evidência de que um
    modelo prevê melhor que outro e deixam um modelo ruim arrastar o
    ensemble. O softmax sobre -LL/T é um meio-termo contínuo: como a
    diferença de log-loss médio vezes N é o log do fator de Bayes entre os
    modelos, esses pesos aproximam probabilidades a posteriori de modelo
    (pseudo-BMA), dando mais peso a quem provou prever melhor sem zerar os
    demais — o que preserva a diversificação que motiva o ensemble. A
    implementação subtrai o máximo do score antes de exponenciar
    (estabilidade numérica padrão do softmax).
    """
    if not logloss_por_modelo:
        raise ValueError("logloss_por_modelo vazio: nenhum modelo para ponderar")
    if temperature <= 0.0 or not np.isfinite(temperature):
        raise ValueError(f"temperature deve ser finita e > 0, recebido {temperature}")

    nomes = list(logloss_por_modelo)
    ll = np.asarray([logloss_por_modelo[m] for m in nomes], dtype=np.float64)
    if not np.all(np.isfinite(ll)):
        raise ValueError("log-loss deve ser finito para todos os modelos")

    score = -ll / temperature
    score -= score.max()
    w = np.exp(score)
    w /= w.sum()
    return {nome: float(wi) for nome, wi in zip(nomes, w, strict=True)}


def _validate_blend(
    probs: Mapping[str, npt.NDArray[np.float64]], weights: Mapping[str, float]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Valida entradas do blend e devolve (P empilhado [M, ...], w normalizado [M]).

    Centraliza as checagens para que o blend falhe cedo com mensagem clara em
    vez de propagar NaN ou broadcast silencioso de shapes incompatíveis.
    """
    if not probs:
        raise ValueError("probs vazio: nenhum modelo para combinar")
    if set(probs) != set(weights):
        raise ValueError(f"chaves de probs e weights diferem: {sorted(probs)} != {sorted(weights)}")

    nomes = list(probs)
    arrays: list[npt.NDArray[np.float64]] = []
    shape_ref: tuple[int, ...] | None = None
    for nome in nomes:
        arr = np.asarray(probs[nome], dtype=np.float64)
        if arr.size == 0:
            raise ValueError(f"probs[{nome!r}] vazio")
        if shape_ref is None:
            shape_ref = arr.shape
        elif arr.shape != shape_ref:
            raise ValueError(
                f"shapes incompatíveis: probs[{nome!r}] tem {arr.shape}, esperado {shape_ref}"
            )
        if np.any(np.isnan(arr)) or np.any(arr < 0.0) or np.any(arr > 1.0):
            raise ValueError(f"probs[{nome!r}] deve estar em [0, 1] e sem NaN")
        arrays.append(arr)

    w = np.asarray([weights[m] for m in nomes], dtype=np.float64)
    if np.any(~np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("weights devem ser finitos e >= 0")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("soma dos weights deve ser > 0")
    return np.stack(arrays, axis=0), w / total


def blend_probs(
    probs: Mapping[str, npt.NDArray[np.float64]],
    weights: Mapping[str, float],
) -> npt.NDArray[np.float64]:
    """Combina probabilidades de vários modelos por pooling em log-odds.

    logit(p_blend) = sum_m w_m * logit(p_m)   =>   p_blend = sigma(sum_m w_m * logit(p_m))

    com os pesos normalizados internamente para somar 1. Isso equivale à média
    geométrica ponderada das odds e é preferível à média linear
    ``sum_m w_m * p_m`` porque a média linear de modelos individualmente
    calibrados é sistematicamente subconfiante (encolhe tudo em direção à
    média e achata os extremos), enquanto a média em logit preserva a
    dispersão — e portanto a calibração — das previsões (logit/geometric
    pooling; Satopää et al. 2014).

    Fallback linear para probabilidades degeneradas: onde algum modelo prevê
    exatamente 0 ou 1 o logit é infinito e o pooling em log-odds deixaria a
    certeza de um único modelo dominar qualquer peso; nessas posições
    (somente nelas) usa-se a média linear ponderada, que trata a certeza
    degenerada na proporção do peso do modelo.

    Aceita arrays 1-D (probabilidade binária por observação; saída em [0, 1])
    ou 2-D com linhas = observações e colunas = classes (ex.: 1x2 calibrado
    por seleção); no caso 2-D cada linha da saída é renormalizada para somar
    1, já que o blend por célula não preserva a soma.
    """
    stacked, w = _validate_blend(probs, weights)
    if stacked.ndim not in (2, 3):
        raise ValueError("probs deve conter arrays 1-D ou 2-D")

    z = np.asarray(logit(np.clip(stacked, _CLIP_EPS, 1.0 - _CLIP_EPS)), dtype=np.float64)
    p_logit = np.asarray(expit(np.tensordot(w, z, axes=1)), dtype=np.float64)
    p_linear = np.tensordot(w, stacked, axes=1)

    degenerado = np.any((stacked == 0.0) | (stacked == 1.0), axis=0)
    blended: npt.NDArray[np.float64] = np.where(degenerado, p_linear, p_logit)

    if blended.ndim == 2:
        soma = blended.sum(axis=1, keepdims=True)
        if np.any(soma <= 0.0):
            raise ValueError("linha com soma zero após o blend: distribuição inválida")
        blended = blended / soma
    return blended
