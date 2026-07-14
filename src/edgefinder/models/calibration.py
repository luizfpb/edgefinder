"""Calibração de probabilidades binárias, por seleção.

Um modelo pode discriminar bem (ordenar jogos por risco) e ainda assim estar
mal calibrado: dizer 60% quando a frequência real é 52%. Para apostar isso é
fatal, porque o valor esperado de uma aposta é ``EV = p * odd - 1`` e um ``p``
sistematicamente inflado transforma edge aparente em prejuízo real. Este
módulo mede (reliability table, Brier, log-loss, ECE) e corrige (isotônica,
Platt) esse desvio.

Escopo: calibração **binária**, por seleção. O mercado 1x2 é tratado como três
problemas um-contra-resto (casa, empate, fora), cada um calibrado em separado
com estas funções; a renormalização das três probabilidades para voltarem a
somar 1 acontece a jusante, no pipeline de blending (ver
``edgefinder.models.ensemble.blend_probs``). Calibração multiclasse conjunta
(ex.: Dirichlet calibration) fica fora de escopo por custar complexidade sem
ganho prático nas amostras que temos.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# Limite de clipping antes de logaritmos/logits: 1e-12 mantém |log(p)| <= ~27.6
# e |logit(p)| <= ~27.6, longe de overflow e sem distorcer probabilidades reais
# (nenhum modelo honesto produz p < 1e-12 em futebol).
_CLIP_EPS: float = 1e-12


def _validate_binary(
    y_true: npt.ArrayLike, y_prob: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Valida e converte o par (rótulos binários, probabilidades) para float64.

    Centralizado aqui para que toda métrica/ajuste falhe cedo e com a mesma
    mensagem diante de entradas degeneradas, em vez de propagar NaN silencioso.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    if y.shape != p.shape:
        raise ValueError(f"y_true e y_prob com tamanhos diferentes: {y.size} != {p.size}")
    if y.size == 0:
        raise ValueError("entradas vazias: nada a calibrar")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("y_true deve conter apenas 0 e 1")
    if np.any(np.isnan(p)) or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("y_prob deve estar em [0, 1] e sem NaN")
    return y, p


def reliability_table(
    y_true: npt.ArrayLike,
    y_prob: npt.ArrayLike,
    n_bins: int = 10,
    strategy: Literal["quantile", "uniform"] = "quantile",
) -> pd.DataFrame:
    """Tabela de confiabilidade (curva de calibração) por bins de probabilidade.

    Para cada bin ``b``: ``prob_mean`` é a confiança média prevista e
    ``freq_observed`` é a frequência empírica de y=1. Modelo calibrado implica
    ``freq_observed ~= prob_mean`` em todo bin (a menos de ruído binomial
    ``sqrt(prob_mean * (1 - prob_mean) / n)``).

    O padrão é ``strategy='quantile'`` (bins equipopulados): probabilidades de
    futebol se concentram em faixas estreitas (ex.: empate quase sempre em
    0.20-0.35), e bins uniformes deixariam a maioria dos bins vazia ou com
    meia dúzia de pontos, tornando ``freq_observed`` puro ruído. Bins com
    bordas duplicadas (previsões constantes) são fundidos; bins vazios são
    omitidos da tabela.

    Retorna DataFrame com colunas [bin_low, bin_high, prob_mean,
    freq_observed, n]; cada bin cobre [bin_low, bin_high) e o último é
    fechado à direita.
    """
    y, p = _validate_binary(y_true, y_prob)
    if n_bins < 1:
        raise ValueError(f"n_bins deve ser >= 1, recebido {n_bins}")

    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
        if edges.size == 1:
            edges = np.array([edges[0], edges[0]])
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(f"strategy desconhecida: {strategy!r}")

    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, edges.size - 2)

    bin_low: list[float] = []
    bin_high: list[float] = []
    prob_mean: list[float] = []
    freq_observed: list[float] = []
    n_por_bin: list[int] = []
    for i in range(edges.size - 1):
        mask = idx == i
        n_i = int(mask.sum())
        if n_i == 0:
            continue
        bin_low.append(float(edges[i]))
        bin_high.append(float(edges[i + 1]))
        prob_mean.append(float(p[mask].mean()))
        freq_observed.append(float(y[mask].mean()))
        n_por_bin.append(n_i)

    return pd.DataFrame(
        {
            "bin_low": np.asarray(bin_low, dtype=np.float64),
            "bin_high": np.asarray(bin_high, dtype=np.float64),
            "prob_mean": np.asarray(prob_mean, dtype=np.float64),
            "freq_observed": np.asarray(freq_observed, dtype=np.float64),
            "n": np.asarray(n_por_bin, dtype=np.int64),
        }
    )


def brier_score(y_true: npt.ArrayLike, y_prob: npt.ArrayLike) -> float:
    """Brier score: BS = (1/N) * sum_i (p_i - y_i)^2.

    Regra de pontuação estritamente própria: é minimizada em esperança por
    ``p_i = P(y_i = 1)``, logo não recompensa exagerar nem suavizar confiança.
    Menor é melhor; o chute constante 0.5 vale 0.25.
    """
    y, p = _validate_binary(y_true, y_prob)
    return float(np.mean((p - y) ** 2))


def log_loss_safe(y_true: npt.ArrayLike, y_prob: npt.ArrayLike, eps: float = 1e-12) -> float:
    """Log-loss com clipping: LL = -(1/N) * sum_i [y_i*ln(p~_i) + (1-y_i)*ln(1-p~_i)].

    onde ``p~ = clip(p, eps, 1-eps)``. Sem o clipping, um único ``p=0`` com
    ``y=1`` devolve infinito e inutiliza a métrica; com ``eps=1e-12`` esse
    erro extremo custa ``-ln(1e-12) ~= 27.6`` — punição enorme, porém finita
    e comparável entre modelos. O clipping enviesa a métrica em no máximo
    ``O(eps)`` para probabilidades interiores, desprezível.
    """
    if eps <= 0.0 or eps >= 0.5:
        raise ValueError(f"eps deve estar em (0, 0.5), recebido {eps}")
    y, p = _validate_binary(y_true, y_prob)
    p_clip = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p_clip) + (1.0 - y) * np.log(1.0 - p_clip)))


@dataclass(frozen=True)
class IsotonicCalibrator:
    """Calibrador isotônico ajustado; aplique com ``.transform(p)``."""

    model: IsotonicRegression

    def transform(self, p: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Mapeia probabilidades cruas para calibradas (não decrescente em p)."""
        arr = np.asarray(p, dtype=np.float64).ravel()
        return np.asarray(self.model.transform(arr), dtype=np.float64)


def fit_isotonic(y_true: npt.ArrayLike, y_prob: npt.ArrayLike) -> IsotonicCalibrator:
    """Ajusta regressão isotônica p_cal = f(p), com f não decrescente.

    A isotônica só assume monotonicidade — se o modelo diz "mais provável",
    a realidade concorda em média —, então corrige qualquer distorção
    monótona (não apenas as sigmoides que o Platt cobre). O preço é precisar
    de mais dados: com poucas centenas de pontos ela sobreajusta degraus.
    ``out_of_bounds='clip'`` garante previsões válidas fora do intervalo
    visto no ajuste; ``y_min/y_max`` prendem a saída a [0, 1].
    """
    y, p = _validate_binary(y_true, y_prob)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(p, y)
    return IsotonicCalibrator(model=iso)


@dataclass(frozen=True)
class PlattCalibrator:
    """Calibrador de Platt ajustado: p_cal = sigma(a * logit(p) + b)."""

    a: float
    b: float

    def transform(self, p: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Aplica p_cal = sigma(a * logit(p~) + b), com p~ = clip(p, eps, 1-eps)."""
        arr = np.asarray(p, dtype=np.float64).ravel()
        z = np.asarray(logit(np.clip(arr, _CLIP_EPS, 1.0 - _CLIP_EPS)), dtype=np.float64)
        return np.asarray(expit(self.a * z + self.b), dtype=np.float64)


def fit_platt(y_true: npt.ArrayLike, y_prob: npt.ArrayLike) -> PlattCalibrator:
    """Ajusta Platt scaling: p_cal = sigma(a * logit(p) + b), sigma(x) = 1/(1+e^-x).

    Regressão logística do rótulo sobre o logit da probabilidade crua:
    ``a`` corrige excesso/falta de confiança global (a < 1 encolhe, a > 1
    amplia a dispersão dos logits) e ``b`` corrige viés de base. Dois
    parâmetros apenas, então funciona com poucos dados e nunca degrada muito —
    mas só conserta distorções que sejam afins no espaço de logit. Usa
    ``C=1e6`` (regularização desprezível) porque o Platt clássico é
    máxima verossimilhança pura sobre 2 parâmetros: regularizar aqui
    encolheria ``a`` em direção a 0 e reintroduziria viés de calibração.
    """
    y, p = _validate_binary(y_true, y_prob)
    if np.unique(y).size < 2:
        raise ValueError("fit_platt requer as duas classes (0 e 1) presentes em y_true")
    z = np.asarray(logit(np.clip(p, _CLIP_EPS, 1.0 - _CLIP_EPS)), dtype=np.float64)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(z.reshape(-1, 1), y.astype(np.int64))
    return PlattCalibrator(a=float(lr.coef_[0, 0]), b=float(lr.intercept_[0]))


def expected_calibration_error(
    y_true: npt.ArrayLike, y_prob: npt.ArrayLike, n_bins: int = 10
) -> float:
    """ECE: soma ponderada dos desvios de calibração por bin.

    ECE = sum_b (n_b / N) * |freq_b - conf_b|

    onde ``conf_b`` é a probabilidade média prevista no bin e ``freq_b`` a
    frequência observada de y=1. Usa bins uniformes em [0, 1] (definição
    padrão, Naeini et al. 2015), o que torna o valor comparável entre
    modelos e datasets; bins vazios têm peso 0 por construção. Resume a
    reliability table em um escalar: 0 = calibração perfeita, e o valor é
    interpretável como "erro médio de probabilidade" (ECE 0.05 = as
    probabilidades erram ~5 p.p. em média).
    """
    table = reliability_table(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    n_total = float(table["n"].sum())
    gaps = (table["freq_observed"] - table["prob_mean"]).abs()
    return float(((table["n"] / n_total) * gaps).sum())
