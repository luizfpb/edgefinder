"""Decaimento exponencial por recência para agregação de histórico.

A forma de um time ou jogador não é estacionária: elenco, técnico e esquema
mudam, então um jogo de dois anos atrás informa menos que um da semana
passada. Em vez de uma janela dura (tudo ou nada), usa-se decaimento
exponencial contínuo, que interpola suavemente entre "usar todo o histórico"
e "usar só o recente":

    w_i = exp(-lambda * dt_i),   lambda = ln(2) / h

onde dt_i >= 0 é a distância em dias entre o jogo i e a data de referência e
h é a meia-vida em dias — um jogo a exatamente h dias de distância pesa 0.5.
A meia-vida é preferida à taxa lambda crua por ser diretamente interpretável.

Todas as funções são puras (Series/arrays já em memória, sem I/O) e
protegidas contra vazamento temporal: uma data posterior à referência nunca
recebe peso — a chamada levanta ValueError em vez de extrapolar w > 1.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import numpy.typing as npt
import pandas as pd

_NS_PER_DAY = 86_400_000_000_000


def exponential_weights(
    dates: pd.Series[pd.Timestamp],
    ref_date: datetime | pd.Timestamp,
    half_life_days: float,
) -> npt.NDArray[np.float64]:
    """Pesos de recência com decaimento exponencial e meia-vida em dias.

    Fórmula: w_i = exp(-lambda * dt_i), com lambda = ln(2) / half_life_days
    e dt_i = (ref_date - data_i) em dias (frações de dia contam). Como
    dt_i >= 0, os pesos ficam sempre em (0, 1], com w = 1 exatamente na
    data de referência e w = 0.5 a uma meia-vida de distância.

    Anti-leakage: qualquer data estritamente posterior a ``ref_date``
    levanta ValueError. Um dt negativo daria peso > 1 a informação do
    futuro, o pior tipo de vazamento em backtest — melhor falhar alto do
    que produzir uma taxa contaminada.

    Parâmetros
    ----------
    dates:
        Datas dos jogos (qualquer coisa que ``pd.to_datetime`` aceite).
        NaT levanta ValueError: peso indefinido não tem interpretação.
    ref_date:
        Instante da previsão; só o passado (ou o exato presente) pesa.
    half_life_days:
        Meia-vida h > 0 em dias; deve ser finita.

    Retorno
    -------
    Array float64 com um peso por linha de ``dates``, na mesma ordem.
    """
    if not math.isfinite(half_life_days) or half_life_days <= 0:
        raise ValueError(f"half_life_days deve ser finito e > 0, recebido {half_life_days}")
    ref = pd.Timestamp(ref_date)
    parsed = pd.to_datetime(dates)
    if bool(parsed.isna().any()):
        raise ValueError("dates contém valores ausentes (NaT)")
    delta_ns = (ref.to_datetime64() - parsed.to_numpy()).astype("timedelta64[ns]")
    dt_days = delta_ns.astype(np.int64).astype(np.float64) / _NS_PER_DAY
    if dt_days.size > 0 and float(dt_days.min()) < 0:
        raise ValueError("dates contém data posterior a ref_date (vazamento temporal)")
    lam = math.log(2.0) / half_life_days
    return np.asarray(np.exp(-lam * dt_days), dtype=np.float64)


def weighted_rate(
    values: npt.NDArray[np.float64],
    exposures: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> float:
    """Taxa por 90 minutos ponderada por recência.

    Fórmula: rate = sum_i(w_i * x_i) / sum_i(w_i * e_i), onde x_i é a
    contagem observada no jogo i (gols, chutes, escanteios...) e e_i é a
    exposição em unidades de 90 minutos (ex.: 63 minutos => e = 0.7).

    Ponderar numerador e denominador pelo MESMO w_i equivale a estimar a
    taxa de um processo de Poisson com intensidade que varia no tempo,
    dando mais crédito à evidência recente sem descartar a antiga. Com
    todos os w_i iguais, reduz-se à taxa bruta sum(x) / sum(e); o resultado
    também é invariante à escala dos pesos (c * w cancela na razão).

    Exposição efetiva nula (sum(w * e) = 0, incluindo arrays vazios)
    devolve NaN: não existe taxa observável sem tempo de exposição, e NaN
    propaga honestamente essa ausência de evidência em vez de inventar 0.

    Pesos ou exposições negativos, e qualquer valor não finito, levantam
    ValueError: são sempre sintoma de bug a montante, nunca dado válido.
    """
    x = np.asarray(values, dtype=np.float64)
    e = np.asarray(exposures, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if not (x.shape == e.shape == w.shape):
        raise ValueError(
            f"shapes incompatíveis: values {x.shape}, exposures {e.shape}, weights {w.shape}"
        )
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(e)) and np.all(np.isfinite(w))):
        raise ValueError("values, exposures e weights devem ser todos finitos")
    if bool(np.any(e < 0)) or bool(np.any(w < 0)):
        raise ValueError("exposures e weights não podem ser negativos")
    denom = float(np.sum(w * e))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(w * x) / denom)


def half_life_grid() -> list[float]:
    """Grade padrão de meias-vidas (dias) para otimização no backtest.

    A meia-vida NÃO é um valor chutado nem fixado a priori: cada candidato
    desta grade é avaliado no backtest por log-loss out-of-sample (validação
    walk-forward, só passado prevê futuro) e vence o que minimiza a perda.
    A grade cobre da memória curta (30 dias, ~1 mês de forma) à memória
    longa (730 dias, ~2 temporadas de nível estrutural), com espaçamento
    aproximadamente geométrico porque o efeito do decaimento é multiplicativo
    — a diferença entre 30 e 60 dias importa muito mais que entre 700 e 730.
    """
    return [30.0, 60.0, 90.0, 120.0, 180.0, 365.0, 730.0]
