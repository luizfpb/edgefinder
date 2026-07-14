"""Métricas de desempenho sobre um DataFrame de apostas simuladas.

Contrato de colunas: ``stake`` (tamanho da aposta), ``odds`` decimais
(alias aceito: ``bet_odds``) e ``result`` em {"win", "lose", "push"}.
Colunas opcionais aproveitadas por :func:`summary`: ``ev``/``bet_ev``,
``clv`` e ``closing_fair`` (probabilidade justa de fechamento).

O P&L por aposta segue a convenção de odds decimais:

    pnl = stake * (odds - 1)   se result == "win"
    pnl = 0                    se result == "push"
    pnl = -stake               se result == "lose"

As fórmulas de Brier e log-loss são reimplementadas localmente de
propósito: o módulo de backtest não deve depender do código de modelagem
que ele próprio está avaliando.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

_ODDS_ALIASES: tuple[str, ...] = ("odds", "bet_odds")
_EPS = 1e-15


def _odds_values(df: pd.DataFrame) -> npt.NDArray[np.float64]:
    for col in _ODDS_ALIASES:
        if col in df.columns:
            return df[col].to_numpy(dtype=np.float64)
    raise KeyError(f"DataFrame de apostas precisa de uma coluna de odds ({_ODDS_ALIASES})")


def _pnl(df: pd.DataFrame) -> npt.NDArray[np.float64]:
    """P&L por aposta a partir de stake, odds e result (win/lose/push)."""
    stake = df["stake"].to_numpy(dtype=np.float64)
    odds = _odds_values(df)
    result = df["result"].to_numpy()
    pnl = np.where(
        result == "win",
        stake * (odds - 1.0),
        np.where(result == "push", 0.0, -stake),
    )
    return pnl.astype(np.float64)


def roi(df: pd.DataFrame) -> float:
    """Retorno sobre o volume apostado: ROI = sum(pnl) / sum(stake).

    Pushes contam no denominador (o capital ficou em risco até a
    liquidação). DataFrame vazio ou sem stake devolve 0.0.
    """
    if df.empty:
        return 0.0
    total_stake = float(df["stake"].to_numpy(dtype=np.float64).sum())
    if total_stake <= 0.0:
        return 0.0
    return float(_pnl(df).sum() / total_stake)


def yield_per_bet(df: pd.DataFrame) -> float:
    """Yield médio por aposta: mean(pnl_i / stake_i) sobre stakes positivos.

    Difere do ROI quando os stakes variam (Kelly): aqui cada aposta pesa
    igual, o que mede a qualidade média da seleção, não do sizing.
    """
    if df.empty:
        return 0.0
    stake = df["stake"].to_numpy(dtype=np.float64)
    mask = stake > 0.0
    if not bool(mask.any()):
        return 0.0
    return float(np.mean(_pnl(df)[mask] / stake[mask]))


def max_drawdown(equity: npt.NDArray[np.float64] | Sequence[float]) -> float:
    """Máximo drawdown relativo de uma curva de patrimônio (equity > 0).

    MDD = max_t (1 - E_t / max_{s<=t} E_s)

    ou seja, a maior queda percentual em relação ao pico anterior. Série
    vazia ou monotonicamente crescente devolve 0.0.
    """
    arr = np.asarray(equity, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(arr)
    return float(np.max(1.0 - arr / peaks))


def sharpe_ratio(returns: npt.NDArray[np.float64] | Sequence[float]) -> float:
    """Sharpe POR APOSTA, deliberadamente sem anualização.

    sharpe = mean(r) / std(r, ddof=1)

    Apostas não têm frequência temporal fixa (um dia pode ter 0 ou 10
    jogos), então anualizar por sqrt(N períodos) seria inventar uma escala
    de tempo — o número reportado é adimensional, na unidade "por aposta".
    Menos de 2 retornos ou variância zero devolve 0.0 (não há dispersão
    para normalizar).
    """
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    sd = float(arr.std(ddof=1))
    if sd == 0.0:
        return 0.0
    return float(arr.mean() / sd)


def hit_rate(df: pd.DataFrame) -> float:
    """Fração de apostas ganhas entre as decididas: wins / (wins + losses).

    Pushes ficam fora do denominador (não foram decididas). Sem apostas
    decididas devolve 0.0.
    """
    if df.empty:
        return 0.0
    result = df["result"].to_numpy()
    wins = int(np.sum(result == "win"))
    losses = int(np.sum(result == "lose"))
    if wins + losses == 0:
        return 0.0
    return float(wins / (wins + losses))


def prob_metrics(
    y_true: npt.NDArray[np.float64] | Sequence[float],
    y_prob: npt.NDArray[np.float64] | Sequence[float],
) -> dict[str, float]:
    """Brier e log-loss binários de probabilidades previstas.

    brier    = mean((p_i - y_i)^2)
    log_loss = -mean(y_i * ln(p_i) + (1 - y_i) * ln(1 - p_i))

    com p recortado em [1e-15, 1 - 1e-15] apenas dentro do logaritmo, para
    evitar ln(0). Brier é próprio e limitado em [0, 1]; log-loss pune
    excesso de confiança de forma ilimitada — reportamos ambos porque um
    modelo pode ganhar em um e perder no outro.
    """
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)
    if y.shape != p.shape:
        raise ValueError(f"shapes incompatíveis: y_true {y.shape} vs y_prob {p.shape}")
    if y.size == 0:
        raise ValueError("y_true vazio")
    if not bool(np.isin(y, (0.0, 1.0)).all()):
        raise ValueError("y_true deve conter apenas 0 e 1")
    if bool(np.any((p < 0.0) | (p > 1.0))):
        raise ValueError("y_prob deve estar em [0, 1]")
    brier = float(np.mean((p - y) ** 2))
    p_clip = np.clip(p, _EPS, 1.0 - _EPS)
    log_loss = float(-np.mean(y * np.log(p_clip) + (1.0 - y) * np.log(1.0 - p_clip)))
    return {"brier": brier, "log_loss": log_loss, "n": float(y.size)}


def significance(
    yields: npt.NDArray[np.float64] | Sequence[float],
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """Significância estatística do yield médio de uma amostra de apostas.

    Testa H0: yield médio <= 0 de duas formas complementares:

    - teste t unilateral de uma amostra: t = m / (s / sqrt(n)),
      p_value = P(T_{n-1} >= t);
    - bootstrap percentil com ``n_boot`` reamostragens: IC95 da média
      (percentis 2.5 e 97.5) e p-valor bootstrap sob H0 (a amostra é
      recentrada em zero e conta-se a fração de médias reamostradas que
      atinge a média observada, com suavização +1 para nunca reportar 0).

    O papel desta função é dizer quando um backtest bonito é
    indistinguível de sorte. Exemplo concreto: +2% de yield em 200
    apostas a odds ~2 tem desvio-padrão por aposta ~1, logo erro padrão
    da média ~1/sqrt(200) ~ 0.071; o t fica em ~0.28 e p ~ 0.39. Esse
    resultado NÃO é evidência de edge, e é exatamente isso que o p-valor
    devolvido vai dizer. Casos degenerados (n < 2 ou variância zero)
    devolvem p = 1.0 por convenção conservadora: sem dispersão observável
    não há como quantificar evidência.
    """
    arr = np.asarray(yields, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "p_value": 1.0,
            "p_value_bootstrap": 1.0,
        }
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    if n < 2 or std == 0.0:
        return {
            "n": float(n),
            "mean": mean,
            "std": std,
            "ci95_low": mean,
            "ci95_high": mean,
            "p_value": 1.0,
            "p_value_bootstrap": 1.0,
        }

    t_res = stats.ttest_1samp(arr, popmean=0.0, alternative="greater")
    p_value = float(t_res.pvalue)

    rng = np.random.default_rng(seed)
    boot_means = rng.choice(arr, size=(n_boot, n), replace=True).mean(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    # Reamostrar (x_i - m) equivale a deslocar cada média bootstrap em -m,
    # então o p-valor sob H0 conta boot_means - m >= m.
    p_boot = float((int(np.sum(boot_means - mean >= mean)) + 1) / (n_boot + 1))

    return {
        "n": float(n),
        "mean": mean,
        "std": std,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_value": p_value,
        "p_value_bootstrap": p_boot,
    }


def summary(df: pd.DataFrame) -> dict[str, float]:
    """Painel consolidado das métricas de aposta de um backtest.

    A curva de patrimônio para o drawdown assume banca inicial de 1.0 e
    stakes já expressos como fração dela (aposta flat sobre a banca
    inicial): E_t = 1 + soma acumulada do pnl, na ordem das linhas do
    DataFrame (que deve estar em ordem cronológica).

    Inclui ``avg_ev`` se houver coluna ``ev``/``bet_ev`` e ``avg_clv`` se
    houver ``clv`` ou ``closing_fair``. Nesse último caso o CLV é derivado:

        clv = odds * p_fair_close - 1

    isto é, o valor esperado da aposta reavaliado pela probabilidade justa
    do fechamento — CLV positivo sistemático é o melhor preditor de edge
    real que existe fora do resultado.
    """
    if df.empty:
        return {
            "n_bets": 0.0,
            "roi": 0.0,
            "yield_per_bet": 0.0,
            "hit_rate": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "total_pnl": 0.0,
        }
    pnl = _pnl(df)
    stake = df["stake"].to_numpy(dtype=np.float64)
    mask = stake > 0.0
    returns = pnl[mask] / stake[mask]
    equity = np.concatenate(([1.0], 1.0 + np.cumsum(pnl)))
    out: dict[str, float] = {
        "n_bets": float(len(df)),
        "roi": roi(df),
        "yield_per_bet": yield_per_bet(df),
        "hit_rate": hit_rate(df),
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(equity),
        "total_pnl": float(pnl.sum()),
    }
    for col in ("ev", "bet_ev"):
        if col in df.columns:
            out["avg_ev"] = float(df[col].to_numpy(dtype=np.float64).mean())
            break
    if "clv" in df.columns:
        out["avg_clv"] = float(df["clv"].to_numpy(dtype=np.float64).mean())
    elif "closing_fair" in df.columns:
        closing = df["closing_fair"].to_numpy(dtype=np.float64)
        out["avg_clv"] = float(np.mean(_odds_values(df) * closing - 1.0))
    return out
