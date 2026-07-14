"""Simulação Monte Carlo de trajetórias de banca sob staking proporcional.

Por que Monte Carlo: o backtest entrega UMA realização do processo; a
distribuição de trajetórias diz o que a variância pode fazer com a mesma
estratégia — em particular a probabilidade de ruína prática e o drawdown
esperado, que nenhuma média pontual revela.

Modelo: cada aposta t arrisca uma fração f_t da banca CORRENTE (estilo
Kelly), logo a banca evolui multiplicativamente:

    B_t = B_{t-1} * m_t,   m_t = 1 + f_t * (o_t - 1)  com prob. p_t
                           m_t = 1 - f_t              caso contrário

e portanto B_T = B_0 * prod_{t=1..T} m_t. Essa forma produto permite
vetorizar toda a simulação num único ``cumprod`` sobre uma matriz
(n_paths x n_bets), sem loop por trajetória.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


def simulate_bankroll(
    p_win: npt.NDArray[np.float64] | Sequence[float],
    odds: npt.NDArray[np.float64] | Sequence[float],
    stakes_frac: npt.NDArray[np.float64] | Sequence[float],
    n_paths: int = 10_000,
    initial: float = 1.0,
    seed: int | None = None,
    ruin_threshold: float = 0.10,
) -> dict[str, float]:
    """Simula ``n_paths`` trajetórias de banca para uma sequência de apostas.

    Cada aposta é um Bernoulli independente com probabilidade ``p_win[t]``,
    odds decimais ``odds[t]`` e fração de banca ``stakes_frac[t]``.

    Retorna um dicionário com:

    - ``final_q05`` .. ``final_q95``: quantis 5/25/50/75/95 da banca final;
    - ``final_mean``: média da banca final;
    - ``prob_ruin``: fração de trajetórias em que a banca cai abaixo de
      ``ruin_threshold * initial`` em QUALQUER ponto (ruína prática — na
      staking proporcional a banca nunca zera exatamente, mas abaixo de
      10% a estratégia está operacionalmente morta);
    - ``expected_max_drawdown``: média, entre trajetórias, do máximo
      drawdown relativo max_t (1 - B_t / max_{s<=t} B_s), incluindo o
      ponto inicial como pico de partida;
    - ``n_bets`` e ``n_paths`` para contexto.

    Levanta ``ValueError`` para entradas inconsistentes (tamanhos
    diferentes, p fora de [0, 1], odds <= 1, frações fora de [0, 1]).
    """
    p = np.asarray(p_win, dtype=np.float64)
    o = np.asarray(odds, dtype=np.float64)
    f = np.asarray(stakes_frac, dtype=np.float64)

    if not (p.shape == o.shape == f.shape) or p.ndim != 1:
        raise ValueError(
            f"p_win, odds e stakes_frac devem ser vetores 1-D do mesmo tamanho "
            f"(shapes: {p.shape}, {o.shape}, {f.shape})"
        )
    if n_paths < 1:
        raise ValueError(f"n_paths deve ser >= 1, recebido {n_paths}")
    if initial <= 0.0:
        raise ValueError(f"initial deve ser > 0, recebido {initial}")
    if not 0.0 <= ruin_threshold < 1.0:
        raise ValueError(f"ruin_threshold deve estar em [0, 1), recebido {ruin_threshold}")
    if bool(np.any((p < 0.0) | (p > 1.0))):
        raise ValueError("p_win deve estar em [0, 1]")
    if bool(np.any(o <= 1.0)):
        raise ValueError("odds decimais devem ser > 1.0")
    if bool(np.any((f < 0.0) | (f > 1.0))):
        raise ValueError("stakes_frac deve estar em [0, 1]")

    n_bets = int(p.size)
    if n_bets == 0:
        return {
            "final_q05": initial,
            "final_q25": initial,
            "final_q50": initial,
            "final_q75": initial,
            "final_q95": initial,
            "final_mean": initial,
            "prob_ruin": 0.0,
            "expected_max_drawdown": 0.0,
            "n_bets": 0.0,
            "n_paths": float(n_paths),
        }

    rng = np.random.default_rng(seed)
    wins = rng.random((n_paths, n_bets)) < p
    multipliers = np.where(wins, 1.0 + f * (o - 1.0), 1.0 - f)
    paths = initial * np.cumprod(multipliers, axis=1)

    finals = paths[:, -1]
    q05, q25, q50, q75, q95 = np.quantile(finals, [0.05, 0.25, 0.50, 0.75, 0.95])

    prob_ruin = float(np.mean(paths.min(axis=1) < ruin_threshold * initial))

    full = np.concatenate([np.full((n_paths, 1), initial), paths], axis=1)
    running_max = np.maximum.accumulate(full, axis=1)
    drawdowns = 1.0 - full / running_max
    expected_mdd = float(np.mean(drawdowns.max(axis=1)))

    return {
        "final_q05": float(q05),
        "final_q25": float(q25),
        "final_q50": float(q50),
        "final_q75": float(q75),
        "final_q95": float(q95),
        "final_mean": float(finals.mean()),
        "prob_ruin": prob_ruin,
        "expected_max_drawdown": expected_mdd,
        "n_bets": float(n_bets),
        "n_paths": float(n_paths),
    }
