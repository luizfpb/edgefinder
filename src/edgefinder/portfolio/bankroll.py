"""Gestão de banca: estado imutável e curva de patrimônio.

O estado da banca é tratado como valor imutável: cada resultado de aposta
gera um NOVO estado, preservando o histórico completo de saldos. Isso evita
bugs de mutação compartilhada em backtests que ramificam cenários a partir
de um mesmo estado inicial, e torna a trajetória da banca auditável.

Convenção de odds: decimais europeias (odds >= 1). O lucro e a perda por
aposta seguem:

    pnl = stake * (odds - 1)   se result == "win"
    pnl = -stake               se result == "lose"
    pnl = 0                    se result == "push" (stake devolvido)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

ResultadoAposta = Literal["win", "lose", "push"]

RESULTADOS_VALIDOS: frozenset[str] = frozenset({"win", "lose", "push"})


@dataclass(frozen=True)
class BankrollState:
    """Fotografia imutável da banca em um instante.

    Atributos:
        current: saldo atual.
        initial: saldo com que a banca começou (referência para medir retorno).
        history: saldos após cada aposta liquidada, em ordem cronológica.

    A classe é congelada (frozen) porque o estado da banca é um valor, não um
    objeto com identidade: transições acontecem via ``apply_bet_result``, que
    devolve uma nova instância.
    """

    current: float
    initial: float
    history: list[float] = field(default_factory=list)


def _pnl_unitario(stake: float, odds: float, result: str) -> float:
    """Calcula o P&L de uma única aposta, validando o rótulo do resultado."""
    if result == "win":
        return stake * (odds - 1.0)
    if result == "lose":
        return -stake
    if result == "push":
        return 0.0
    raise ValueError(f"resultado invalido: {result!r} (esperado 'win', 'lose' ou 'push')")


def apply_bet_result(
    state: BankrollState,
    stake: float,
    odds: float,
    result: ResultadoAposta,
) -> BankrollState:
    """Aplica o resultado de uma aposta e devolve um NOVO estado da banca.

    Fórmula da transição de saldo:

        current' = current + pnl
        pnl = stake * (odds - 1)  se win; -stake se lose; 0 se push

    O estado original não é modificado (imutabilidade): o histórico é copiado
    e recebe o novo saldo ao final. Isso permite que o chamador ramifique
    cenários de simulação sem cópias defensivas.
    """
    if stake < 0:
        raise ValueError(f"stake deve ser nao negativo, recebido {stake}")
    if odds < 1.0:
        raise ValueError(f"odds decimais devem ser >= 1, recebido {odds}")
    novo_saldo = state.current + _pnl_unitario(stake, odds, result)
    return BankrollState(
        current=novo_saldo,
        initial=state.initial,
        history=[*state.history, novo_saldo],
    )


def equity_curve(bets_df: pd.DataFrame, initial: float = 0.0) -> pd.Series[float]:
    """Curva de patrimônio acumulada após cada aposta liquidada.

    Para a linha i do DataFrame (colunas ``stake``, ``odds``, ``result``):

        pnl_i = stake_i * (odds_i - 1)   se result_i == "win"
        pnl_i = -stake_i                 se result_i == "lose"
        pnl_i = 0                        se result_i == "push"

        equity_i = initial + soma_{j <= i} pnl_j

    O parâmetro ``initial`` ancora a curva no saldo inicial da banca. Isso é
    necessário quando a curva alimenta o ``drawdown_guard``, que mede quedas
    percentuais em relação ao pico e portanto exige valores absolutos de
    patrimônio, não apenas o P&L acumulado.
    """
    obrigatorias = {"stake", "odds", "result"}
    faltantes = obrigatorias - set(bets_df.columns)
    if faltantes:
        raise ValueError(f"colunas ausentes em bets_df: {sorted(faltantes)}")
    if bets_df.empty:
        return pd.Series(dtype=float, name="equity")

    invalidos = set(bets_df["result"].astype(str)) - RESULTADOS_VALIDOS
    if invalidos:
        raise ValueError(f"resultados invalidos em bets_df: {sorted(invalidos)}")

    stake = bets_df["stake"].to_numpy(dtype=float)
    odds = bets_df["odds"].to_numpy(dtype=float)
    result = bets_df["result"].to_numpy(dtype=str)
    pnl = np.select(
        [result == "win", result == "lose"],
        [stake * (odds - 1.0), -stake],
        default=0.0,
    )
    return pd.Series(initial + np.cumsum(pnl), index=bets_df.index, name="equity")
