"""Controles de risco da carteira de apostas.

A disciplina de risco importa mais que o modelo: um edge real de poucos
pontos percentuais só se materializa ao longo de centenas ou milhares de
apostas, e qualquer sequência ruim pode quebrar a banca antes disso se a
exposição não for limitada. Estas funções são o disjuntor entre o stake
teórico (Kelly) e o dinheiro real: limitam concentração por jogo e por dia
e interrompem a operação quando o drawdown passa do tolerável.

Todos os limites são expressos como FRAÇÃO do bankroll (ex.: 0.05 significa
5% da banca), o que mantém o risco proporcional ao capital conforme a banca
cresce ou encolhe.
"""

from __future__ import annotations

from typing import TypedDict

import pandas as pd

_EPS = 1e-9


class NovaAposta(TypedDict):
    """Aposta candidata submetida ao controle de exposição."""

    match_id: str | int
    stake: float


def check_exposure(
    bets_do_dia: pd.DataFrame,
    nova: NovaAposta,
    bankroll: float,
    max_per_match: float,
    max_per_day: float,
) -> tuple[bool, str]:
    """Verifica se uma nova aposta cabe nos limites de exposição.

    Limites absolutos derivados das frações do bankroll:

        limite_jogo = max_per_match * bankroll
        limite_dia  = max_per_day * bankroll

    A aposta é aprovada se, e somente se:

        exposicao_jogo + stake <= limite_jogo   E
        exposicao_dia + stake <= limite_dia

    onde ``exposicao_jogo`` soma os stakes já comprometidos no mesmo
    ``match_id`` e ``exposicao_dia`` soma todos os stakes do dia. Comparações
    usam tolerância numérica para que uma aposta exatamente no limite não
    seja rejeitada por erro de ponto flutuante.

    Retorna ``(aprovada, motivo)``; o motivo explica a rejeição ou confirma
    a aprovação.
    """
    if bankroll <= 0:
        raise ValueError(f"bankroll deve ser positivo, recebido {bankroll}")
    stake = float(nova["stake"])
    if stake < 0:
        raise ValueError(f"stake deve ser nao negativo, recebido {stake}")

    if bets_do_dia.empty:
        exposicao_jogo = 0.0
        exposicao_dia = 0.0
    else:
        obrigatorias = {"match_id", "stake"}
        faltantes = obrigatorias - set(bets_do_dia.columns)
        if faltantes:
            raise ValueError(f"colunas ausentes em bets_do_dia: {sorted(faltantes)}")
        stakes = bets_do_dia["stake"].astype(float)
        exposicao_dia = float(stakes.sum())
        mesmo_jogo = bets_do_dia["match_id"] == nova["match_id"]
        exposicao_jogo = float(stakes[mesmo_jogo].sum())

    limite_jogo = max_per_match * bankroll
    limite_dia = max_per_day * bankroll
    tol = _EPS * max(1.0, bankroll)

    if exposicao_jogo + stake > limite_jogo + tol:
        return (
            False,
            f"limite por jogo excedido: exposicao {exposicao_jogo + stake:.2f} "
            f"> {limite_jogo:.2f} ({max_per_match:.1%} do bankroll)",
        )
    if exposicao_dia + stake > limite_dia + tol:
        return (
            False,
            f"limite diario excedido: exposicao {exposicao_dia + stake:.2f} "
            f"> {limite_dia:.2f} ({max_per_day:.1%} do bankroll)",
        )
    return (True, "aprovada: dentro dos limites por jogo e por dia")


def drawdown_guard(equity: pd.Series[float], max_dd: float = 0.25) -> bool:
    """Sinaliza PARE de apostar quando o drawdown atual excede o limite.

    Drawdown atual em relação ao pico histórico da curva de patrimônio:

        dd = (max(equity) - equity_final) / max(equity)

    Retorna ``True`` (parar) quando dd >= max_dd. Se o pico for não positivo,
    a banca já está tecnicamente quebrada e a resposta também é parar.

    Por que este guardião existe: a disciplina de risco importa mais que o
    modelo. Nenhuma estimativa de edge sobrevive a uma banca zerada; parar em
    um drawdown pré-definido protege contra os dois modos de falha mais
    comuns, o modelo estar errado (edge ilusório) e a variância normal de uma
    sequência ruim, e força uma reavaliação fria antes de repor capital.
    """
    if max_dd <= 0:
        raise ValueError(f"max_dd deve ser positivo, recebido {max_dd}")
    if equity.empty:
        return False
    valores = equity.astype(float)
    pico = float(valores.max())
    if pico <= 0:
        return True
    atual = float(valores.iloc[-1])
    dd = (pico - atual) / pico
    return dd >= max_dd - _EPS


def stake_com_limites(
    kelly_stake_sugerido: float,
    exposicao_atual_jogo: float,
    exposicao_atual_dia: float,
    bankroll: float,
    max_per_match: float,
    max_per_day: float,
) -> float:
    """Trunca o stake sugerido pelo Kelly à folga restante nos limites.

    Fórmula:

        folga_jogo = max(0, max_per_match * bankroll - exposicao_atual_jogo)
        folga_dia  = max(0, max_per_day * bankroll - exposicao_atual_dia)
        stake      = max(0, min(kelly_stake_sugerido, folga_jogo, folga_dia))

    O truncamento (em vez de rejeitar a aposta) preserva parte do valor
    esperado quando o Kelly pede mais do que os limites permitem: apostar a
    folga disponível ainda captura edge, apenas com menos variância. Um
    Kelly negativo (sem edge) resulta em stake zero.
    """
    if bankroll <= 0:
        raise ValueError(f"bankroll deve ser positivo, recebido {bankroll}")
    folga_jogo = max(0.0, max_per_match * bankroll - exposicao_atual_jogo)
    folga_dia = max(0.0, max_per_day * bankroll - exposicao_atual_dia)
    return max(0.0, min(kelly_stake_sugerido, folga_jogo, folga_dia))
