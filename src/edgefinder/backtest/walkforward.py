"""Motor genérico de backtest walk-forward, agnóstico ao modelo.

O motor não sabe nada de Dixon-Coles, Poisson ou qualquer modelo concreto:
recebe um par ``fit_fn``/``predict_fn`` e garante apenas o protocolo
temporal honesto. As duas garantias anti-leakage são estruturais:

1. o frame de treino contém somente jogos com ``match_date`` ESTRITAMENTE
   anterior ao início do bloco previsto (t_treino < t_bloco, nunca <=);
2. o frame de predição chega a ``predict_fn`` SEM colunas de resultado
   (``home_goals``/``away_goals`` e qualquer coluna extra listada em
   ``result_cols``) — um modelo mal escrito não consegue sequer ler o
   futuro, porque as colunas não existem.

Ambas são cobertas por um teste que envenena todos os resultados futuros
e exige previsões bit a bit idênticas (tests/test_walkforward.py).
"""

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

DEFAULT_RESULT_COLS: tuple[str, ...] = ("home_goals", "away_goals")


@dataclass(frozen=True)
class StepResult:
    """Um passo (bloco temporal) do walk-forward, guardado para auditoria.

    ``period_start``/``period_end`` delimitam o bloco previsto; ``n_train``
    é o tamanho do frame efetivamente passado a ``fit_fn``; ``predictions``
    é a saída crua de ``predict_fn`` para o bloco.
    """

    period_start: pd.Timestamp
    period_end: pd.Timestamp
    n_train: int
    predictions: pd.DataFrame


@dataclass(frozen=True)
class WalkForwardResult:
    """Resultado completo do walk-forward: passos individuais + concatenação.

    ``predictions`` é a união (``concat``) das previsões de todos os passos,
    na ordem cronológica dos blocos.
    """

    steps: list[StepResult]
    predictions: pd.DataFrame


def walk_forward_detailed[M](
    matches: pd.DataFrame,
    fit_fn: Callable[[pd.DataFrame], M],
    predict_fn: Callable[[M, pd.DataFrame], pd.DataFrame],
    freq: str = "W",
    min_train: int = 380,
    expanding: bool = True,
    result_cols: list[str] | None = None,
) -> WalkForwardResult:
    """Executa o walk-forward e devolve também os passos individuais.

    Os jogos são ordenados por ``match_date`` e agrupados em blocos pelo
    período de ``freq`` (ex.: ``"W"`` = semanas). Para cada bloco com início
    em t: treina-se em {jogos com match_date < t} e prevê-se o bloco inteiro.

    - ``expanding=True``: janela expansiva (todo o passado disponível).
    - ``expanding=False``: janela deslizante com os últimos ``min_train``
      jogos — útil para medir se o passado remoto ajuda ou atrapalha.
    - Blocos cujo passado tem menos de ``min_train`` jogos são pulados
      (período de aquecimento), porque um modelo mal alimentado só gera
      ruído no início da série.

    O frame de treino é uma cópia (mutações dentro de ``fit_fn`` não vazam
    para os blocos seguintes) e o frame de predição perde as colunas de
    resultado antes de chegar a ``predict_fn``.
    """
    if "match_date" not in matches.columns:
        raise ValueError("matches precisa de uma coluna 'match_date'")
    if min_train < 1:
        raise ValueError(f"min_train deve ser >= 1, recebido {min_train}")

    df = matches.copy()
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date", kind="stable").reset_index(drop=True)

    drop_cols = set(DEFAULT_RESULT_COLS)
    if result_cols is not None:
        drop_cols.update(result_cols)

    periods = df["match_date"].dt.to_period(freq)
    steps: list[StepResult] = []
    frames: list[pd.DataFrame] = []

    for period in periods.unique():
        start = period.start_time
        train = df[df["match_date"] < start]
        if len(train) < min_train:
            continue
        if not expanding:
            train = train.iloc[-min_train:]

        block = df.loc[periods == period]
        features = block.drop(columns=[c for c in block.columns if c in drop_cols])

        model = fit_fn(train.copy())
        preds = predict_fn(model, features)

        steps.append(
            StepResult(
                period_start=start,
                period_end=period.end_time,
                n_train=len(train),
                predictions=preds,
            )
        )
        if not preds.empty:
            frames.append(preds)

    predictions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return WalkForwardResult(steps=steps, predictions=predictions)


def walk_forward[M](
    matches: pd.DataFrame,
    fit_fn: Callable[[pd.DataFrame], M],
    predict_fn: Callable[[M, pd.DataFrame], pd.DataFrame],
    freq: str = "W",
    min_train: int = 380,
    expanding: bool = True,
    result_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Walk-forward que devolve apenas as previsões concatenadas.

    Atalho sobre :func:`walk_forward_detailed` para quem não precisa dos
    passos individuais. ``predict_fn`` deve preservar a identificação do
    jogo (``match_id`` ou índice) nas linhas que devolve.
    """
    result = walk_forward_detailed(
        matches,
        fit_fn,
        predict_fn,
        freq=freq,
        min_train=min_train,
        expanding=expanding,
        result_cols=result_cols,
    )
    return result.predictions
