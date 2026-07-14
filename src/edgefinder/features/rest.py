"""Features de descanso e congestão de calendário por time.

Fadiga é um dos poucos efeitos de curto prazo com suporte empírico em
futebol: times com menos de ~4 dias de descanso ou em sequência congestionada
(copa + liga na mesma semana) rendem abaixo do seu nível base. Este módulo
deriva, para cada partida de cada time, duas medidas complementares:

- ``days_since_last``: descanso pontual — dias desde o jogo anterior do
  mesmo time (NaN no primeiro jogo do histórico, quando não há passado).
- ``matches_last_14d``: carga acumulada — quantos jogos o time disputou na
  janela [d - 14 dias, d), ou seja, nos 14 dias ANTERIORES à partida,
  excluindo a própria partida.

Anti-leakage por construção: ambas as features olham SOMENTE para o passado.
``days_since_last`` vem de um ``shift(1)`` dentro do grupo do time (o jogo
anterior, nunca o próximo) e a janela de ``matches_last_14d`` é fechada em
d - 14 e aberta em d — a própria partida e qualquer jogo futuro ficam fora.

Funções puras: operam sobre o DataFrame recebido, sem rede, banco ou arquivo.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

_REQUIRED_COLUMNS = ("team", "match_date")
_WINDOW_DAYS = 14
_SECONDS_PER_DAY = 86400.0
_NS_PER_DAY = 86_400_000_000_000


def _window_counts(dates_ns: npt.NDArray[np.int64], window_ns: int) -> npt.NDArray[np.int64]:
    """Conta, para cada jogo i, os jogos j do mesmo time com t_i - W <= t_j < t_i.

    Requer ``dates_ns`` ordenado crescente (garantido pelo sort do chamador).
    Busca binária vetorizada: count_i = hi_i - lo_i, com
    lo_i = searchsorted(t, t_i - W, 'left') e hi_i = searchsorted(t, t_i,
    'left'). O lado 'left' em hi_i torna o limite superior estrito: a
    própria partida (e qualquer duplicata na mesma data) não conta.
    """
    lo = np.searchsorted(dates_ns, dates_ns - window_ns, side="left")
    hi = np.searchsorted(dates_ns, dates_ns, side="left")
    return np.asarray(hi - lo, dtype=np.int64)


def rest_features(match_dates_by_team: pd.DataFrame) -> pd.DataFrame:
    """Deriva descanso e congestão de calendário para cada (team, match_date).

    Para cada linha do calendário:

    - ``days_since_last`` = (d_i - d_{i-1}) em dias (frações contam se as
      datas tiverem horário), onde d_{i-1} é o jogo imediatamente anterior
      do MESMO time; NaN quando não existe jogo anterior no histórico.
    - ``matches_last_14d`` = #{j : d_i - 14 dias <= d_j < d_i}, jogos do
      mesmo time na janela dos 14 dias anteriores, excluindo a própria
      partida. O limite inferior é inclusivo: um jogo exatamente 14 dias
      antes ainda conta como carga recente.

    Só o passado entra: o cálculo usa ``groupby("team")`` + ``shift(1)``
    para o descanso e uma janela estritamente anterior a d_i para a carga,
    de modo que adicionar jogos futuros ao calendário não altera as
    features das partidas já existentes.

    Parâmetros
    ----------
    match_dates_by_team:
        DataFrame com colunas ``team`` e ``match_date`` (qualquer coisa que
        ``pd.to_datetime`` aceite); colunas extras são ignoradas. A ordem
        das linhas é livre — o resultado volta alinhado ao índice original.
        NaT em ``match_date`` levanta ValueError.

    Retorno
    -------
    DataFrame com o MESMO índice e ordem de linhas da entrada e colunas
    ``team``, ``match_date`` (normalizada para datetime), ``days_since_last``
    (float64) e ``matches_last_14d`` (int64).
    """
    missing = [col for col in _REQUIRED_COLUMNS if col not in match_dates_by_team.columns]
    if missing:
        raise ValueError(f"match_dates_by_team sem colunas obrigatórias: {missing}")

    df = match_dates_by_team.loc[:, list(_REQUIRED_COLUMNS)].copy()
    original_index = df.index
    df = df.reset_index(drop=True)
    df["match_date"] = pd.to_datetime(df["match_date"])
    if bool(df["match_date"].isna().any()):
        raise ValueError("match_date contém valores ausentes (NaT)")

    ordered = df.sort_values(["team", "match_date"], kind="stable")
    prev = ordered.groupby("team", sort=False)["match_date"].shift(1)
    days_since_last = (ordered["match_date"] - prev).dt.total_seconds() / _SECONDS_PER_DAY

    dates_ns = ordered["match_date"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    counts = np.zeros(len(ordered), dtype=np.int64)
    for positions in ordered.groupby("team", sort=False).indices.values():
        counts[positions] = _window_counts(dates_ns[positions], _WINDOW_DAYS * _NS_PER_DAY)

    result = pd.DataFrame(
        {
            "team": ordered["team"],
            "match_date": ordered["match_date"],
            "days_since_last": days_since_last.astype(np.float64),
            "matches_last_14d": pd.Series(counts, index=ordered.index, dtype=np.int64),
        }
    )
    result = result.sort_index()
    result.index = original_index
    return result
