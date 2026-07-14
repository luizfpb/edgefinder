"""Modelo de minutos esperados por jogador.

Props de jogador (finalizações, desarmes, gols etc.) são taxas por tempo em
campo: sem uma estimativa de minutos, qualquer projeção de contagem é inútil.
Este módulo estima os minutos esperados via a decomposição

    E[min] = P(joga) * E[min | joga]

que separa dois fenômenos distintos: a chance de o jogador entrar em campo
(lesão, rotação, perda de espaço no elenco) e o volume de minutos quando
entra (titular que joga 90 vs. reserva que entra aos 70). Ambos os termos
usam ponderação exponencial por recência com meia-vida configurável:

    w_i = 2^(-dt_i / h) = exp(-ln(2) * dt_i / h)

onde dt_i é a distância em dias entre a partida i e ``as_of`` e h é a
meia-vida em dias. Assim, um jogo a h dias de distância pesa metade de um
jogo de hoje.

Todas as funções são puras: operam sobre DataFrames/arrays já em memória,
sem rede, banco ou leitura de arquivo.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pandas as pd

type FloatInput = float | npt.NDArray[np.float64] | pd.Series[float]

_REQUIRED_COLUMNS = ("player_id", "match_date", "minutes")
_TEAM_WINDOW = 10
_DEFAULT_GAP_DAYS = 7.0
_SECONDS_PER_DAY = 86400.0


def _recency_weights(
    dates: pd.DatetimeIndex, as_of: pd.Timestamp, half_life_days: float
) -> npt.NDArray[np.float64]:
    """Pesos exponenciais de recência com meia-vida.

    Fórmula: w_i = exp(-ln(2) * dt_i / h), com dt_i = (as_of - data_i) em
    dias e h = ``half_life_days``. A meia-vida é mais interpretável que uma
    taxa de decaimento crua: h dias atrás o peso é exatamente 0.5.
    """
    delta_sec = (as_of.to_datetime64() - dates.to_numpy()).astype("timedelta64[s]")
    dt_days = delta_sec.astype(np.float64) / _SECONDS_PER_DAY
    return np.asarray(np.exp(-math.log(2.0) * dt_days / half_life_days), dtype=np.float64)


def _fallback_schedule(own_dates: pd.DatetimeIndex, as_of: pd.Timestamp) -> pd.DatetimeIndex:
    """Aproxima o calendário do time a partir do histórico do próprio jogador.

    Sem ``team_dates`` não sabemos quais jogos do time o jogador perdeu, e o
    histórico dele sozinho daria p_play = 1 para qualquer um (só contém jogos
    em que ele aparece). A simplificação documentada é preencher o vão entre
    o último jogo do jogador e ``as_of`` com datas sintéticas de ausência,
    espaçadas pela cadência típica do time, estimada pela mediana dos
    intervalos entre os últimos jogos do próprio jogador:

        g = mediana(data_{i+1} - data_i)  [dias]

    (ou 7 dias se houver menos de dois jogos). Assim, um jogador parado há
    meses acumula ausências sintéticas e p_play cai, como deve.
    """
    if len(own_dates) == 0:
        return pd.DatetimeIndex([])
    tail = own_dates[-(_TEAM_WINDOW + 1) :]
    if len(tail) >= 2:
        diffs_sec = np.diff(tail.to_numpy()).astype("timedelta64[s]").astype(np.float64)
        gap_days = float(np.median(diffs_sec)) / _SECONDS_PER_DAY
    else:
        gap_days = _DEFAULT_GAP_DAYS
    if gap_days <= 0:
        gap_days = _DEFAULT_GAP_DAYS

    synthetic: list[pd.Timestamp] = []
    current = own_dates[-1] + pd.Timedelta(days=gap_days)
    while current < as_of and len(synthetic) < _TEAM_WINDOW:
        synthetic.append(current)
        current = current + pd.Timedelta(days=gap_days)
    combined = pd.DatetimeIndex(list(own_dates) + synthetic)
    return combined[-_TEAM_WINDOW:]


def _weighted_presence(
    schedule: pd.DatetimeIndex,
    played_dates: set[pd.Timestamp],
    as_of: pd.Timestamp,
    half_life_days: float,
) -> float:
    """Fração de presença ponderada por recência sobre o calendário do time.

    Fórmula: p_play = sum_i w_i * 1[jogou na data i] / sum_i w_i, sobre as
    últimas ``_TEAM_WINDOW`` datas de jogo antes de ``as_of``. Como o
    indicador está em {0, 1} e os pesos são positivos, o resultado fica
    sempre em [0, 1]. Calendário vazio devolve NaN (não há evidência).
    """
    if len(schedule) == 0:
        return float("nan")
    weights = _recency_weights(schedule, as_of, half_life_days)
    presence = np.asarray([1.0 if ts in played_dates else 0.0 for ts in schedule], dtype=np.float64)
    return float(np.sum(weights * presence) / np.sum(weights))


def expected_minutes(
    history: pd.DataFrame,
    as_of: pd.Timestamp,
    half_life_days: float = 60.0,
    team_dates: pd.Series[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Estima minutos esperados por jogador na próxima partida.

    Decomposição: E[min] = P(joga) * E[min | joga].

    - P(joga): presença ponderada por recência nas últimas 10 datas de jogo
      do time antes de ``as_of`` (ver ``_weighted_presence``). "Presença"
      significa minutos > 0 naquela data; linha com minutos 0 (banco sem
      entrar) conta como ausência. Se ``team_dates`` não for fornecido, o
      calendário do time é aproximado pelas datas do próprio jogador mais
      datas sintéticas de ausência até ``as_of`` (ver ``_fallback_schedule``).
    - E[min | joga]: média ponderada por recência dos minutos nos jogos em
      que jogou: sum_j w_j * min_j / sum_j w_j, com w_j = 2^(-dt_j / h).

    Anti-leakage: qualquer linha do histórico com ``match_date >= as_of``
    levanta ValueError — a partida a prever (ou posterior) nunca pode
    alimentar a estimativa. Datas de ``team_dates`` no futuro (o próprio
    jogo a prever, por exemplo) são apenas ignoradas, pois um calendário
    contém fixtures futuros por natureza.

    Parâmetros
    ----------
    history:
        Colunas obrigatórias: ``player_id``, ``match_date``, ``minutes``.
        A coluna ``started`` (bool | NaN) é aceita e ignorada: a decomposição
        adotada não distingue titular de reserva, apenas minutos observados.
        ``minutes`` NaN é tratado como ausência (excluído de E[min | joga]).
    as_of:
        Instante da previsão; só entra informação estritamente anterior.
    half_life_days:
        Meia-vida h > 0 dos pesos de recência, em dias.
    team_dates:
        Datas de jogo do time (opcional). Comparadas por dia (normalizadas),
        pois calendários costumam vir sem horário.

    Retorno
    -------
    DataFrame com uma linha por jogador (ordenado por ``player_id``) e
    colunas ``player_id``, ``exp_minutes``, ``p_play``,
    ``exp_minutes_if_plays``. Jogador sem nenhum jogo com minutos > 0 tem
    ``exp_minutes_if_plays`` NaN e ``exp_minutes`` 0.0 (p_play = 0 domina).
    """
    missing = [col for col in _REQUIRED_COLUMNS if col not in history.columns]
    if missing:
        raise ValueError(f"history sem colunas obrigatórias: {missing}")
    if not math.isfinite(half_life_days) or half_life_days <= 0:
        raise ValueError(f"half_life_days deve ser finito e > 0, recebido {half_life_days}")

    df = history.loc[:, list(_REQUIRED_COLUMNS)].copy()
    df["match_date"] = pd.to_datetime(df["match_date"])
    if bool((df["match_date"] >= as_of).any()):
        raise ValueError("history contém partida com match_date >= as_of (vazamento temporal)")
    df["minutes"] = pd.to_numeric(df["minutes"]).astype(np.float64)
    if bool((df["minutes"] < 0).any()):
        raise ValueError("minutes não pode ser negativo")

    team_schedule: pd.DatetimeIndex | None = None
    if team_dates is not None:
        raw = pd.DatetimeIndex(pd.to_datetime(team_dates))
        past = raw[raw < as_of].normalize().unique().sort_values()
        team_schedule = past[-_TEAM_WINDOW:]

    player_ids: list[object] = []
    exp_minutes_col: list[float] = []
    p_play_col: list[float] = []
    exp_if_plays_col: list[float] = []

    for player_id, group in df.groupby("player_id", sort=True):
        played_mask = group["minutes"] > 0
        played = group.loc[played_mask]

        if len(played) > 0:
            played_idx = pd.DatetimeIndex(played["match_date"])
            weights = _recency_weights(played_idx, as_of, half_life_days)
            minutes_arr = played["minutes"].to_numpy(dtype=np.float64)
            exp_if_plays = float(np.sum(weights * minutes_arr) / np.sum(weights))
        else:
            exp_if_plays = float("nan")

        own_dates = pd.DatetimeIndex(group["match_date"]).normalize().unique().sort_values()
        played_dates = set(pd.DatetimeIndex(played["match_date"]).normalize())
        schedule = (
            team_schedule if team_schedule is not None else _fallback_schedule(own_dates, as_of)
        )
        p_play = _weighted_presence(schedule, played_dates, as_of, half_life_days)

        if p_play == 0.0:
            exp_min = 0.0
        elif math.isnan(p_play) or math.isnan(exp_if_plays):
            exp_min = float("nan")
        else:
            exp_min = p_play * exp_if_plays

        player_ids.append(player_id)
        exp_minutes_col.append(exp_min)
        p_play_col.append(p_play)
        exp_if_plays_col.append(exp_if_plays)

    return pd.DataFrame(
        {
            "player_id": pd.Series(player_ids, dtype=object),
            "exp_minutes": pd.Series(exp_minutes_col, dtype=np.float64),
            "p_play": pd.Series(p_play_col, dtype=np.float64),
            "exp_minutes_if_plays": pd.Series(exp_if_plays_col, dtype=np.float64),
        }
    )


def _match_container(result: npt.NDArray[np.float64], *inputs: object) -> FloatInput:
    """Devolve o resultado no mesmo "recipiente" da entrada mais rica.

    Prioridade Series > ndarray > float, para que o chamador receba de volta
    o tipo que forneceu (com o índice da primeira Series encontrada). Series
    são combinadas posicionalmente, não por alinhamento de índice.
    """
    for obj in inputs:
        if isinstance(obj, pd.Series):
            return pd.Series(result, index=obj.index)
    for obj in inputs:
        if isinstance(obj, np.ndarray):
            return result
    return float(result)


def to_per90(value: FloatInput, minutes: FloatInput) -> FloatInput:
    """Converte uma contagem em taxa por 90 minutos.

    Fórmula: rate90 = 90 * value / minutes.

    A normalização por 90 coloca jogadores com volumes de minutos distintos
    na mesma escala (a de um jogo completo), pré-requisito para comparar
    taxas e reescalá-las depois com ``rescale_to_expected``. É nan-safe:
    minutes <= 0 (jogador que não atuou) devolve NaN em vez de dividir por
    zero, pois não há taxa observável sem tempo em campo.
    """
    value_arr = np.asarray(value, dtype=np.float64)
    minutes_arr = np.asarray(minutes, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.divide(90.0 * value_arr, minutes_arr)
    rate = np.asarray(np.where(minutes_arr > 0, raw, np.nan), dtype=np.float64)
    return _match_container(rate, value, minutes)


def rescale_to_expected(rate90: FloatInput, exp_minutes: FloatInput) -> FloatInput:
    """Reescala uma taxa por 90 para o valor esperado na partida.

    Fórmula: E[valor na partida] = rate90 * exp_minutes / 90.

    É a inversa de ``to_per90`` avaliada nos minutos esperados: assume que a
    taxa do jogador é constante por minuto, de modo que o valor esperado
    escala linearmente com o tempo em campo previsto. NaN em qualquer
    entrada propaga para o resultado.
    """
    rate_arr = np.asarray(rate90, dtype=np.float64)
    minutes_arr = np.asarray(exp_minutes, dtype=np.float64)
    expected = np.asarray(rate_arr * minutes_arr / 90.0, dtype=np.float64)
    return _match_container(expected, rate90, exp_minutes)
