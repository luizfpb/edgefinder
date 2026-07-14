"""Confiabilidade e persistência de estatísticas de jogador por mercado.

Nem toda estatística é igualmente previsível. Finalizações por jogo carregam
muito sinal estável de habilidade (volume de chute é um traço do jogador),
enquanto gols em amostras curtas são dominados por variância amostral.
Modelar as duas com a mesma confiança levaria a exigir a mesma margem de EV
sobre projeções de qualidade muito diferente.

Este módulo mede a previsibilidade de cada estatística com duas lentes
complementares:

- Confiabilidade split-half (consistência interna, entre jogadores): divide
  os jogos de cada jogador em posições pares e ímpares e correlaciona as
  médias das duas metades. Se a estatística carrega sinal estável, as duas
  metades concordam.
- Autocorrelação lag-1 (persistência temporal, dentro do jogador): quanto o
  valor de um jogo informa o do jogo seguinte do mesmo jogador.

O `stability_report` consolida ambas por mercado e alimenta a modulação de
thresholds de EV: mercado menos confiável exige edge maior antes de apostar,
porque a projeção construída sobre ele é mais ruidosa.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pandas as pd

_SPLIT_HALF_COLUMNS = ("player_id", "value")
# 2r/(1+r) e indefinida em r = -1; com erro de ponto flutuante r pode chegar
# como -1 +- 1e-16, e o quociente viraria um numero absurdo em vez de NaN.
_SB_DENOM_TOL = 1e-12
_AUTOCORR_COLUMNS = ("player_id", "match_date", "value")
_REPORT_COLUMNS = ("player_id", "match_date", "market", "value")


def _validate_columns(df: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    """Falha cedo, com mensagem clara, quando o contrato de colunas é violado."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"colunas obrigatórias ausentes em {name}: {missing}")


def _validate_min_games(min_games: int) -> None:
    """O split par/ímpar exige pelo menos um jogo em cada metade, logo min_games >= 2."""
    if min_games < 2:
        raise ValueError(f"min_games deve ser >= 2, recebido {min_games}")


def _validate_values(values: pd.Series[float]) -> npt.NDArray[np.float64]:
    """Converte para float64 rejeitando NaN/inf: dado faltante silencioso viraria viés."""
    arr = np.asarray(values.to_numpy(dtype=np.float64), dtype=np.float64)
    if arr.size and not bool(np.all(np.isfinite(arr))):
        raise ValueError("value deve conter apenas valores finitos (sem NaN/inf)")
    return arr


def _validate_dates(dates: pd.Series[pd.Timestamp]) -> None:
    """Data nula quebraria a ordenação temporal sem aviso; melhor falhar explícito."""
    if bool(dates.isna().any()):
        raise ValueError("match_date não pode conter nulos")


def _pearson(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    """Correlação de Pearson: r = sum(da * db) / sqrt(sum(da^2) * sum(db^2)).

    Implementada à mão para devolver NaN quando alguma das séries tem
    variância zero (r é indefinido nesse caso), em vez de emitir warning
    numérico e propagar lixo como `np.corrcoef` faria.
    """
    da = a - a.mean()
    db = b - b.mean()
    denom = math.sqrt(float(da @ da)) * math.sqrt(float(db @ db))
    if denom == 0.0:
        return float("nan")
    return float(da @ db) / denom


def split_half_reliability(player_stat: pd.DataFrame, min_games: int = 10) -> float:
    """Confiabilidade split-half entre jogadores, corrigida por Spearman-Brown.

    Para cada jogador j com pelo menos `min_games` jogos, divide os jogos em
    posições pares e ímpares — na ordem das linhas, que deve ser cronológica —
    e calcula as médias m_par(j) e m_impar(j). Entre jogadores:

        r    = corr_Pearson(m_par, m_impar)
        r_sb = 2r / (1 + r)                    (correção de Spearman-Brown)

    r mede a concordância entre duas MEIAS amostras; a correção de
    Spearman-Brown projeta essa concordância para a amostra completa (o dobro
    do comprimento) — sem ela a confiabilidade seria sistematicamente
    subestimada. O split par/ímpar, em vez de primeira x segunda metade,
    intercala as metades no tempo e protege a medida de tendências lentas
    (forma, mudança de papel tático), que inflariam a discordância entre
    metades contíguas.

    Args:
        player_stat: DataFrame com colunas [player_id, value], uma linha por
            (jogador, jogo), em ordem cronológica dentro de cada jogador.
        min_games: mínimo de jogos para o jogador entrar no cálculo (>= 2).

    Returns:
        r_sb, ou NaN quando indefinido: menos de dois jogadores elegíveis,
        variância zero entre jogadores em alguma das metades, ou r = -1
        (a menos de erro de ponto flutuante), que anula o denominador da
        correção.
    """
    _validate_columns(player_stat, _SPLIT_HALF_COLUMNS, "player_stat")
    _validate_min_games(min_games)
    if player_stat.empty:
        return float("nan")
    values = _validate_values(player_stat["value"])
    work = pd.DataFrame({"player_id": player_stat["player_id"].to_numpy(), "value": values})

    even_means: list[float] = []
    odd_means: list[float] = []
    for _, group in work.groupby("player_id", sort=True):
        x = group["value"].to_numpy(dtype=np.float64)
        if x.size < min_games:
            continue
        even_means.append(float(x[0::2].mean()))
        odd_means.append(float(x[1::2].mean()))
    if len(even_means) < 2:
        return float("nan")

    r = _pearson(
        np.asarray(even_means, dtype=np.float64),
        np.asarray(odd_means, dtype=np.float64),
    )
    if not math.isfinite(r) or 1.0 + r < _SB_DENOM_TOL:
        return float("nan")
    return 2.0 * r / (1.0 + r)


def autocorrelation_lag1(player_stat: pd.DataFrame, min_games: int = 10) -> float:
    """Média das autocorrelações lag-1 por jogador (persistência temporal).

    Para cada jogador j com pelo menos `min_games` jogos, ordenados por
    match_date (x_1, ..., x_n):

        r_j = corr_Pearson((x_1, ..., x_{n-1}), (x_2, ..., x_n))

    e o resultado é a média simples dos r_j. Calcular por jogador e depois
    tirar a média — em vez de empilhar todos os pares — evita que diferenças
    de nível médio entre jogadores se disfarcem de persistência: um pool
    misturado teria autocorrelação alta só porque jogador bom segue bom, o
    que é sinal de habilidade (capturado pelo split-half), não de dinâmica
    jogo a jogo.

    Args:
        player_stat: DataFrame com colunas [player_id, match_date, value];
            match_date conversível para datetime. A ordenação temporal é
            feita internamente (empates de data mantêm a ordem das linhas).
        min_games: mínimo de jogos para o jogador entrar na média (>= 2).

    Returns:
        Média dos r_j, ou NaN se nenhum jogador elegível tiver r definido.
        Jogador com série constante tem r indefinido e fica de fora da média
        (uma constante não informa nada sobre persistência).
    """
    _validate_columns(player_stat, _AUTOCORR_COLUMNS, "player_stat")
    _validate_min_games(min_games)
    if player_stat.empty:
        return float("nan")
    work = player_stat.loc[:, list(_AUTOCORR_COLUMNS)].copy()
    work["match_date"] = pd.to_datetime(work["match_date"])
    _validate_dates(work["match_date"])
    _validate_values(work["value"])
    work = work.sort_values("match_date", kind="stable")

    corrs: list[float] = []
    for _, group in work.groupby("player_id", sort=True):
        x = group["value"].to_numpy(dtype=np.float64)
        if x.size < min_games:
            continue
        r = _pearson(x[:-1], x[1:])
        if math.isfinite(r):
            corrs.append(r)
    if not corrs:
        return float("nan")
    return float(np.mean(corrs))


def stability_report(df: pd.DataFrame, min_games: int = 10) -> pd.DataFrame:
    """Relatório de previsibilidade por mercado para modular thresholds de EV.

    Para cada mercado no formato longo [player_id, match_date, market, value]:

    - split_half: `split_half_reliability` sobre os jogos do mercado,
      ordenados cronologicamente dentro de cada jogador;
    - autocorr: `autocorrelation_lag1` sobre os mesmos jogos;
    - n_players: quantos jogadores atingem `min_games` no mercado — o tamanho
      efetivo da evidência; com poucos jogadores, as duas métricas acima são
      elas próprias ruidosas e merecem desconfiança.

    O uso a jusante: mercado com split_half baixo tem projeções mais ruidosas
    e o threshold de EV para apostar nele deve subir proporcionalmente.

    Args:
        df: DataFrame longo com colunas [player_id, match_date, market, value].
        min_games: mínimo de jogos por jogador dentro de cada mercado (>= 2).

    Returns:
        DataFrame [market, split_half, autocorr, n_players], uma linha por
        mercado, ordenado por market. Entrada vazia devolve DataFrame vazio
        com as mesmas colunas.
    """
    _validate_columns(df, _REPORT_COLUMNS, "df")
    _validate_min_games(min_games)
    if df.empty:
        return pd.DataFrame(
            {
                "market": pd.Series(dtype=object),
                "split_half": pd.Series(dtype=np.float64),
                "autocorr": pd.Series(dtype=np.float64),
                "n_players": pd.Series(dtype=np.int64),
            }
        )

    work = df.loc[:, list(_REPORT_COLUMNS)].copy()
    work["match_date"] = pd.to_datetime(work["match_date"])
    _validate_dates(work["match_date"])
    _validate_values(work["value"])
    work = work.sort_values(["player_id", "match_date"], kind="stable")

    markets: list[object] = []
    split_halves: list[float] = []
    autocorrs: list[float] = []
    n_players_col: list[int] = []
    for market, sub in work.groupby("market", sort=True):
        games_per_player = sub.groupby("player_id").size()
        markets.append(market)
        split_halves.append(split_half_reliability(sub[["player_id", "value"]], min_games))
        autocorrs.append(autocorrelation_lag1(sub[["player_id", "match_date", "value"]], min_games))
        n_players_col.append(int((games_per_player >= min_games).sum()))

    return pd.DataFrame(
        {
            "market": pd.Series(markets, dtype=object),
            "split_half": pd.Series(split_halves, dtype=np.float64),
            "autocorr": pd.Series(autocorrs, dtype=np.float64),
            "n_players": pd.Series(n_players_col, dtype=np.int64),
        }
    )
