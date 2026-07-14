"""Ajuste de contagens pela força do adversário (estilo ataque x defesa de Dixon-Coles).

Uma contagem bruta (chutes, escanteios, desarmes) mistura a habilidade de quem
produz com a permissividade de quem sofre: 4 chutes contra a defesa mais vazada
da liga carregam menos sinal do que 4 chutes contra a mais sólida. Este módulo
estima um fator multiplicativo de permissividade por time e o usa para
normalizar contagens antes da modelagem, no mesmo espírito em que o termo de
defesa do modelo de Dixon-Coles separa ataque e defesa nos gols.

Anti-leakage: `concession_factors` usa toda a amostra e serve para prever jogos
FUTUROS à amostra. Para construir features históricas de treino/backtest, use
`rolling_concession_factors`, que para cada linha só enxerga jogos com data
estritamente anterior à daquela linha.
"""

import math

import numpy as np
import numpy.typing as npt
import pandas as pd

_REQUIRED_COLUMNS = ("team", "match_date", "stat_conceded")
_SECONDS_PER_DAY = 86_400.0


def _decay_constant(half_life_days: float) -> float:
    """Converte meia-vida em constante de decaimento: xi = ln(2) / half_life_days.

    Com essa parametrização o peso de um jogo cai pela metade a cada
    `half_life_days` dias: w = exp(-xi * dt) e w(half_life_days) = 1/2.
    """
    if not (math.isfinite(half_life_days) and half_life_days > 0):
        raise ValueError(f"half_life_days deve ser finito e positivo, recebido {half_life_days}")
    return math.log(2.0) / half_life_days


def _validate_team_match(team_match: pd.DataFrame) -> None:
    """Falha cedo em entrada malformada: coluna ausente, valor negativo/NaN, data nula."""
    missing = [col for col in _REQUIRED_COLUMNS if col not in team_match.columns]
    if missing:
        raise ValueError(f"colunas obrigatórias ausentes em team_match: {missing}")
    values = team_match["stat_conceded"].to_numpy(dtype=np.float64)
    if values.size and (not np.all(np.isfinite(values)) or bool(np.any(values < 0.0))):
        raise ValueError("stat_conceded deve ser finito e não negativo")
    if bool(team_match["match_date"].isna().any()):
        raise ValueError("match_date não pode conter nulos")


def concession_factors(team_match: pd.DataFrame, half_life_days: float) -> pd.DataFrame:
    """Fator de permissividade por time: taxa concedida ponderada por recência / média da liga.

    Com t_ref = data mais recente da amostra e xi = ln(2) / half_life_days:

        w_i      = exp(-xi * (t_ref - t_i))              (t em dias)
        taxa_j   = sum_i w_i * x_i / sum_i w_i           (i percorre jogos do time j)
        taxa_lig = sum_i w_i * x_i / sum_i w_i           (i percorre todos os jogos)
        factor_j = taxa_j / taxa_lig

    Um atacante que chuta contra um time com factor 1.3 está chutando contra
    uma defesa 30% mais permissiva que a média da liga. A ponderação por
    recência existe porque elencos e treinadores mudam: um jogo de meses atrás
    diz menos sobre a defesa de hoje do que o da semana passada.

    Usa TODA a amostra — adequado apenas para prever jogos posteriores a ela.
    Para features históricas sem vazamento, use `rolling_concession_factors`.

    Args:
        team_match: DataFrame com colunas [team, match_date, stat_conceded],
            uma linha por (time, jogo); match_date conversível para datetime.
        half_life_days: meia-vida do peso exponencial, em dias.

    Returns:
        DataFrame [team, factor], um time por linha, ordenado por team.
        Degenerados: entrada vazia devolve DataFrame vazio; liga que não
        concedeu nada (taxa 0) devolve factor 1.0 para todos (não há como
        distinguir defesas quando ninguém sofre nada).
    """
    xi = _decay_constant(half_life_days)
    _validate_team_match(team_match)
    if team_match.empty:
        return pd.DataFrame(
            {"team": pd.Series(dtype=object), "factor": pd.Series(dtype=np.float64)}
        )

    dates = pd.to_datetime(team_match["match_date"])
    values = team_match["stat_conceded"].to_numpy(dtype=np.float64)
    dt_days = (dates.max() - dates).dt.total_seconds().to_numpy(dtype=np.float64) / _SECONDS_PER_DAY
    weights = np.exp(-xi * dt_days)

    grouped = (
        pd.DataFrame(
            {
                "team": team_match["team"].to_numpy(),
                "wx": weights * values,
                "w": weights,
            }
        )
        .groupby("team", sort=True)[["wx", "w"]]
        .sum()
    )
    num = grouped["wx"].to_numpy(dtype=np.float64)
    den = grouped["w"].to_numpy(dtype=np.float64)

    w_total = float(weights.sum())
    league_rate = float(weights @ values) / w_total if w_total > 0.0 else 0.0

    rate = np.divide(num, den, out=np.zeros_like(num), where=den > 0.0)
    if league_rate > 0.0:
        factor = np.where(den > 0.0, rate / league_rate, 1.0)
    else:
        factor = np.ones_like(num)

    return pd.DataFrame({"team": grouped.index.to_numpy(), "factor": factor})


def adjust_counts(
    counts: npt.ArrayLike,
    opponents_factors: npt.ArrayLike,
    k: float = 1.0,
) -> npt.NDArray[np.float64]:
    """Normaliza contagens pela permissividade do adversário: x_adj = x / factor_j**k.

    `k` modula a confiança no ajuste e é otimizado no backtest: os fatores são
    estimativas ruidosas, e aplicá-los integralmente (k = 1) pode injetar mais
    ruído do que remove viés. k = 0 ignora o adversário, k = 1 confia
    integralmente no fator, e valores intermediários encolhem o ajuste na
    proporção sinal/ruído que o backtest revelar.

    Args:
        counts: contagens observadas (array-like numérico).
        opponents_factors: fator do adversário enfrentado em cada observação,
            alinhado posicionalmente a `counts`; estritamente positivo.
        k: expoente de confiança no ajuste.

    Returns:
        Array float64 com as contagens normalizadas, mesma forma da entrada.
    """
    x = np.asarray(counts, dtype=np.float64)
    f = np.asarray(opponents_factors, dtype=np.float64)
    if x.shape != f.shape:
        raise ValueError(f"shapes incompatíveis: counts {x.shape} vs opponents_factors {f.shape}")
    if not math.isfinite(k):
        raise ValueError(f"k deve ser finito, recebido {k}")
    if f.size and not (bool(np.all(np.isfinite(f))) and bool(np.all(f > 0.0))):
        raise ValueError("opponents_factors deve ser estritamente positivo e finito")
    return x / f**k


def rolling_concession_factors(team_match: pd.DataFrame, half_life_days: float) -> pd.DataFrame:
    """Fator de permissividade por LINHA usando apenas o passado estrito daquela linha.

    Para a linha do time j na data t, o fator usa somente jogos com data < t
    (de qualquer time), com a mesma fórmula de `concession_factors`:

        w_i(t)    = exp(-xi * (t - t_i)),  t_i < t
        factor(t) = taxa_j(t) / taxa_liga(t)

    Isso elimina o vazamento temporal: o valor concedido no próprio jogo em t
    (e em qualquer jogo futuro) nunca entra no fator daquele jogo. Jogos na
    mesma data t também ficam de fora uns dos outros — o mercado da rodada não
    conhece os resultados da rodada.

    Implementação e complexidade: O(n log n) pela ordenação por data; depois
    uma única varredura O(n) mantém, por time e para a liga, somas ponderadas
    (num, den) atualizadas pela recorrência

        S(t) = S(t_ant) * exp(-xi * (t - t_ant)) + x

    que é matematicamente idêntica a recalcular todos os pesos a cada linha
    (custo O(n^2)), porém numericamente estável — evita o overflow do truque
    alternativo de cumsum com exp(+xi * t_i). A leitura da taxa não precisa
    decair o estado até t: num e den decaem pelo mesmo fator, que cancela na
    razão num/den.

    Args:
        team_match: DataFrame com colunas [team, match_date, stat_conceded].
        half_life_days: meia-vida do peso exponencial, em dias.

    Returns:
        Cópia de `team_match` (mesma ordem e índice) com a coluna `factor`.
        Linhas sem informação (primeiro jogo do time, ou liga sem histórico /
        com taxa passada zero) recebem 1.0 — o prior neutro de defesa média,
        equivalente a encolher totalmente para a liga.
    """
    xi = _decay_constant(half_life_days)
    _validate_team_match(team_match)
    out = team_match.copy()
    if team_match.empty:
        out["factor"] = pd.Series(dtype=np.float64)
        return out

    dates = pd.to_datetime(team_match["match_date"])
    day = (dates - dates.min()).dt.total_seconds().to_numpy(dtype=np.float64) / _SECONDS_PER_DAY
    values = team_match["stat_conceded"].to_numpy(dtype=np.float64)
    teams = team_match["team"].to_numpy()

    n = len(team_match)
    order = np.argsort(day, kind="stable")
    factors = np.ones(n, dtype=np.float64)

    league_num = 0.0
    league_den = 0.0
    league_t = 0.0
    # time -> (soma ponderada de x, soma de pesos, data da última atualização)
    state: dict[object, tuple[float, float, float]] = {}

    i = 0
    while i < n:
        t = float(day[order[i]])
        j = i
        while j < n and day[order[j]] == t:
            j += 1
        block = order[i:j]

        if league_den > 0.0:
            decay = math.exp(-xi * (t - league_t))
            league_num *= decay
            league_den *= decay
        league_t = t
        league_rate = league_num / league_den if league_den > 0.0 else 0.0

        for idx in block:
            st = state.get(teams[idx])
            if st is not None and league_rate > 0.0:
                num, den, _ = st
                factors[idx] = (num / den) / league_rate

        for idx in block:
            team = teams[idx]
            x = float(values[idx])
            st = state.get(team)
            if st is None:
                state[team] = (x, 1.0, t)
            else:
                num, den, t_last = st
                decay = math.exp(-xi * (t - t_last))
                state[team] = (num * decay + x, den * decay + 1.0, t)
            league_num += x
            league_den += 1.0
        i = j

    out["factor"] = factors
    return out
