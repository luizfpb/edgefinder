"""Props de jogador (chutes, chutes no gol, gols, cartões) com pooling parcial.

O problema central de props é o tamanho amostral: um jogador tem poucas dezenas
de jogos (às vezes dois ou três) e a taxa bruta x/e é ruidosa demais para
precificar uma linha. A solução é a hierarquia conjugada Gamma-Poisson por
empirical Bayes — exata, rápida e sem MCMC:

    x_ij ~ Poisson(theta_i * e_ij),    theta_i ~ Gamma(alpha, beta)

onde x_ij é a contagem do jogador i no jogo j e e_ij = minutos / 90 é a
exposição. A população (posição ou liga) fornece (alpha, beta); por conjugação,
o posterior de cada jogador é Gamma(alpha + x_i, beta + e_i), com média

    E[theta_i | dados] = (alpha + x_i) / (beta + e_i)

que interpola entre a taxa bruta x_i / e_i e a média populacional alpha / beta,
com peso proporcional à exposição acumulada: pouca amostra implica encolhimento
forte para a média da população (shrinkage), muita amostra deixa os dados do
próprio jogador dominarem.

A distribuição preditiva de uma nova contagem com exposição e, dado
theta ~ Gamma(a, b), é Binomial Negativa (mistura Gamma de Poissons):

    P(X = k) = NB(k; r = a, p = b / (b + e))
    E[X] = a * e / b,    Var[X] = E[X] * (b + e) / b > E[X]

ou seja, a incerteza sobre theta engrossa a cauda — exatamente o que uma
Poisson avaliada na média posterior ignora. Se a NB de fato precifica melhor
que a Poisson é uma questão EMPÍRICA, respondida por model_comparison
(log-loss preditivo fora da amostra e AIC/BIC no treino), não uma suposição.
"""

from __future__ import annotations

import math
from typing import overload

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import nbinom, poisson

__all__ = [
    "adjusted_rate",
    "fit_population",
    "model_comparison",
    "posterior_rate",
    "prob_over",
    "prob_over_poisson",
]

FloatArray = npt.NDArray[np.float64]

# Piso de Var(theta) relativo a m^2 quando o método dos momentos não detecta
# heterogeneidade (s2 <= m / e_barra): a variância amostral pode ficar abaixo
# do ruído de Poisson por puro acaso, e um piso pequeno e positivo produz um
# prior quase degenerado na média — shrinkage quase total, que é a resposta
# correta quando não há evidência de diferença entre jogadores.
_MIN_REL_VAR = 1e-6

# Caixas em log-espaço para o MLE da NB; o limite superior de log(alpha) = 20
# permite alpha ~ 5e8, na prática o limite Poisson (a NB aninha a Poisson
# quando alpha -> inf com alpha/beta fixo).
_NB_BOUNDS: list[tuple[float, float]] = [(-10.0, 20.0), (-15.0, 20.0)]


def _validate_counts_exposures(
    counts: npt.ArrayLike,
    exposures: npt.ArrayLike,
    *,
    integer_counts: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Converte e valida os vetores de contagem e exposição.

    Exposições devem ser estritamente positivas: e = 0 significa que o jogador
    não entrou em campo e não carrega informação sobre a taxa; e < 0 é erro de
    dado. Contagens podem ser fracionárias (somas ponderadas por decaimento),
    exceto onde integer_counts=True (verossimilhanças exigem contagens reais).
    """
    x = np.asarray(counts, dtype=np.float64)
    e = np.asarray(exposures, dtype=np.float64)
    if x.ndim != 1 or e.ndim != 1 or x.shape != e.shape:
        raise ValueError("counts e exposures devem ser vetores 1-d do mesmo tamanho")
    if x.size == 0:
        raise ValueError("counts e exposures não podem ser vazios")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(e))):
        raise ValueError("counts e exposures devem ser finitos")
    if np.any(e <= 0):
        raise ValueError("exposições devem ser estritamente positivas (minutos / 90 > 0)")
    if np.any(x < 0):
        raise ValueError("contagens devem ser não negativas")
    if integer_counts and np.any(x != np.floor(x)):
        raise ValueError("contagens devem ser inteiras para avaliar a verossimilhança")
    return x, e


def _check_gamma_params(alpha: float, beta: float) -> None:
    if not (math.isfinite(alpha) and math.isfinite(beta)) or alpha <= 0 or beta <= 0:
        raise ValueError("alpha e beta devem ser finitos e estritamente positivos")


def _fit_single_population(x: FloatArray, e: FloatArray) -> tuple[float, float]:
    """Método dos momentos ponderado pela exposição (estimador de Marshall, 1991).

    Com r_i = x_i / e_i e pesos w_i = e_i (cada taxa pesa a informação que
    carrega, já que Var(r_i | theta_i) = theta_i / e_i):

        m  = sum_i e_i r_i / sum_i e_i = sum_i x_i / sum_i e_i
        s2 = sum_i e_i (r_i - m)^2 / sum_i e_i
        Var(theta) ~= s2 - m / e_barra,    e_barra = (1/n) sum_i e_i

    O termo -m / e_barra desconta a parcela de s2 que é puro ruído de Poisson
    (E[média ponderada de Var(r_i | theta_i)] = m * n / sum_i e_i): sem ele,
    jogadores com exposição pequena inflariam a heterogeneidade estimada, pois
    suas taxas brutas oscilam muito mesmo com theta idêntico. Da média e
    variância da Gamma (E = alpha/beta, Var = alpha/beta^2) seguem

        alpha = m^2 / Var(theta),    beta = m / Var(theta).
    """
    n = x.size
    if n < 2:
        raise ValueError("são necessários ao menos 2 jogadores para estimar a população")
    sum_e = float(e.sum())
    m = float(x.sum()) / sum_e
    if m <= 0:
        raise ValueError("todas as contagens são zero; não há taxa média para ancorar o prior")
    rates = x / e
    s2 = float(np.sum(e * (rates - m) ** 2)) / sum_e
    e_bar = sum_e / n
    var = max(s2 - m / e_bar, m * m * _MIN_REL_VAR)
    return m * m / var, m / var


@overload
def fit_population(
    counts: npt.ArrayLike,
    exposures: npt.ArrayLike,
    groups: None = ...,
) -> tuple[float, float]: ...


@overload
def fit_population(
    counts: npt.ArrayLike,
    exposures: npt.ArrayLike,
    groups: npt.ArrayLike,
) -> dict[str, tuple[float, float]]: ...


def fit_population(
    counts: npt.ArrayLike,
    exposures: npt.ArrayLike,
    groups: npt.ArrayLike | None = None,
) -> tuple[float, float] | dict[str, tuple[float, float]]:
    """Estima o prior populacional Gamma(alpha, beta) das taxas individuais.

    Cada elemento é UM jogador com contagem total (pode ser soma ponderada por
    decaimento, logo fracionária) e exposição total em unidades de 90 minutos.
    A estimativa usa o método dos momentos ponderado descrito em
    _fit_single_population, que corrige o viés causado por exposições pequenas.

    Com groups (posição ou liga por jogador), ajusta um prior independente por
    grupo e devolve {rótulo: (alpha, beta)} — pooling parcial dentro de cada
    população, sem misturar atacantes com zagueiros na mesma Gamma. Cada grupo
    precisa de ao menos 2 jogadores.
    """
    x, e = _validate_counts_exposures(counts, exposures)
    if groups is None:
        return _fit_single_population(x, e)
    g = np.asarray(groups)
    if g.ndim != 1 or g.shape != x.shape:
        raise ValueError("groups deve ser um vetor 1-d do mesmo tamanho de counts")
    result: dict[str, tuple[float, float]] = {}
    for label in np.unique(g):
        mask = g == label
        result[str(label)] = _fit_single_population(x[mask], e[mask])
    return result


def posterior_rate(
    x_total_w: float,
    e_total_w: float,
    alpha: float,
    beta: float,
) -> tuple[float, float]:
    """Atualização conjugada: posterior Gamma(alpha + x, beta + e) da taxa.

    Com prior theta ~ Gamma(alpha, beta) e verossimilhança Poisson ponderada
    por pesos de decaimento temporal w_j (ex.: w_j = exp(-xi * dt_j)):

        log L(theta) = sum_j w_j [x_j log(theta e_j) - theta e_j] + const
                     = (sum_j w_j x_j) log(theta) - theta (sum_j w_j e_j) + const

    que é o kernel de uma Gamma. Por isso as entradas são as somas PONDERADAS
    x_total_w = sum_j w_j x_j e e_total_w = sum_j w_j e_j: o decaimento reduz
    simultaneamente a pseudo-contagem e a pseudo-exposição, encolhendo jogos
    antigos sem distorcer a taxa. A média posterior é

        E[theta | dados] = (alpha + x_total_w) / (beta + e_total_w).

    Devolve (a, b) do posterior, prontos para prob_over / adjusted_rate.
    """
    _check_gamma_params(alpha, beta)
    if not (math.isfinite(x_total_w) and math.isfinite(e_total_w)):
        raise ValueError("x_total_w e e_total_w devem ser finitos")
    if x_total_w < 0 or e_total_w < 0:
        raise ValueError("x_total_w e e_total_w devem ser não negativos")
    return alpha + x_total_w, beta + e_total_w


def adjusted_rate(a: float, b: float, multiplier: float = 1.0) -> float:
    """Taxa pontual ajustada: rate_final = (a / b) * multiplier.

    O multiplicador m compõe ajustes contextuais calculados fora deste módulo
    (força do adversário, casa/fora, árbitro para cartões): a hierarquia
    Gamma-Poisson estima a taxa base do jogador, e o contexto entra como fator
    multiplicativo sobre a média posterior a / b.
    """
    _check_gamma_params(a, b)
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("multiplier deve ser finito e estritamente positivo")
    return (a / b) * multiplier


def _over_threshold(line: float) -> int:
    """Menor inteiro que ganha a aposta over: k = floor(line) + 1.

    Para linhas .5 isso coincide com ceil(line + 0.5) (ex.: over 2.5 paga com
    X >= 3); para linhas inteiras devolve o over excluindo o push (over 2.0
    paga com X >= 3, X = 2 devolve a aposta e fica fora dos dois lados).
    """
    if not math.isfinite(line):
        raise ValueError("line deve ser finita")
    return math.floor(line) + 1


def prob_over(
    line: float,
    a: float,
    b: float,
    exposure: float,
    multiplier: float = 1.0,
) -> float:
    """P(X > line) sob a preditiva Binomial Negativa da hierarquia Gamma-Poisson.

    Dado theta ~ Gamma(a, b) (tipicamente o posterior do jogador) e intensidade
    theta * exposure * multiplier, a marginal de X é

        X ~ NB(r = a, p = b / (b + e_eff)),    e_eff = exposure * multiplier

    pois a mistura Gamma de Poissons é Binomial Negativa. O multiplicador entra
    escalando a exposição efetiva, o que equivale exatamente a escalar a taxa
    (Poisson(theta * m * e) = Poisson(theta * (m e))) e, ao contrário de mexer
    só na média, preserva a incerteza de theta na cauda. Devolve

        P(X >= k),    k = floor(line) + 1

    via a função de sobrevivência da NB do scipy (estável na cauda).
    """
    _check_gamma_params(a, b)
    if not (math.isfinite(exposure) and math.isfinite(multiplier)):
        raise ValueError("exposure e multiplier devem ser finitos")
    if exposure <= 0 or multiplier <= 0:
        raise ValueError("exposure e multiplier devem ser estritamente positivos")
    k = _over_threshold(line)
    if k <= 0:
        return 1.0
    e_eff = exposure * multiplier
    p = b / (b + e_eff)
    return float(nbinom.sf(k - 1, a, p))


def prob_over_poisson(
    line: float,
    a: float,
    b: float,
    exposure: float,
    multiplier: float = 1.0,
) -> float:
    """P(X > line) sob Poisson avaliada apenas na média posterior (baseline).

    Usa X ~ Poisson(mu) com mu = (a / b) * exposure * multiplier: mesma média
    da preditiva NB de prob_over, mas ignorando a incerteza sobre theta. Serve
    de comparação — a diferença entre as duas é a gordura de cauda que o
    pooling parcial adiciona, e model_comparison decide empiricamente qual das
    duas precifica melhor.
    """
    _check_gamma_params(a, b)
    if not (math.isfinite(exposure) and math.isfinite(multiplier)):
        raise ValueError("exposure e multiplier devem ser finitos")
    if exposure <= 0 or multiplier <= 0:
        raise ValueError("exposure e multiplier devem ser estritamente positivos")
    k = _over_threshold(line)
    if k <= 0:
        return 1.0
    mu = (a / b) * exposure * multiplier
    return float(poisson.sf(k - 1, mu))


def _nb_logpmf(x: FloatArray, e: FloatArray, alpha: float, beta: float) -> FloatArray:
    """log NB(x; r = alpha, p = beta / (beta + e)), a marginal de Gamma-Poisson."""
    p = beta / (beta + e)
    return np.asarray(nbinom.logpmf(x, alpha, p), dtype=np.float64)


def model_comparison(
    train_counts: npt.ArrayLike,
    train_exposures: npt.ArrayLike,
    test_counts: npt.ArrayLike,
    test_exposures: npt.ArrayLike,
) -> pd.DataFrame:
    """Compara EMPIRICAMENTE Poisson pura vs NB (Gamma-Poisson) nos dados.

    A NB só deve ser adotada se os dados exibirem sobredispersão real; assumir
    que ela ganha é erro de metodologia. A comparação usa:

    - Poisson homogênea: MLE em forma fechada, lambda = sum(x) / sum(e), e
      x_i ~ Poisson(lambda * e_i). 1 parâmetro.
    - Gamma-Poisson: MLE de (alpha, beta) em log-espaço (L-BFGS-B, partida no
      método dos momentos) na marginal x_i ~ NB(r = alpha, p = beta/(beta+e_i)).
      2 parâmetros; aninha a Poisson no limite alpha -> inf com alpha/beta fixo.

    Métricas reportadas (linhas 'poisson' e 'gamma_poisson'):

        log_loss_test = -(1/n_test) sum_i log p(x_i | modelo ajustado no treino)
        AIC = 2k - 2 logL_treino,    BIC = k ln(n_treino) - 2 logL_treino

    Log-loss preditivo menor fora da amostra é o critério decisivo; AIC/BIC no
    treino mostram se o ganho de verossimilhança da NB paga o parâmetro extra.
    Contagens devem ser inteiras (verossimilhanças exatas, sem ponderação).
    """
    x_tr, e_tr = _validate_counts_exposures(train_counts, train_exposures, integer_counts=True)
    x_te, e_te = _validate_counts_exposures(test_counts, test_exposures, integer_counts=True)
    n_tr = x_tr.size
    if n_tr < 2:
        raise ValueError("são necessárias ao menos 2 observações de treino")
    if float(x_tr.sum()) <= 0:
        raise ValueError("treino sem nenhum evento; não há taxa para estimar")

    lam = float(x_tr.sum()) / float(e_tr.sum())
    ll_pois_train = float(np.sum(poisson.logpmf(x_tr, lam * e_tr)))
    log_loss_pois = -float(np.mean(poisson.logpmf(x_te, lam * e_te)))

    alpha0, beta0 = _fit_single_population(x_tr, e_tr)
    x0 = np.clip(
        np.array([math.log(alpha0), math.log(beta0)], dtype=np.float64),
        [b[0] for b in _NB_BOUNDS],
        [b[1] for b in _NB_BOUNDS],
    )

    def negloglik(params: FloatArray) -> float:
        alpha = math.exp(float(params[0]))
        beta = math.exp(float(params[1]))
        return -float(np.sum(_nb_logpmf(x_tr, e_tr, alpha, beta)))

    res = minimize(negloglik, x0, method="L-BFGS-B", bounds=_NB_BOUNDS)
    alpha_hat = math.exp(float(res.x[0]))
    beta_hat = math.exp(float(res.x[1]))
    ll_nb_train = -float(res.fun)
    log_loss_nb = -float(np.mean(_nb_logpmf(x_te, e_te, alpha_hat, beta_hat)))

    k_pois, k_nb = 1, 2
    log_n = math.log(n_tr)
    return pd.DataFrame(
        {
            "n_params": [k_pois, k_nb],
            "loglik_train": [ll_pois_train, ll_nb_train],
            "aic_train": [2 * k_pois - 2 * ll_pois_train, 2 * k_nb - 2 * ll_nb_train],
            "bic_train": [k_pois * log_n - 2 * ll_pois_train, k_nb * log_n - 2 * ll_nb_train],
            "log_loss_test": [log_loss_pois, log_loss_nb],
        },
        index=pd.Index(["poisson", "gamma_poisson"], name="model"),
    )
