"""Poisson bivariado de Karlis & Ntzoufras (2003) para placares de futebol.

O placar (X, Y) é construído a partir de três contagens independentes
W1 ~ Poisson(lambda1), W2 ~ Poisson(lambda2) e W3 ~ Poisson(lambda3):

    X = W1 + W3  (gols do mandante)
    Y = W2 + W3  (gols do visitante)

O choque comum W3 induz Cov(X, Y) = lambda3 >= 0 e E[X] = lambda1 + lambda3,
E[Y] = lambda2 + lambda3. Isso corrige a principal deficiência do produto de
Poissons independentes: a subestimação da dependência positiva entre os gols
dos dois times (e, em consequência, dos empates). A pmf conjunta é

    P(X=x, Y=y) = exp(-(l1+l2+l3)) * (l1^x / x!) * (l2^y / y!)
                  * sum_{k=0}^{min(x,y)} C(x,k) C(y,k) k! (l3 / (l1 l2))^k

e aqui é avaliada inteiramente em log-espaço (gammaln + logsumexp) para não
estourar em placares altos nem em taxas pequenas.

As taxas seguem a parametrização de Dixon-Coles: para o time i em casa contra j,

    log lambda1 = mu + gamma + atk_i + def_j
    log lambda2 = mu + atk_j + def_i

onde atk é força de ataque, def é fraqueza defensiva (maior = sofre mais gols),
gamma é a vantagem de casa e mu o nível médio de gols. A identificação usa
sum(atk) = sum(def) = 0, e lambda3 é global (um único parâmetro de correlação
compartilhado por todas as partidas). A verossimilhança é ponderada no tempo
por w = exp(-xi * dt), xi = ln(2) / half_life_days.

Referência: Karlis, D. e Ntzoufras, I. (2003), "Analysis of sports data by
using bivariate Poisson models", Journal of the Royal Statistical Society D,
52(3), 381-393.
"""

from __future__ import annotations

import math
from datetime import date, datetime

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp

__all__ = ["BivariatePoisson", "log_pmf", "time_decay_weights"]

FloatArray = npt.NDArray[np.float64]
DateLike = str | date | datetime | pd.Timestamp

_REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
)

# Caixas amplas o bastante para qualquer liga real; servem só para impedir que
# o L-BFGS-B visite regiões onde exp() estoura durante a busca de linha.
_BOUND_MU: tuple[float, float] = (-5.0, 5.0)
_BOUND_HOME_ADV: tuple[float, float] = (-3.0, 3.0)
_BOUND_STRENGTH: tuple[float, float] = (-5.0, 5.0)
_BOUND_LOG_LAMBDA3: tuple[float, float] = (-12.0, 3.0)


def log_pmf(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    lambda1: npt.ArrayLike,
    lambda2: npt.ArrayLike,
    lambda3: npt.ArrayLike,
) -> FloatArray:
    """Log da pmf conjunta do Poisson bivariado, vetorizada por broadcasting.

    A soma interna é avaliada em log-espaço porque seus termos crescem como
    k! e a soma direta perde precisão (ou estoura) já em placares moderados:

        log S(x, y) = logsumexp_{k=0}^{min(x,y)} [ log C(x,k) + log C(y,k)
                       + log k! + k * (log l3 - log l1 - log l2) ]

    com log C(n,k) = gammaln(n+1) - gammaln(k+1) - gammaln(n-k+1). O log da
    pmf completa é

        log P = -(l1+l2+l3) + x log l1 - gammaln(x+1)
                + y log l2 - gammaln(y+1) + log S.

    lambda3 = 0 é aceito: sobra apenas o termo k = 0 (S = 1) e a pmf reduz ao
    produto de Poissons independentes, como exige a construção X = W1 + W3.
    """
    xb, yb, l1, l2, l3 = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(lambda1, dtype=np.float64),
        np.asarray(lambda2, dtype=np.float64),
        np.asarray(lambda3, dtype=np.float64),
    )
    if np.any(xb < 0) or np.any(yb < 0) or np.any(xb != np.floor(xb)) or np.any(yb != np.floor(yb)):
        raise ValueError("x e y devem ser contagens inteiras não negativas")
    if np.any(l1 <= 0) or np.any(l2 <= 0):
        raise ValueError("lambda1 e lambda2 devem ser estritamente positivos")
    if np.any(l3 < 0):
        raise ValueError("lambda3 deve ser não negativo")
    if xb.size == 0:
        return np.zeros(xb.shape, dtype=np.float64)

    k_max = int(np.max(np.minimum(xb, yb)))
    kk = np.arange(k_max + 1, dtype=np.float64)
    x3 = xb[..., None]
    y3 = yb[..., None]
    mask = kk <= np.minimum(x3, y3)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log(l3) - np.log(l1) - np.log(l2)
        # k = 0 é tratado à parte para evitar 0 * (-inf) = nan quando l3 = 0.
        k_term = np.where(kk == 0.0, 0.0, kk * log_ratio[..., None])
    # Fora da região k <= min(x, y) os argumentos de gammaln ficariam não
    # positivos (polos); os valores são substituídos por -inf logo abaixo.
    safe_xk = np.where(mask, x3 - kk, 0.0)
    safe_yk = np.where(mask, y3 - kk, 0.0)
    inner = (
        gammaln(x3 + 1.0)
        + gammaln(y3 + 1.0)
        - gammaln(kk + 1.0)
        - gammaln(safe_xk + 1.0)
        - gammaln(safe_yk + 1.0)
        + k_term
    )
    inner = np.where(mask, inner, -np.inf)
    log_s = logsumexp(inner, axis=-1)
    base = (
        -(l1 + l2 + l3) + xb * np.log(l1) - gammaln(xb + 1.0) + yb * np.log(l2) - gammaln(yb + 1.0)
    )
    return np.asarray(base + log_s, dtype=np.float64)


def time_decay_weights(
    dates: pd.Series[pd.Timestamp],
    ref_date: DateLike,
    half_life_days: float,
) -> FloatArray:
    """Pesos de decaimento exponencial no tempo: w = exp(-xi * dt).

    xi = ln(2) / half_life_days e dt são os dias entre a partida e ref_date,
    de modo que a partida com dt = half_life_days pesa exatamente 0.5 e uma
    partida em ref_date pesa 1. Datas posteriores a ref_date são recusadas:
    ponderá-las (com w > 1, ainda por cima) seria vazamento de futuro.
    """
    if math.isnan(half_life_days) or half_life_days <= 0:
        raise ValueError("half_life_days deve ser positivo")
    ref_ts = pd.Timestamp(ref_date)
    dt_days = (ref_ts - pd.to_datetime(dates)).dt.total_seconds().to_numpy(dtype=np.float64)
    dt_days = dt_days / 86400.0
    if np.any(dt_days < 0):
        raise ValueError("há partidas posteriores a ref_date; filtre antes de ponderar")
    xi = math.log(2.0) / half_life_days
    return np.asarray(np.exp(-xi * dt_days), dtype=np.float64)


class BivariatePoisson:
    """Modelo de placares com forças de ataque/defesa por time e lambda3 global.

    A interface pública (fit / score_matrix / probs_1x2 / probs_over_under)
    espelha a do Dixon-Coles por duck-typing, para que o ensemble troque um
    modelo pelo outro sem adaptação. A diferença estrutural é o mecanismo de
    dependência: aqui um choque comum aditivo (lambda3) em vez do ajuste tau
    de placares baixos do Dixon-Coles.
    """

    def __init__(self) -> None:
        self.teams_: list[str] = []
        self.attack_: pd.Series[float] = pd.Series(dtype=np.float64)
        self.defence_: pd.Series[float] = pd.Series(dtype=np.float64)
        self.intercept_: float = 0.0
        self.home_advantage_: float = 0.0
        self.lambda3_: float = 0.0
        self.loglik_: float = float("nan")
        self.n_matches_: int = 0
        self.converged_: bool = False
        self._fitted: bool = False

    def fit(
        self,
        matches: pd.DataFrame,
        ref_date: DateLike,
        half_life_days: float = 365.0,
    ) -> BivariatePoisson:
        """Ajusta (mu, gamma, atk, def, log lambda3) por máxima verossimilhança ponderada.

        A função objetivo é

            max_theta  sum_i w_i * log P(X=x_i, Y=y_i | lambda1_i, lambda2_i, lambda3)

        com w_i = exp(-xi * dt_i) e xi = ln(2) / half_life_days. lambda3 é
        otimizado em log-espaço (log_lambda3) para garantir positividade sem
        restrição explícita. A identificação sum(atk) = sum(def) = 0 é imposta
        otimizando n-1 coeficientes livres por vetor e definindo o último time
        como o negativo da soma dos demais. Partidas posteriores a ref_date e
        partidas sem resultado são descartadas antes do ajuste, o que impede
        vazamento de futuro por construção.
        """
        missing = [c for c in _REQUIRED_COLUMNS if c not in matches.columns]
        if missing:
            raise ValueError(f"colunas ausentes em matches: {missing}")
        df = matches.loc[:, list(_REQUIRED_COLUMNS)].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["date", "home_goals", "away_goals"])
        ref_ts = pd.Timestamp(ref_date)
        df = df[df["date"] <= ref_ts]
        if df.empty:
            raise ValueError("nenhuma partida disputada até ref_date para ajustar o modelo")

        x = df["home_goals"].to_numpy(dtype=np.float64)
        y = df["away_goals"].to_numpy(dtype=np.float64)
        if np.any(x < 0) or np.any(y < 0) or np.any(x != np.floor(x)) or np.any(y != np.floor(y)):
            raise ValueError("home_goals e away_goals devem ser contagens inteiras não negativas")

        teams = sorted(set(df["home_team"].astype(str)) | set(df["away_team"].astype(str)))
        if len(teams) < 2:
            raise ValueError("são necessários ao menos 2 times distintos")
        index = {team: i for i, team in enumerate(teams)}
        h_idx = df["home_team"].astype(str).map(index).to_numpy(dtype=np.int64)
        a_idx = df["away_team"].astype(str).map(index).to_numpy(dtype=np.int64)
        w = time_decay_weights(df["date"], ref_ts, half_life_days)

        n = len(teams)

        def unpack(theta: FloatArray) -> tuple[float, float, FloatArray, FloatArray, float]:
            mu = float(theta[0])
            gamma = float(theta[1])
            atk_free = theta[2 : 2 + (n - 1)]
            def_free = theta[2 + (n - 1) : 2 + 2 * (n - 1)]
            atk = np.append(atk_free, -float(np.sum(atk_free)))
            dfc = np.append(def_free, -float(np.sum(def_free)))
            return mu, gamma, atk, dfc, float(theta[-1])

        def nll(theta: FloatArray) -> float:
            mu, gamma, atk, dfc, log_l3 = unpack(theta)
            log_l1 = mu + gamma + atk[h_idx] + dfc[a_idx]
            log_l2 = mu + atk[a_idx] + dfc[h_idx]
            ll = log_pmf(x, y, np.exp(log_l1), np.exp(log_l2), math.exp(log_l3))
            return -float(np.sum(w * ll))

        x0 = np.zeros(2 * n + 1, dtype=np.float64)
        x0[0] = math.log(max(float(np.mean(np.concatenate([x, y]))), 0.05))
        x0[1] = 0.25
        x0[-1] = math.log(0.1)
        bounds = (
            [_BOUND_MU, _BOUND_HOME_ADV] + [_BOUND_STRENGTH] * (2 * (n - 1)) + [_BOUND_LOG_LAMBDA3]
        )
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
        theta = np.asarray(res.x, dtype=np.float64)
        mu, gamma, atk, dfc, log_l3 = unpack(theta)

        self.teams_ = teams
        self.attack_ = pd.Series(atk, index=teams, dtype=np.float64)
        self.defence_ = pd.Series(dfc, index=teams, dtype=np.float64)
        self.intercept_ = mu
        self.home_advantage_ = gamma
        self.lambda3_ = math.exp(log_l3)
        self.loglik_ = -float(res.fun)
        self.n_matches_ = len(df)
        self.converged_ = bool(res.success)
        self._fitted = True
        return self

    def match_rates(self, home: str, away: str) -> tuple[float, float, float]:
        """Taxas (lambda1, lambda2, lambda3) da partida.

        Os gols esperados são E[X] = lambda1 + lambda3 e E[Y] = lambda2 +
        lambda3, porque o choque comum W3 entra nas duas contagens.
        """
        self._check_fitted()
        for team in (home, away):
            if team not in self.attack_.index:
                raise ValueError(f"time desconhecido pelo modelo: {team!r}")
        log_l1 = (
            self.intercept_
            + self.home_advantage_
            + float(self.attack_[home])
            + float(self.defence_[away])
        )
        log_l2 = self.intercept_ + float(self.attack_[away]) + float(self.defence_[home])
        return math.exp(log_l1), math.exp(log_l2), self.lambda3_

    def score_matrix(self, home: str, away: str, max_goals: int = 10) -> FloatArray:
        """Matriz P(X=i, Y=j) para i, j em 0..max_goals (linha = gols do mandante).

        A grade é truncada em max_goals; com o padrão de 10 gols a massa
        perdida é desprezível para taxas típicas de futebol. Os derivados
        (1x2, over/under) renormalizam pela massa da grade para que as
        probabilidades reportadas somem exatamente 1.
        """
        if max_goals < 1:
            raise ValueError("max_goals deve ser >= 1")
        l1, l2, l3 = self.match_rates(home, away)
        goals = np.arange(max_goals + 1, dtype=np.float64)
        grid = log_pmf(goals[:, None], goals[None, :], l1, l2, l3)
        return np.asarray(np.exp(grid), dtype=np.float64)

    def probs_1x2(self, home: str, away: str, max_goals: int = 10) -> dict[str, float]:
        """Probabilidades de vitória do mandante, empate e vitória do visitante.

        Vitória do mandante é a massa abaixo da diagonal da matriz de placares
        (X > Y), empate é o traço e vitória do visitante a massa acima. As três
        são normalizadas pela massa total da grade truncada, somando 1.
        """
        m = self.score_matrix(home, away, max_goals=max_goals)
        total = float(m.sum())
        return {
            "home": float(np.tril(m, k=-1).sum()) / total,
            "draw": float(np.trace(m)) / total,
            "away": float(np.triu(m, k=1).sum()) / total,
        }

    def probs_over_under(
        self,
        home: str,
        away: str,
        line: float = 2.5,
        max_goals: int = 10,
    ) -> dict[str, float]:
        """P(X + Y > line) e P(X + Y < line), normalizadas pela grade truncada.

        Para linhas fracionárias (0.5, 1.5, ...) não existe push e over + under
        = 1. Para linhas inteiras a massa exatamente na linha não pertence a
        nenhum dos dois lados, então over + under < 1 (o push fica de fora).
        """
        m = self.score_matrix(home, away, max_goals=max_goals)
        goals = np.arange(max_goals + 1, dtype=np.float64)
        totals = goals[:, None] + goals[None, :]
        total = float(m.sum())
        return {
            "over": float(m[totals > line].sum()) / total,
            "under": float(m[totals < line].sum()) / total,
        }

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("modelo não ajustado; chame fit() antes de prever")
