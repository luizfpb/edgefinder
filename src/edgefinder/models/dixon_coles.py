"""Modelo de Dixon-Coles (1997) para placares de futebol.

O placar (X, Y) parte de dois Poissons X ~ Poisson(lambda) e Y ~ Poisson(mu)
com taxas log-lineares por time: para o time i em casa contra o time j,

    lambda = exp(atk_i - def_j + home_adv)
    mu     = exp(atk_j - def_i)

onde atk mede a força de ataque e def a força de defesa (def maior = sofre
menos gols, por isso entra com sinal negativo). A contribuição central do
artigo é a correção de dependência tau nos placares baixos {0,1} x {0,1},
região em que o produto de Poissons independentes erra sistematicamente:

    tau(0,0) = 1 - lambda * mu * rho
    tau(0,1) = 1 + lambda * rho
    tau(1,0) = 1 + mu * rho
    tau(1,1) = 1 - rho
    tau(x,y) = 1   nos demais placares.

rho < 0 realoca massa dos placares 1-0 e 0-1 para 0-0 e 1-1 (mais empates
magros), que é o padrão empírico do futebol. A verossimilhança é ponderada
por recência com w = exp(-xi * dt), xi = ln(2) / half_life_days, para que a
forma atual dos times pese mais que a de temporadas passadas.

Identificação: atk e def só entram via diferenças, então o modelo é
invariante à translação (atk + c, def + c). As restrições média(atk) = 0 e
média(def) = 0 — impostas por reparametrização: n-1 coeficientes livres por
vetor, o último é o negativo da soma dos demais — removem essa invariância
e ancoram a escala: um visitante médio marca exp(0) = 1 gol esperado e um
mandante médio, exp(home_adv).

Referência: Dixon, M. J. e Coles, S. G. (1997), "Modelling association
football scores and inefficiencies in the football betting market",
Journal of the Royal Statistical Society C, 46(2), 265-280.
"""

from __future__ import annotations

import math
from datetime import date, datetime

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

__all__ = ["RHO_BOUNDS", "DixonColes", "tau_correction"]

FloatArray = npt.NDArray[np.float64]
DateLike = str | date | datetime | pd.Timestamp

_REQUIRED_COLUMNS: tuple[str, ...] = (
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "match_date",
)

# tau só é uma pmf válida se permanecer positivo, o que exige
# |rho| < min(1, 1/lambda, 1/mu, 1/(lambda*mu)). Em ligas reais o rho
# estimado fica tipicamente em [-0.15, 0], então a caixa [-0.2, 0.2] cobre
# qualquer valor plausível e mantém o otimizador longe da região instável.
# Qualquer tau residualmente não positivo durante a busca de linha (possível
# com lambda*mu grande) é assoalhado em _TAU_FLOOR antes do log.
RHO_BOUNDS: tuple[float, float] = (-0.2, 0.2)
_TAU_FLOOR: float = 1e-10

# Caixas amplas para forças e vantagem de casa: não restringem nenhuma liga
# real, só impedem que exp() estoure durante a busca de linha do L-BFGS-B.
_BOUND_STRENGTH: tuple[float, float] = (-5.0, 5.0)
_BOUND_HOME_ADV: tuple[float, float] = (-3.0, 3.0)


def tau_correction(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    lam: npt.ArrayLike,
    mu: npt.ArrayLike,
    rho: float,
) -> FloatArray:
    """Fator tau de Dixon-Coles, vetorizado por broadcasting.

    Implementa exatamente a tabela do artigo:

        tau(0,0) = 1 - lam*mu*rho;  tau(0,1) = 1 + lam*rho
        tau(1,0) = 1 + mu*rho;      tau(1,1) = 1 - rho
        tau(x,y) = 1  para os demais placares.

    O valor bruto é retornado sem recorte: pode ser <= 0 se rho estiver fora
    da faixa estável para as taxas dadas. Quem consome decide o recorte
    (a verossimilhança assoalha antes do log; a matriz de placares zera).
    """
    xb, yb, lb, mb = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(lam, dtype=np.float64),
        np.asarray(mu, dtype=np.float64),
    )
    out = np.ones(xb.shape, dtype=np.float64)
    out = np.where((xb == 0) & (yb == 0), 1.0 - lb * mb * rho, out)
    out = np.where((xb == 0) & (yb == 1), 1.0 + lb * rho, out)
    out = np.where((xb == 1) & (yb == 0), 1.0 + mb * rho, out)
    out = np.where((xb == 1) & (yb == 1), 1.0 - rho, out)
    return np.asarray(out, dtype=np.float64)


def _time_decay_weights(
    dates: pd.Series[pd.Timestamp],
    ref_date: pd.Timestamp,
    half_life_days: float,
) -> FloatArray:
    """Pesos de decaimento exponencial: w = exp(-xi * dt), xi = ln(2) / half_life.

    dt é a distância em dias entre a partida e ref_date, de modo que uma
    partida a half_life_days de distância pesa exatamente 0.5 e uma partida
    em ref_date pesa 1. O chamador já filtrou partidas futuras.
    """
    if math.isnan(half_life_days) or half_life_days <= 0:
        raise ValueError("half_life_days deve ser positivo")
    dt_days = (ref_date - dates).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
    xi = math.log(2.0) / half_life_days
    return np.asarray(np.exp(-xi * dt_days), dtype=np.float64)


class DixonColes:
    """Forças de ataque/defesa por time com correção tau de placares baixos.

    Após ``fit``, os parâmetros ficam em ``attack_``, ``defence_``,
    ``home_advantage_`` e ``rho_``; o vetor bruto do otimizador fica em
    ``params_`` e pode ser reinjetado via ``fit(..., init_params=...)``
    para acelerar reajustes em walk-forward (warm start).
    """

    def __init__(self) -> None:
        self.teams_: list[str] = []
        self.attack_: pd.Series[float] = pd.Series(dtype=np.float64)
        self.defence_: pd.Series[float] = pd.Series(dtype=np.float64)
        self.home_advantage_: float = 0.0
        self.rho_: float = 0.0
        self.params_: FloatArray = np.empty(0, dtype=np.float64)
        self.loglik_: float = float("nan")
        self.n_matches_: int = 0
        self.converged_: bool = False
        self._fitted: bool = False

    def fit(
        self,
        matches: pd.DataFrame,
        ref_date: DateLike,
        half_life_days: float = 180.0,
        init_params: npt.ArrayLike | None = None,
    ) -> DixonColes:
        """Ajusta (atk, def, home_adv, rho) por máxima verossimilhança ponderada.

        A função objetivo é a log-verossimilhança ponderada por recência

            log L = sum_t w_t * [ log tau(x_t, y_t; lambda_t, mu_t, rho)
                                  + x_t log lambda_t - lambda_t
                                  + y_t log mu_t - mu_t
                                  - log x_t! - log y_t! ]

        com w_t = exp(-xi * dt_t) e xi = ln(2) / half_life_days. Os termos
        log x! são constantes em theta e entram só para que ``loglik_`` seja
        a verossimilhança verdadeira (comparável entre modelos). A avaliação
        é inteiramente vetorizada: taxas, tau e pesos são arrays por jogo.

        rho é restrito a RHO_BOUNDS = [-0.2, 0.2] (ver comentário no topo do
        módulo): fora dessa faixa o tau pode ficar não positivo e a pmf deixa
        de ser válida. Partidas posteriores a ref_date e partidas sem
        resultado são descartadas antes do ajuste, o que impede vazamento de
        futuro por construção.

        Warm start: passe ``init_params=modelo_anterior.params_`` para
        iniciar o L-BFGS-B do ótimo anterior. O vetor só é aproveitado se o
        comprimento bater com o conjunto de times atual (2*(n-1) + 2, na
        ordem: atk livres, def livres, home_adv, rho); caso contrário cai
        silenciosamente no chute padrão — em walk-forward o pool de times
        muda em viradas de temporada e abortar o ajuste seria pior que
        recomeçar frio.
        """
        missing = [c for c in _REQUIRED_COLUMNS if c not in matches.columns]
        if missing:
            raise ValueError(f"colunas ausentes em matches: {missing}")
        df = matches.loc[:, list(_REQUIRED_COLUMNS)].copy()
        df["match_date"] = pd.to_datetime(df["match_date"])
        df = df.dropna(subset=["match_date", "home_goals", "away_goals"])
        ref_ts = pd.Timestamp(ref_date)
        df = df[df["match_date"] <= ref_ts]
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
        w = _time_decay_weights(df["match_date"], ref_ts, half_life_days)

        n = len(teams)
        n_params = 2 * (n - 1) + 2

        # Máscaras dos placares baixos, fixas ao longo da otimização: o tau
        # só difere de 1 nesses quatro placares, então o log tau é calculado
        # apenas onde importa.
        m00 = (x == 0) & (y == 0)
        m01 = (x == 0) & (y == 1)
        m10 = (x == 1) & (y == 0)
        m11 = (x == 1) & (y == 1)
        # Constante da pmf de Poisson (-log x! - log y!), ponderada uma vez.
        log_fact = -float(np.sum(w * (gammaln(x + 1.0) + gammaln(y + 1.0))))

        def unpack(theta: FloatArray) -> tuple[FloatArray, FloatArray, float, float]:
            atk_free = theta[: n - 1]
            def_free = theta[n - 1 : 2 * (n - 1)]
            atk = np.concatenate([atk_free, [-float(np.sum(atk_free))]])
            dfn = np.concatenate([def_free, [-float(np.sum(def_free))]])
            return atk, dfn, float(theta[-2]), float(theta[-1])

        def nll(theta: FloatArray) -> float:
            atk, dfn, home_adv, rho = unpack(theta)
            log_lam = atk[h_idx] - dfn[a_idx] + home_adv
            log_mu = atk[a_idx] - dfn[h_idx]
            lam = np.exp(log_lam)
            mu = np.exp(log_mu)
            tau = np.ones_like(lam)
            tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
            tau[m01] = 1.0 + lam[m01] * rho
            tau[m10] = 1.0 + mu[m10] * rho
            tau[m11] = 1.0 - rho
            ll = np.log(np.maximum(tau, _TAU_FLOOR)) + x * log_lam - lam + y * log_mu - mu
            return -float(np.sum(w * ll))

        bounds: list[tuple[float, float]] = [_BOUND_STRENGTH] * (2 * (n - 1)) + [
            _BOUND_HOME_ADV,
            RHO_BOUNDS,
        ]
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        x0 = np.zeros(n_params, dtype=np.float64)
        x0[-2] = 0.25
        if init_params is not None:
            arr = np.asarray(init_params, dtype=np.float64).ravel()
            if arr.shape[0] == n_params and bool(np.all(np.isfinite(arr))):
                x0 = np.clip(arr, lo, hi)

        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
        theta = np.asarray(res.x, dtype=np.float64)
        atk, dfn, home_adv, rho = unpack(theta)

        self.teams_ = teams
        self.attack_ = pd.Series(atk, index=teams, dtype=np.float64)
        self.defence_ = pd.Series(dfn, index=teams, dtype=np.float64)
        self.home_advantage_ = home_adv
        self.rho_ = rho
        self.params_ = theta
        self.loglik_ = -float(res.fun) + log_fact
        self.n_matches_ = len(df)
        self.converged_ = bool(res.success)
        self._fitted = True
        return self

    def match_rates(self, home: str, away: str) -> tuple[float, float]:
        """Taxas (lambda, mu) da partida: gols esperados do mandante e do visitante.

        lambda = exp(atk_home - def_away + home_adv) e
        mu = exp(atk_away - def_home). Times não vistos no treino levantam
        KeyError com o nome do time.
        """
        self._check_fitted()
        for team in (home, away):
            if team not in self.attack_.index:
                raise KeyError(f"time desconhecido pelo modelo: {team!r} (não presente no treino)")
        log_lam = float(self.attack_[home]) - float(self.defence_[away]) + self.home_advantage_
        log_mu = float(self.attack_[away]) - float(self.defence_[home])
        return math.exp(log_lam), math.exp(log_mu)

    def score_matrix(self, home: str, away: str, max_goals: int = 10) -> FloatArray:
        """Matriz P(X=i, Y=j) para i, j em 0..max_goals (linha = gols do mandante).

        Cada célula parte do produto de Poissons independentes

            P(i, j) = tau(i, j) * e^{-lambda} lambda^i / i! * e^{-mu} mu^j / j!

        com o tau aplicado às quatro células {0,1} x {0,1}. Células com tau
        negativo (só possível com taxas extremas) são zeradas, e a matriz é
        renormalizada para somar exatamente 1: isso absorve tanto o efeito
        líquido do tau (que não preserva massa exatamente) quanto a cauda
        truncada além de max_goals.
        """
        if max_goals < 1:
            raise ValueError("max_goals deve ser >= 1")
        lam, mu = self.match_rates(home, away)
        goals = np.arange(max_goals + 1, dtype=np.float64)
        log_px = goals * math.log(lam) - lam - gammaln(goals + 1.0)
        log_py = goals * math.log(mu) - mu - gammaln(goals + 1.0)
        m = np.exp(log_px)[:, None] * np.exp(log_py)[None, :]
        tau = tau_correction(goals[:, None], goals[None, :], lam, mu, self.rho_)
        m = m * np.maximum(tau, 0.0)
        total = float(m.sum())
        if total <= 0.0:
            raise ValueError("matriz de placares degenerou; verifique as taxas do modelo")
        return np.asarray(m / total, dtype=np.float64)

    def probs_1x2(self, home: str, away: str, max_goals: int = 10) -> tuple[float, float, float]:
        """Probabilidades (p_home, p_draw, p_away) do mercado 1X2.

        Da matriz de placares M (que já soma 1):

            p_home = sum_{i > j} M[i, j]   (massa abaixo da diagonal)
            p_draw = sum_{i = j} M[i, i]   (traço)
            p_away = sum_{i < j} M[i, j]   (massa acima da diagonal)
        """
        m = self.score_matrix(home, away, max_goals=max_goals)
        return (
            float(np.tril(m, k=-1).sum()),
            float(np.trace(m)),
            float(np.triu(m, k=1).sum()),
        )

    def prob_over(self, home: str, away: str, line: float = 2.5, max_goals: int = 10) -> float:
        """P(X + Y > line) na matriz de placares.

        Para linhas fracionárias (2.5, 3.5, ...) vale over + under = 1. Para
        linhas inteiras a massa exatamente na linha é push e não pertence a
        nenhum lado, então over + under < 1.
        """
        m = self.score_matrix(home, away, max_goals=max_goals)
        totals = self._totals_grid(max_goals)
        return float(m[totals > line].sum())

    def prob_under(self, home: str, away: str, line: float = 2.5, max_goals: int = 10) -> float:
        """P(X + Y < line) na matriz de placares (complemento de prob_over em linhas .5)."""
        m = self.score_matrix(home, away, max_goals=max_goals)
        totals = self._totals_grid(max_goals)
        return float(m[totals < line].sum())

    def prob_ah(
        self,
        home: str,
        away: str,
        handicap: float,
        max_goals: int = 10,
    ) -> tuple[float, float, float]:
        """Handicap asiático do mandante: (p_home_win_bet, p_push, p_away_win_bet).

        O handicap h é somado aos gols do mandante e a aposta é decidida pela
        margem ajustada d = X + h - Y:

            p_home_win_bet = P(d > 0)
            p_push         = P(d = 0)   (só possível em linhas inteiras)
            p_away_win_bet = P(d < 0)

        Em linhas .5 a margem ajustada nunca zera, então p_push = 0 e
        p_home + p_away = 1. Em linhas inteiras (0, -1, +2, ...) o empate na
        linha devolve a aposta (push) e as três probabilidades somam 1.
        Linhas de quarto (0.25, -0.75, ...) são a média de duas apostas e
        devem ser decompostas pelo chamador; aqui levantam ValueError.
        """
        if not float(2.0 * handicap).is_integer():
            raise ValueError(
                "handicap deve ser linha inteira ou meia (múltiplo de 0.5); "
                "decomponha linhas de quarto em duas metades"
            )
        m = self.score_matrix(home, away, max_goals=max_goals)
        goals = np.arange(max_goals + 1, dtype=np.float64)
        margin = goals[:, None] + handicap - goals[None, :]
        p_home = float(m[margin > 1e-9].sum())
        p_push = float(m[np.abs(margin) <= 1e-9].sum())
        p_away = float(m[margin < -1e-9].sum())
        return p_home, p_push, p_away

    @staticmethod
    def _totals_grid(max_goals: int) -> FloatArray:
        """Grade i + j do total de gols, alinhada com score_matrix."""
        goals = np.arange(max_goals + 1, dtype=np.float64)
        return np.asarray(goals[:, None] + goals[None, :], dtype=np.float64)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("modelo não ajustado; chame fit() antes de prever")
