"""Orquestra o backtest walk-forward de ponta a ponta e grava os artefatos.

Protocolo honesto (o desenho importa mais que o código):

- Treino em [t0, t), previsão do bloco [t, t+freq). Zero acesso ao futuro —
  o motor genérico (walkforward.py) droppa colunas de resultado do frame de
  predição e tem teste que falha se o futuro vazar.
- A aposta simulada é feita À ODD DE FECHAMENTO real (não à de abertura):
  é o cenário conservador — sem assumir que pegamos linha melhor que o
  fechamento. O benchmark de probabilidade é o fechamento DE-VIGADO (Shin).
- O sistema aposta apenas quando EV > threshold do tier e dimensiona com
  Kelly fracionário. Se o resultado agregado não for significativo, o
  relatório diz isso em destaque — não existe maquiagem de métrica aqui.

Artefatos em data/reports/: predictions_<comp>.parquet, backtest_summary.json.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import structlog
from sqlalchemy import Engine

from edgefinder.backtest.data import closing_odds_1x2, matches_frame
from edgefinder.backtest.metrics import prob_metrics, significance, summary
from edgefinder.backtest.monte_carlo import simulate_bankroll
from edgefinder.backtest.walkforward import walk_forward
from edgefinder.config import COMPETITION_TIERS, settings
from edgefinder.edge.ev import expected_value
from edgefinder.edge.kelly import kelly_stake
from edgefinder.market.devig import devig
from edgefinder.models.dixon_coles import DixonColes

log = structlog.get_logger()

MIN_EV_BY_TIER = {1: "min_ev_tier1", 2: "min_ev_tier2", 3: "min_ev_tier3"}


@dataclass
class BacktestConfig:
    competitions: list[str]
    half_life_days: float = 180.0
    freq: str = "W"
    # >= 2 temporadas de treino mínimo: com menos que isso o Dixon-Coles produz
    # parâmetros ruidosos e "edges" gigantes que são erro de estimação, não valor.
    min_train: int = 760
    # start_season delimita o PERÍODO DE APOSTA/AVALIAÇÃO; o treino sempre usa
    # toda a história anterior disponível (filtrar o treino aqui seria bug).
    start_season: str = "1920"
    kelly_cap: float = 0.05
    devig_method: str = "shin"
    # EV acima disso é tratado como erro do modelo, não como valor: nenhum
    # modelo simples tem 15%+ de edge real contra closing line de liga líquida.
    max_ev: float = 0.15


def _make_fit_predict(
    half_life_days: float,
) -> tuple[Any, Any]:
    """fit/predict para o walk-forward, com warm start entre blocos."""
    state: dict[str, Any] = {"prev": None}

    def fit_fn(past: pd.DataFrame) -> DixonColes:
        model = DixonColes()
        ref = past["match_date"].max()
        try:
            model.fit(
                past,
                ref_date=ref,
                half_life_days=half_life_days,
                init_params=state["prev"],
            )
        except TypeError:  # implementação sem warm start
            model.fit(past, ref_date=ref, half_life_days=half_life_days)
        state["prev"] = getattr(model, "params_", None)
        return model

    def predict_fn(model: DixonColes, block: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in block.iterrows():
            try:
                p_home, p_draw, p_away = model.probs_1x2(row["home_team"], row["away_team"])
            except KeyError:
                continue  # time recém-promovido, sem histórico no treino
            rows.append(
                {
                    "match_id": row["match_id"],
                    "p_home": p_home,
                    "p_draw": p_draw,
                    "p_away": p_away,
                }
            )
        return pd.DataFrame(rows)

    return fit_fn, predict_fn


def run_1x2_backtest(engine: Engine, config: BacktestConfig) -> dict[str, Any]:
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    all_bets: list[pd.DataFrame] = []
    report: dict[str, Any] = {"config": config.__dict__ | {"ran_at": datetime.now().isoformat()}}
    leagues_out: dict[str, Any] = {}

    for comp in config.competitions:
        matches = matches_frame(engine, [comp], min_season=None)
        odds = closing_odds_1x2(engine, [comp])
        if matches.empty or odds.empty:
            log.warning("backtest.sem_dados", competition=comp)
            continue

        fit_fn, predict_fn = _make_fit_predict(config.half_life_days)
        preds = walk_forward(
            matches,
            fit_fn=fit_fn,
            predict_fn=predict_fn,
            freq=config.freq,
            min_train=config.min_train,
            result_cols=["home_goals", "away_goals"],
        )
        if preds.empty:
            log.warning("backtest.sem_previsoes", competition=comp)
            continue

        df = preds.merge(matches, on="match_id").merge(odds, on="match_id", how="inner")
        df = df[df["season"] >= config.start_season].reset_index(drop=True)
        if df.empty:
            log.warning("backtest.sem_jogos_no_periodo", competition=comp)
            continue
        df = _evaluate_1x2(df, comp, config)
        df.to_parquet(settings.reports_dir / f"predictions_{comp.replace(' ', '_')}.parquet")

        bets = df[df["bet_selection"].notna()].copy()
        all_bets.append(bets)
        leagues_out[comp] = _league_summary(df, bets)
        log.info("backtest.liga_ok", competition=comp, n_pred=len(df), n_bets=len(bets))

    if not all_bets:
        report["verdict"] = "SEM DADOS SUFICIENTES"
        return report

    bets_df = pd.concat(all_bets, ignore_index=True)
    report["leagues"] = leagues_out
    report["overall"] = _overall_summary(bets_df)
    report["verdict"] = _verdict(report["overall"])
    (settings.reports_dir / "backtest_summary.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    bets_df.to_parquet(settings.reports_dir / "bets_all.parquet")
    return report


def _evaluate_1x2(df: pd.DataFrame, comp: str, config: BacktestConfig) -> pd.DataFrame:
    """De-vig do fechamento, EV, decisão de aposta e liquidação por linha."""
    tier = COMPETITION_TIERS.get(comp, 2)
    min_ev = getattr(settings, MIN_EV_BY_TIER[tier])

    probs_market = np.vstack(
        [
            devig(np.array([r.odds_home, r.odds_draw, r.odds_away]), method=config.devig_method)
            for r in df.itertuples()
        ]
    )
    df[["mkt_home", "mkt_draw", "mkt_away"]] = probs_market

    outcome = np.where(
        df["home_goals"] > df["away_goals"],
        "home",
        np.where(df["home_goals"] < df["away_goals"], "away", "draw"),
    )
    df["outcome"] = outcome

    selections = ("home", "draw", "away")
    p_arr = df[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    o_arr = df[["odds_home", "odds_draw", "odds_away"]].to_numpy(dtype=float)
    outcomes = df["outcome"].to_numpy()

    sel_data: list[tuple[str | None, float, float, float, float, float]] = []
    for i in range(len(df)):
        evs = [float(expected_value(p_arr[i, j], o_arr[i, j])) for j in range(3)]
        j_best = int(np.argmax(evs))
        best_ev = evs[j_best]
        if best_ev < min_ev or best_ev > config.max_ev:
            sel_data.append((None, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        p, odds = float(p_arr[i, j_best]), float(o_arr[i, j_best])
        stake = float(
            kelly_stake(
                p, odds, bankroll=1.0, fraction=settings.kelly_fraction, cap=config.kelly_cap
            )
        )
        won = selections[j_best] == outcomes[i]
        pnl = stake * (odds - 1.0) if won else -stake
        sel_data.append((selections[j_best], best_ev, odds, p, stake, pnl))
    df[["bet_selection", "bet_ev", "bet_odds", "p_bet", "stake", "pnl"]] = pd.DataFrame(
        sel_data, index=df.index
    )
    result = pd.Series(None, index=df.index, dtype=object)
    placed = df["bet_selection"].notna()
    result[placed] = np.where(
        df.loc[placed, "bet_selection"] == df.loc[placed, "outcome"], "win", "lose"
    )
    df["result"] = result
    return df


def _league_summary(df: pd.DataFrame, bets: pd.DataFrame) -> dict[str, Any]:
    y_true = np.asarray((df["outcome"] == "home").to_numpy(), dtype=np.float64)
    model_ll = prob_metrics(y_true, df["p_home"].to_numpy(dtype=np.float64))
    market_ll = prob_metrics(y_true, df["mkt_home"].to_numpy(dtype=np.float64))
    out: dict[str, Any] = {
        "n_predictions": len(df),
        "model_logloss_home": model_ll["log_loss"],
        "market_logloss_home": market_ll["log_loss"],
        "model_brier_home": model_ll["brier"],
        "market_brier_home": market_ll["brier"],
        "model_beats_market_logloss": bool(model_ll["log_loss"] < market_ll["log_loss"]),
    }
    if not bets.empty:
        out["bets"] = summary(bets)
    return out


def _overall_summary(bets: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = dict(summary(bets))
    yields = np.where(bets["result"] == "win", bets["bet_odds"] - 1.0, -1.0)
    out["significance"] = significance(yields)
    stakes = bets["stake"].to_numpy(dtype=float)
    mask = stakes > 0
    out["monte_carlo"] = simulate_bankroll(
        p_win=bets["p_bet"].to_numpy(dtype=float)[mask],
        odds=bets["bet_odds"].to_numpy(dtype=float)[mask],
        stakes_frac=stakes[mask],
        n_paths=10_000,
        seed=42,
    )
    return out


def _verdict(overall: dict[str, Any]) -> str:
    sig = overall.get("significance", {})
    p_value = sig.get("p_value", 1.0)
    yield_mean = sig.get("mean", overall.get("yield_per_bet", 0.0))
    if yield_mean <= 0:
        return (
            "O MODELO NAO BATE O MERCADO: yield <= 0 no backtest. "
            "NAO use este sistema para apostar."
        )
    if p_value > 0.05:
        return (
            "INCONCLUSIVO: yield positivo mas NAO significativo "
            f"(p={p_value:.3f}). Indistinguivel de sorte. Nao aposte com base nisso."
        )
    return (
        f"Yield positivo e significativo (p={p_value:.3f}) NESTE backtest. "
        "Isso ainda nao garante o futuro; monitore CLV prospectivo."
    )
