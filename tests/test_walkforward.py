"""Testes do motor walk-forward, com foco nas garantias anti-leakage."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from edgefinder.backtest.walkforward import (
    WalkForwardResult,
    walk_forward,
    walk_forward_detailed,
)

FitFn = Callable[[pd.DataFrame], dict[str, float]]
PredictFn = Callable[[dict[str, float], pd.DataFrame], pd.DataFrame]


def _synthetic_matches(n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "match_id": np.arange(n),
            "match_date": pd.date_range("2023-08-01", periods=n, freq="D"),
            "home_team": [f"t{i % 12}" for i in range(n)],
            "away_team": [f"t{(i + 5) % 12}" for i in range(n)],
            "elo_diff": rng.normal(0.0, 1.0, n),
            "home_goals": rng.poisson(1.5, n).astype(np.int64),
            "away_goals": rng.poisson(1.1, n).astype(np.int64),
            "xg_home": rng.gamma(2.0, 0.7, n),
        }
    )


def _make_fns(record: dict[str, list[Any]] | None = None) -> tuple[FitFn, PredictFn]:
    """fit/predict instrumentados e deliberadamente 'curiosos'.

    O fit depende dos RESULTADOS do treino (média de gols); o predict tenta
    trapacear lendo home_goals/xg_home do bloco se essas colunas existirem.
    Se o motor vazar futuro para o treino ou esquecer de dropar colunas de
    resultado, as previsões mudam e os testes de leakage falham.
    """

    def fit_fn(train: pd.DataFrame) -> dict[str, float]:
        if record is not None:
            record["fit"].append({"max_date": train["match_date"].max(), "n": len(train)})
        mu = float(train["home_goals"].mean() - train["away_goals"].mean())
        return {"mu": mu}

    def predict_fn(model: dict[str, float], block: pd.DataFrame) -> pd.DataFrame:
        if record is not None:
            record["predict"].append(
                {"min_date": block["match_date"].min(), "columns": set(block.columns)}
            )
        cheat = 0.0
        if "home_goals" in block.columns:
            cheat += float(block["home_goals"].sum())
        if "xg_home" in block.columns:
            cheat += float(block["xg_home"].sum())
        z = model["mu"] + 0.1 * block["elo_diff"].to_numpy(dtype=np.float64) + cheat
        p_home = 1.0 / (1.0 + np.exp(-z))
        p_draw = (1.0 - p_home) * 0.4
        return pd.DataFrame(
            {
                "match_id": block["match_id"].to_numpy(),
                "p_home": p_home,
                "p_draw": p_draw,
                "p_away": 1.0 - p_home - p_draw,
            }
        )

    return fit_fn, predict_fn


def test_happy_path_predicts_after_warmup() -> None:
    matches = _synthetic_matches()
    fit_fn, predict_fn = _make_fns()
    preds = walk_forward(matches, fit_fn, predict_fn, freq="W", min_train=60)

    assert not preds.empty
    assert {"match_id", "p_home", "p_draw", "p_away"} <= set(preds.columns)
    assert preds["match_id"].is_unique
    probs = preds[["p_home", "p_draw", "p_away"]].to_numpy()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-12)
    assert bool((probs > 0).all()) and bool((probs < 1).all())
    # os 60 primeiros jogos nunca têm treino suficiente
    assert preds["match_id"].min() >= 60


def test_result_columns_never_reach_predict_fn() -> None:
    matches = _synthetic_matches()
    record: dict[str, list[Any]] = {"fit": [], "predict": []}
    fit_fn, predict_fn = _make_fns(record)
    walk_forward(matches, fit_fn, predict_fn, freq="W", min_train=60, result_cols=["xg_home"])

    assert record["predict"], "nenhum bloco foi previsto"
    for call in record["predict"]:
        assert "home_goals" not in call["columns"]
        assert "away_goals" not in call["columns"]
        assert "xg_home" not in call["columns"]
        assert {"match_id", "match_date", "elo_diff"} <= call["columns"]


def test_train_is_strictly_before_block() -> None:
    matches = _synthetic_matches()
    record: dict[str, list[Any]] = {"fit": [], "predict": []}
    fit_fn, predict_fn = _make_fns(record)
    walk_forward(matches, fit_fn, predict_fn, freq="W", min_train=60)

    assert len(record["fit"]) == len(record["predict"]) > 0
    for fit_call, pred_call in zip(record["fit"], record["predict"], strict=True):
        assert fit_call["max_date"] < pred_call["min_date"]


def test_min_train_respected_and_expanding_grows() -> None:
    matches = _synthetic_matches()
    record: dict[str, list[Any]] = {"fit": [], "predict": []}
    fit_fn, predict_fn = _make_fns(record)
    walk_forward(matches, fit_fn, predict_fn, freq="W", min_train=60, expanding=True)

    sizes = [c["n"] for c in record["fit"]]
    assert all(n >= 60 for n in sizes)
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


def test_rolling_window_caps_train_size() -> None:
    matches = _synthetic_matches()
    record: dict[str, list[Any]] = {"fit": [], "predict": []}
    fit_fn, predict_fn = _make_fns(record)
    walk_forward(matches, fit_fn, predict_fn, freq="W", min_train=60, expanding=False)

    assert record["fit"]
    assert all(c["n"] == 60 for c in record["fit"])


def test_shuffled_input_gives_same_predictions() -> None:
    matches = _synthetic_matches()
    fit_fn, predict_fn = _make_fns()
    preds_a = walk_forward(matches, fit_fn, predict_fn, freq="W", min_train=60)
    shuffled = matches.sample(frac=1.0, random_state=3).reset_index(drop=True)
    preds_b = walk_forward(shuffled, fit_fn, predict_fn, freq="W", min_train=60)

    a = preds_a.sort_values("match_id").reset_index(drop=True)
    b = preds_b.sort_values("match_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_leakage_future_results_do_not_change_predictions() -> None:
    """Teste anti-leakage exigido pelo briefing.

    Para cada bloco do walk-forward, envenena TODOS os resultados a partir
    do início do bloco (home_goals += 5 etc.) e exige que as previsões até
    o fim desse bloco fiquem bit a bit idênticas: o treino só viu o passado
    estrito e o frame de predição não carrega colunas de resultado. Se
    qualquer feature usar o futuro, a comparação exata falha.
    """
    base = _synthetic_matches()
    fit_fn, predict_fn = _make_fns()
    kwargs: dict[str, Any] = {"freq": "W", "min_train": 60, "result_cols": ["xg_home"]}

    detail = walk_forward_detailed(base, fit_fn, predict_fn, **kwargs)
    preds_base = detail.predictions
    assert len(detail.steps) >= 5

    date_by_id = base.set_index("match_id")["match_date"]

    def _subset(preds: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
        keep = preds["match_id"].map(date_by_id) <= cutoff
        return preds[keep].sort_values("match_id").reset_index(drop=True)

    changed_after_somewhere = False
    for step in detail.steps:
        cut = step.period_start
        poisoned = base.copy()
        future = poisoned["match_date"] >= cut
        poisoned.loc[future, "home_goals"] += 5
        poisoned.loc[future, "away_goals"] += 7
        poisoned.loc[future, "xg_home"] += 3.0

        preds_pois = walk_forward(poisoned, fit_fn, predict_fn, **kwargs)

        # até o fim do bloco que começa em `cut`, nada pode mudar:
        # o treino é estritamente anterior e os resultados do próprio
        # bloco foram dropados antes do predict.
        pd.testing.assert_frame_equal(
            _subset(preds_base, step.period_end),
            _subset(preds_pois, step.period_end),
            check_exact=True,
        )

        after_base = preds_base[preds_base["match_id"].map(date_by_id) > step.period_end]
        after_pois = preds_pois[preds_pois["match_id"].map(date_by_id) > step.period_end]
        if len(after_base) > 0 and not after_base.reset_index(drop=True).equals(
            after_pois.reset_index(drop=True)
        ):
            changed_after_somewhere = True

    # sanidade do próprio teste: o veneno precisa afetar blocos seguintes
    # (via treino), senão o teste não teria poder para detectar leakage.
    assert changed_after_somewhere


def test_empty_and_insufficient_data() -> None:
    fit_fn, predict_fn = _make_fns()
    empty = pd.DataFrame(columns=["match_id", "match_date", "home_goals", "away_goals"])
    assert walk_forward(empty, fit_fn, predict_fn).empty

    small = _synthetic_matches(n=30)
    result = walk_forward_detailed(small, fit_fn, predict_fn, min_train=380)
    assert isinstance(result, WalkForwardResult)
    assert result.steps == []
    assert result.predictions.empty


def test_step_results_are_recorded() -> None:
    matches = _synthetic_matches()
    fit_fn, predict_fn = _make_fns()
    result = walk_forward_detailed(matches, fit_fn, predict_fn, freq="W", min_train=60)

    assert result.steps
    for step in result.steps:
        assert step.period_start <= step.period_end
        assert step.n_train >= 60
        assert not step.predictions.empty
    starts = [s.period_start for s in result.steps]
    assert starts == sorted(starts)
    total = sum(len(s.predictions) for s in result.steps)
    assert total == len(result.predictions)


def test_invalid_inputs_raise() -> None:
    fit_fn, predict_fn = _make_fns()
    no_date = pd.DataFrame({"match_id": [1], "home_goals": [1], "away_goals": [0]})
    with pytest.raises(ValueError, match="match_date"):
        walk_forward(no_date, fit_fn, predict_fn)
    with pytest.raises(ValueError, match="min_train"):
        walk_forward(_synthetic_matches(), fit_fn, predict_fn, min_train=0)
