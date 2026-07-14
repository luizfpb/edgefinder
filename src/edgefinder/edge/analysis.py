"""Análise de apostas defensáveis — distinta de "edge comprovado".

A trava de honestidade (`backtest_gate`) continua valendo para o comando
`edges`: nenhum modelo nosso provou bater o mercado, então o sistema não
emite recomendação validada. Este módulo responde uma pergunta mais modesta
e MUITO mais honesta: dado que você VAI apostar, o que é defensável?

Três lentes, todas com regras transparentes:

1. **Preço vs. consenso** (a única vantagem objetiva disponível): se a melhor
   odd entre as casas está ACIMA da odd justa do consenso de-vigado (Shin),
   você está comprando melhor que o preço médio do mercado — line shopping.
   ev_consenso = melhor_odd * p_consenso - 1.
2. **Modelo** (rotulado com o próprio status): Dixon-Coles para ligas de clube
   (REPROVADO no backtest — serve de sinal fraco, não de verdade) e Elo de
   seleções para torneios internacionais (NUNCA validado — Tier 3).
3. **Forma recente** (o "bateu X dos últimos Y" do grupo, com o tamanho da
   amostra na cara).

Veredicto por seleção:
- "defensavel": ev_consenso >= 0 e o modelo não grita contra (ev_modelo > -10%);
- "neutro": pagando até 3% acima do justo do consenso;
- "evitar": pagando caro (>3% acima do justo) ou modelo fortemente contra.
"""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import structlog
from sqlalchemy import Engine

from edgefinder.config import settings
from edgefinder.ingest.eloratings import read_national_elo
from edgefinder.market.devig import devig
from edgefinder.storage.repository import read_df

log = structlog.get_logger()

# Códigos do eloratings.net verificados no TSV. Cobertura parcial e deliberada:
# seleção fora do mapa fica sem modelo Elo (e a análise diz isso), nunca com
# um palpite de código errado.
ELO_CODES: dict[str, str] = {
    "france": "FR",
    "spain": "ES",
    "england": "EN",
    "argentina": "AR",
    "brazil": "BR",
    "portugal": "PT",
    "netherlands": "NL",
    "germany": "GE",
    "italy": "IT",
    "uruguay": "UY",
    "colombia": "CO",
    "mexico": "MX",
    "united states": "US",
    "usa": "US",
    "croatia": "HR",
    "morocco": "MA",
    "belgium": "BE",
    "japan": "JP",
    "norway": "NO",
    "switzerland": "SZ",
}

_SELECTIONS = ("home", "draw", "away")


def _upcoming_odds(engine: Engine) -> pd.DataFrame:
    sql = """
        SELECT o.match_id, o.bookmaker, o.market, o.selection, o.line,
               o.odds_decimal, o.collected_at, o.commence_time,
               m.competition_id, th.name AS home_team, ta.name AS away_team
        FROM odds_snapshots o
        JOIN matches m ON m.id = o.match_id
        JOIN teams th ON th.id = m.home_team_id
        JOIN teams ta ON ta.id = m.away_team_id
        WHERE o.source = 'theoddsapi' AND o.commence_time > :now
    """
    df = read_df(engine, sql, {"now": datetime.now().isoformat(sep=" ")})
    if df.empty:
        return df
    df["collected_at"] = pd.to_datetime(df["collected_at"])
    return (
        df.sort_values("collected_at")
        .groupby(["match_id", "bookmaker", "market", "selection", "line"], as_index=False)
        .last()
    )


def _elo_probs_1x2(elo_home: float, elo_away: float) -> tuple[float, float, float]:
    """1X2 de 90 min a partir do Elo em campo neutro.

    Expectativa Elo (empate vale 0.5): We = 1/(10^(-dr/400) + 1).
    Empate aproximado: p_E = 0.27·exp(-(dr/600)^2) — ~27% em jogo parelho,
    caindo com a diferença de força. p_casa = We - p_E/2.
    """
    dr = elo_home - elo_away
    we = 1.0 / (10.0 ** (-dr / 400.0) + 1.0)
    p_draw = 0.27 * float(np.exp(-((dr / 600.0) ** 2)))
    p_home = we - p_draw / 2.0
    return p_home, p_draw, 1.0 - p_home - p_draw


def _team_form(engine: Engine, team: str, n: int = 6) -> str:
    sql = """
        SELECT th.name AS home, ta.name AS away, m.home_goals, m.away_goals
        FROM matches m
        JOIN teams th ON th.id = m.home_team_id
        JOIN teams ta ON ta.id = m.away_team_id
        WHERE (th.name = :t OR ta.name = :t) AND m.home_goals IS NOT NULL
        ORDER BY m.match_date DESC LIMIT :n
    """
    df = read_df(engine, sql, {"t": team, "n": n})
    if df.empty:
        return "sem historico no banco"
    mine = np.where(df["home"] == team, df["home_goals"], df["away_goals"])
    theirs = np.where(df["home"] == team, df["away_goals"], df["home_goals"])
    total = df["home_goals"] + df["away_goals"]
    wins = int((mine > theirs).sum())
    over = int((total > 2.5).sum())
    return (
        f"{wins}V em {len(df)} jogos, {int(mine.sum())}GP/{int(theirs.sum())}GC, "
        f"over 2.5 em {over}/{len(df)}"
    )


def _model_probs(
    engine: Engine, competition: str, home: str, away: str, elo: pd.DataFrame
) -> tuple[tuple[float, float, float] | None, str]:
    """Probabilidades do modelo disponível para a competição, com rótulo honesto."""
    if competition.startswith("INT-"):
        codes = dict(zip(elo["country_code"], elo["elo"], strict=False))
        code_h, code_a = ELO_CODES.get(home), ELO_CODES.get(away)
        if code_h in codes and code_a in codes:
            probs = _elo_probs_1x2(float(codes[code_h]), float(codes[code_a]))
            return probs, "elo-selecoes (Tier 3, nunca validado)"
        return None, "sem modelo (selecao fora do mapa Elo)"
    matches = read_df(
        engine,
        """
        SELECT th.name AS home_team, ta.name AS away_team, m.home_goals,
               m.away_goals, m.match_date
        FROM matches m
        JOIN teams th ON th.id = m.home_team_id
        JOIN teams ta ON ta.id = m.away_team_id
        WHERE m.competition_id = :c AND m.home_goals IS NOT NULL
        """,
        {"c": competition},
    )
    if len(matches) < 760:
        return None, "sem modelo (historico insuficiente)"
    from edgefinder.models.dixon_coles import DixonColes

    matches["match_date"] = pd.to_datetime(matches["match_date"])
    model = DixonColes().fit(matches, ref_date=matches["match_date"].max(), half_life_days=365.0)
    try:
        return model.probs_1x2(home, away), "dixon-coles (REPROVADO no backtest 1X2)"
    except KeyError:
        return None, "sem modelo (time sem historico)"


def analyze_today(engine: Engine) -> pd.DataFrame:
    """Analisa todos os jogos futuros com odds coletadas. Grava o artefato."""
    odds = _upcoming_odds(engine)
    if odds.empty:
        return pd.DataFrame()
    elo = read_national_elo()

    rows: list[dict[str, Any]] = []
    for match_id, ev in odds.groupby("match_id"):
        home = str(ev["home_team"].iloc[0])
        away = str(ev["away_team"].iloc[0])
        comp = str(ev["competition_id"].iloc[0])
        kickoff = ev["commence_time"].iloc[0]
        model_probs, model_label = _model_probs(engine, comp, home, away, elo)
        form = {home: _team_form(engine, home), away: _team_form(engine, away)}

        # 1X2
        x = ev[ev["market"] == "1x2"].pivot_table(
            index="bookmaker", columns="selection", values="odds_decimal", aggfunc="last"
        )
        if set(_SELECTIONS).issubset(x.columns):
            x = x.dropna(subset=list(_SELECTIONS))
            consensus = np.mean(
                [devig(r[list(_SELECTIONS)].to_numpy()) for _, r in x.iterrows()], axis=0
            )
            for i, sel in enumerate(_SELECTIONS):
                rows.append(
                    _build_row(
                        match_id,
                        home,
                        away,
                        comp,
                        kickoff,
                        "1x2",
                        sel,
                        0.0,
                        best_odds=float(x[sel].to_numpy(dtype=float).max()),
                        n_books=len(x),
                        p_consensus=float(consensus[i]),
                        p_model=float(model_probs[i]) if model_probs else None,
                        model_label=model_label,
                        form=form,
                    )
                )
        # Over/Under por linha
        for line_val, ou_all in ev[ev["market"] == "ou"].groupby("line"):
            line = float(str(line_val))
            ou = ou_all.pivot_table(
                index="bookmaker", columns="selection", values="odds_decimal", aggfunc="last"
            )
            if not {"over", "under"}.issubset(ou.columns):
                continue
            ou = ou.dropna(subset=["over", "under"])
            if ou.empty:
                continue
            cons = np.mean(
                [devig(r[["over", "under"]].to_numpy()) for _, r in ou.iterrows()], axis=0
            )
            for i, sel in enumerate(("over", "under")):
                rows.append(
                    _build_row(
                        match_id,
                        home,
                        away,
                        comp,
                        kickoff,
                        "ou",
                        sel,
                        float(line),
                        best_odds=float(ou[sel].to_numpy(dtype=float).max()),
                        n_books=len(ou),
                        p_consensus=float(cons[i]),
                        p_model=None,
                        model_label="sem modelo de gols p/ esta linha",
                        form=form,
                    )
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(settings.reports_dir / "analysis_today.parquet")
    log.info("analise.gravada", selecoes=len(out))
    return out


def _build_row(
    match_id: Any,
    home: str,
    away: str,
    comp: str,
    kickoff: Any,
    market: str,
    selection: str,
    line: float,
    best_odds: float,
    n_books: int,
    p_consensus: float,
    p_model: float | None,
    model_label: str,
    form: dict[str, str],
) -> dict[str, Any]:
    fair_odds = 1.0 / p_consensus if p_consensus > 0 else float("nan")
    ev_consensus = best_odds * p_consensus - 1.0
    ev_model = (p_model * best_odds - 1.0) if p_model is not None else None

    if ev_consensus >= 0.0 and (ev_model is None or ev_model > -0.10):
        verdict = "defensavel"
    elif ev_consensus >= -0.03 and (ev_model is None or ev_model > -0.15):
        verdict = "neutro"
    else:
        verdict = "evitar"
    # score p/ ordenacao: execucao (preco vs consenso) + sinal do modelo (capado:
    # EV de modelo acima de +25% e quase certamente erro, nao valor)
    score = ev_consensus + 0.5 * min(ev_model if ev_model is not None else 0.0, 0.25)

    parts = [
        f"consenso {p_consensus:.1%} (odd justa {fair_odds:.2f}); "
        f"melhor odd {best_odds:.2f} em {n_books} casas -> ev vs consenso {ev_consensus:+.1%}"
    ]
    if p_model is not None:
        parts.append(f"modelo [{model_label}]: {p_model:.1%} -> ev {ev_model:+.1%}")
    else:
        parts.append(model_label)
    parts.append(f"forma {home}: {form[home]} | {away}: {form[away]}")

    return {
        "match_id": int(match_id),
        "jogo": f"{home} x {away}",
        "competicao": comp,
        "kickoff": kickoff,
        "mercado": market,
        "selecao": selection,
        "linha": line,
        "melhor_odd": best_odds,
        "p_consenso": p_consensus,
        "odd_justa": fair_odds,
        "ev_consenso": ev_consensus,
        "p_modelo": p_model,
        "ev_modelo": ev_model,
        "modelo": model_label,
        "veredicto": verdict,
        "score": score,
        "explicacao": "; ".join(parts),
    }
