"""Dashboard do EdgeFinder (Streamlit).

Lê apenas artefatos já produzidos (data/reports/*, banco SQLite) — nenhum
cálculo pesado acontece aqui. O aviso obrigatório aparece em toda página, e o
veredito do backtest é mostrado como ele é, inclusive quando é ruim.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from edgefinder.config import settings

st.set_page_config(page_title="EdgeFinder", layout="wide")

st.warning(
    "A esmagadora maioria dos apostadores perde dinheiro no longo prazo. "
    "Um modelo com EV+ no backtest nao garante EV+ no futuro. Este e um projeto "
    "de ciencia de dados aplicada, nao uma promessa de lucro. Nunca aposte o que "
    "voce nao pode perder."
)

tab_bt, tab_cal, tab_edges, tab_cov, tab_meta = st.tabs(
    ["Backtest", "Calibracao", "Edges do dia", "Cobertura de dados", "Metodologia"]
)


def _load_summary() -> dict[str, Any] | None:
    path = settings.reports_dir / "backtest_summary.json"
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _load_bets() -> pd.DataFrame | None:
    path = settings.reports_dir / "bets_all.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


with tab_bt:
    summary = _load_summary()
    if summary is None:
        st.info("Nenhum backtest gravado. Rode: uv run edgefinder backtest")
    else:
        verdict = summary.get("verdict", "?")
        if "NAO BATE" in verdict or "INCONCLUSIVO" in verdict:
            st.error(f"VEREDITO: {verdict}")
        else:
            st.success(f"VEREDITO: {verdict}")

        if leagues := summary.get("leagues"):
            rows = []
            for comp, s in leagues.items():
                bets = s.get("bets", {})
                rows.append(
                    {
                        "liga": comp,
                        "previsoes": s["n_predictions"],
                        "logloss modelo": round(s["model_logloss_home"], 4),
                        "logloss mercado": round(s["market_logloss_home"], 4),
                        "modelo bate mercado (logloss)": s["model_beats_market_logloss"],
                        "apostas": int(bets.get("n_bets", 0)),
                        "yield": bets.get("yield_per_bet"),
                        "roi": bets.get("roi"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        if overall := summary.get("overall"):
            sig = overall.get("significance", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Apostas", int(overall.get("n_bets", 0)))
            c2.metric("Yield", f"{overall.get('yield_per_bet', 0):.2%}")
            c3.metric("p-valor", f"{sig.get('p_value', 1):.4f}")
            c4.metric("Max drawdown", f"{overall.get('max_drawdown', 0):.1%}")
            if mc := overall.get("monte_carlo"):
                st.caption(
                    f"Monte Carlo 10k trajetorias: mediana {mc.get('final_q50', 0):.2f}x bankroll, "
                    f"quantis 5/95: {mc.get('final_q05', 0):.2f}x / {mc.get('final_q95', 0):.2f}x, "
                    f"P(ruina) {mc.get('prob_ruin', 0):.1%}, "
                    f"drawdown esperado {mc.get('expected_max_drawdown', 0):.1%}"
                )
                st.caption(
                    "Atencao: o Monte Carlo usa as probabilidades DO PROPRIO MODELO. "
                    "Se o veredito acima diz que o modelo nao bate o mercado, essas "
                    "probabilidades sao otimistas e a distribuicao simulada e um teto "
                    "irreal — o contraste entre ela e o yield realizado e justamente o "
                    "diagnostico do excesso de confianca."
                )

        bets = _load_bets()
        if bets is not None and not bets.empty:
            bets = bets.sort_values("match_date")
            bets["equity"] = 1.0 + bets["pnl"].cumsum()
            fig = px.line(
                bets,
                x="match_date",
                y="equity",
                title="Curva de bankroll (stake Kelly fracionario, banca=1)",
            )
            st.plotly_chart(fig, use_container_width=True)

with tab_cal:
    preds_files = sorted(Path(settings.reports_dir).glob("predictions_*.parquet"))
    if not preds_files:
        st.info("Sem previsoes gravadas ainda.")
    else:
        frames = [pd.read_parquet(p) for p in preds_files]
        df = pd.concat(frames, ignore_index=True)
        st.caption(f"{len(df)} previsoes walk-forward de {len(preds_files)} ligas")
        options = {
            "casa": ("p_home", "home"),
            "empate": ("p_draw", "draw"),
            "fora": ("p_away", "away"),
        }
        sel = st.selectbox("Selecao", list(options))
        pcol, out = options[sel]
        dfx = df.dropna(subset=[pcol, "outcome"]).copy()
        dfx["hit"] = (dfx["outcome"] == out).astype(float)
        dfx["bin"] = pd.qcut(dfx[pcol], 10, duplicates="drop")
        # observed=True: silencioso e correto para categorias de qcut
        grouped = (
            dfx.groupby("bin", observed=True)
            .agg(prob_media=(pcol, "mean"), freq_observada=("hit", "mean"), n=("hit", "size"))
            .reset_index(drop=True)
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=grouped["prob_media"],
                y=grouped["freq_observada"],
                mode="markers+lines",
                name="modelo",
                marker={"size": (grouped["n"] / grouped["n"].max() * 20 + 4)},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="calibracao perfeita", line={"dash": "dash"}
            )
        )
        fig.update_layout(
            title=f"Curva de calibracao ({sel})",
            xaxis_title="probabilidade prevista",
            yaxis_title="frequencia observada",
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_edges:
    path = settings.reports_dir / "edges_today.parquet"
    if not path.exists():
        st.info(
            "Sem edges gravados. Configure ODDS_API_KEY no .env, rode "
            "'uv run edgefinder collect-odds' e depois 'uv run edgefinder edges'. "
            "Sem chave, o sistema opera apenas em modo backtest historico."
        )
    else:
        edges = pd.read_parquet(path)
        st.dataframe(
            edges[
                [
                    "home_team",
                    "away_team",
                    "competition",
                    "commence_time",
                    "selection",
                    "p_model",
                    "p_market",
                    "odds",
                    "ev",
                    "stake_frac",
                ]
            ],
            use_container_width=True,
        )
        for _, r in edges.iterrows():
            with st.expander(
                f"{r['home_team']} x {r['away_team']} - {r['selection']} (EV {r['ev']:.1%})"
            ):
                st.write(r["explanation"])

with tab_cov:
    try:
        from edgefinder.storage.repository import get_engine, read_df

        cov = read_df(
            get_engine(),
            "SELECT competition_id, dataset, status, notes, checked_at FROM data_coverage",
        )
        if cov.empty:
            st.info("Cobertura ainda nao registrada no banco.")
        else:
            st.dataframe(cov, use_container_width=True)
    except Exception as exc:
        st.error(f"Banco indisponivel: {exc}")
    st.markdown(
        "Contexto essencial: o FBref perdeu todos os dados avancados da Opta "
        "(xG incluido) em 20/01/2026. A matriz completa com evidencias esta no "
        "PLAN.md / README.md do projeto."
    )

with tab_meta:
    st.markdown(
        """
### Como ler este dashboard

- **Backtest**: walk-forward estrito (treina no passado, preve o proximo bloco,
  avanca). A aposta simulada usa a odd de FECHAMENTO real; o benchmark de
  probabilidade e o fechamento de-vigado pelo metodo de Shin. Se o veredito
  diz que o modelo nao bate o mercado, acredite no veredito.
- **Calibracao**: probabilidade prevista vs. frequencia observada em decis.
  Pontos acima da diagonal = modelo subestima; abaixo = superestima.
- **Edges**: so aparecem mercados com validacao nivel A (PLAN.md secao 3) e
  EV acima do threshold do tier. Tier 3 (selecoes) nasce desligado: nao ha
  odds historicas gratuitas para provar CLV.
- **CLV**: a metrica que importa. Bater o fechamento consistentemente e a
  unica evidencia real de edge; resultado de curto prazo e ruido.
        """
    )
