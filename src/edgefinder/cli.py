"""CLI do EdgeFinder (typer + rich).

Imports de modelos/backtest são feitos dentro dos comandos: a CLI sobe mesmo
que um módulo pesado esteja quebrado, e o aviso obrigatório aparece sempre.
"""

import json
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(add_completion=False, help="EdgeFinder: probabilidades calibradas vs. mercado.")
console = Console()

DISCLAIMER = (
    "A esmagadora maioria dos apostadores perde dinheiro no longo prazo. "
    "Um modelo com EV+ no backtest nao garante EV+ no futuro. Projeto de "
    "ciencia de dados aplicada, nao promessa de lucro. Nunca aposte o que "
    "voce nao pode perder."
)

BIG5 = ["ENG-Premier League", "ESP-La Liga", "ITA-Serie A", "GER-Bundesliga", "FRA-Ligue 1"]
DEFAULT_BACKTEST = [*BIG5, "ENG-Championship", "NED-Eredivisie", "POR-Liga Portugal", "BRA-Serie A"]


@app.callback()
def _banner() -> None:
    console.print(Panel(DISCLAIMER, title="AVISO", border_style="red"))


@app.command()
def ingest(
    source: Annotated[str, typer.Option(help="all|matchhistory|understat|fbref")] = "all",
) -> None:
    """Carrega o banco a partir do cache bruto (idempotente)."""
    from edgefinder.ingest.etl import load_matchhistory
    from edgefinder.storage.repository import get_engine, init_db

    engine = get_engine()
    init_db(engine)
    if source in ("all", "matchhistory"):
        totals = load_matchhistory(engine)
        console.print(f"matchhistory: {totals}")
    if source in ("all", "understat", "fbref"):
        console.print(
            "Understat/FBref: use scripts/load_db.py (carga longa, roda fora da CLI) "
            "ou rode novamente apos aquecer o cache com scripts/warm_*.py"
        )


@app.command("collect-odds")
def collect_odds(
    competitions: Annotated[list[str] | None, typer.Option("--comp")] = None,
) -> None:
    """Snapshot de odds correntes (The Odds API; exige ODDS_API_KEY no .env)."""
    from edgefinder.edge.tracking import link_snapshots_to_matches, update_clv
    from edgefinder.ingest.odds_api import collect_h2h_snapshot
    from edgefinder.storage.repository import get_engine, init_db

    engine = get_engine()
    init_db(engine)
    comps = competitions or ["ENG-Premier League", "BRA-Serie A", "INT-World Cup"]
    n = collect_h2h_snapshot(engine, comps)
    linked = link_snapshots_to_matches(engine)
    clv_updates = update_clv(engine)
    console.print(
        f"{n} linhas de odds gravadas para {comps} "
        f"({linked} snapshots vinculados a jogos, {clv_updates} apostas com CLV atualizado)"
    )


@app.command()
def backtest(
    market: Annotated[str, typer.Option(help="por ora: 1x2")] = "1x2",
    competitions: Annotated[list[str] | None, typer.Option("--comp")] = None,
    half_life: Annotated[float, typer.Option(help="meia-vida do decaimento (dias)")] = 180.0,
    start_season: str = "1920",
    freq: str = "W",
) -> None:
    """Backtest walk-forward contra odds de fechamento historicas."""
    from edgefinder.backtest.runner import BacktestConfig, run_1x2_backtest
    from edgefinder.storage.repository import get_engine

    if market != "1x2":
        console.print(
            "[red]Somente o mercado 1x2 tem odds historicas para backtest completo "
            "(niveis de validacao: PLAN.md secao 3).[/red]"
        )
        raise typer.Exit(1)
    config = BacktestConfig(
        competitions=competitions or DEFAULT_BACKTEST,
        half_life_days=half_life,
        start_season=start_season,
        freq=freq,
    )
    report = run_1x2_backtest(get_engine(), config)
    _print_backtest_report(report)


def _print_backtest_report(report: dict) -> None:  # type: ignore[type-arg]
    verdict = report.get("verdict", "?")
    style = "red" if "NAO BATE" in verdict or "INCONCLUSIVO" in verdict else "green"
    if leagues := report.get("leagues"):
        table = Table(title="Backtest por liga (1X2, walk-forward)")
        for col in (
            "liga",
            "n",
            "logloss modelo",
            "logloss mercado",
            "bate?",
            "n apostas",
            "yield",
        ):
            table.add_column(col)
        for comp, s in leagues.items():
            bets = s.get("bets", {})
            table.add_row(
                comp,
                str(s["n_predictions"]),
                f"{s['model_logloss_home']:.4f}",
                f"{s['market_logloss_home']:.4f}",
                "sim" if s["model_beats_market_logloss"] else "NAO",
                str(int(bets.get("n_bets", 0))),
                f"{bets.get('yield_per_bet', float('nan')):.3%}" if bets else "-",
            )
        console.print(table)
    if overall := report.get("overall"):
        sig = overall.get("significance", {})
        console.print(
            f"GERAL: {int(overall.get('n_bets', 0))} apostas, ROI {overall.get('roi', 0):.3%}, "
            f"yield {overall.get('yield_per_bet', 0):.3%}, p-valor {sig.get('p_value', 1):.4f}, "
            f"IC95 [{sig.get('ci95_low', 0):.3%}, {sig.get('ci95_high', 0):.3%}]"
        )
        if mc := overall.get("monte_carlo"):
            console.print(
                f"Monte Carlo (10k trajetorias): mediana {mc.get('final_q50', 0):.2f}x, "
                f"quantis 5/95 {mc.get('final_q05', 0):.2f}x/{mc.get('final_q95', 0):.2f}x, "
                f"P(ruina) {mc.get('prob_ruin', 0):.1%}, drawdown esperado {mc.get('expected_max_drawdown', 0):.1%}"
            )
    console.print(Panel(verdict, title="VEREDITO", border_style=style))


@app.command()
def edges(
    min_ev: float = 0.03,
) -> None:
    """Edges nos jogos futuros com odds correntes coletadas (exige snapshots)."""
    from edgefinder.edge.today import backtest_gate, today_edges
    from edgefinder.storage.repository import get_engine

    approved, verdict = backtest_gate()
    if not approved:
        console.print(
            Panel(
                f"SUGESTOES BLOQUEADAS PELO BACKTEST.\n\n{verdict}\n\n"
                "Esta trava e deliberada: um modelo que nao bate a closing line no "
                "passado nao tem por que bater no futuro. O sistema segue util para "
                "estudo (backtest, calibracao, dashboard), nao para apostar.",
                title="TRAVA DE HONESTIDADE",
                border_style="red",
            )
        )
        raise typer.Exit(0)
    df = today_edges(get_engine(), min_ev=min_ev)
    if df.empty:
        console.print(
            "[yellow]Sem edges: ou nao ha snapshots de odds correntes (configure "
            "ODDS_API_KEY e rode collect-odds), ou nenhum jogo passa do threshold.[/yellow]"
        )
        raise typer.Exit(0)
    table = Table(title=f"Edges (EV >= {min_ev:.0%})")
    for col in (
        "jogo",
        "data",
        "mercado",
        "selecao",
        "p modelo",
        "p mercado",
        "odd",
        "EV",
        "kelly",
        "explicacao",
    ):
        table.add_column(col)
    for _, r in df.iterrows():
        table.add_row(
            f"{r['home_team']} x {r['away_team']}",
            str(r["commence_time"])[:16],
            r["market"],
            r["selection"],
            f"{r['p_model']:.1%}",
            f"{r['p_market']:.1%}",
            f"{r['odds']:.2f}",
            f"{r['ev']:.1%}",
            f"{r['stake_frac']:.2%}",
            r["explanation"],
        )
    console.print(table)


@app.command()
def analise(
    top: Annotated[int, typer.Option(help="quantas selecoes mostrar")] = 12,
    so_defensaveis: bool = typer.Option(False, "--so-defensaveis"),
) -> None:
    """Analisa os jogos futuros com odds coletadas: o que e defensavel apostar.

    ANALISE, nao recomendacao validada: nenhum modelo nosso provou bater o
    mercado (por isso `edges` segue travado). Aqui as lentes sao (1) preco vs
    consenso de-vigado das casas, (2) modelo com rotulo honesto do proprio
    status e (3) forma recente com o tamanho da amostra na cara.
    """
    from edgefinder.edge.analysis import analyze_today
    from edgefinder.storage.repository import get_engine

    df = analyze_today(get_engine())
    if df.empty:
        console.print(
            "[yellow]Nenhum jogo futuro com odds coletadas. Rode collect-odds "
            "(e confira ODDS_API_KEY no .env).[/yellow]"
        )
        raise typer.Exit(0)
    if so_defensaveis:
        df = df[df["veredicto"] == "defensavel"]
    df = df.head(top)
    table = Table(title="Analise do dia (ordenada por defensabilidade)")
    for col in (
        "jogo",
        "kickoff",
        "mercado",
        "selecao",
        "linha",
        "melhor odd",
        "odd justa",
        "EV vs consenso",
        "EV modelo",
        "veredicto",
    ):
        table.add_column(col)
    import pandas as pd

    for _, r in df.iterrows():
        style = {"defensavel": "green", "neutro": "yellow", "evitar": "red"}[r["veredicto"]]
        table.add_row(
            r["jogo"],
            str(r["kickoff"])[:16],
            r["mercado"],
            r["selecao"],
            f"{r['linha']:g}" if r["linha"] else "-",
            f"{r['melhor_odd']:.2f}",
            f"{r['odd_justa']:.2f}",
            f"{r['ev_consenso']:+.1%}",
            f"{r['ev_modelo']:+.1%}" if pd.notna(r["ev_modelo"]) else "-",
            f"[{style}]{r['veredicto']}[/{style}]",
        )
    console.print(table)
    console.print(
        "[dim]Detalhes de cada linha (consenso, modelo, forma): "
        "expanda no dashboard ou leia data/reports/analysis_today.parquet[/dim]"
    )


@app.command()
def coverage() -> None:
    """Matriz de cobertura de dados (a tabela viva do banco)."""
    from edgefinder.storage.repository import get_engine, read_df

    df = read_df(get_engine(), "SELECT competition_id, dataset, status, notes FROM data_coverage")
    if df.empty:
        console.print("Cobertura ainda nao registrada; rode a ingestao.")
        return
    table = Table(title="Cobertura de dados")
    for col in df.columns:
        table.add_column(col)
    for _, r in df.iterrows():
        table.add_row(*(str(v) for v in r))
    console.print(table)


@app.command()
def clv(report: bool = typer.Option(True, "--report")) -> None:
    """Relatorio de CLV das apostas registradas (paper ou reais)."""
    from edgefinder.storage.repository import get_engine, read_df

    df = read_df(get_engine(), "SELECT * FROM bets WHERE clv IS NOT NULL")
    if df.empty:
        console.print(
            "[yellow]Nenhuma aposta com CLV ainda. O CLV e a metrica mais importante do "
            "sistema: registre apostas (paper) e rode collect-odds ate o fechamento.[/yellow]"
        )
        return
    console.print(df[["match_id", "market", "selection", "odds_taken", "closing_odds", "clv"]])
    console.print(f"CLV medio: {df['clv'].mean():.2%} | % positivo: {(df['clv'] > 0).mean():.1%}")


@app.command("summary")
def summary_cmd() -> None:
    """Resumo do ultimo backtest gravado em data/reports."""
    from edgefinder.config import settings

    path = settings.reports_dir / "backtest_summary.json"
    if not path.exists():
        console.print("Nenhum backtest gravado; rode: edgefinder backtest")
        raise typer.Exit(1)
    _print_backtest_report(json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    app()
