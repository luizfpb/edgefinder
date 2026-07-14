"""CLI do EdgeFinder (typer + rich).

Imports de modelos/backtest são feitos dentro dos comandos: a CLI sobe mesmo
que um módulo pesado esteja quebrado, e o aviso obrigatório aparece sempre.
"""

import json
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    import pandas as pd
    from sqlalchemy import Engine
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
    half_life: Annotated[
        float | None,
        typer.Option(help="meia-vida do decaimento (dias); default: otimo do H1 em config"),
    ] = None,
    start_season: str = "1920",
    freq: str = "W",
) -> None:
    """Backtest walk-forward contra odds de fechamento historicas."""
    from edgefinder.backtest.runner import BacktestConfig, run_1x2_backtest
    from edgefinder.config import settings
    from edgefinder.storage.repository import get_engine

    if market != "1x2":
        console.print(
            "[red]Somente o mercado 1x2 tem odds historicas para backtest completo "
            "(niveis de validacao: PLAN.md secao 3).[/red]"
        )
        raise typer.Exit(1)
    config = BacktestConfig(
        competitions=competitions or DEFAULT_BACKTEST,
        half_life_days=half_life if half_life is not None else settings.dc_half_life_days,
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
    _print_analysis_table(df)
    console.print(
        "[dim]Detalhes de cada linha (consenso, modelo, sequencia, forma): "
        "expanda no dashboard ou leia data/reports/analysis_today.parquet[/dim]"
    )


def _print_analysis_table(df: "pd.DataFrame") -> None:
    import pandas as pd

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
        "sequencia",
        "veredicto",
    ):
        table.add_column(col)
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
            str(r.get("sequencia", "") or "-"),
            f"[{style}]{r['veredicto']}[/{style}]",
        )
    console.print(table)


@app.command()
def sequencias(
    time: Annotated[str, typer.Argument(help="nome canonico do time (ex.: 'arsenal')")],
    adversario: Annotated[str | None, typer.Argument(help="opcional: o outro time")] = None,
    n: Annotated[int, typer.Option("--n", help="janela: ultimos N jogos")] = 5,
    comp: Annotated[str | None, typer.Option("--comp", help="filtrar por competicao")] = None,
) -> None:
    """Tabela de sequencias: em quantos dos ultimos N jogos aconteceu cada coisa.

    Com dois times, mostra casa, fora e o combinado ("N de 2N"). Alem dos
    gols: escanteios, cartoes e chutes no gol (onde a fonte tem o dado) e a
    tabela por jogador (quem marcou em quantos dos ultimos N, chutes, cartoes,
    faltas). Sequencia e contexto descritivo, NAO probabilidade: amostra de 5
    jogos e minuscula e o mercado ja precifica tendencia obvia.
    """
    from edgefinder.edge.streaks import last_matches, match_streak_table, team_streaks
    from edgefinder.storage.repository import get_engine

    engine = get_engine()
    if adversario:
        df = match_streak_table(engine, time, adversario, n)
        if (df[time] == "0/0").all():
            console.print(f"[yellow]Sem historico para '{time}' no banco.[/yellow]")
            raise typer.Exit(1)
        table = Table(title=f"Sequencias - ultimos {n} jogos de cada time")
        for col in df.columns:
            table.add_column(str(col))
        for _, r in df.iterrows():
            table.add_row(*(str(v) for v in r))
        console.print(table)
        for team_name in (time, adversario):
            _print_team_stats_sections(engine, team_name, n)
    else:
        view = last_matches(engine, time, n, competition=comp)
        if view.empty:
            console.print(f"[yellow]Sem historico para '{time}' no banco.[/yellow]")
            raise typer.Exit(1)
        table = Table(title=f"Sequencias de {time} - ultimos {len(view)} jogos")
        table.add_column("condicao")
        table.add_column("contagem")
        for streak_line in team_streaks(view, n):
            table.add_row(streak_line.label, f"{streak_line.hits} de {streak_line.total}")
        console.print(table)
        jogos = Table(title="Jogos considerados (mais recente primeiro)")
        for col in ("data", "adversario", "local", "placar"):
            jogos.add_column(col)
        for _, r in view.iterrows():
            jogos.add_row(
                str(r["match_date"])[:10],
                str(r["opponent"]),
                str(r["venue"]),
                f"{int(r['gf'])}x{int(r['ga'])}",
            )
        console.print(jogos)
        _print_team_stats_sections(engine, time, n)
    console.print(
        "[dim]Sequencia nao e probabilidade: use como contexto, nunca como prova de valor.[/dim]"
    )


def _print_team_stats_sections(engine: "Engine", team: str, n: int) -> None:
    """Seções de escanteios/cartões/médias e de jogadores dos últimos n jogos."""
    import pandas as pd

    from edgefinder.edge.streaks import (
        player_summary,
        team_players_last,
        team_stats_averages,
        team_stats_last,
        team_stats_streaks,
    )

    stats_view = team_stats_last(engine, team, n)
    stat_lines = team_stats_streaks(stats_view, n)
    if stat_lines:
        table = Table(title=f"{team} - escanteios/cartoes/chutes (ultimos {n} com dado)")
        table.add_column("condicao")
        table.add_column("contagem")
        for s in stat_lines:
            table.add_row(s.label, f"{s.hits} de {s.total}")
        console.print(table)
    medias = team_stats_averages(stats_view, n)
    if not medias.empty:
        table = Table(title=f"{team} - medias por jogo (ultimos {n} com dado)")
        for col in medias.columns:
            table.add_column(str(col))
        for _, r in medias.iterrows():
            table.add_row(
                *("-" if pd.isna(v) else f"{v:g}" if isinstance(v, float) else str(v) for v in r)
            )
        console.print(table)
    resumo = player_summary(team_players_last(engine, team, n))
    if resumo.empty:
        console.print(f"[dim]{team}: sem stats de jogador no banco para estes jogos.[/dim]")
        return
    table = Table(title=f"{team} - jogadores (ultimos {n} jogos do time)")
    for col in resumo.columns:
        table.add_column(str(col))
    for _, r in resumo.head(18).iterrows():
        table.add_row(
            *("-" if pd.isna(v) else f"{v:g}" if isinstance(v, float) else str(v) for v in r)
        )
    console.print(table)


bet_app = typer.Typer(help="Registro e liquidacao de apostas (paper por default).")
app.add_typer(bet_app, name="bet")


@bet_app.command("add")
def bet_add(
    match_id: Annotated[int, typer.Argument(help="id do jogo (coluna match_id da analise)")],
    market: Annotated[str, typer.Argument(help="1x2 | ou")],
    selection: Annotated[str, typer.Argument(help="home|draw|away|over|under")],
    odds: Annotated[float, typer.Argument(help="odd decimal tomada")],
    stake: Annotated[float, typer.Option("--stake")] = 1.0,
    line: Annotated[float, typer.Option("--line", help="linha (OU); 0 para 1x2")] = 0.0,
    bookmaker: Annotated[str, typer.Option("--book")] = "",
    real: Annotated[bool, typer.Option("--real", help="aposta real (default: paper)")] = False,
) -> None:
    """Registra uma aposta manualmente para rastrear CLV e resultado."""
    from edgefinder.edge.tracking import record_paper_bet
    from edgefinder.storage.repository import get_engine

    created = record_paper_bet(
        get_engine(),
        match_id=match_id,
        market=market,
        selection=selection,
        odds_taken=odds,
        stake=stake,
        bookmaker=bookmaker,
        line=line,
        is_paper=not real,
    )
    if created:
        console.print(f"Aposta registrada: match {match_id} {market} {selection} @ {odds:.2f}")
    else:
        console.print("[yellow]Aposta identica ja registrada (dedup) — nada feito.[/yellow]")


@bet_app.command("settle")
def bet_settle() -> None:
    """Liquida apostas de jogos ja disputados e atualiza o CLV das encerradas."""
    from edgefinder.edge.tracking import link_snapshots_to_matches, settle_bets, update_clv
    from edgefinder.storage.repository import get_engine

    engine = get_engine()
    linked = link_snapshots_to_matches(engine)
    clv_n = update_clv(engine)
    settled = settle_bets(engine)
    console.print(
        f"{linked} snapshots vinculados, {clv_n} apostas com CLV atualizado, "
        f"{settled} apostas liquidadas."
    )


@bet_app.command("list")
def bet_list(
    limit: Annotated[int, typer.Option("--limit")] = 30,
) -> None:
    """Lista as apostas registradas com CLV e resultado."""
    from edgefinder.storage.repository import get_engine, read_df

    df = read_df(
        get_engine(),
        """
        SELECT b.id, th.name || ' x ' || ta.name AS jogo, b.market, b.selection,
               b.line, b.odds_taken, b.closing_odds, b.clv, b.result, b.pnl,
               b.is_paper, b.placed_at
        FROM bets b
        JOIN matches m ON m.id = b.match_id
        JOIN teams th ON th.id = m.home_team_id
        JOIN teams ta ON ta.id = m.away_team_id
        ORDER BY b.placed_at DESC LIMIT :n
        """,
        {"n": limit},
    )
    if df.empty:
        console.print(
            "[yellow]Nenhuma aposta registrada. O comando 'daily' registra paper bets "
            "automaticamente para as selecoes defensaveis; ou use 'bet add'.[/yellow]"
        )
        return
    import pandas as pd

    table = Table(title=f"Apostas registradas (ultimas {len(df)})")
    for col in ("jogo", "mercado", "selecao", "linha", "odd", "fechamento", "CLV", "resultado"):
        table.add_column(col)
    for _, r in df.iterrows():
        table.add_row(
            str(r["jogo"]),
            str(r["market"]),
            str(r["selection"]),
            f"{r['line']:g}" if r["line"] else "-",
            f"{r['odds_taken']:.2f}",
            f"{r['closing_odds']:.2f}" if pd.notna(r["closing_odds"]) else "-",
            f"{r['clv']:+.2%}" if pd.notna(r["clv"]) else "-",
            str(r["result"] or "aberta"),
        )
    console.print(table)


@app.command()
def daily(
    top: Annotated[int, typer.Option(help="quantas selecoes mostrar")] = 12,
    sem_odds: Annotated[
        bool, typer.Option("--sem-odds", help="pula a coleta (nao gasta creditos)")
    ] = False,
) -> None:
    """Fluxo diario completo em um comando: coleta, analisa, registra e mede.

    Passos: (1) snapshot de odds (The Odds API), (2) vincula snapshots e
    atualiza CLV, (3) liquida apostas de jogos encerrados, (4) analisa os
    jogos futuros, (5) registra paper bets das selecoes defensaveis, (6)
    mostra a analise, o estado do CLV e os creditos consumidos no mes.
    """
    from edgefinder.config import settings
    from edgefinder.edge.analysis import analyze_today
    from edgefinder.edge.tracking import (
        link_snapshots_to_matches,
        record_defensible_paper_bets,
        settle_bets,
        update_clv,
    )
    from edgefinder.ingest.odds_api import collect_h2h_snapshot, credits_used_this_month
    from edgefinder.storage.repository import get_engine, init_db

    engine = get_engine()
    init_db(engine)

    if not sem_odds:
        comps = ["ENG-Premier League", "BRA-Serie A", "INT-World Cup"]
        n = collect_h2h_snapshot(engine, comps)
        console.print(f"1/6 coleta: {n} linhas de odds ({comps})")
    else:
        console.print("1/6 coleta: pulada (--sem-odds)")

    linked = link_snapshots_to_matches(engine)
    clv_updates = update_clv(engine)
    console.print(f"2/6 vinculo: {linked} snapshots ligados a jogos, {clv_updates} CLV atualizados")

    settled = settle_bets(engine)
    console.print(f"3/6 liquidacao: {settled} apostas resolvidas")

    df = analyze_today(engine)
    if df.empty:
        console.print(
            "[yellow]4/6 analise: nenhum jogo futuro com odds coletadas "
            "(confira ODDS_API_KEY no .env).[/yellow]"
        )
    else:
        console.print(f"4/6 analise: {len(df)} selecoes avaliadas")

    recorded = record_defensible_paper_bets(engine, df)
    console.print(f"5/6 paper bets: {recorded} novas registradas (defensaveis)")

    from edgefinder.edge.streaks import export_streak_snapshots

    exported = export_streak_snapshots(engine)
    console.print(
        f"snapshots p/ dashboard publicado: {exported.get('matches', 0)} jogos, "
        f"{exported.get('team_stats', 0)} linhas de stats de time, "
        f"{exported.get('players', 0)} linhas de jogador"
    )

    if not df.empty:
        _print_analysis_table(df.head(top))

    from edgefinder.storage.repository import read_df

    bets = read_df(engine, "SELECT clv FROM bets WHERE clv IS NOT NULL")
    if bets.empty:
        console.print("CLV: ainda sem apostas fechadas — o historico esta sendo construido.")
    else:
        console.print(
            f"CLV: media {bets['clv'].mean():+.2%} | positivo em {(bets['clv'] > 0).mean():.0%} "
            f"de {len(bets)} apostas"
        )
    used = credits_used_this_month(engine)
    console.print(f"6/6 creditos The Odds API no mes: {used}/{settings.odds_api_monthly_budget}")

    summary_path = settings.reports_dir / "backtest_summary.json"
    if summary_path.exists():
        import time

        age_days = (time.time() - summary_path.stat().st_mtime) / 86_400
        if age_days > 30:
            console.print(
                f"[yellow]Aviso: o backtest gravado tem {age_days:.0f} dias; "
                "rode 'edgefinder backtest' para renovar o veredito.[/yellow]"
            )
    else:
        console.print(
            "[yellow]Aviso: nenhum backtest gravado; rode 'edgefinder backtest'.[/yellow]"
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
