"""Registro de apostas (paper) e atualização de CLV.

Fluxo: `record_paper_bet` no momento da sugestão; o coletor de odds continua
tirando snapshots até o kickoff; `update_clv` casa cada aposta com o último
snapshot antes do início do jogo (nossa closing line construída) e grava

    CLV = odds_tomada / odds_fechamento - 1.

CLV positivo consistente é a única prova real de edge — resultado de curto
prazo é ruído. Enquanto não houver histórico próprio suficiente, o dashboard
mostra o CLV como "em construção", não como validado.
"""

from datetime import datetime
from typing import Any

import pandas as pd
import structlog
from sqlalchemy import Engine, select, update

from edgefinder.storage import schema
from edgefinder.timeutil import utcnow_naive

log = structlog.get_logger()


def record_paper_bet(
    engine: Engine,
    match_id: int,
    market: str,
    selection: str,
    odds_taken: float,
    stake: float,
    bookmaker: str = "",
    line: float = 0.0,
    is_paper: bool = True,
) -> bool:
    """Registra uma aposta; retorna False se já existia (dedup por seleção).

    A chave de dedup é (match_id, market, selection, line, is_paper): rodar o
    fluxo diário duas vezes não pode duplicar a mesma aposta — duplicata
    inflaria o CLV médio com pseudo-amostras.
    """
    with engine.begin() as conn:
        exists = conn.execute(
            select(schema.bets.c.id).where(
                schema.bets.c.match_id == match_id,
                schema.bets.c.market == market,
                schema.bets.c.selection == selection,
                schema.bets.c.line == line,
                schema.bets.c.is_paper == is_paper,
            )
        ).fetchone()
        if exists:
            return False
        conn.execute(
            schema.bets.insert().values(
                match_id=match_id,
                market=market,
                selection=selection,
                line=line,
                odds_taken=odds_taken,
                bookmaker=bookmaker,
                stake=stake,
                placed_at=utcnow_naive(),
                is_paper=is_paper,
            )
        )
    return True


def record_defensible_paper_bets(engine: Engine, analysis: pd.DataFrame) -> int:
    """Registra paper bets para toda seleção "defensavel" da análise do dia.

    É isso que fecha o loop de CLV sem depender de disciplina manual: cada
    análise defensável vira uma aposta de papel com stake fixo 1.0; o coletor
    segue tirando snapshots até o kickoff e `update_clv` mede se a odd tomada
    bateu o fechamento. Stake flat porque o objetivo é MEDIR CLV, não simular
    banca — Kelly aqui exigiria confiar em probabilidades não validadas.
    """
    if analysis.empty or "veredicto" not in analysis.columns:
        return 0
    recorded = 0
    for _, r in analysis[analysis["veredicto"] == "defensavel"].iterrows():
        if record_paper_bet(
            engine,
            match_id=int(r["match_id"]),
            market=str(r["mercado"]),
            selection=str(r["selecao"]),
            odds_taken=float(r["melhor_odd"]),
            stake=1.0,
            line=float(r["linha"]) if pd.notna(r["linha"]) else 0.0,
        ):
            recorded += 1
    if recorded:
        log.info("clv.paper_bets_registradas", bets=recorded)
    return recorded


def link_snapshots_to_matches(engine: Engine) -> int:
    """Vincula snapshots de odds (match_id nulo) aos jogos do banco.

    Casamento por nome canônico dos dois times + data do jogo dentro de ±1 dia
    do commence_time (fusos divergem entre fontes). Roda após cada coleta —
    sem esse vínculo o CLV não tem como casar aposta com fechamento.
    """
    from edgefinder.data.teamnames import normalize_team_name

    with engine.begin() as conn:
        pending = conn.execute(
            select(
                schema.odds_snapshots.c.id,
                schema.odds_snapshots.c.home_team_raw,
                schema.odds_snapshots.c.away_team_raw,
                schema.odds_snapshots.c.commence_time,
            ).where(
                schema.odds_snapshots.c.match_id.is_(None),
                schema.odds_snapshots.c.commence_time.is_not(None),
            )
        ).fetchall()
        if not pending:
            return 0
        match_rows = conn.execute(
            select(
                schema.matches.c.id,
                schema.matches.c.match_date,
                schema.teams.c.name.label("home_name"),
            ).select_from(
                schema.matches.join(
                    schema.teams, schema.matches.c.home_team_id == schema.teams.c.id
                )
            )
        ).fetchall()
        away_rows = conn.execute(
            select(schema.matches.c.id, schema.teams.c.name).select_from(
                schema.matches.join(
                    schema.teams, schema.matches.c.away_team_id == schema.teams.c.id
                )
            )
        ).fetchall()
        away_by_id = {int(mid): str(name) for mid, name in away_rows}
        index: dict[tuple[str, str], list[tuple[int, datetime]]] = {}
        for mid, mdate, home_name in match_rows:
            key = (str(home_name), away_by_id.get(int(mid), ""))
            index.setdefault(key, []).append((int(mid), mdate))

        linked = 0
        for snap_id, home_raw, away_raw, commence in pending:
            key = (normalize_team_name(str(home_raw)), normalize_team_name(str(away_raw)))
            candidates = [
                mid
                for mid, mdate in index.get(key, [])
                if abs((mdate - commence).total_seconds()) <= 86_400
            ]
            if len(candidates) == 1:
                conn.execute(
                    update(schema.odds_snapshots)
                    .where(schema.odds_snapshots.c.id == snap_id)
                    .values(match_id=candidates[0])
                )
                linked += 1
    if linked:
        log.info("clv.snapshots_vinculados", snapshots=linked)
    return linked


def update_clv(engine: Engine) -> int:
    """Preenche closing_odds/clv das apostas cujo jogo já começou."""
    now = utcnow_naive()  # commence_time no banco é UTC-naive
    with engine.begin() as conn:
        pending = conn.execute(
            select(
                schema.bets.c.id,
                schema.bets.c.match_id,
                schema.bets.c.market,
                schema.bets.c.selection,
                schema.bets.c.line,
                schema.bets.c.odds_taken,
            ).where(schema.bets.c.closing_odds.is_(None))
        ).fetchall()
        updated = 0
        for bet_id, match_id, market, selection, line, odds_taken in pending:
            snap = conn.execute(
                select(schema.odds_snapshots.c.odds_decimal)
                .where(
                    schema.odds_snapshots.c.match_id == match_id,
                    schema.odds_snapshots.c.market == market,
                    schema.odds_snapshots.c.selection == selection,
                    schema.odds_snapshots.c.line == line,
                    schema.odds_snapshots.c.commence_time.is_not(None),
                    schema.odds_snapshots.c.commence_time <= now,
                )
                .order_by(schema.odds_snapshots.c.collected_at.desc())
                .limit(1)
            ).fetchone()
            if snap is None:
                continue
            closing = float(snap[0])
            conn.execute(
                update(schema.bets)
                .where(schema.bets.c.id == bet_id)
                .values(closing_odds=closing, clv=odds_taken / closing - 1.0)
            )
            updated += 1
    if updated:
        log.info("clv.atualizado", bets=updated)
    return updated


def settle_bets(engine: Engine) -> int:
    """Liquida apostas 1x2 e over/under de jogos já disputados (win/lose/push + pnl)."""
    with engine.begin() as conn:
        pending = conn.execute(
            select(
                schema.bets.c.id,
                schema.bets.c.market,
                schema.bets.c.selection,
                schema.bets.c.line,
                schema.bets.c.odds_taken,
                schema.bets.c.stake,
                schema.matches.c.home_goals,
                schema.matches.c.away_goals,
            )
            .select_from(schema.bets.join(schema.matches))
            .where(
                schema.bets.c.result.is_(None),
                schema.bets.c.market.in_(["1x2", "ou"]),
                schema.matches.c.home_goals.is_not(None),
            )
        ).fetchall()
        settled = 0
        for bet_id, market, selection, line, odds_taken, stake, hg, ag in pending:
            result = _settle_one(market, selection, float(line or 0.0), int(hg), int(ag))
            if result is None:
                continue
            pnl = {
                "win": stake * (odds_taken - 1.0),
                "lose": -stake,
                "push": 0.0,
            }[result]
            values: dict[str, Any] = {"result": result, "pnl": pnl}
            conn.execute(update(schema.bets).where(schema.bets.c.id == bet_id).values(**values))
            settled += 1
    return settled


def _settle_one(market: str, selection: str, line: float, hg: int, ag: int) -> str | None:
    """Resultado de uma aposta dado o placar: win|lose|push (None = mercado desconhecido)."""
    if market == "1x2":
        outcome = "home" if hg > ag else "away" if hg < ag else "draw"
        return "win" if selection == outcome else "lose"
    if market == "ou":
        total = hg + ag
        if total == line:  # linha inteira exata devolve a stake
            return "push"
        over = total > line
        return "win" if (selection == "over") == over else "lose"
    return None
