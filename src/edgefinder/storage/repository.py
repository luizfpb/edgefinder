"""Acesso ao banco: engine, criação de schema e upserts idempotentes.

Idempotência é requisito: toda escrita usa INSERT ... ON CONFLICT (upsert do
SQLite). Rodar a mesma ingestão duas vezes produz o mesmo banco.
"""

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import Engine, Table, create_engine, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from edgefinder.config import COMPETITION_TIERS, settings
from edgefinder.storage import schema


def get_engine(url: str | None = None) -> Engine:
    # timeout alto: cargas em lote e leitores (backtest/dashboard) convivem no
    # mesmo arquivo SQLite; o leitor espera o escritor em vez de estourar
    # "database is locked".
    engine = create_engine(url or settings.sqlalchemy_url, connect_args={"timeout": 60})
    return engine


def init_db(engine: Engine) -> None:
    """Cria as tabelas (no-op se já existem) e semeia as competições."""
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    schema.metadata.create_all(engine)
    rows = [
        {
            "id": comp_id,
            "name": comp_id.split("-", 1)[1],
            "country": comp_id.split("-", 1)[0],
            "tier": tier,
            "kind": (
                "international"
                if comp_id.startswith("INT-")
                else "cup"
                if "Champions" in comp_id or "Libertadores" in comp_id
                else "league"
            ),
        }
        for comp_id, tier in COMPETITION_TIERS.items()
    ]
    upsert(engine, schema.competitions, rows, conflict_cols=["id"])


def upsert(
    engine: Engine,
    table: Table,
    rows: Iterable[Mapping[str, Any]],
    conflict_cols: list[str],
) -> int:
    """INSERT ... ON CONFLICT DO UPDATE em lote. Retorna o total processado."""
    rows = list(rows)
    if not rows:
        return 0
    with engine.begin() as conn:
        for chunk_start in range(0, len(rows), 500):
            chunk = rows[chunk_start : chunk_start + 500]
            stmt = sqlite_insert(table).values(chunk)
            update_cols = {
                c.name: getattr(stmt.excluded, c.name)
                for c in table.columns
                if c.name not in conflict_cols and not c.primary_key
            }
            conn.execute(
                stmt.on_conflict_do_update(index_elements=conflict_cols, set_=update_cols)
                if update_cols
                else stmt.on_conflict_do_nothing(index_elements=conflict_cols)
            )
    return len(rows)


def get_or_create_team(engine: Engine, source: str, alias: str, country: str | None = None) -> int:
    """Resolve um nome de time vindo de `source` para o id canônico.

    A resolução usa a tabela de aliases; se o alias é novo, o nome
    normalizado decide se ele se junta a um time existente ou cria um novo.
    """
    from edgefinder.data.teamnames import normalize_team_name

    with engine.begin() as conn:
        row = conn.execute(
            select(schema.team_aliases.c.team_id).where(
                schema.team_aliases.c.source == source,
                schema.team_aliases.c.alias == alias,
            )
        ).fetchone()
        if row:
            return int(row[0])

        canonical = normalize_team_name(alias)
        row = conn.execute(
            select(schema.teams.c.id).where(schema.teams.c.name == canonical)
        ).fetchone()
        if row:
            team_id = int(row[0])
        else:
            result = conn.execute(schema.teams.insert().values(name=canonical, country=country))
            pk = result.inserted_primary_key
            assert pk is not None
            team_id = int(pk[0])
        conn.execute(
            schema.team_aliases.insert().values(team_id=team_id, source=source, alias=alias)
        )
        return team_id


def read_df(engine: Engine, sql: str, params: Mapping[str, Any] | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(sql, conn, params=params)


def set_coverage(
    engine: Engine, competition_id: str, dataset: str, status: str, notes: str = ""
) -> None:
    upsert(
        engine,
        schema.data_coverage,
        [
            {
                "competition_id": competition_id,
                "dataset": dataset,
                "status": status,
                "notes": notes,
                "checked_at": datetime.utcnow(),
            }
        ],
        conflict_cols=["competition_id", "dataset"],
    )
