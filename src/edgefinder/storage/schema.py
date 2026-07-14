"""Schema do banco (SQLAlchemy 2.0 Core).

Decisões estruturais:

- `tier` é coluna de primeira classe em `competitions` — o tratamento
  estatístico difere por tier e isso não pode ser um detalhe de aplicação.
- `odds_snapshots` é append-only com `collected_at`: a closing line é o último
  snapshot antes do kickoff, nunca um UPDATE. Odds históricas dos CSVs entram
  com `is_closing` já marcado pela fonte.
- `data_coverage` é a matriz de cobertura (README) como tabela viva: a UI e o
  confidence badge leem daqui o que cada competição realmente tem.
- Chaves naturais + UNIQUE em tudo que é ingerido: rodar a ingestão duas vezes
  não duplica nada (idempotência via INSERT OR REPLACE / ON CONFLICT).
"""

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)

# Convenção de nomes explícita: necessária para migrations determinísticas.
metadata = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)

competitions = Table(
    "competitions",
    metadata,
    Column("id", String, primary_key=True),  # ex.: "ENG-Premier League"
    Column("name", String, nullable=False),
    Column("country", String),
    Column("tier", Integer, nullable=False),
    Column("kind", String, nullable=False, default="league"),  # league|cup|international
)

teams = Table(
    "teams",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String, nullable=False, unique=True),  # nome canônico
    Column("country", String),
)

# Cada fonte chama o mesmo time de um jeito ("Man United", "Manchester Utd",
# "Manchester United"). O alias resolve (source, alias) -> team canônico.
team_aliases = Table(
    "team_aliases",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("team_id", Integer, ForeignKey("teams.id"), nullable=False),
    Column("source", String, nullable=False),
    Column("alias", String, nullable=False),
    UniqueConstraint("source", "alias"),
)

players = Table(
    "players",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String, nullable=False),
    Column("source", String, nullable=False),  # fonte do id nativo
    Column("source_player_id", String, nullable=False),
    UniqueConstraint("source", "source_player_id"),
)

matches = Table(
    "matches",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("competition_id", String, ForeignKey("competitions.id"), nullable=False),
    Column("season", String, nullable=False),  # "2425" ou "2024" (single-year)
    Column("match_date", DateTime, nullable=False),
    Column("home_team_id", Integer, ForeignKey("teams.id"), nullable=False),
    Column("away_team_id", Integer, ForeignKey("teams.id"), nullable=False),
    Column("home_goals", Integer),
    Column("away_goals", Integer),
    Column("status", String, nullable=False, default="scheduled"),  # scheduled|played
    Column("fbref_id", String, unique=True),
    Column("understat_id", String, unique=True),
    UniqueConstraint("competition_id", "season", "match_date", "home_team_id", "away_team_id"),
)

team_match_stats = Table(
    "team_match_stats",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("match_id", Integer, ForeignKey("matches.id"), nullable=False),
    Column("team_id", Integer, ForeignKey("teams.id"), nullable=False),
    Column("is_home", Boolean, nullable=False),
    Column("goals", Integer),
    Column("shots", Integer),
    Column("shots_on_target", Integer),
    Column("corners", Integer),
    Column("fouls", Integer),
    Column("yellow_cards", Integer),
    Column("red_cards", Integer),
    Column("ht_goals", Integer),
    Column("xg", Float),
    Column("np_xg", Float),
    Column("deep_completions", Integer),
    Column("ppda", Float),
    UniqueConstraint("match_id", "team_id"),
)

player_match_stats = Table(
    "player_match_stats",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("match_id", Integer, ForeignKey("matches.id"), nullable=False),
    Column("team_id", Integer, ForeignKey("teams.id")),
    Column("player_id", Integer, ForeignKey("players.id"), nullable=False),
    Column("source", String, nullable=False),  # fbref|understat
    Column("minutes", Integer),
    Column("position", String),
    Column("started", Boolean),
    Column("shots", Integer),
    Column("shots_on_target", Integer),
    Column("goals", Integer),
    Column("assists", Integer),
    Column("yellow_cards", Integer),
    Column("red_cards", Integer),
    Column("fouls_committed", Integer),
    Column("fouls_drawn", Integer),
    Column("tackles_won", Integer),
    Column("interceptions", Integer),
    Column("crosses", Integer),
    Column("xg", Float),
    Column("xa", Float),
    Column("key_passes", Integer),
    UniqueConstraint("match_id", "player_id", "source"),
)

odds_snapshots = Table(
    "odds_snapshots",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("match_id", Integer, ForeignKey("matches.id")),
    # Snapshots de API chegam antes de o jogo existir no banco: os campos raw
    # permitem vincular depois (match_id nulo até o link).
    Column("home_team_raw", String),
    Column("away_team_raw", String),
    Column("commence_time", DateTime),
    Column("source", String, nullable=False),  # football-data.co.uk | theoddsapi
    Column("bookmaker", String, nullable=False),  # pinnacle|bet365|market_max|market_avg|...
    Column("market", String, nullable=False),  # 1x2|ou|ah|player_shots|...
    Column("selection", String, nullable=False),  # home|draw|away|over|under|<player>
    Column("line", Float),  # 2.5 para OU, handicap para AH, linha da prop
    Column("odds_decimal", Float, nullable=False),
    Column("collected_at", DateTime, nullable=False),
    Column("is_closing", Boolean, nullable=False, default=False),
    UniqueConstraint(
        "match_id", "source", "bookmaker", "market", "selection", "line", "collected_at"
    ),
)

predictions = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("match_id", Integer, ForeignKey("matches.id"), nullable=False),
    Column("model", String, nullable=False),
    Column("model_version", String, nullable=False),
    Column("market", String, nullable=False),
    Column("selection", String, nullable=False),
    Column("line", Float),
    Column("probability", Float, nullable=False),
    Column("created_at", DateTime, nullable=False),
    Column("explanation", Text),  # JSON com os drivers da previsão
    UniqueConstraint("match_id", "model", "model_version", "market", "selection", "line"),
)

bets = Table(
    "bets",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("match_id", Integer, ForeignKey("matches.id"), nullable=False),
    Column("market", String, nullable=False),
    Column("selection", String, nullable=False),
    Column("line", Float),
    Column("odds_taken", Float, nullable=False),
    Column("bookmaker", String),
    Column("stake", Float, nullable=False),
    Column("placed_at", DateTime, nullable=False),
    Column("is_paper", Boolean, nullable=False, default=True),
    Column("closing_odds", Float),  # preenchida quando a closing line chega
    Column("clv", Float),  # log(odds_taken / closing_odds_justa)
    Column("result", String),  # win|lose|push|void
    Column("pnl", Float),
)

data_coverage = Table(
    "data_coverage",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("competition_id", String, ForeignKey("competitions.id"), nullable=False),
    Column("dataset", String, nullable=False),  # results|team_stats|player_stats|xg|odds_closing
    Column("status", String, nullable=False),  # ok|partial|missing
    Column("notes", Text),
    Column("checked_at", DateTime, nullable=False),
    UniqueConstraint("competition_id", "dataset"),
)

credit_ledger = Table(
    "credit_ledger",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("api", String, nullable=False),
    Column("used_at", DateTime, nullable=False),
    Column("credits", Integer, nullable=False),
    Column("endpoint", String),
)
