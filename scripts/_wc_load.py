import sys

sys.path.insert(0, "src")
from edgefinder.ingest.etl import load_fbref_matches
from edgefinder.storage.repository import get_engine, read_df

engine = get_engine()
n = load_fbref_matches(engine, "INT-World Cup", "2026")
print(f"WC 2026: {n} jogos processados")
print(
    read_df(
        engine,
        "SELECT status, COUNT(1) AS n FROM matches WHERE competition_id='INT-World Cup' GROUP BY status",
    )
)
