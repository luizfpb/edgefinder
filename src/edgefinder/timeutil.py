"""Tempo do sistema em um único padrão: UTC-naive.

O banco inteiro guarda datetimes UTC sem tzinfo (o SQLite não tem tipo com
fuso). Misturar `datetime.now()` (hora local, UTC-3 no Brasil) com esses
valores desloca filtros de "jogo futuro" em ~3h perto do kickoff — bug real
que este módulo existe para impedir. Toda comparação com colunas do banco
deve usar `utcnow_naive()`.
"""

from datetime import datetime, timezone


def utcnow_naive() -> datetime:
    """Agora em UTC, sem tzinfo — o formato canônico do banco."""
    # timezone.utc em vez de datetime.UTC (3.11+): o Streamlit Cloud roda o
    # dashboard num Python mais antigo que o 3.12 do desenvolvimento, e este
    # módulo está na cadeia de import da aba de sequências.
    return datetime.now(timezone.utc).replace(tzinfo=None)  # noqa: UP017
