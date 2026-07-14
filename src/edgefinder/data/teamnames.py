"""Canonicalização de nomes de time entre fontes.

O problema: cada fonte nomeia o mesmo clube de um jeito ("Man United",
"Manchester Utd", "Manchester United"). A junção entre fontes é por nome — não
existe id compartilhado.

A estratégia tem duas camadas, ambas determinísticas:

1. `normalize_team_name`: normalização de texto (minúsculas, sem acento, sem
   sufixos de clube) + um mapa curado de apelidos consagrados. Resolve a
   maioria dos casos.
2. `match_team_sets`: para juntar duas fontes numa liga-temporada, os DOIS
   conjuntos têm os mesmos N clubes. Casamos por similaridade de tokens com
   verificação de unicidade — qualquer ambiguidade vira erro explícito, nunca
   um palpite silencioso (um palpite errado aqui corrompe features rio abaixo).
"""

import re
import unicodedata

# Apelidos consagrados -> forma canônica (aplicado APÓS a normalização básica).
_CURATED: dict[str, str] = {
    "man united": "manchester united",
    "man utd": "manchester united",
    "manchester utd": "manchester united",
    "man city": "manchester city",
    "wolves": "wolverhampton wanderers",
    "wolverhampton": "wolverhampton wanderers",
    "spurs": "tottenham hotspur",
    "tottenham": "tottenham hotspur",
    "newcastle": "newcastle united",
    "west ham": "west ham united",
    "west brom": "west bromwich albion",
    "brighton": "brighton hove albion",
    "brighton and hove albion": "brighton hove albion",
    "nottm forest": "nottingham forest",
    "nott ham forest": "nottingham forest",
    "nott m forest": "nottingham forest",
    "nottingham": "nottingham forest",
    "sheffield utd": "sheffield united",
    "leeds": "leeds united",
    "leicester": "leicester city",
    "norwich": "norwich city",
    "ipswich": "ipswich town",
    "luton": "luton town",
    "psg": "paris saint germain",
    "paris s g": "paris saint germain",
    "paris sg": "paris saint germain",
    "marseille": "olympique marseille",
    "lyon": "olympique lyonnais",
    "olympique lyon": "olympique lyonnais",
    "saint etienne": "saint etienne",
    "st etienne": "saint etienne",
    "inter": "internazionale",
    "inter milan": "internazionale",
    "milan": "ac milan",
    "ath madrid": "atletico madrid",
    "atl madrid": "atletico madrid",
    "ath bilbao": "athletic bilbao",
    "athletic club": "athletic bilbao",
    "betis": "real betis",
    "sociedad": "real sociedad",
    "celta": "celta vigo",
    "espanol": "espanyol",
    "vallecano": "rayo vallecano",
    "alaves": "deportivo alaves",
    "la coruna": "deportivo la coruna",
    "leverkusen": "bayer leverkusen",
    "bayern": "bayern munich",
    "bayern munchen": "bayern munich",
    "dortmund": "borussia dortmund",
    "m gladbach": "borussia monchengladbach",
    "gladbach": "borussia monchengladbach",
    "monchengladbach": "borussia monchengladbach",
    "ein frankfurt": "eintracht frankfurt",
    "frankfurt": "eintracht frankfurt",
    "hamburger sv": "hamburg",
    "nuernberg": "nurnberg",
    "fc nurnberg": "nurnberg",
    "duesseldorf": "fortuna dusseldorf",
    "fortuna duesseldorf": "fortuna dusseldorf",
    "moenchengladbach": "borussia monchengladbach",
    "greuther fuerth": "greuther furth",
    "wuerzburger kickers": "wurzburger kickers",
    "fc koln": "koln",
    "cologne": "koln",
    "mainz 05": "mainz",
    "rb leipzig": "rb leipzig",
    "leipzig": "rb leipzig",
    "hoffenheim": "tsg hoffenheim",
    "st pauli": "st pauli",
    "stuttgart": "vfb stuttgart",
    "wolfsburg": "vfl wolfsburg",
    "bochum": "vfl bochum",
    "athletico pr": "athletico paranaense",
    "athletico-pr": "athletico paranaense",
    "atletico-mg": "atletico mineiro",
    "atletico mg": "atletico mineiro",
    "atletico-go": "atletico goianiense",
    "atletico go": "atletico goianiense",
    "america-mg": "america mineiro",
    "america mg": "america mineiro",
    "bragantino": "red bull bragantino",
    "rb bragantino": "red bull bragantino",
    "vasco": "vasco da gama",
    "vasco gama": "vasco da gama",
    "flamengo rj": "flamengo",
    "fluminense rj": "fluminense",
    "atl goianiense": "atletico goianiense",
    "operario pr": "operario",
    "gremio novorizontino": "novorizontino",
    "sao paulo fc": "sao paulo",
    "botafogo rj": "botafogo",
    "botafogo fr": "botafogo",
}

# Tokens que não distinguem clubes (sufixos jurídicos e afixos comuns).
_STOP_TOKENS = {
    "fc",
    "cf",
    "afc",
    "cfc",
    "ac",
    "as",
    "ss",
    "ssc",
    "us",
    "sc",
    "sv",
    "vfb",
    "vfl",
    "fsv",
    "tsv",
    "rcd",
    "rc",
    "cd",
    "ca",
    "club",
    "clube",
    "de",
    "futebol",
    "regatas",
    "esporte",
    "esportes",
    "1",
    "04",
    "05",
    "1899",
    "1846",
    "1860",
    "e",
}


def normalize_team_name(name: str) -> str:
    """Normaliza um nome de time para a forma canônica de junção."""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    # o en-dash e intencional: o FBref separa clube-estado com U+2013 ("America-MG")
    text = re.sub(r"[''`\.\-/–—]", " ", text)  # noqa: RUF001
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text in _CURATED:
        return _CURATED[text]
    tokens = [t for t in text.split() if t not in _STOP_TOKENS]
    result = " ".join(tokens) if tokens else text
    return _CURATED.get(result, result)


def _token_set(name: str) -> frozenset[str]:
    return frozenset(normalize_team_name(name).split())


class AmbiguousTeamMatch(ValueError):
    """Dois candidatos igualmente plausíveis: exige curadoria manual."""


def match_team_sets(source_a: list[str], source_b: list[str]) -> dict[str, str]:
    """Casa dois conjuntos de nomes da MESMA liga-temporada (bijeção esperada).

    Retorna {nome_em_a: nome_em_b}. Estratégia: igualdade canônica primeiro;
    depois melhor sobreposição de Jaccard entre conjuntos de tokens, exigindo
    vencedor único e não-empatado. Ambiguidade -> exceção com os candidatos.
    """
    result: dict[str, str] = {}
    remaining_b = list(source_b)

    for a in list(source_a):
        canon_a = normalize_team_name(a)
        exact = [b for b in remaining_b if normalize_team_name(b) == canon_a]
        if len(exact) == 1:
            result[a] = exact[0]
            remaining_b.remove(exact[0])

    for a in source_a:
        if a in result:
            continue
        tokens_a = _token_set(a)
        scored: list[tuple[float, str]] = []
        for b in remaining_b:
            tokens_b = _token_set(b)
            union = tokens_a | tokens_b
            score = len(tokens_a & tokens_b) / len(union) if union else 0.0
            scored.append((score, b))
        scored.sort(reverse=True)
        if not scored or scored[0][0] == 0.0:
            raise AmbiguousTeamMatch(f"Sem candidato para '{a}' entre {remaining_b}")
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            raise AmbiguousTeamMatch(
                f"Empate para '{a}': {scored[0][1]} vs {scored[1][1]} (score {scored[0][0]:.2f})"
            )
        result[a] = scored[0][1]
        remaining_b.remove(scored[0][1])

    return result
