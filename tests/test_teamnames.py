"""Testes da canonicalização de nomes de time (junção entre fontes)."""

import pytest

from edgefinder.data.teamnames import (
    AmbiguousTeamMatch,
    match_team_sets,
    normalize_team_name,
)


def test_variantes_do_mesmo_clube_convergem() -> None:
    variantes = ["Man United", "Manchester Utd", "Manchester United", "Manchester United FC"]
    canonicos = {normalize_team_name(v) for v in variantes}
    assert canonicos == {"manchester united"}


def test_acentos_e_sufixos() -> None:
    assert normalize_team_name("Atlético-MG") == "atletico mineiro"
    assert normalize_team_name("São Paulo FC") == "sao paulo"
    assert normalize_team_name("Bayern München") == "bayern munich"
    assert normalize_team_name("Paris SG") == "paris saint germain"


def test_clubes_diferentes_nao_colidem() -> None:
    assert normalize_team_name("Manchester United") != normalize_team_name("Manchester City")
    assert normalize_team_name("Botafogo") != normalize_team_name("Botafogo SP")


def test_match_team_sets_bijecao_simples() -> None:
    fonte_a = ["Man United", "Wolves", "Nott'm Forest"]
    fonte_b = ["Manchester United", "Wolverhampton Wanderers", "Nottingham Forest"]
    mapping = match_team_sets(fonte_a, fonte_b)
    assert mapping["Man United"] == "Manchester United"
    assert mapping["Wolves"] == "Wolverhampton Wanderers"
    assert mapping["Nott'm Forest"] == "Nottingham Forest"


def test_match_team_sets_por_tokens() -> None:
    mapping = match_team_sets(["Ath Bilbao", "Celta"], ["Athletic Bilbao", "Celta de Vigo"])
    assert mapping["Ath Bilbao"] == "Athletic Bilbao"
    assert mapping["Celta"] == "Celta de Vigo"


def test_sem_candidato_levanta_erro() -> None:
    with pytest.raises(AmbiguousTeamMatch):
        match_team_sets(["Flamengo"], ["Palmeiras"])
