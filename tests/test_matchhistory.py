"""Testes do parser do football-data.co.uk (formatos clássico e 'new')."""

import io
from pathlib import Path

import pandas as pd

from edgefinder.ingest.matchhistory import parse_classic, parse_new_format

CLASSIC_CSV = """Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HS,AS,HST,AST,HC,AC,HF,AF,HY,AY,HR,AR,B365H,B365D,B365A,PSH,PSD,PSA,MaxH,MaxD,MaxA,AvgH,AvgD,AvgA,B365>2.5,B365<2.5,P>2.5,P<2.5,AHh,B365AHH,B365AHA,PAHH,PAHA,B365CH,B365CD,B365CA,PSCH,PSCD,PSCA,MaxCH,MaxCD,MaxCA,AvgCH,AvgCD,AvgCA,B365C>2.5,B365C<2.5,PC>2.5,PC<2.5,AHCh,B365CAHH,B365CAHA,PCAHH,PCAHA
E0,17/08/2024,15:00,Arsenal,Wolves,2,0,H,1,0,15,8,7,2,6,3,10,12,1,2,0,0,1.30,5.50,10.00,1.32,5.60,10.50,1.33,5.75,11.00,1.30,5.40,10.20,1.60,2.40,1.62,2.42,-1.5,1.90,2.00,1.92,1.98,1.28,5.75,11.00,1.31,5.80,11.20,1.33,6.00,11.50,1.29,5.60,10.80,1.57,2.45,1.59,2.47,-1.5,1.88,2.02,1.90,2.00
E0,17/08/2024,17:30,Ipswich,Liverpool,0,2,A,0,0,5,18,1,9,2,8,11,9,2,1,0,0,8.50,5.00,1.36,8.80,5.10,1.37,9.00,5.25,1.38,8.40,4.90,1.35,1.70,2.20,1.72,2.22,1.0,2.05,1.85,2.03,1.87,9.00,5.25,1.33,9.20,5.30,1.34,9.50,5.50,1.36,8.80,5.10,1.32,1.65,2.25,1.67,2.27,1.25,2.00,1.90,1.98,1.92
"""

NEW_CSV = """Country,League,Season,Date,Time,Home,Away,HG,AG,Res,PSCH,PSCD,PSCA,MaxCH,MaxCD,MaxCA,AvgCH,AvgCD,AvgCA,B365CH,B365CD,B365CA
Brazil,Serie A,2024,13/04/2024,21:00,Flamengo,Palmeiras,2,1,H,2.10,3.20,3.80,2.15,3.30,3.90,2.08,3.15,3.70,2.10,3.25,3.75
Brazil,Serie A,2012/2013,20/05/2012,16:00,Fluminense,Vasco,1,0,H,1.80,3.40,4.50,1.85,3.50,4.60,1.78,3.35,4.40,,,
"""


def _df(csv: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(csv))


def test_parse_classic_jogos_e_stats() -> None:
    parsed = parse_classic(_df(CLASSIC_CSV), "E0", "2425")
    assert len(parsed["matches"]) == 2
    m = parsed["matches"][0]
    assert m["home"] == "Arsenal" and m["home_goals"] == 2 and m["status"] == "played"
    stats = next(s for s in parsed["team_stats"] if s["team"] == "Arsenal")
    assert stats["shots"] == 15 and stats["corners"] == 6 and stats["ht_goals"] == 1


def test_parse_classic_odds_abertura_e_fechamento() -> None:
    parsed = parse_classic(_df(CLASSIC_CSV), "E0", "2425")
    odds = [o for o in parsed["odds"] if o["home"] == "Arsenal"]
    pinn_close = [
        o for o in odds if o["bookmaker"] == "pinnacle" and o["market"] == "1x2" and o["is_closing"]
    ]
    assert {o["selection"]: o["odds_decimal"] for o in pinn_close} == {
        "home": 1.31,
        "draw": 5.80,
        "away": 11.20,
    }
    pinn_open = [
        o
        for o in odds
        if o["bookmaker"] == "pinnacle" and o["market"] == "1x2" and not o["is_closing"]
    ]
    assert {o["selection"]: o["odds_decimal"] for o in pinn_open} == {
        "home": 1.32,
        "draw": 5.60,
        "away": 10.50,
    }
    ou_close = [
        o for o in odds if o["market"] == "ou" and o["is_closing"] and o["bookmaker"] == "pinnacle"
    ]
    assert {o["selection"]: o["odds_decimal"] for o in ou_close} == {"over": 1.59, "under": 2.47}
    ah = [o for o in odds if o["market"] == "ah" and o["bookmaker"] == "pinnacle"]
    assert all(o["line"] == -1.5 for o in ah)


def test_parse_new_format(tmp_path: Path) -> None:
    p = tmp_path / "BRA.csv"
    p.write_text(NEW_CSV, encoding="utf-8-sig")
    parsed = parse_new_format(p, "BRA")
    assert len(parsed["matches"]) == 2
    assert parsed["matches"][0]["season"] == "2024"
    assert parsed["matches"][1]["season"] == "1213"  # 2012/2013 -> 1213
    assert parsed["team_stats"] == []  # formato new nao tem stats
    flam = [o for o in parsed["odds"] if o["home"] == "Flamengo"]
    assert all(o["is_closing"] for o in flam)
    pinn = {o["selection"]: o["odds_decimal"] for o in flam if o["bookmaker"] == "pinnacle"}
    assert pinn == {"home": 2.10, "draw": 3.20, "away": 3.80}
    # odds vazias (B365 na linha antiga) nao geram linha
    flum_b365 = [
        o for o in parsed["odds"] if o["home"] == "Fluminense" and o["bookmaker"] == "bet365"
    ]
    assert flum_b365 == []
