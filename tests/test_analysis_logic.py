"""Testes da lógica de decisão da análise do dia (funções puras).

Cobrem exatamente o que o usuário vê: o Elo de seleções (que já produziu
probabilidade negativa em mismatch extremo), o veredicto de cada seleção e a
trava de honestidade do backtest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edgefinder.backtest.runner import _verdict
from edgefinder.config import settings
from edgefinder.edge.analysis import _build_row, _elo_probs_1x2
from edgefinder.edge.today import backtest_gate


class TestEloProbs:
    def test_probabilidades_validas_em_jogo_parelho(self) -> None:
        p_home, p_draw, p_away = _elo_probs_1x2(1800.0, 1800.0)
        assert p_home == pytest.approx(p_away, abs=1e-12)
        assert p_draw == pytest.approx(0.27, abs=1e-6)
        assert p_home + p_draw + p_away == pytest.approx(1.0, abs=1e-12)

    def test_mismatch_extremo_nao_produz_probabilidade_negativa(self) -> None:
        # dr = -800: sem o clip, p_home saia negativo e p_away > 1
        p_home, p_draw, p_away = _elo_probs_1x2(1200.0, 2000.0)
        assert p_home >= 0.0
        assert p_away <= 1.0
        assert p_home + p_draw + p_away == pytest.approx(1.0, abs=1e-12)

    def test_favorito_tem_mais_probabilidade(self) -> None:
        p_home, _, p_away = _elo_probs_1x2(2000.0, 1700.0)
        assert p_home > p_away

    def test_simetria(self) -> None:
        ph1, pd1, _pa1 = _elo_probs_1x2(1900.0, 1600.0)
        _ph2, pd2, pa2 = _elo_probs_1x2(1600.0, 1900.0)
        assert ph1 == pytest.approx(pa2, abs=1e-12)
        assert pd1 == pytest.approx(pd2, abs=1e-12)


class TestBuildRow:
    def _row(self, **kwargs: object) -> dict[str, object]:
        base: dict[str, object] = {
            "match_id": 1,
            "home": "alfa",
            "away": "beta",
            "comp": "ENG-Premier League",
            "kickoff": "2026-07-20 15:00",
            "market": "1x2",
            "selection": "home",
            "line": 0.0,
            "best_odds": 2.10,
            "n_books": 5,
            "p_consensus": 0.50,
            "p_model": None,
            "model_label": "sem modelo",
            "form": {"alfa": "3V", "beta": "1V"},
        }
        base.update(kwargs)
        return _build_row(**base)  # type: ignore[arg-type]

    def test_preco_acima_do_justo_sem_modelo_contra_e_defensavel(self) -> None:
        row = self._row(best_odds=2.10, p_consensus=0.50)  # ev_consenso = +5%
        assert row["veredicto"] == "defensavel"

    def test_modelo_gritando_contra_derruba_para_neutro_ou_evitar(self) -> None:
        row = self._row(best_odds=2.10, p_consensus=0.50, p_model=0.40)  # ev_modelo = -16%
        assert row["veredicto"] == "evitar"

    def test_pagando_caro_e_evitar(self) -> None:
        row = self._row(best_odds=1.80, p_consensus=0.50)  # ev_consenso = -10%
        assert row["veredicto"] == "evitar"

    def test_levemente_caro_e_neutro(self) -> None:
        row = self._row(best_odds=1.96, p_consensus=0.50)  # ev_consenso = -2%
        assert row["veredicto"] == "neutro"

    def test_score_capa_ev_de_modelo_absurdo(self) -> None:
        # EV de modelo de +80% e tratado como erro de estimacao: capado em 25%
        row_absurdo = self._row(p_model=0.90, best_odds=2.0, p_consensus=0.50)
        row_alto = self._row(p_model=0.625, best_odds=2.0, p_consensus=0.50)  # ev = +25%
        assert row_absurdo["score"] == pytest.approx(float(row_alto["score"]))  # type: ignore[arg-type]

    def test_sequencia_entra_na_explicacao_e_na_coluna(self) -> None:
        row = self._row(streak="over 2.5: alfa 4/5, beta 3/5 (7/10)")
        assert row["sequencia"] == "over 2.5: alfa 4/5, beta 3/5 (7/10)"
        assert "sequencia: over 2.5" in str(row["explicacao"])


class TestBacktestGate:
    @pytest.fixture()
    def reports_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.setattr(settings, "reports_dir", tmp_path)
        return tmp_path

    def test_sem_arquivo_bloqueia(self, reports_dir: Path) -> None:
        approved, verdict = backtest_gate()
        assert not approved
        assert "Nenhum backtest" in verdict

    def test_veredito_negativo_bloqueia(self, reports_dir: Path) -> None:
        (reports_dir / "backtest_summary.json").write_text(
            json.dumps({"verdict": "O MODELO NAO BATE O MERCADO: yield <= 0."}),
            encoding="utf-8",
        )
        approved, _ = backtest_gate()
        assert not approved

    def test_inconclusivo_bloqueia(self, reports_dir: Path) -> None:
        (reports_dir / "backtest_summary.json").write_text(
            json.dumps({"verdict": "INCONCLUSIVO: yield positivo mas NAO significativo"}),
            encoding="utf-8",
        )
        approved, _ = backtest_gate()
        assert not approved

    def test_veredito_vazio_bloqueia(self, reports_dir: Path) -> None:
        (reports_dir / "backtest_summary.json").write_text(json.dumps({}), encoding="utf-8")
        approved, _ = backtest_gate()
        assert not approved

    def test_veredito_positivo_libera(self, reports_dir: Path) -> None:
        (reports_dir / "backtest_summary.json").write_text(
            json.dumps({"verdict": "Yield positivo e significativo (p=0.010) NESTE backtest."}),
            encoding="utf-8",
        )
        approved, _ = backtest_gate()
        assert approved


class TestRunnerVerdict:
    def test_yield_negativo_e_nao_bate(self) -> None:
        v = _verdict({"significance": {"p_value": 0.01, "mean": -0.04}})
        assert "NAO BATE" in v

    def test_yield_positivo_sem_significancia_e_inconclusivo(self) -> None:
        v = _verdict({"significance": {"p_value": 0.30, "mean": 0.02}})
        assert "INCONCLUSIVO" in v

    def test_yield_positivo_significativo_libera_com_ressalva(self) -> None:
        v = _verdict({"significance": {"p_value": 0.01, "mean": 0.03}})
        assert "NAO BATE" not in v
        assert "INCONCLUSIVO" not in v
        assert "nao garante o futuro" in v

    def test_sem_significancia_usa_yield_per_bet(self) -> None:
        v = _verdict({"yield_per_bet": -0.02})
        assert "NAO BATE" in v


class TestMinEvForTier:
    def test_tiers_conhecidos(self) -> None:
        assert settings.min_ev_for_tier(1) == settings.min_ev_tier1
        assert settings.min_ev_for_tier(2) == settings.min_ev_tier2
        assert settings.min_ev_for_tier(3) == settings.min_ev_tier3

    def test_tier_desconhecido_recebe_o_threshold_hostil(self) -> None:
        assert settings.min_ev_for_tier(99) == settings.min_ev_tier3
