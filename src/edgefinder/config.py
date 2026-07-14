"""Configuração central do EdgeFinder (pydantic-settings).

Tudo que é ajustável vive aqui: caminhos, chaves de API (via .env), tiers das
competições e thresholds de EV por tier. O resto do código importa `settings`
e nunca lê variável de ambiente diretamente.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- caminhos -----------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    db_path: Path = PROJECT_ROOT / "data" / "edgefinder.db"
    models_dir: Path = PROJECT_ROOT / "data" / "models"
    reports_dir: Path = PROJECT_ROOT / "data" / "reports"

    # --- chaves (opcionais; sem elas os coletores de API ficam inativos) ----
    odds_api_key: str = ""
    football_data_org_key: str = ""

    # --- The Odds API: orçamento de créditos --------------------------------
    # 500/mês no free tier; custo por chamada = mercados x regiões.
    odds_api_monthly_budget: int = 500

    # --- política de risco (Fase 5+) ----------------------------------------
    # Kelly fracionário: Kelly cheio maximiza o crescimento logarítmico apenas
    # se p for exata. Como p vem com erro de estimativa, Kelly cheio
    # sobre-aposta sistematicamente (o custo de apostar demais é maior que o
    # de apostar de menos: a função de crescimento é assimétrica em torno do
    # ótimo). 1/4 de Kelly mantém ~93% do crescimento com ~1/4 da variância.
    kelly_fraction: float = 0.25
    max_exposure_per_match: float = 0.05  # fração do bankroll
    max_exposure_per_day: float = 0.15

    # --- thresholds de EV por tier (Tier 3 é hostil por construção) ---------
    min_ev_tier1: float = 0.03
    min_ev_tier2: float = 0.05
    min_ev_tier3: float = 0.10

    # --- Dixon-Coles: fonte única dos hiperparâmetros de produção ------------
    # Meia-vida ótima medida no H1 (research-log): ~365d, curva rasa 180-730.
    dc_half_life_days: float = 365.0
    # >= 2 temporadas de treino mínimo: com menos, o DC produz parâmetros
    # ruidosos e "edges" gigantes que são erro de estimação, não valor.
    dc_min_train_matches: int = 760

    # --- sequências ("N dos últimos N"): tamanho default da janela ----------
    streak_window: int = 5

    def min_ev_for_tier(self, tier: int) -> float:
        """Threshold de EV do tier (3 é o default hostil para tier desconhecido)."""
        return {1: self.min_ev_tier1, 2: self.min_ev_tier2, 3: self.min_ev_tier3}.get(
            tier, self.min_ev_tier3
        )

    @property
    def sqlalchemy_url(self) -> str:
        return f"sqlite:///{self.db_path}"


settings = Settings()

# Tier é atributo da competição, de primeira classe no schema; este mapa é a
# fonte da verdade usada no seed do banco.
COMPETITION_TIERS: dict[str, int] = {
    "ENG-Premier League": 1,
    "ESP-La Liga": 1,
    "ITA-Serie A": 1,
    "GER-Bundesliga": 1,
    "FRA-Ligue 1": 1,
    "BRA-Serie A": 1,
    "BRA-Serie B": 2,
    "ENG-Championship": 2,
    "POR-Liga Portugal": 2,
    "NED-Eredivisie": 2,
    "EUR-Champions League": 2,
    "SAM-Copa Libertadores": 2,
    "INT-World Cup": 3,
    "INT-European Championship": 3,
    "INT-Copa America": 3,
}
