# EdgeFinder — contexto do projeto

Motor de análise quantitativa para futebol: probabilidades calibradas vs. odds
de mercado, com backtest walk-forward honesto. Leia PLAN.md antes de mexer em
qualquer coisa — a matriz de cobertura de dados e as decisões técnicas estão
lá, com evidências.

## Princípios inegociáveis

- Honestidade estatística acima de features. Se o backtest não bate o mercado,
  o sistema diz isso e não sugere apostas. Nunca maquiar resultado.
- Custo zero: só fontes gratuitas. Sem trial, sem API paga.
- Nunca dizer que algo funciona sem ter rodado.
- Sem emojis em código, commits e respostas. Docs e respostas em pt-BR;
  commits em inglês (conventional commits).

## Fatos do ambiente de dados (Fase 0, 13/07/2026 — não redescobrir)

- FBref perdeu TODOS os dados avançados da Opta (xG incluído, retroativo) em
  20/01/2026. Só restam stats básicas ("summary") por jogador por partida.
- FBref exige seleniumbase UC + Chrome local (rate limit 7s+jitter; IP
  residencial; máx. 10 req/min ou "jail" de 1 dia). Cache em
  data/raw/soccerdata (SOCCERDATA_DIR — configurado por edgefinder/ingest/_sd.py).
- Understat: única fonte de xG (Big 5 apenas). Funciona via soccerdata 1.9.0.
- football-data.co.uk: NUNCA usar via soccerdata (o tls-client da lib leva 503
  do WAF); usar o cliente httpx próprio (edgefinder/ingest/matchhistory.py).
- The Odds API: 500 CRÉDITOS/mês (custo = mercados x regiões por chamada);
  histórico é pago. Closing line própria = snapshots diários acumulados.
- Ligas custom do FBref (Brasileirão etc.): league_dict.json exige o NOME
  EXATO da competição com acentos (ver ingest/_sd.py).

## Comandos

- `uv sync --all-extras` / `uv run pytest` / `uv run ruff check src tests` /
  `uv run mypy`
- Aquecer cache (lento, FBref ~80 min por liga-temporada de player stats):
  `uv run python scripts/warm_fbref.py` (idem warm_understat, warm_fast)
- Carga: `uv run edgefinder ingest ...`; backtest: `uv run edgefinder backtest ...`
- Dashboard: `uv run streamlit run src/edgefinder/dashboard/app.py`

## Regras de trabalho

- Claude NÃO executa git commit/push/branch — entrega os comandos prontos
  (cmd do Windows, uma linha, sem acentos na mensagem).
- Toda hipótese de modelagem testada entra em docs/research-log.md ANTES do
  teste, com o resultado depois (inclusive negativo) — antídoto de overfitting
  por tentativa e erro.
- mypy --strict e ruff limpos; cobertura >= 80% no núcleo estatístico.
- Testes não tocam rede nem banco real: fixtures sintéticas.
