# EdgeFinder

Motor de análise quantitativa para futebol: estima probabilidades calibradas de
eventos (mercados de time e props de jogador), remove o overround das odds e
identifica apostas de valor esperado positivo — **e prova, via backtest
walk-forward honesto, se o modelo bate ou não o mercado**. Se não bate, o
sistema diz isso e não sugere apostas.

> **Aviso.** A esmagadora maioria dos apostadores perde dinheiro no longo
> prazo. Um modelo com EV+ no backtest não garante EV+ no futuro. Este é um
> projeto de ciência de dados aplicada, não uma promessa de lucro. Nunca
> aposte o que você não pode perder.

## Estado atual (backtest de 14/07/2026)

**O MODELO NÃO BATE O MERCADO.** Walk-forward estrito em 9 ligas (25.194
previsões, 2019-20 a 2025-26, 1X2 contra closing line de-vigada por Shin):
log-loss do modelo pior que o do mercado nas 9 de 9 ligas; simulação de
apostas com 8.886 entradas deu yield de **-3,85%** [IC95: -7,0%, -0,8%].
Por isso a trava `backtest_gate()` **bloqueia sugestões de aposta** — o
sistema, hoje, é um instrumento de medição honesto, não uma máquina de
apostar. Detalhes, hipóteses testadas e próximos passos:
[docs/research-log.md](docs/research-log.md). Isso era o resultado esperado
para um Dixon-Coles básico contra mercados líquidos; as alavancas de melhoria
estão priorizadas no research-log (H4).

## Matriz de cobertura real (verificada em 13/07/2026)

Cada célula reflete um teste real contra a fonte, não a documentação dela.
Contexto essencial: **o FBref perdeu todos os dados avançados da Opta
(incluindo xG e histórico) em 20/01/2026** — a matriz abaixo já é o mundo
pós-Opta. Detalhes, evidências e consequências: [PLAN.md](PLAN.md).

Legenda: OK = verificado · PARC = existe com limitações · PROV = provável,
pendente de verificação · NÃO = não existe a custo zero.

| Competição (tier) | Resultados | Stats time/partida | Stats jogador/partida | xG | Escalações | Odds hist. fechamento | Odds correntes (API) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Big 5 europeias (T1) | OK | OK (chutes/SoT/escanteios/faltas/cartões) | OK (FBref + Understat) | OK (Understat) | OK (ESPN) | OK (1X2, OU 2.5, AH) | OK (+ props via casas US) |
| Brasileirão Série A (T1) | OK | PARC (sem escanteios) | OK (FBref) | NÃO | PROV (Cartola) | PARC (só 1X2, desde 2012) | OK (sem props) |
| Brasileirão Série B (T2) | OK | PARC | OK (FBref) | NÃO | PROV | NÃO | PROV |
| Championship / Primeira Liga / Eredivisie (T2) | OK | OK | PROV | NÃO | PROV | OK (1X2, OU 2.5, AH) | OK |
| Champions / Europa League (T2) | OK | PROV | PROV | NÃO | PROV | NÃO | OK |
| Libertadores / Sudamericana (T2) | OK | PROV | PROV | NÃO | PROV | NÃO | OK |
| Copa do Mundo (T3) | OK | PROV | OK (FBref) | NÃO | PROV | NÃO | OK |
| Eurocopa (T3) | OK | PROV | PROV | NÃO | PROV | NÃO | OK |
| Copa América / Eliminatórias (T3) | PROV | PROV | PROV | NÃO | PROV | NÃO | OK |

Consequência honesta (detalhe em PLAN.md, seção 3): **sugestões de aposta só
nos mercados com backtest completo contra closing line histórica** (1X2/OU/AH
das ligas europeias e 1X2 do Brasileirão Série A). Player props e escanteios
são validados por calibração contra resultados + CLV prospectivo com odds
coletadas pelo próprio sistema. O Tier 3 (seleções) nasce **desligado**: não
há odds históricas gratuitas para comprovar CLV, e a regra do projeto é não
sugerir o que não se pode validar.

## Fontes de dados (todas gratuitas)

| Fonte | Papel | Acesso |
| --- | --- | --- |
| FBref (soccerdata) | Calendários, resultados, stats de jogador por partida | Scraping (Chrome headless, 7 s+/req) |
| Understat (soccerdata) | xG por time/jogador/chute (Big 5) | Scraping leve |
| football-data.co.uk | Odds históricas de abertura/fechamento + stats de partida | CSV via httpx |
| The Odds API | Odds correntes (snapshot diário → closing próprio) | Chave gratuita (500 créditos/mês) |
| football-data.org | Fixtures/tabelas (12 comps, inclui BSA) | Chave gratuita (10 req/min) |
| ESPN (soccerdata) | Escalações confirmadas, stats de time | JSON aberto |
| Cartola FC | Status provável/dúvida (Brasileirão) | API não oficial |
| eloratings.net | Elo de seleções (prior do Tier 3) | TSV aberto |
| ClubElo | Elo de clubes europeus (prior entre ligas) | API aberta |

## Uso

```text
uv sync --all-extras          # instala o ambiente (Python 3.12)
uv run edgefinder --help      # CLI
uv run edgefinder daily       # fluxo diario completo: coleta, analisa, registra paper bets, mede CLV
uv run edgefinder sequencias arsenal chelsea --n 5   # "N dos ultimos N" dos dois times
uv run edgefinder bet list    # apostas registradas, CLV e resultados
uv run edgefinder backtest --market 1x2
uv run edgefinder edges --min-ev 0.03
uv run streamlit run src/edgefinder/dashboard/app.py
```

Chaves de API (opcionais, para odds correntes e fixtures): copie
`.env.example` para `.env` e preencha `ODDS_API_KEY` e `FOOTBALL_DATA_ORG_KEY`.

## Operação contínua (Agendador de Tarefas do Windows)

O coletor de odds precisa rodar todo dia — a closing line própria é construída
snapshot a snapshot, e histórico perdido não se recupera de graça. Registre as
duas tarefas (cmd, uma linha cada):

```text
schtasks /Create /TN "EdgeFinder Daily" /TR "C:\Users\luizf\Desktop\PreditorLuib\scripts\publish.cmd" /SC DAILY /ST 10:00
schtasks /Create /TN "EdgeFinder Update" /TR "C:\Users\luizf\Desktop\PreditorLuib\scripts\update.cmd" /SC WEEKLY /D MON /ST 08:00
```

A tarefa diária roda `edgefinder daily` (coleta odds, vincula snapshots,
liquida apostas, analisa o dia, registra paper bets das seleções defensáveis
e atualiza o CLV) e publica `data/reports` para o dashboard.

Justificativa da escolha (vs. Prefect/APScheduler): o Task Scheduler sobrevive
a reboot sem processo Python residente, tem retry e log nativos — Prefect é um
orquestrador de frota (overkill para 1 máquina) e APScheduler morre junto com
o processo num desktop que dorme. Detalhes: PLAN.md, seção 4.5.

## Publicar o painel na internet (grátis)

A opção gratuita que roda Streamlit é o **Streamlit Community Cloud**
(deploy direto de um repositório GitHub). Cloudflare Pages e GitHub Pages
**não servem**: hospedam só conteúdo estático/JS, e o dashboard é um servidor
Python. E o "cérebro" (scraping + backtest) **não pode ir para nuvem nenhuma**:
o FBref bloqueia IPs de datacenter (verificado na Fase 0) — a arquitetura é
o PC local produzir os artefatos e o site publicá-los.

Passos (uma vez):

1. Crie um repositório no GitHub e envie o projeto (`data/reports` já é
   versionado de propósito — é o que o site lê).
2. Entre em <https://share.streamlit.io> com a conta do GitHub.
3. "Create app" → escolha o repositório e branch `main` → Main file path:
   `src/edgefinder/dashboard/app.py` → Deploy.
4. Para atualizar o site: rode `scripts\publish.cmd` (coleta odds, refaz a
   análise e faz commit+push de `data/reports`) — ou agende no Task Scheduler.

Atenção: o repositório precisa ser público no plano gratuito (ou conceda
acesso ao app), e o site mostra tudo que estiver em `data/reports`.

## Estrutura de scripts

| Script | Papel | Duração típica |
| --- | --- | --- |
| `scripts/warm_fast.py` | CSVs de odds + Elo (rede leve) | ~3 min |
| `scripts/warm_understat.py` | Cache Understat (xG, jogadores) | ~2 h por 9 liga-temporadas |
| `scripts/warm_fbref.py` | Cache FBref (jogador por partida; 7 s+/req) | ~80 min por liga-temporada |
| `scripts/load_db.py` | Cache -> SQLite (idempotente, sem rede) | ~15 min |
| `scripts/run_backtest.py` | Backtest walk-forward completo | ~20 min |
