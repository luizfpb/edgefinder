# PLAN.md — EdgeFinder, Fase 0

Data da sondagem: 13/07/2026, executada desta máquina (IP residencial, Windows 11).
Método: cada fonte foi testada com requisições reais (não com base em documentação).
Scripts de sondagem e saídas brutas ficaram no scratchpad da sessão; as evidências
citadas (issues, PRs, snapshots do Wayback Machine) têm URL neste documento.

---

## 0. Sumário executivo

O projeto é viável, mas **duas premissas do briefing caíram na verificação empírica**:

1. **O FBref perdeu TODOS os dados avançados da Opta em 20/01/2026** — xG, xA,
   npxG, SCA/GCA, shot-level, tabelas de passe/posse/defesa por jogador, e até
   escanteios. A remoção foi retroativa (o histórico também sumiu). Não é
   bloqueio anti-bot: o site inteiro está servindo a versão "básica".
   Fonte: rescisão de contrato pela Stats Perform, 8 dias após ela se tornar
   distribuidora exclusiva de dados da FIFA para a Copa 2026
   (https://awfulannouncing.com/soccer/sports-reference-pulls-advanced-data-agreement-violation-dispute.html).
   Verificado empiricamente: zero ocorrências de xG em schedule, match report,
   matchlogs e season stats servidos hoje.

2. **Não existe fonte gratuita de odds históricas de fechamento para player
   props, escanteios e seleções.** Ponto. Para esses mercados, o backtest
   "contra o mercado" descrito no briefing é impossível a custo zero olhando
   para trás. A consequência honesta está na seção 3.

O que **sobrevive e sustenta o projeto**:

- FBref ainda entrega **stats de jogador por partida** (minutos, posição,
  chutes, chutes no gol, gols, assistências, cartões, faltas, pênaltis) para
  Premier League, **Brasileirão Série A e Série B** e seleções — verificado
  jogo a jogo. É o suficiente para props de chutes/SoT/gols/cartões.
- Understat funciona 100% e vira a **única fonte de xG** (por time, por
  jogador e por chute) — mas só para as 5 grandes ligas europeias.
- football-data.co.uk tem **odds de fechamento históricas** (Pinnacle, Bet365,
  Máx/Média) para 1X2, Over/Under 2.5 e handicap asiático nas ligas europeias,
  mais estatísticas de partida (chutes, escanteios, cartões). Para o
  Brasileirão Série A: fechamento 1X2 desde 2012 (só 1X2, sem stats).
- Dixon-Coles modela **gols**, não xG — o modelo de time não depende do que
  foi perdido. xG entra como refinamento onde existir (Europa).

**Componente mais urgente do projeto**: um coletor diário de odds via The Odds
API, rodando desde a Fase 1. Cada dia sem coletar é um dia a menos de
histórico próprio de closing line para props — o dado que não se compra de
graça depois. Por isso movi a coleta de odds para a Fase 1.

---

## 1. O que foi testado

| Fonte | Como | Resultado em uma linha |
| --- | --- | --- |
| FBref (soccerdata 1.9.0) | ~35 requests reais via seleniumbase UC + Chrome local | Funciona; só dados básicos; 7s+/request |
| Understat (soccerdata) | schedule, team stats, player match stats, shot events | Funciona 100%; só Big 5; barato (1 req/temporada p/ xG de time) |
| Sofascore (soccerdata) | schedule, introspecção da API | Só schedule + tabela; **sem** ratings/lineups via lib |
| ESPN (soccerdata) | schedule, lineup, matchsheet | Escalações grátis + stats de time **com escanteios** |
| WhoScored (soccerdata) | instância + schedule | Anti-bot passa, mas bug de parsing na lib; descartado por ora |
| football-data.co.uk | 5 CSVs + data.php via httpx | Funciona via httpx; **soccerdata quebrado p/ esta fonte (503)** |
| ClubElo | API aberta, 3 requests | Só clubes europeus; **zero clubes brasileiros** |
| football-data.org | /v4/competitions sem chave | Free tier: 12 comps, inclui Brasileirão A, CL, Copa, Euro |
| The Odds API | contrato de erro + docs oficiais | 500 créditos/mês; histórico é pago; props só Big 5 + MLS |
| eloratings.net | GET World.tsv | TSV aberto com Elo de 244 seleções |
| Cartola FC API | pesquisa (não sondado ao vivo) | Provável escalação do BR-A; verificar na Fase 1 |

Configuração persistente já criada pela sondagem:
`C:/Users/luizf/soccerdata/config/league_dict.json` com BRA-Serie A (comp 24),
BRA-Serie B (38), EUR-Champions League (8), SAM-Copa Libertadores (14) — o
campo `FBref` exige o nome exato da competição (com acentos), não o id.

---

## 2. Matriz de cobertura real

Legenda: **OK** = verificado hoje com requisição real · **PARC** = existe com
limitações (descritas) · **PROV** = provável, verificar na fase indicada ·
**NÃO** = não existe a custo zero.

| Competição (tier) | Resultados | Stats time/partida | Stats jogador/partida | xG | Escalações | Odds hist. fechamento | Odds correntes (API) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Big 5 europeias (T1) | OK (FBref, fd.co.uk) | OK (fd.co.uk: chutes/SoT/escanteios/faltas/cartões; FBref: posse/formação) | OK (FBref summary + Understat) | OK (Understat: time/jogador/chute) | OK (ESPN confirmadas) | OK 1X2+OU2.5+AH (Pinnacle/B365/Máx/Méd) | OK h2h/totals; props EPL+top ligas (casas US) |
| Brasileirão Série A (T1) | OK (FBref; fd.org BSA free) | PARC (FBref shooting/misc; **sem escanteios**; ESPN PROV F1) | OK (FBref summary) | **NÃO** | PROV (Cartola prováveis; FBref/ESPN lineup F1) | PARC: só 1X2 desde 2012 (PSCH/Máx/Méd; sem OU/AH) | OK h2h/totals; **props NÃO** |
| Brasileirão Série B (T2) | OK (FBref) | PARC (idem Série A) | OK (FBref summary — melhor que o briefing esperava) | NÃO | PROV | **NÃO** | PROV (verificar chave na F1) |
| Championship, Primeira Liga, Eredivisie (T2) | OK (fd.co.uk) | OK (fd.co.uk completo) | PROV (FBref summary, F2) | NÃO (Understat não cobre) | PROV (ESPN) | OK 1X2+OU2.5+AH | OK h2h/totals |
| Champions/Europa League (T2) | OK (FBref, 189 jogos 24-25; fd.org CL free) | PROV | PROV (FBref summary, F2) | NÃO | PROV | **NÃO** | OK |
| Libertadores/Sudamericana (T2) | OK (FBref, 125 jogos 2024) | PROV | PROV | NÃO | PROV | **NÃO** | OK (chave confirmada p/ Libertadores) |
| Copa do Mundo (T3) | OK (FBref 2022: 64 jogos) | PROV | OK (FBref summary, verificado Qatar-Equador) | NÃO (2026 sem xG: Stats Perform exclusiva FIFA) | PROV | **NÃO** (só um XLSX avulso 2026) | OK (h2h/totals) |
| Eurocopa (T3) | OK (dicionário padrão FBref) | PROV | PROV | NÃO | PROV | **NÃO** | OK |
| Copa América / Eliminatórias (T3) | PROV (league_dict custom; match reports summary confirmados p/ WCQ via snapshot) | PROV | PROV | NÃO (nunca teve p/ WCQ) | PROV | **NÃO** | OK (WCQ Conmebol/Europa) |

Notas da matriz:

- "Odds hist. fechamento" nas ligas europeias: colunas de fechamento existem
  a partir de ~2019-20 no formato clássico (6-7 temporadas — suficiente).
  Validar o corte exato no início da Fase 6.
- Aviso do próprio football-data.co.uk: odds de fechamento da **Pinnacle
  não confiáveis desde 23/07/2025**. Mitigação: usar MáxC/MédC (agregado de
  casas) como benchmark alternativo e comparar os dois.
- Understat: schedule de 1 temporada = 1 request (xG de time das 380 partidas
  de uma vez); player stats + shots = 1 request por jogo.
- FBref: rate limit oficial de 10 req/min ("jail" de até 1 dia se violar);
  soccerdata usa 7s fixos + jitter (~12,5 s/página observados). Backfill de
  player stats = 1 request/jogo → **~80 min por liga-temporada**. Requer
  Chrome (existe: v150) e IP residencial (IPs de datacenter levam 403).

---

## 3. O que o sistema pode prometer, honestamente, por mercado

A pergunta certa não é "que mercados dá para modelar", é "que mercados dá
para **validar**". Três níveis de validação possíveis:

**Nível A — backtest completo contra closing line histórica** (o padrão-ouro
do briefing). Possível para:
- 1X2, Over/Under 2.5 e handicap asiático: Big 5 + Championship + Primeira
  Liga + Eredivisie (2019-20 em diante, e 1X2 de abertura desde ~2000).
- 1X2 do Brasileirão Série A (2012 em diante).

**Nível B — backtest de calibração contra resultados + CLV prospectivo.**
O modelo é validado contra o que aconteceu (Brier, log-loss, curva de
calibração, comparação com baseline ingênuo), e contra o mercado apenas
**daqui para frente**, com odds que nós mesmos coletarmos. Vale para:
- Player props (chutes, SoT, gols, cartões) em qualquer liga.
- Escanteios (Europa: temos as contagens históricas; odds só prospectivas).
- OU/AH do Brasileirão.

**Nível C — nem calibração histórica de mercado, nem odds correntes de props.**
- Tier 3 inteiro (sem odds históricas de seleções a custo zero) e props do
  Brasileirão (The Odds API não tem props fora de Big 5 + MLS).

Consequência aplicada (regra do próprio briefing, antecipada para a Fase 0):
**sugestões de aposta só serão emitidas em mercados nível A** até que o
histórico próprio de odds acumule o suficiente para promover mercados do
nível B via CLV comprovado. O Tier 3 nasce desligado por construção, não por
opção. Props do Brasileirão: o modelo roda e publica probabilidades com
explicação, mas sem EV/stake — não há odd de API para comparar (a interface
aceitará entrada manual de odd para consulta pontual).

---

## 4. Decisões técnicas (com justificativa)

1. **Python 3.12 (pin), gerenciado por uv.** O 3.14 do sistema fica fora:
   PyMC/PyTensor ainda não o suportam; soccerdata exige <3.15. O uv baixa e
   isola o 3.12 sozinho (`requires-python = ">=3.12,<3.13"`).

2. **soccerdata 1.9.0 como cliente de FBref e Understat; NÃO usar para
   football-data.co.uk.** Verificado: o reader `MatchHistory` usa tls-client,
   cujo fingerprint TLS o WAF do site bloqueia com 503 (5/5 tentativas),
   enquanto httpx puro recebe 200. Além disso, para a temporada corrente o
   soccerdata força `no_cache=True`, ou seja, quebraria sempre. Escreveremos
   um coletor próprio (~60 linhas): URLs previsíveis
   (`/mmz4281/{season}/{div}.csv`, `/new/{PAIS}.csv`), encoding `utf-8-sig`
   para 2425+ e `latin-1` para temporadas antigas (verificado).

3. **Sofascore sai do núcleo.** Via soccerdata só há schedule e tabela — os
   ratings e lineups do briefing não são acessíveis pela lib. Chamar a API
   não documentada diretamente é frágil e juridicamente cinza; não vamos
   construir dependência disso. Substitutos: **ESPN** (escalações confirmadas
   + stats de time com escanteios, verificado; sem anti-bot) e **Cartola FC**
   (status Provável/Dúvida/Suspenso/Contundido por rodada do BR-A — o melhor
   proxy gratuito de escalação provável; verificar na Fase 1).

4. **xG por fonte, com o modelo tolerante à ausência.** Understat para as Big
   5 (time + jogador + chute). Brasileirão e copas: **sem xG** — o Dixon-Coles
   opera sobre gols (é a formulação original), e props operam sobre contagens
   (chutes/SoT/gols/cartões) que temos por jogador por partida. xG entra como
   feature adicional onde existir, nunca como dependência do pipeline. Isso
   também elimina a assimetria "Série A do Brasil de segunda classe" — hoje
   nenhuma competição tem xG no FBref.

5. **Orquestração: Agendador de Tarefas do Windows chamando a CLI, não
   Prefect nem APScheduler.** Justificativa: Prefect é um servidor + UI para
   orquestrar frotas — overkill absurdo para 1 máquina. APScheduler exige um
   processo Python vivo 24/7, que morre a cada reboot/suspensão de um desktop.
   O Task Scheduler sobrevive a reboot, tem retry nativo e loga — e o comando
   agendado é só `edgefinder collect odds`. APScheduler fica como opção
   futura apenas para polling intra-dia (steam move), se chegarmos lá.

6. **SQLite + SQLAlchemy 2.0 Core**, `tier` como coluna de primeira classe,
   e snapshots de odds com timestamp (`odds_snapshots` é append-only — closing
   line = último snapshot antes do kickoff). Alembic para migrations.

7. **Cache bruto**: `SOCCERDATA_DIR` apontado para `data/raw/soccerdata`
   (gitignored) para o cache da lib viver dentro do projeto; CSVs do
   football-data.co.uk também em `data/raw/`. Dado bruto nunca é deletado —
   com o precedente da Opta, **o cache é um ativo**: o que está salvo hoje
   pode não estar disponível amanhã.

8. **PyMC com plano B explícito.** PyTensor no Windows depende de toolchain
   C para performance; se a compilação for dolorosa, o fallback é numpyro
   (NUTS via JAX, CPU no Windows funciona) com o mesmo modelo hierárquico.
   Decidido na Fase 4, com benchmark; a interface do módulo `hierarchical.py`
   não muda.

9. **Duas chaves gratuitas a registrar (ação sua, seção 10)**:
   football-data.org (fixtures/tabelas, 10 req/min, cobre BSA/CL/WC/EC) e
   The Odds API (500 créditos/mês).

10. **Orçamento de créditos The Odds API** (custo = mercados × regiões por
    chamada; histórico é pago — confirmado na doc oficial):
    - Snapshot diário h2h+totals, região eu, BR-A + EPL: 2×1×2 ligas×30d = 120/mês.
    - Props EPL (player_shots, player_shots_on_target, região us, endpoint
      por evento): ~10 jogos/rodada × 2 mercados × 2 snapshots (T-24h e
      T-1h) ≈ 160/mês.
    - Total ≈ 280 de 500 — folga para ajustes. Escala-se depois; o
      contador vem nos headers (`x-requests-remaining`).

11. **Qualidade**: ruff + mypy --strict + pytest (cobertura ≥80% no núcleo
    estatístico) + CI GitHub Actions com smoke test do pipeline usando
    fixtures de cache gravadas (CI não pode depender de scraping ao vivo —
    IPs do GitHub Actions levam 403 do FBref, verificado em issues da lib).

---

## 5. Discordâncias e correções ao briefing

1. **"FBref dá xG, xA, passes, desarmes por partida"** — não dá mais desde
   20/01/2026. Dá: minutos, posição, chutes, SoT, gols, assistências,
   cartões, faltas, pênaltis sofridos/cometidos, cruzamentos, desarmes com
   sucesso (TklW), interceptações. Sem xG, sem histórico avançado.

2. **"Sofascore dá ratings, lineups"** — não via soccerdata 1.9.0 (só
   schedule/tabela). Removido do núcleo; ESPN + Cartola no lugar.

3. **"The Odds API: 500 req/mês"** — são 500 **créditos**/mês e 1 chamada
   custa `mercados × regiões`. Odds **históricas são plano pago** (a doc é
   explícita). E props de futebol só existem para Big 5 + MLS via casas dos
   EUA — **não há props de Brasileirão** em API gratuita.

4. **O backtest de props contra closing line histórica que você pediu é
   impossível a custo zero.** Ninguém arquiva props de graça (OddsPortal/
   BetExplorer não cobrem props; scrapers maduros confirmam). A redefinição
   honesta está na seção 3 (níveis A/B/C). Se isso for inaceitável, as
   alternativas são pagar por histórico ou aceitar meses de coleta própria
   antes de validar props contra o mercado.

5. **Tier 3 nasce desligado**, não "com badge vermelho": sem odds históricas
   de seleções, não há como demonstrar CLV positivo no backtest — e a sua
   própria regra manda desligar nesse caso. Reavaliação quando houver
   histórico próprio coletado (Copa 2026 está em andamento agora — o coletor
   da Fase 1 já pode capturar esses jogos).

6. **Escanteios**: sem odds históricas em lugar nenhum, e contagens
   históricas só nas ligas europeias do formato clássico. FBref não tem mais
   escanteios. Mercado de escanteios do BR: inviável por ora (nem contagem
   histórica gratuita confirmada — ESPN pode suprir; verificar na Fase 1).

7. **Prefect OU APScheduler** — nenhum dos dois (seção 4.5).

8. **ClubElo não serve para o Brasil** (zero clubes brasileiros, verificado).
   O fator de força entre ligas para Libertadores/copas será estimado pelo
   nosso próprio modelo hierárquico a partir dos confrontos inter-liga, com
   Elo próprio calculado dos resultados como prior — não haverá atalho de
   ClubElo fora da Europa. Para seleções, eloratings.net (TSV aberto,
   verificado) entra como prior fraco do Tier 3.

9. **Understat cobre 6 ligas, não 5** (Big 5 + liga russa, não exposta pelo
   soccerdata). Irrelevante para o escopo, registrado por precisão.

---

## 6. Arquitetura revisada

Mudanças em relação à árvore do briefing (o resto permanece):

```
src/edgefinder/
├── ingest/
│   ├── base.py            # Protocol comum + retry/backoff + rate limit
│   ├── fbref.py           # via soccerdata (seleniumbase UC, 7s/req)
│   ├── understat.py       # via soccerdata (xG Europa)
│   ├── matchhistory.py    # httpx PRÓPRIO (football-data.co.uk; soccerdata quebrado)
│   ├── espn.py            # escalações + stats de time (escanteios)
│   ├── cartola.py         # prováveis do Brasileirão (API não oficial)
│   ├── eloratings.py      # Elo de seleções (TSV aberto)
│   ├── fixtures.py        # football-data.org (chave free)
│   ├── odds.py            # The Odds API: snapshots com orçamento de créditos
│   └── cache.py           # TTL por recurso + cache bruto imutável
```

- `odds_snapshots` append-only com `collected_at`; closing = último snapshot
  pré-kickoff. Tabela `credit_ledger` simples para rastrear consumo da The
  Odds API contra o teto mensal.
- `storage/schema.py`: `competitions.tier` (1/2/3) e `data_coverage`
  (competição × dataset × status) — a matriz da seção 2 vira tabela viva no
  banco e alimenta o confidence badge e a UI ("este mercado não tem odds
  históricas; validação nível B").
- WhoScored, FotMob e Sofascore-API-direta: fora. FotMob foi removido do
  soccerdata a pedido do próprio site; WhoScored tem bug na lib e exige
  Chrome visível; Sofascore direto é frágil/cinza.

---

## 7. Riscos e mitigações

| Risco | Impacto | Mitigação |
| --- | --- | --- |
| FBref degradar mais (paywall, novo provedor só pago) | Perda da fonte de player stats | Cache bruto imutável desde o dia 1; backfill cedo das temporadas-alvo; schema desacoplado da fonte |
| "Jail" de IP no FBref (10 req/min) | 1 dia sem coleta | 7s+jitter do soccerdata (nunca sobrescrever para menos); backfills longos em lotes noturnos com checkpoint |
| WAF do football-data.co.uk se estender ao httpx | Perda das odds históricas | Baixar TODO o histórico-alvo na Fase 1 e versionar em `data/raw` (são CSVs pequenos, ~100 KB/temporada) |
| Pinnacle closing não confiável pós-07/2025 | Benchmark de CLV viciado | Duplo benchmark: PSCH e MáxC/MédC; reportar os dois |
| Docs conflitantes sobre histórico da The Odds API no free tier | Planejamento de coleta | Testar com a chave real na Fase 1; assumir "pago" até prova contrária |
| Cartola/ESPN mudarem API não oficial | Perda de prováveis/escanteios BR | Degradação graciosa: props BR caem para "minutos estimados por histórico" e escanteios BR ficam fora |
| PyMC/PyTensor no Windows | Fase 4 emperrar | Plano B numpyro decidido por benchmark; interface do módulo estável |
| Overfitting por tentativa e erro no backtest | Sistema que "mente" | Diário de pesquisa versionado (`docs/research-log.md`): toda hipótese registrada ANTES do teste, com resultado (inclusive negativo) |
| Cobertura PROV da matriz não se confirmar | Escopo encolher | Cada PROV tem fase dona (F1/F2); a matriz no banco é atualizada por teste automatizado, não por fé |

---

## 8. Fases revisadas (entregas e critérios de aceite)

- **Fase 1 — Ingestão + schema + cache + COLETOR DE ODDS.**
  Schema completo (com `tier` e `data_coverage`), coletores FBref/Understat/
  MatchHistory-httpx/fixtures, backfill de 3+ temporadas do Tier 1 europeu e
  BR-A, download integral do histórico football-data.co.uk alvo, coletor
  diário The Odds API agendado no Task Scheduler, verificação Cartola/ESPN
  (os PROV da matriz com dono na F1). Aceite: banco populado e idempotente
  (rodar 2x não duplica), snapshots de odds chegando há 3+ dias, testes de
  ingestão passando.
- **Fase 2 — Features + testes de leakage.** Decaimento, força de adversário,
  casa/fora, minutos esperados, descanso, estabilidade por stat. Aceite:
  teste que FALHA se qualquer feature enxergar o futuro; cobertura ≥80%.
- **Fase 3 — Dixon-Coles + bivariado.** Aceite: log-loss out-of-sample ≤
  benchmark ingênuo e ≤ mercado de-vigado em pelo menos validação preliminar
  (senão, dizer em letras garrafais).
- **Fase 4 — Props + hierárquico.** Poisson vs NB vs Poisson-Gamma por
  AIC/BIC + log-loss OOS; PyMC ou numpyro por benchmark.
- **Fase 5 — De-vig (proporcional/aditivo/Shin) + EV + Kelly fracionário.**
- **Fase 6 — Backtest walk-forward + calibração + Monte Carlo.** Métricas
  completas, IC e p-valor do yield, simulação de bankroll, e o veredito
  honesto por mercado (nível A/B/C da seção 3).
- **Fase 7 — CLI + dashboard** (explicabilidade, badges, aviso obrigatório).
- **Fase 8 — CLV tracking + line shopping + steam move + CI completo.**

Commits atômicos por fase, mensagens em inglês, formato convencional — os
comandos git serão entregues prontos para você colar (regra do ambiente).

---

## 9. Ações que dependem de você

1. **OK (ou objeções) a este plano** — em particular à seção 3 (níveis de
   validação) e à seção 5 (correções de escopo).
2. Registrar duas chaves gratuitas quando aprovar a Fase 1:
   - https://www.football-data.org/client/register
   - https://the-odds-api.com/ (botão "Get API Key", plano free)
   As chaves vão em `.env` (nunca no git; `.env.example` documenta).
3. Decisão de risco (pode ficar para depois): aceitar ou não scraping de
   BetExplorer/OddsPortal para odds históricas de seleções (viola ToS,
   anti-bot ativo; eu recomendo NÃO — colecionar prospectivamente).

---

## Aviso

A esmagadora maioria dos apostadores perde dinheiro no longo prazo. Um modelo
com EV+ no backtest não garante EV+ no futuro. Este é um projeto de ciência de
dados aplicada, não uma promessa de lucro. Nunca aposte o que você não pode
perder. (Este aviso será impresso pela CLI e pelo dashboard, sempre.)
