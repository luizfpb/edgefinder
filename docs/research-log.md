# Diário de pesquisa

Regra: toda hipótese entra AQUI antes de ser testada, com data, motivação e
critério de decisão. O resultado entra depois, inclusive (principalmente)
quando negativo. Isso existe para impedir overfitting por tentativa e erro:
se testamos 20 variantes e reportamos só a melhor, o "ganho" é ruído.

---

## 2026-07-13 — H1: meia-vida do decaimento temporal

- Hipótese: a meia-vida ótima do decaimento exponencial para o Dixon-Coles
  está entre 90 e 365 dias (literatura sugere ξ ≈ 0.0018/dia ≈ meia-vida ~380d
  em (Dixon & Coles, 1997); mercados mais líquidos hoje podem favorecer menos
  memória).
- Teste: grade half_life ∈ {30, 60, 90, 120, 180, 365, 730}, log-loss 1X2
  out-of-sample no walk-forward, por liga.
- Decisão: escolher por log-loss médio; reportar sensibilidade.
- Resultado (2026-07-14, scripts/exp_h1_halflife.py, EPL 2019-20→2025-26,
  n=2654, artefato data/reports/h1_halflife_grid.json):
  - 60d: log-loss 0.6370 (Brier 0.2198)
  - 90d: 0.6249 (0.2155)
  - 180d: 0.6167 (0.2124)
  - **365d: 0.6151 (0.2126) — ótimo**
  - 730d: 0.6176 (0.2142)

  Ótimo em ~365 dias — consistente com ξ≈0,0018/dia do paper original de
  Dixon & Coles (meia-vida ln2/ξ ≈ 385d). A curva é rasa entre 180 e 730:
  pouca sensibilidade. Mercado no mesmo período: 0.5970 — o decaimento certo
  melhora o modelo, mas não fecha o gap (ver H4).

## 2026-07-13 — H2: Poisson vs Negativa Binomial para chutes de jogador

- Hipótese: chutes por jogo têm overdispersion (variância > média) e a NB
  (via posterior preditiva Gamma-Poisson) vence o Poisson puro em log-loss
  preditivo out-of-sample.
- Teste: models/player_props.model_comparison em dados reais (EPL 2223-2425,
  Understat; BR-A 2024-2025, FBref quando o cache completar).
- Decisão: AIC/BIC no treino E log-loss no holdout; a NB só entra se vencer
  nos dois.
- Resultado (2026-07-14, scripts/exp_h2_props.py, artefato
  data/reports/h2_props_comparison.json):
  - **Chutes: NB vence com folga.** EPL/Understat (34k jogador-jogos):
    log-loss teste 1.2633 (NB) vs 1.3328 (Poisson); AIC 58238 vs 61599.
    BR-A/FBref (36k): 1.2483 vs 1.3015. Overdispersion confirmada.
  - **Chutes no gol: NB vence.** EPL/FBref: 0.6558 vs 0.6656.
  - **Cartões amarelos: EMPATE — Poisson basta.** BR-A: 0.44415 (NB) vs
    0.44413 (Poisson); o parâmetro extra da NB não compra nada. Cartão por
    jogador-jogo é ~Bernoulli raro, sem heterogeneidade extra além da taxa.
  - Decisão: NB (posterior preditiva Gamma-Poisson) para chutes/SoT/gols;
    Poisson para cartões. Confirma que "assumir NB em tudo" seria erro.

## 2026-07-13 — H3: de-vig de Shin vs proporcional como benchmark de mercado

- Hipótese: Shin produz probabilidade implícita mais calibrada que o método
  proporcional em odds de fechamento da Pinnacle (corrige favourite-longshot
  bias), medido por log-loss contra resultados.
- Teste: ambos sobre PSCH/PSCD/PSCA das ligas europeias; comparar log-loss.
- Decisão: o método vencedor vira o benchmark oficial do backtest.
- Resultado (2026-07-14, scripts/exp_h3_devig.py, 9 ligas, artefato
  data/reports/h3_devig_comparison.json). Log-loss multiclasse sobre o
  fechamento da Pinnacle (n=36.306): **proporcional 0.97173 > Shin 0.97147 >
  aditivo 0.97142**. Mesmo ordenamento em market_avg; em market_max o Shin
  fica marginalmente melhor que o aditivo.
  - Shin bate o proporcional, como a hipótese previa — mas a diferença total
    entre os TRÊS métodos é ~0.0003 nats, essencialmente ruído.
  - Registro honesto: o aditivo ficou 5e-5 à frente do Shin na Pinnacle, o
    que não é distinguível de acaso e inverte de casa para casa.
  - Decisão: **Shin permanece o benchmark oficial** — vence o proporcional
    (a comparação da hipótese), tem fundamento teórico para o viés
    favorito-azarão, e a diferença para o aditivo não é material. Nota de
    processo: o Shin já estava como default do código antes deste teste
    (decisão antecipada à evidência — o que este diário existe para evitar);
    o resultado o confirma, mas a ordem foi errada e fica registrada.

## 2026-07-13 — H4: o modelo bate o mercado? (a pergunta do projeto)

- Hipótese nula: o Dixon-Coles com decaimento NÃO bate a closing line
  de-vigada da Pinnacle em log-loss nem gera CLV/ROI positivo significativo.
- Teste: walk-forward 2019-20 → 2025-26 nas Big 5 + BR-A (1X2), apostando
  apenas quando EV > threshold do tier; IC 95% bootstrap do yield; p-valor.
- Decisão honesta: se não rejeitarmos a nula, o sistema reporta o fato em
  destaque e NÃO emite sugestões — esse é o comportamento correto do produto.
- Resultado (2026-07-14, run completo: 9 ligas, 25.194 previsões walk-forward,
  2019-20→2025-26, half-life 180d, artefatos em data/reports/):
  **A HIPÓTESE NULA NÃO FOI REJEITADA — O MODELO NÃO BATE O MERCADO.**
  - Log-loss do modelo pior que o do fechamento de-vigado (Shin) nas 9 de 9
    ligas (ex.: EPL 0.6167 vs 0.5970; BR-A 0.6732 vs 0.6493).
  - Simulação de apostas (EV>threshold, Kelly 1/4, à odd de fechamento):
    8.886 apostas, yield −3,85% [IC95: −7,0%, −0,8%], p=0,99. Perda
    estatisticamente significativa, não ruído.
  - Diagnóstico: os "edges" que o modelo enxerga são erro de estimação
    (excesso de confiança), não valor — o Monte Carlo sob as probabilidades
    do próprio modelo prevê crescimento absurdo, e a realidade dá −3,9%.
  - Consequência aplicada: a trava `backtest_gate()` (edge/today.py) BLOQUEIA
    sugestões de aposta enquanto o veredito for negativo. O sistema funciona
    como instrumento de medição honesto; não como máquina de apostar.
  - Próximas alavancas com chance real (ordem de prioridade): calibração
    isotônica das saídas + ensemble com o próprio mercado de abertura como
    feature (prever o fechamento, não o resultado), xG do Understat como
    entrada do DC (Europa), e mercados menos eficientes (props/escanteios)
    quando o histórico próprio de odds acumular.
- Adendo (2026-07-14, tarde): re-rodado com a meia-vida no ótimo do H1 (365d
  — o run original usara 180d, incoerência corrigida) e com o benchmark do
  H5 (market_avg a partir de 23/07/2025). Resultado: 8.706 apostas, yield
  **−4,36%** [IC95: −7,5%, −1,2%], p=0,997; log-loss pior que o mercado nas
  9 de 9 ligas (EPL 0.6151 vs 0.5967). A conclusão não muda — a meia-vida
  melhor não fecha o gap — e a trava segue acionada.

## 2026-07-13 — H5: Pinnacle closing pós-07/2025 é benchmark confiável?

- Contexto: o próprio football-data.co.uk avisa que as closing odds da
  Pinnacle ficaram não confiáveis a partir de 23/07/2025.
- Teste: comparar overround e log-loss de `PSC*` vs `MaxC*`/`AvgC*` antes/depois
  de 2025-07; se divergirem, usar AvgC como benchmark no período recente.
- Resultado (2026-07-14, scripts/exp_h5_pinnacle.py, 8 ligas europeias,
  artefato data/reports/h5_pinnacle_reliability.json). Corte em 23/07/2025:
  - **Cobertura despenca**: pós-corte a Pinnacle só tem fechamento em 1.464
    de 2.916 jogos (50%); antes cobria ~100% (17.614/17.617).
  - **Divergência sobe 60%**: |p_pinnacle − p_avg| médio vai de 0.0045 (pré)
    para 0.0072 (pós).
  - **Log-loss PAREADO** (mesmos jogos para as duas casas — sem isso a
    comparação tem viés de seleção): pré-corte a Pinnacle vence a média
    (0.97193 vs 0.97221); pós-corte ela PERDE (0.96894 vs 0.96878).
  - Overround da própria Pinnacle sobe de 2,90% para 3,25% — comportamento
    de casa que deixou de ser a referência sharp.
  - Decisão aplicada: `backtest/data.py` passa a preferir **market_avg como
    benchmark a partir de 23/07/2025** (antes disso, Pinnacle segue primeiro
    na ordem). O aviso do football-data.co.uk se confirma nos dados.
