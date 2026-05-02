# Inferência Estatística + Power + Decisão CMO

> **A pergunta do CMO:** *Os achados estatísticos sustentam decisões de campanha? A amostra é grande o suficiente? Quais segmentos priorizar com R$ 500k Q1?*

Pipeline de inferência estatística rigorosa transformado em **decisão executiva de marketing**: testes não-paramétricos + power analysis + segmentação cruzada idade/renda/gênero + alocação de budget.

---

## Por que existe

Mann-Whitney U + Kruskal-Wallis + Cohen's d são técnica acadêmica. CMO precisa traduzir para três respostas:

| Pergunta | Sinal técnico |
|---|---|
| Os achados são confiáveis? | Power analysis (1 − β ≥ 0.80) |
| Quem priorizar? | Segmentos cruzados com IC 90% |
| Quanto investir em cada? | Alocação proporcional a spending × n |

---

## A história em três atos

### Ato 1 — A reunião
Você é Head of Marketing do shopping. Conselho aprovou R$ 500k para campanhas Q1. Pediu lista de segmentos prioritários com fundamento estatístico. Você roda:
```bash
python -m src.exec_report
```

### Ato 2 — A evidência
1 segundo depois:
```
Clientes                   200
Spending médio             50.2/100
Segmentos prioridade ALTA  2
H3 Power (Jovem vs Maduro) 1.00
```

### Ato 3 — A decisão
H1 (Gênero) tem power baixo — não rejeitar H0 não significa "iguais", apenas "amostra insuficiente". H3 (Idade) tem power 1.00 — diferença robusta. **Decisão**: campanhas focam em **jovens com renda alta**, ignoram divisão por gênero.

---

## Modelos

### Testes Não-Paramétricos
```
Mann-Whitney U : 2 grupos       (H1, H2)
Kruskal-Wallis : 3+ grupos      (H3, H4)
Dunn post-hoc  : pairwise depois de Kruskal significativo
```

### Effect Size
```
Cohen's d  = (μ₁ − μ₂) / σ_pooled       (2 grupos)
η²         = SS_between / SS_total       (ANOVA)
```

| d | Interpretação |
|---|---|
| < 0.20 | Negligível |
| 0.20–0.50 | Pequeno |
| 0.50–0.80 | Médio |
| > 0.80 | Grande |

### Power Analysis
```
Power = 1 − β = P(rejeitar H0 | H1 verdadeira)
n_necessário(d, α=0.05, power=0.80) = 2 × ((z_α + z_β) / d)²
```
Convenção: power ≥ 0.80 = adequado.

### Bootstrap CI 95%
10.000 reamostragens com reposição → CI da diferença de médias (não-paramétrico).

### Segmentação Acionável
Cruza **Idade × Renda × Gênero** (8 combinações) → spending médio + IC 90% → prioridade ALTA / MEDIA / BAIXA por quartis.

### Alocação de Budget
Proporcional a `spending_medio × n_clientes` por segmento.

---

## Stack

| Camada | Tecnologia |
|---|---|
| Dados | Mall Customers (Kaggle) — 200 clientes |
| Inferência | scipy (Mann-Whitney, Kruskal-Wallis, Bootstrap) |
| Power | NumPy + scipy.norm.ppf |
| Visualização | matplotlib |

---

## Como rodar

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Pipeline executivo (recomendado)
python -m src.exec_report

# Pipeline original (testes + bootstrap)
python -m src.main
```

---

## Outputs

```
output/
├── relatorio_cmo.md             # ⭐ Briefing CMO + alocação Q1
├── power_analysis.png           # ⭐ Power por hipótese
├── power_analysis.csv
├── segmentos_marketing.png      # ⭐ Segmentos × spending + IC
├── segmentos_marketing.csv
├── alocacao_budget.csv          # ⭐ R$ 500k distribuído
├── boxplots_genero.png          # Original
├── boxplots_grupos.png
├── bootstrap_h1.png
└── forest_plot.png
```

⭐ = adicionado nesta versão.

---

## Estrutura

```
├── src/
│   ├── exec_report.py             # ⭐ Pipeline executivo CMO
│   ├── power_analysis.py          # ⭐ Cohen's d + power + n necessário
│   ├── marketing_segments.py      # ⭐ Segmentos cruzados + alocação
│   ├── ingest.py
│   ├── hypothesis.py              # Mann-Whitney + Kruskal + Bootstrap
│   ├── report.py
│   └── main.py
├── data/
├── output/
├── tests/
├── pytest.ini
└── requirements.txt
```
