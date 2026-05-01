# Testes de Hipótese — Mall Customers

Pipeline de inferência estatística rigorosa aplicada a dados de clientes de shopping, cobrindo testes não-paramétricos, tamanho de efeito, bootstrap e post-hoc.

## Dataset

**Mall Customers** — Kaggle (`vjchoudhary7/customer-segmentation-tutorial-in-python`)  
200 clientes | Gender, Age, Annual Income, Spending Score

## Stack

| Camada | Tecnologia |
|---|---|
| Testes | Mann-Whitney U · Kruskal-Wallis |
| Post-hoc | Dunn (pairwise Mann-Whitney) |
| Efeito | Cohen's d · Eta² |
| Incerteza | Bootstrap 95% CI (10k iterações) |

## Hipóteses Testadas

| H | Comparação | p-value | Efeito | Resultado |
|---|---|---|---|---|
| H1 | Spending: Mulheres vs Homens | 0.571 | Negligível | Não rejeita H0 |
| H2 | Renda: Mulheres vs Homens | 0.414 | Negligível | Não rejeita H0 |
| H3 | Spending por Grupo de Idade | <0.001 | **Médio** | **Rejeita H0** |
| H4 | Spending por Nível de Renda | 0.944 | Negligível | Não rejeita H0 |

**Insight principal**: A idade é o único fator com impacto significativo no Spending Score — clientes jovens (18–30) gastam significativamente mais que os maduros (46+), com CI Bootstrap [+16.5, +31.0].

## Estrutura

```
├── src/
│   ├── ingest.py      # Carga e grupos etários/renda
│   ├── hypothesis.py  # Mann-Whitney · Kruskal-Wallis · Cohen's d · Bootstrap
│   ├── report.py      # Painel CLI + 4 gráficos
│   └── main.py        # CLI argparse
├── data/
├── output/
├── tests/
└── requirements.txt
```
