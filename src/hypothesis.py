"""
Testes de Hipótese completos com efeito, bootstrap e poder.

H1: Mulheres vs Homens — Spending Score (Mann-Whitney U)
H2: Mulheres vs Homens — Renda Anual (Mann-Whitney U)
H3: Grupos de Idade   — Spending Score (Kruskal-Wallis + Dunn post-hoc)
H4: Quartis de Renda  — Spending Score (Kruskal-Wallis)

Para cada teste:
  - Estatística + p-value
  - Tamanho de efeito (Cohen's d / eta²)
  - Bootstrap 95% CI da diferença de médias
  - Interpretação (pequeno/médio/grande)
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

logger = logging.getLogger(__name__)

ALPHA = 0.05
N_BOOTSTRAP = 10_000
SEED = 42


@dataclass
class TesteHipotese:
    nome: str
    estatistica: float
    p_value: float
    efeito: float
    efeito_label: str
    ci_lower: float
    ci_upper: float
    conclusao: str
    grupos: dict = field(default_factory=dict)


@dataclass
class ResultadoHipoteses:
    h1_spending_genero: TesteHipotese
    h2_renda_genero:    TesteHipotese
    h3_spending_idade:  TesteHipotese
    h4_spending_renda:  TesteHipotese
    posthoc_idade:      pd.DataFrame


def executar(df: pd.DataFrame, random_state: int = SEED) -> ResultadoHipoteses:
    rng = np.random.default_rng(random_state)

    h1 = _mann_whitney(
        df.loc[df["gender"] == "Female", "spending"].values,
        df.loc[df["gender"] == "Male",   "spending"].values,
        "H1 Spending: Mulheres vs Homens", rng,
    )
    h2 = _mann_whitney(
        df.loc[df["gender"] == "Female", "income"].values,
        df.loc[df["gender"] == "Male",   "income"].values,
        "H2 Renda: Mulheres vs Homens", rng,
    )
    h3, posthoc = _kruskal_idade(df, rng)
    h4          = _kruskal_renda(df, rng)

    for h in [h1, h2, h3, h4]:
        logger.info("%s — p=%.4f | efeito=%s", h.nome, h.p_value, h.efeito_label)

    return ResultadoHipoteses(
        h1_spending_genero=h1,
        h2_renda_genero=h2,
        h3_spending_idade=h3,
        h4_spending_renda=h4,
        posthoc_idade=posthoc,
    )


def _mann_whitney(g1: np.ndarray, g2: np.ndarray, nome: str,
                  rng: np.random.Generator) -> TesteHipotese:
    stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    d = _cohens_d(g1, g2)
    ci = _bootstrap_ci(g1, g2, rng)
    return TesteHipotese(
        nome=nome,
        estatistica=float(stat),
        p_value=float(p),
        efeito=abs(d),
        efeito_label=_label_d(abs(d)),
        ci_lower=ci[0],
        ci_upper=ci[1],
        conclusao="Rejeita H0" if p < ALPHA else "Não rejeita H0",
        grupos={"g1_mean": float(g1.mean()), "g2_mean": float(g2.mean()),
                "g1_n": len(g1), "g2_n": len(g2)},
    )


def _kruskal_idade(df: pd.DataFrame, rng: np.random.Generator):
    from .ingest import AGE_LABELS
    grupos = [
        df.loc[df["age_group"] == g, "spending"].dropna().values
        for g in AGE_LABELS
    ]
    grupos = [g for g in grupos if len(g) >= 5]
    stat, p = stats.kruskal(*grupos)
    eta2    = _eta_squared(grupos)
    ci      = _bootstrap_ci_multi(grupos, rng)

    posthoc = _dunn_test(df, "age_group", "spending", AGE_LABELS)

    return TesteHipotese(
        nome="H3 Spending por Grupo de Idade",
        estatistica=float(stat),
        p_value=float(p),
        efeito=eta2,
        efeito_label=_label_eta2(eta2),
        ci_lower=ci[0],
        ci_upper=ci[1],
        conclusao="Rejeita H0" if p < ALPHA else "Não rejeita H0",
        grupos={g: float(df.loc[df["age_group"] == g, "spending"].mean())
                for g in AGE_LABELS},
    ), posthoc


def _kruskal_renda(df: pd.DataFrame, rng: np.random.Generator) -> TesteHipotese:
    from .ingest import INC_LABELS
    grupos = [
        df.loc[df["inc_group"] == g, "spending"].dropna().values
        for g in INC_LABELS
    ]
    grupos = [g for g in grupos if len(g) >= 5]
    stat, p = stats.kruskal(*grupos)
    eta2    = _eta_squared(grupos)
    ci      = _bootstrap_ci_multi(grupos, rng)
    return TesteHipotese(
        nome="H4 Spending por Nível de Renda",
        estatistica=float(stat),
        p_value=float(p),
        efeito=eta2,
        efeito_label=_label_eta2(eta2),
        ci_lower=ci[0],
        ci_upper=ci[1],
        conclusao="Rejeita H0" if p < ALPHA else "Não rejeita H0",
        grupos={g: float(df.loc[df["inc_group"] == g, "spending"].mean())
                for g in INC_LABELS},
    )


def _cohens_d(g1: np.ndarray, g2: np.ndarray) -> float:
    n1, n2 = len(g1), len(g2)
    pooled = np.sqrt(((n1-1)*g1.std(ddof=1)**2 + (n2-1)*g2.std(ddof=1)**2) / (n1+n2-2))
    return (g1.mean() - g2.mean()) / pooled if pooled > 0 else 0.0


def _eta_squared(grupos: list) -> float:
    all_vals = np.concatenate(grupos)
    grand_mean = all_vals.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in grupos)
    ss_total   = sum((v - grand_mean)**2 for v in all_vals)
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def _bootstrap_ci(g1: np.ndarray, g2: np.ndarray, rng: np.random.Generator,
                  n: int = N_BOOTSTRAP) -> tuple[float, float]:
    diffs = np.array([
        rng.choice(g1, len(g1)).mean() - rng.choice(g2, len(g2)).mean()
        for _ in range(n)
    ])
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _bootstrap_ci_multi(grupos: list, rng: np.random.Generator,
                         n: int = N_BOOTSTRAP) -> tuple[float, float]:
    ranges = np.array([
        max(rng.choice(g, len(g)).mean() for g in grupos) -
        min(rng.choice(g, len(g)).mean() for g in grupos)
        for _ in range(n)
    ])
    return float(np.percentile(ranges, 2.5)), float(np.percentile(ranges, 97.5))


def _dunn_test(df: pd.DataFrame, group_col: str, val_col: str, labels: list) -> pd.DataFrame:
    rows = []
    for a, b in combinations(labels, 2):
        ga = df.loc[df[group_col] == a, val_col].dropna().values
        gb = df.loc[df[group_col] == b, val_col].dropna().values
        if len(ga) < 5 or len(gb) < 5:
            continue
        _, p = stats.mannwhitneyu(ga, gb, alternative="two-sided")
        rows.append({"par": f"{a} vs {b}", "p_value": float(p),
                     "significativo": p < ALPHA})
    return pd.DataFrame(rows)


def _label_d(d: float) -> str:
    if d < 0.2:   return "Negligível"
    if d < 0.5:   return "Pequeno"
    if d < 0.8:   return "Médio"
    return "Grande"


def _label_eta2(e: float) -> str:
    if e < 0.01:  return "Negligível"
    if e < 0.06:  return "Pequeno"
    if e < 0.14:  return "Médio"
    return "Grande"
