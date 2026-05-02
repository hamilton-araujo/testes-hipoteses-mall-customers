"""
Segmentos de marketing acionáveis cruzando Idade × Renda × Gênero.

Por que existe:
    Achado: idade impacta Spending. Insight de negócio precisa ir além:
    qual segmento (idade × renda × gênero) tem maior Spending Score?
    Quem priorizar nas campanhas Q1?

Saída:
    Tabela de 8 segmentos (2 idade × 2 renda × 2 gênero):
        n, spending médio, IC 90%, prioridade.
"""

import numpy as np
import pandas as pd


def construir_segmentos(df: pd.DataFrame) -> pd.DataFrame:
    """Constrói tabela de segmentos cruzados."""
    df = df.copy()

    if "grupo_idade" not in df.columns:
        df["grupo_idade"] = pd.cut(
            df["age"], bins=[0, 35, 100],
            labels=["Jovem (18-35)", "Maduro (36+)"],
        )
    if "nivel_renda" not in df.columns:
        df["nivel_renda"] = pd.cut(
            df["income"], bins=[0, 60, 200],
            labels=["Renda Média", "Renda Alta"],
        )

    rows = []
    for (idade, renda, genero), sub in df.groupby(
        ["grupo_idade", "nivel_renda", "gender"], observed=True
    ):
        if len(sub) < 5:
            continue
        spending = sub["spending"]
        ic_lower = float(spending.quantile(0.05))
        ic_upper = float(spending.quantile(0.95))
        rows.append({
            "idade":       str(idade),
            "renda":       str(renda),
            "genero":      genero,
            "n":           len(sub),
            "spending_medio": float(spending.mean()),
            "ic_lower":    ic_lower,
            "ic_upper":    ic_upper,
        })

    df_seg = pd.DataFrame(rows).sort_values("spending_medio", ascending=False)

    # Prioridade: top 25% spending → ALTA · meio 50% → MÉDIA · resto → BAIXA
    if len(df_seg) > 0:
        q75 = df_seg["spending_medio"].quantile(0.75)
        q25 = df_seg["spending_medio"].quantile(0.25)
        def _prio(s):
            if s >= q75: return "ALTA"
            if s >= q25: return "MEDIA"
            return "BAIXA"
        df_seg["prioridade"] = df_seg["spending_medio"].apply(_prio)
    return df_seg


def alocacao_budget(
    df_seg: pd.DataFrame,
    budget_total: float = 500_000.0,
) -> pd.DataFrame:
    """Aloca budget proporcional ao spending médio × n do segmento."""
    df = df_seg.copy()
    df["weight"] = df["spending_medio"] * df["n"]
    total_w = df["weight"].sum()
    df["alocacao"] = df["weight"] / total_w * budget_total
    return df.drop(columns="weight")
