"""Carga e preparação do Mall Customers dataset."""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "Mall_Customers.csv"

AGE_BINS  = [17, 30, 45, 70]
AGE_LABELS = ["Jovem (18–30)", "Adulto (31–45)", "Maduro (46+)"]

INC_BINS  = [0, 40, 70, 140]
INC_LABELS = ["Baixa (≤40k)", "Média (41–70k)", "Alta (71k+)"]


def carregar() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {CSV_PATH}.\n"
            "Baixe via: kaggle datasets download -d vjchoudhary7/customer-segmentation-tutorial-in-python"
        )
    df = pd.read_csv(CSV_PATH)
    df = _limpar(df)
    logger.info("Dataset carregado: %d clientes", len(df))
    return df


def _limpar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns={
        "Annual Income (k$)": "income",
        "Spending Score (1-100)": "spending",
        "CustomerID": "id",
        "Gender": "gender",
        "Age": "age",
    })
    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS)
    df["inc_group"] = pd.cut(df["income"], bins=INC_BINS, labels=INC_LABELS)
    df["is_female"]  = (df["gender"] == "Female").astype(int)
    return df.reset_index(drop=True)


def resumo(df: pd.DataFrame) -> None:
    print(f"\n{'─'*50}")
    print(f"  Mall Customers — {len(df):,} clientes")
    print(f"{'─'*50}")
    print(f"  Mulheres        : {df['is_female'].mean():.1%}")
    print(f"  Idade média     : {df['age'].mean():.1f} anos")
    print(f"  Renda média     : ${df['income'].mean():.0f}k")
    print(f"  Spending médio  : {df['spending'].mean():.1f}/100")
    print(f"{'─'*50}\n")
