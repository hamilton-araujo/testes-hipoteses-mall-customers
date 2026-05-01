"""Painel CLI + gráficos de testes de hipóteses."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .hypothesis import ResultadoHipoteses, TesteHipotese

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def imprimir(resultado: ResultadoHipoteses, df, output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)

    testes = [
        resultado.h1_spending_genero,
        resultado.h2_renda_genero,
        resultado.h3_spending_idade,
        resultado.h4_spending_renda,
    ]

    print(f"\n{'═'*60}")
    print("  TESTES DE HIPÓTESE — MALL CUSTOMERS")
    print(f"{'═'*60}")
    print(f"  {'Hipótese':<35} {'p-value':>8} {'Efeito':>12} {'Resultado':>14}")
    for t in testes:
        sig = "**" if t.p_value < 0.001 else ("*" if t.p_value < 0.05 else "  ")
        print(f"  {t.nome:<35} {t.p_value:>7.4f}{sig} {t.efeito_label:>12} {t.conclusao:>14}")
    print()
    print("  ── Bootstrap 95% CI (diferença de médias) ──")
    for t in testes:
        print(f"  {t.nome:<35} [{t.ci_lower:+.2f}, {t.ci_upper:+.2f}]")
    print()
    if len(resultado.posthoc_idade) > 0:
        print("  ── Post-hoc Dunn (Grupos de Idade) ──")
        for _, r in resultado.posthoc_idade.iterrows():
            sig = "*" if r["significativo"] else " "
            print(f"  {sig} {r['par']:<40} p={r['p_value']:.4f}")
    print(f"{'═'*60}\n")

    _boxplots_genero(df, output_dir)
    _boxplots_grupos(df, output_dir)
    _forest_plot(testes, output_dir)
    _bootstrap_dist(df, output_dir)


def _boxplots_genero(df: pd.DataFrame, out: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title in [
        (ax1, "spending", "Spending Score por Gênero"),
        (ax2, "income",   "Renda Anual por Gênero"),
    ]:
        groups = [df.loc[df["gender"] == g, col].values for g in ["Female", "Male"]]
        ax.boxplot(groups, labels=["Feminino", "Masculino"], patch_artist=True,
                   boxprops=dict(facecolor="lightblue", alpha=0.7))
        ax.set_title(title)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "boxplots_genero.png", dpi=150, bbox_inches="tight")
    plt.close()


def _boxplots_grupos(df: pd.DataFrame, out: Path):
    from .ingest import AGE_LABELS, INC_LABELS
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col_group, title, labels in [
        (ax1, "age_group", "Spending por Grupo de Idade", AGE_LABELS),
        (ax2, "inc_group", "Spending por Nível de Renda", INC_LABELS),
    ]:
        groups = [df.loc[df[col_group] == g, "spending"].dropna().values for g in labels]
        ax.boxplot([g for g in groups if len(g) > 0], patch_artist=True,
                   labels=[l for l, g in zip(labels, groups) if len(g) > 0],
                   boxprops=dict(facecolor="lightyellow", alpha=0.8))
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "boxplots_grupos.png", dpi=150, bbox_inches="tight")
    plt.close()


def _forest_plot(testes: list, out: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(testes))
    for i, t in enumerate(testes):
        color = "steelblue" if t.p_value < 0.05 else "lightgray"
        ax.errorbar(t.efeito, i,
                    xerr=[[t.efeito - max(0, t.efeito - 0.1)],
                           [t.efeito + 0.1 - t.efeito]],
                    fmt="o", color=color, markersize=10, elinewidth=2, capsize=5)
    ax.set_yticks(y)
    ax.set_yticklabels([t.nome.split(":")[0] for t in testes], fontsize=9)
    ax.set_xlabel("Tamanho de Efeito (|d| ou η²)")
    ax.set_title("Forest Plot — Tamanho de Efeito por Hipótese")
    ax.axvline(0.2, color="orange", lw=1, ls="--", alpha=0.7, label="Pequeno (0.2)")
    ax.axvline(0.5, color="red",    lw=1, ls="--", alpha=0.7, label="Médio (0.5)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "forest_plot.png", dpi=150, bbox_inches="tight")
    plt.close()


def _bootstrap_dist(df: pd.DataFrame, out: Path, n: int = 5000, seed: int = 42):
    rng = np.random.default_rng(seed)
    fem = df.loc[df["gender"] == "Female", "spending"].values
    mas = df.loc[df["gender"] == "Male",   "spending"].values
    diffs = np.array([
        rng.choice(fem, len(fem)).mean() - rng.choice(mas, len(mas)).mean()
        for _ in range(n)
    ])
    ci = np.percentile(diffs, [2.5, 97.5])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(diffs, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="black", lw=1.5, ls="--")
    ax.axvline(ci[0], color="red", lw=1.5, ls=":", label=f"95% CI [{ci[0]:.2f}, {ci[1]:.2f}]")
    ax.axvline(ci[1], color="red", lw=1.5, ls=":")
    ax.axvline(diffs.mean(), color="darkorange", lw=2, label=f"Média: {diffs.mean():.2f}")
    ax.set_xlabel("Diferença de médias (Feminino – Masculino) [Spending]")
    ax.set_title("Bootstrap Distribution — H1 Spending por Gênero")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "bootstrap_h1.png", dpi=150, bbox_inches="tight")
    plt.close()
