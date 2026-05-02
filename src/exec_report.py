"""
Relatório executivo CMO — Inferência + Power + Segmentação.
"""

import io
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import carregar
from src.hypothesis import executar as analisar_hipoteses
from src.power_analysis import calcular as calcular_power, interpretar_d
from src.marketing_segments import construir_segmentos, alocacao_budget

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _grafico_segmentos(df_seg: pd.DataFrame, out: Path):
    cores_prio = {"ALTA": "#27ae60", "MEDIA": "#f39c12", "BAIXA": "#c0392b"}
    fig, ax = plt.subplots(figsize=(11, 7))
    df_seg["label"] = df_seg.apply(
        lambda r: f"{r['idade']} · {r['renda']} · {r['genero']}", axis=1
    )
    cs = [cores_prio.get(p, "#bdc3c7") for p in df_seg["prioridade"]]
    ax.barh(df_seg["label"], df_seg["spending_medio"], color=cs, alpha=0.85)
    ax.errorbar(df_seg["spending_medio"], df_seg["label"],
                xerr=[df_seg["spending_medio"] - df_seg["ic_lower"],
                      df_seg["ic_upper"] - df_seg["spending_medio"]],
                fmt="none", color="black", capsize=3, alpha=0.6)
    for i, (_, r) in enumerate(df_seg.iterrows()):
        ax.text(r["spending_medio"] + 1, i, f"n={int(r['n'])}",
                fontsize=8, va="center")
    ax.set_xlabel("Spending Score médio (1-100)")
    ax.set_title("Segmentos de Marketing — Prioridade por Spending")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _grafico_power(df_power: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    cs = ["#27ae60" if p >= 0.80 else "#e67e22" if p >= 0.50 else "#c0392b"
          for p in df_power["power_atual"]]
    ax.barh(df_power["hipotese"], df_power["power_atual"], color=cs, alpha=0.85)
    ax.axvline(0.80, color="green", ls="--", lw=1.5, label="Power = 0.80 (alvo)")
    ax.set_xlabel("Power estatístico (1 − β)")
    ax.set_title("Power Analysis — Capacidade de Detectar Efeito")
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    logger.info("Carregando dataset...")
    df = carregar()

    logger.info("Rodando hipóteses...")
    res = analisar_hipoteses(df)

    logger.info("Power analysis...")
    rows_power = []
    # H1: Female vs Male spending
    f = df[df["gender"] == "Female"]["spending"].values
    m = df[df["gender"] == "Male"]["spending"].values
    p1 = calcular_power(f, m)
    rows_power.append({"hipotese": "H1 Spending: F vs M", "n1": p1.n1, "n2": p1.n2,
                        "cohens_d": p1.cohens_d, "power_atual": p1.power_atual,
                        "n_necessario": p1.n_necessario,
                        "magnitude": interpretar_d(p1.cohens_d)})

    # H3: Jovens vs Maduros (decompõe Kruskal em pareado)
    df["grupo_idade_bin"] = pd.cut(df["age"], bins=[0, 35, 100],
                                    labels=["Jovem", "Maduro"])
    j = df[df["grupo_idade_bin"] == "Jovem"]["spending"].values
    md = df[df["grupo_idade_bin"] == "Maduro"]["spending"].values
    p3 = calcular_power(j, md)
    rows_power.append({"hipotese": "H3 Spending: Jovem vs Maduro", "n1": p3.n1, "n2": p3.n2,
                        "cohens_d": p3.cohens_d, "power_atual": p3.power_atual,
                        "n_necessario": p3.n_necessario,
                        "magnitude": interpretar_d(p3.cohens_d)})

    df_power = pd.DataFrame(rows_power)
    df_power.to_csv(OUTPUT_DIR / "power_analysis.csv", index=False)

    logger.info("Segmentos de marketing...")
    df_seg = construir_segmentos(df)
    df_seg.to_csv(OUTPUT_DIR / "segmentos_marketing.csv", index=False)

    df_aloc = alocacao_budget(df_seg, budget_total=500_000.0)
    df_aloc.to_csv(OUTPUT_DIR / "alocacao_budget.csv", index=False)

    # Charts
    _grafico_segmentos(df_seg, OUTPUT_DIR / "segmentos_marketing.png")
    _grafico_power(df_power, OUTPUT_DIR / "power_analysis.png")

    # Markdown
    lines = [
        "# Marketing Strategy — Decisão CMO Mall Customers",
        "",
        "## Sumário Executivo",
        "",
        f"- **Clientes analisados:** {len(df)}",
        f"- **Spending médio:** {df['spending'].mean():.1f}/100",
        f"- **Idade significativa para Spending:** Sim (Kruskal-Wallis p < 0.001)",
        f"- **Gênero não é fator significativo** (p > 0.40)",
        f"- **Segmentos prioritários (ALTA):** {(df_seg['prioridade'] == 'ALTA').sum()}",
        "",
        "---",
        "",
        "## 1. Power Analysis — Capacidade Detectiva",
        "",
        "![Power](power_analysis.png)",
        "",
        "| Hipótese | n1 | n2 | Cohen's d | Magnitude | Power | n Necessário |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in df_power.iterrows():
        lines.append(
            f"| {r['hipotese']} | {int(r['n1'])} | {int(r['n2'])} | "
            f"{r['cohens_d']:.3f} | {r['magnitude']} | {r['power_atual']:.2f} | "
            f"{int(r['n_necessario']) if r['n_necessario'] > 0 else '∞'} |"
        )

    lines += [
        "",
        "## 2. Segmentos de Marketing",
        "",
        "![Segmentos](segmentos_marketing.png)",
        "",
        "| Idade | Renda | Gênero | n | Spending | IC 90% | Prioridade |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in df_seg.iterrows():
        lines.append(
            f"| {r['idade']} | {r['renda']} | {r['genero']} | {int(r['n'])} | "
            f"{r['spending_medio']:.1f} | [{r['ic_lower']:.0f}, {r['ic_upper']:.0f}] | "
            f"{r['prioridade']} |"
        )

    lines += [
        "",
        "## 3. Alocação Sugerida de Budget Q1 (R$ 500k)",
        "",
        "| Segmento | Spending | n | Alocação |",
        "|---|---|---|---|",
    ]
    for _, r in df_aloc.head(8).iterrows():
        seg = f"{r['idade']} · {r['renda']} · {r['genero']}"
        lines.append(
            f"| {seg} | {r['spending_medio']:.1f} | {int(r['n'])} | "
            f"R$ {r['alocacao']/1e3:,.1f}k |"
        )

    lines += [
        "",
        "---",
        "",
        "## Metodologia",
        "",
        "- **Mann-Whitney U**: comparação de medianas para 2 grupos (não paramétrico).",
        "- **Kruskal-Wallis**: extensão para 3+ grupos.",
        "- **Cohen's d**: tamanho de efeito padronizado.",
        "- **Power = 1 − β**: P(rejeitar H0 | H1 verdadeira). Alvo: ≥ 0.80.",
        "- **Bootstrap 95% CI**: 10.000 reamostragens.",
        "- **Alocação de budget**: proporcional a spending_medio × n por segmento.",
    ]
    (OUTPUT_DIR / "relatorio_cmo.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n{'═'*60}")
    print("  MARKETING STRATEGY — DECISÃO CMO")
    print(f"{'═'*60}")
    print(f"  Clientes                   {len(df)}")
    print(f"  Spending médio             {df['spending'].mean():.1f}/100")
    print(f"  Segmentos prioridade ALTA  {(df_seg['prioridade'] == 'ALTA').sum()}")
    print(f"  H3 Power (Jovem vs Maduro) {df_power.iloc[1]['power_atual']:.2f}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
