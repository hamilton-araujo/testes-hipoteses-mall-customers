"""
Power Analysis — poder estatístico e tamanho de amostra.

Por que existe:
    Achados não-significativos (p > 0.05) podem ser:
    a) ausência real de efeito
    b) **falta de poder** (amostra pequena demais)

    Power = 1 − β = P(rejeitar H0 | H1 verdadeira).
    Convenção: power ≥ 0.80 considera-se adequado.

    Para detectar Cohen's d = 0.20 (efeito pequeno) com α = 0.05 e power = 0.80,
    precisa-se de ~393 observações por grupo. Com n=100 (Mall Customers tem 88
    homens, 112 mulheres), só detectamos Cohen's d ≥ 0.40 (efeito médio).

Implementação simplificada (fórmulas analíticas):
    n_per_group(d, α=0.05, power=0.80) = 2 × ((z_α + z_β) / d)²
    Onde z_α = 1.96 (two-tailed) e z_β = 0.84.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class PowerResultado:
    n1:                int
    n2:                int
    cohens_d:          float
    power_atual:       float        # poder com amostra atual
    n_necessario:      int          # n por grupo para power = 0.80
    detectavel_p80:    float        # menor d detectável com power=0.80


def calcular(
    g1: np.ndarray,
    g2: np.ndarray,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> PowerResultado:
    """Cohen's d + power analysis."""
    n1, n2 = len(g1), len(g2)
    pooled_std = np.sqrt(((n1 - 1) * g1.var(ddof=1) + (n2 - 1) * g2.var(ddof=1))
                          / (n1 + n2 - 2))
    d = (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else 0.0

    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(target_power)

    # Power atual: para d observado e n atual
    n_harmonic = 2 / (1/n1 + 1/n2)
    ncp = abs(d) * np.sqrt(n_harmonic / 2)
    power_atual = float(1 - stats.norm.cdf(z_a - ncp) + stats.norm.cdf(-z_a - ncp))

    # n necessário para target_power dado d observado
    if abs(d) > 0.001:
        n_nec = 2 * ((z_a + z_b) / d) ** 2
    else:
        n_nec = float("inf")

    # menor d detectável com power=0.80 dada amostra atual
    n_min = min(n1, n2)
    d_detectavel = (z_a + z_b) * np.sqrt(2 / n_min)

    return PowerResultado(
        n1=n1, n2=n2,
        cohens_d=float(d),
        power_atual=float(power_atual),
        n_necessario=int(np.ceil(n_nec)) if np.isfinite(n_nec) else -1,
        detectavel_p80=float(d_detectavel),
    )


def interpretar_d(d: float) -> str:
    """Convenção Cohen para magnitude de efeito."""
    abs_d = abs(d)
    if abs_d < 0.20:
        return "Negligível"
    if abs_d < 0.50:
        return "Pequeno"
    if abs_d < 0.80:
        return "Médio"
    return "Grande"
