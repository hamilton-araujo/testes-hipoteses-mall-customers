"""Testes unitários do pipeline de hipóteses."""

import numpy as np
import pandas as pd
import pytest

from src.ingest import _limpar
from src.hypothesis import executar, _cohens_d, _eta_squared, _bootstrap_ci


def _df_sintetico(n=200, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "CustomerID": range(n),
        "Gender":     rng.choice(["Female", "Male"], n),
        "Age":        rng.integers(18, 70, n),
        "Annual Income (k$)":       rng.integers(15, 137, n),
        "Spending Score (1-100)":   rng.integers(1, 101, n),
    })


class TestIngest:
    def test_colunas_renomeadas(self):
        df = _limpar(_df_sintetico())
        assert "spending" in df.columns
        assert "income" in df.columns

    def test_grupos_idade_criados(self):
        df = _limpar(_df_sintetico())
        assert "age_group" in df.columns
        assert df["age_group"].notna().sum() > 0

    def test_is_female_binario(self):
        df = _limpar(_df_sintetico())
        assert set(df["is_female"].unique()).issubset({0, 1})


class TestHypothesis:
    def setup_method(self):
        self.df = _limpar(_df_sintetico())

    def test_cohens_d_range(self):
        g1 = np.array([1.0, 2.0, 3.0])
        g2 = np.array([4.0, 5.0, 6.0])
        d = abs(_cohens_d(g1, g2))
        assert d > 0

    def test_eta_squared_range(self):
        grupos = [np.array([1.0, 2.0]), np.array([5.0, 6.0]), np.array([10.0, 11.0])]
        e = _eta_squared(grupos)
        assert 0 <= e <= 1

    def test_bootstrap_ci_ordenado(self):
        rng = np.random.default_rng(42)
        g1 = np.random.randn(50) + 2
        g2 = np.random.randn(50)
        lo, hi = _bootstrap_ci(g1, g2, rng, n=1000)
        assert lo < hi

    def test_executar_retorna_4_hipoteses(self):
        r = executar(self.df)
        assert r.h1_spending_genero is not None
        assert r.h2_renda_genero is not None
        assert r.h3_spending_idade is not None
        assert r.h4_spending_renda is not None

    def test_pvalues_validos(self):
        r = executar(self.df)
        for h in [r.h1_spending_genero, r.h2_renda_genero,
                  r.h3_spending_idade, r.h4_spending_renda]:
            assert 0 <= h.p_value <= 1
