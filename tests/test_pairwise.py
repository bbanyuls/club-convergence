import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.adfvalues import mackinnonp
from statsmodels.tsa.stattools import adfuller

import convergence_clubs as cc


def _arma(rng, n, phi):
    e = rng.normal(size=n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t] + 0.3 * e[t - 1]
    return x


@pytest.mark.parametrize("regression", ["c", "ct"])
@pytest.mark.parametrize("autolag", ["AIC", "BIC", None])
def test_adf_matches_statsmodels(regression, autolag):
    rng = np.random.default_rng(7)
    for _ in range(60):
        n = int(rng.integers(20, 120))
        x = _arma(rng, n, rng.choice([1.0, 0.95, 0.7, 0.3])) + 0.02 * rng.normal() * np.arange(n)
        stat, p, lags, nobs, *_ = adfuller(
            x, regression=regression, autolag=autolag, result_object=False
        )
        ours = cc.adf_test(x, regression=regression, autolag=autolag)
        assert ours.lags == lags and ours.nobs == nobs
        assert ours.stat == pytest.approx(stat, abs=1e-9)
        assert ours.p_value == pytest.approx(p, abs=1e-9)


def test_mackinnon_pvalues():
    for regression in ["c", "ct"]:
        for stat in np.linspace(-20, 3, 200):
            assert cc.mackinnon_pvalue(stat, regression) == pytest.approx(
                mackinnonp(stat, regression=regression), abs=1e-12
            )


def test_adf_errors():
    with pytest.raises(ValueError, match="regression"):
        cc.adf_test(np.arange(30.0), regression="n")
    with pytest.raises(ValueError, match="autolag"):
        cc.adf_test(np.arange(30.0), autolag="t-stat")
    with pytest.raises(ValueError, match="too short"):
        cc.adf_test(np.arange(30.0), max_lag=20)


def _two_groups(seed=3, per_group=8, t=80):
    """Units share a group-specific random walk plus stationary noise."""
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for g in range(2):
        trend = np.cumsum(rng.normal(0, 0.05, t))
        for _ in range(per_group):
            noise = np.zeros(t)
            for s in range(1, t):
                noise[s] = 0.5 * noise[s - 1] + rng.normal(0, 0.03)
            rows.append(np.exp(5 + trend + noise))
            labels.append(f"club{g + 1}")
    index = [f"u{i:02d}" for i in range(len(rows))]
    return pd.DataFrame(rows, index=index, columns=range(t)), pd.Series(labels, index=index)


def test_within_club_gaps_are_stationary_between_clubs_are_not():
    data, clubs = _two_groups()
    result = cc.pairwise_convergence(data, clubs=clubs)
    summary = result.summary()
    assert summary["n_pairs"].tolist() == [120, 56, 64]
    assert summary.loc["within clubs", "share_rejected"] > 0.8
    assert summary.loc["between clubs", "share_rejected"] < 0.3
    assert list(result.by_club().index) == ["club1", "club2"]


def test_pairs_and_matrix():
    data, clubs = _two_groups(per_group=4, t=50)
    result = cc.pairwise_convergence(data, clubs=clubs, alpha=0.1)
    assert len(result.pairs) == 8 * 7 // 2
    first = result.pairs.iloc[0]
    gap = np.log(data.loc[first["unit_i"]]) - np.log(data.loc[first["unit_j"]])
    assert first["adf_stat"] == pytest.approx(cc.adf_test(gap.to_numpy()).stat)
    assert (result.pairs["reject"] == (result.pairs["p_value"] < 0.1)).all()
    matrix = result.matrix()
    assert matrix.shape == (8, 8) and np.isnan(np.diag(matrix)).all()
    np.testing.assert_allclose(matrix.fillna(0), matrix.T.fillna(0))
    assert matrix.notna().to_numpy().sum() == 8 * 7
    logs = cc.pairwise_convergence(np.log(data), log=False, clubs=clubs, alpha=0.1)
    pd.testing.assert_frame_equal(logs.pairs, result.pairs)


def test_divergent_units_count_as_between_clubs():
    data, clubs = _two_groups(per_group=4, t=50)
    clubs.iloc[0] = "divergent"
    result = cc.pairwise_convergence(data, clubs=clubs)
    with_divergent = result.pairs[(result.pairs["club_i"] == "divergent")]
    assert not with_divergent["same_club"].any() and with_divergent["between_clubs"].all()


def test_gdp_panel(filtered_gdp):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cc.ClubConvergenceWarning)
        merged = cc.find_clubs(filtered_gdp).merge("PS")
    result = cc.pairwise_convergence(cc.load_gdp(), clubs=merged)
    summary = result.summary()
    assert summary.loc["all pairs", "n_pairs"] == 152 * 151 // 2
    # Short annual panels give ADF tests little power, but the ordering holds.
    assert (
        summary.loc["within clubs", "share_rejected"]
        > summary.loc["between clubs", "share_rejected"]
        > summary.loc["all pairs", "expected_under_null"]
    )
    assert "pairs" in repr(result)
    ax = result.plot(theme="dark")
    assert [t.get_text() for t in ax.get_yticklabels()][0] == "club1"
    plt.close("all")


def test_errors():
    data, clubs = _two_groups(per_group=3, t=40)
    with pytest.raises(ValueError, match="alpha"):
        cc.pairwise_convergence(data, alpha=1.5)
    with pytest.raises(ValueError, match="positive"):
        cc.pairwise_convergence(-data)
    with pytest.raises(ValueError, match="clubs="):
        cc.pairwise_convergence(data).by_club()
    gappy = data.copy()
    gappy.iloc[0, 3] = np.nan
    with pytest.raises(ValueError, match="missing"):
        cc.pairwise_convergence(gappy)
