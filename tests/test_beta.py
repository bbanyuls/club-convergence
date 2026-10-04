import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.stats
import statsmodels.api as sm

import convergence_clubs as cc
from convergence_clubs._stats import betainc, ols, t_two_sided_p


@pytest.fixture(scope="module")
def gdp():
    return cc.load_gdp()


@pytest.fixture(scope="module")
def clubs():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cc.ClubConvergenceWarning)
        return cc.find_clubs(cc.load_filtered_gdp()).merge("PS")


def _reference(gdp, X_extra=None, cov="HC1"):
    y0 = np.log(gdp[1970]).to_numpy()
    growth = (np.log(gdp[2003]).to_numpy() - y0) / 33
    X = np.column_stack([np.ones_like(y0), y0] + ([] if X_extra is None else [X_extra]))
    return sm.OLS(growth, X).fit(cov_type=cov, use_t=True)


# -- statistics helpers -------------------------------------------------------


def test_t_distribution_matches_scipy():
    rng = np.random.default_rng(0)
    for t, df in zip(rng.normal(0, 4, 300), rng.uniform(1, 300, 300), strict=True):
        assert t_two_sided_p(t, df) == pytest.approx(2 * scipy.stats.t.sf(abs(t), df), abs=1e-11)
    for a, b, x in rng.uniform([0.1, 0.1, 0], [40, 40, 1], (300, 3)):
        assert betainc(a, b, x) == pytest.approx(scipy.special.betainc(a, b, x), abs=1e-12)
    assert t_two_sided_p(math.inf, 5) == 0.0 and math.isnan(t_two_sided_p(1.0, 0))


@pytest.mark.parametrize("cov", ["nonrobust", "HC0", "HC1"])
def test_ols_matches_statsmodels(cov):
    rng = np.random.default_rng(1)
    X = sm.add_constant(rng.normal(size=(50, 2)))
    y = X @ [1.0, -0.5, 2.0] + rng.normal(size=50) * (1 + np.abs(X[:, 1]))
    ours, ref = ols(y, X, cov), sm.OLS(y, X).fit(cov_type=cov, use_t=True)
    np.testing.assert_allclose(ours.params, ref.params, rtol=1e-12)
    np.testing.assert_allclose(ours.std_err, ref.bse, rtol=1e-10)
    np.testing.assert_allclose(ours.p_values(), ref.pvalues, rtol=1e-8, atol=1e-14)
    assert ours.r_squared == pytest.approx(ref.rsquared)


def test_newey_west_matches_statsmodels():
    rng = np.random.default_rng(2)
    X = sm.add_constant(np.arange(40.0))
    y = 0.3 - 0.01 * X[:, 1] + np.cumsum(rng.normal(0, 0.05, 40)) * 0.2
    ref = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3, "use_correction": False})
    np.testing.assert_allclose(ols(y, X, "HAC", lags=3).std_err, ref.bse, rtol=1e-10)
    with pytest.raises(ValueError, match="cov_type"):
        ols(y, X, "HC3")
    with pytest.raises(ValueError, match="observations"):
        ols(y[:2], X[:2])


# -- β-convergence ------------------------------------------------------------


def test_absolute_beta_matches_statsmodels(gdp):
    result = cc.beta_convergence(gdp)
    ref = _reference(gdp)
    np.testing.assert_allclose(result.summary()["coef"], ref.params, rtol=1e-10)
    np.testing.assert_allclose(result.summary()["std_err"], ref.bse, rtol=1e-10)
    np.testing.assert_allclose(result.summary()["p_value"], ref.pvalues, rtol=1e-8)
    assert result.r_squared == pytest.approx(ref.rsquared)
    assert result.periods == 33 and result.n == 152
    assert not result.converges()  # no absolute convergence in this panel


def test_speed_and_half_life(gdp):
    result = cc.beta_convergence(gdp, cov_type="nonrobust")
    beta = result.beta
    assert result.speed == pytest.approx(-math.log(1 + beta * 33) / 33)
    assert result.half_life == pytest.approx(math.log(2) / result.speed)


def test_conditional_on_controls(gdp):
    rng = np.random.default_rng(3)
    controls = pd.DataFrame({"schooling": rng.normal(size=len(gdp))}, index=gdp.index)
    result = cc.beta_convergence(gdp, controls=controls)
    ref = _reference(gdp, controls["schooling"].to_numpy())
    np.testing.assert_allclose(result.summary()["coef"], ref.params, rtol=1e-10)
    assert list(result.summary().index) == ["const", "log_initial", "schooling"]
    assert result.control_means == pytest.approx((controls["schooling"].mean(),))
    with pytest.raises(ValueError, match="controls missing"):
        cc.beta_convergence(gdp, controls=controls.iloc[1:])
    with pytest.raises(ValueError, match="clash"):
        cc.beta_convergence(gdp, controls=controls.rename(columns={"schooling": "const"}))


def test_club_effects_and_by_club(gdp, clubs):
    result = cc.beta_convergence(gdp, clubs=clubs, club_effects=True)
    assert [t for t in result.summary().index if t.startswith("club[")] == [
        f"club[club{i}]" for i in range(2, 7)
    ]
    assert result.converges() and 20 < result.half_life < 50
    dummies = pd.get_dummies(clubs.memberships()).drop(columns="club1").astype(float)
    ref = _reference(gdp, dummies.to_numpy())
    np.testing.assert_allclose(result.summary()["coef"], ref.params, rtol=1e-9)

    by_club = result.by_club()
    assert list(by_club.index) == [c.name for c in clubs]
    assert by_club["n_units"].tolist() == [50, 30, 21, 38, 11, 2]
    assert np.isnan(by_club.loc["club6", "beta"])  # two units: not estimated
    within = gdp.loc[list(clubs.units("club1"))]
    assert by_club.loc["club1", "beta"] == pytest.approx(cc.beta_convergence(within).beta)


def test_options_and_errors(gdp, clubs):
    sub = cc.beta_convergence(gdp, start=1980, end=2000)
    assert sub.periods == 20 and sub.start == 1980
    logs = cc.beta_convergence(np.log(gdp), log=False)
    assert logs.beta == pytest.approx(cc.beta_convergence(gdp).beta)
    strings = gdp.copy()
    strings.columns = [f"Y{c}" for c in gdp.columns]
    assert cc.beta_convergence(strings).periods == 33
    with pytest.raises(ValueError, match="positive"):
        cc.beta_convergence(-gdp)
    with pytest.raises(ValueError, match="after start"):
        cc.beta_convergence(gdp, start=2000, end=1990)
    with pytest.raises(KeyError):
        cc.beta_convergence(gdp, start=1900)
    with pytest.raises(ValueError, match="needs clubs"):
        cc.beta_convergence(gdp, club_effects=True)
    with pytest.raises(ValueError, match="by club"):
        cc.beta_convergence(gdp).by_club()
    with pytest.raises(ValueError, match="no club"):
        cc.beta_convergence(gdp, clubs=clubs.memberships().iloc[1:])
    with pytest.raises(ValueError, match="cov_type"):
        cc.beta_convergence(gdp, cov_type="HAC")
    assert "absolute" in repr(cc.beta_convergence(gdp))


def test_plot(gdp, clubs):
    ax = cc.beta_convergence(gdp).plot(label=["China", "Atlantis"])
    assert [t.get_text() for t in ax.texts][1:] == ["China"]
    assert "not significantly negative" in ax.texts[0].get_text()
    ax = cc.beta_convergence(gdp, clubs=clubs, club_effects=True).plot(theme="dark")
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [c.name for c in clubs]
    assert len(ax.get_lines()) == 1 + 6  # zero line + one fitted line per club
    assert "half-life: 33 years" in ax.texts[0].get_text()
    plt.close("all")


def test_collinear_regressors_raise_a_clear_error(gdp):
    duplicate = pd.DataFrame({"copy": np.log(gdp[1970])}, index=gdp.index)
    with pytest.raises(ValueError, match="collinear"):
        cc.beta_convergence(gdp, controls=duplicate)
