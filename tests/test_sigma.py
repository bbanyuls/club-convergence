import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import convergence_clubs as cc
from convergence_clubs._stats import newey_west_lags


@pytest.fixture(scope="module")
def gdp():
    return cc.load_gdp()


@pytest.fixture(scope="module")
def clubs():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cc.ClubConvergenceWarning)
        return cc.find_clubs(cc.load_filtered_gdp()).merge("PS")


def test_measures_match_direct_computation(gdp):
    logs = np.log(gdp)
    sd = cc.sigma_convergence(gdp).series["all units"]
    pd.testing.assert_series_equal(sd, logs.std(ddof=1), check_names=False)
    var = cc.sigma_convergence(gdp, measure="var").series["all units"]
    np.testing.assert_allclose(var, logs.var(ddof=1))
    cv = cc.sigma_convergence(gdp, measure="cv").series["all units"]
    np.testing.assert_allclose(cv, gdp.std(ddof=1) / gdp.mean())
    already_logs = cc.sigma_convergence(logs, log=False, measure="cv").series["all units"]
    np.testing.assert_allclose(already_logs, cv)


def test_trend_matches_statsmodels_newey_west(gdp):
    result = cc.sigma_convergence(gdp)
    assert result.hac_lags == newey_west_lags(34) == 3
    y = result.series["all units"].to_numpy()
    ref = sm.OLS(y, sm.add_constant(np.arange(1970, 2004.0))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 3, "use_correction": False}
    )
    trend = result.trend().loc["all units"]
    assert trend["slope"] == pytest.approx(ref.params[1])
    assert trend["std_err"] == pytest.approx(ref.bse[1], rel=1e-10)
    assert trend["p_value"] == pytest.approx(ref.pvalues[1], abs=1e-12)
    assert trend["change"] == pytest.approx(y[-1] / y[0] - 1)


def test_clubs_converge_while_the_panel_diverges(gdp, clubs):
    result = cc.sigma_convergence(gdp, clubs=clubs)
    trend = result.trend()
    assert list(trend.index) == ["all units", "club1", "club2", "club3", "club4", "club5"]
    assert "club6" not in result.series
    assert not result.converges()
    assert all(result.converges(c) for c in ["club1", "club2", "club3", "club4", "club5"])
    assert trend.loc["all units", "slope"] > 0
    assert trend["n_units"].tolist() == [152, 50, 30, 21, 38, 11]
    assert "club6" in cc.sigma_convergence(gdp, clubs=clubs, min_units=2).series
    assert "not converging" in repr(result)


def test_non_numeric_periods_use_positions(gdp):
    strings = gdp.copy()
    strings.columns = [f"Y{c}" for c in gdp.columns]
    a = cc.sigma_convergence(strings).trend().loc["all units"]
    b = cc.sigma_convergence(gdp).trend().loc["all units"]
    assert a["slope"] == pytest.approx(b["slope"]) and a["t_value"] == pytest.approx(b["t_value"])


def test_errors(gdp):
    with pytest.raises(ValueError, match="measure"):
        cc.sigma_convergence(gdp, measure="gini")
    with pytest.raises(ValueError, match="positive"):
        cc.sigma_convergence(-gdp)
    gappy = gdp.copy()
    gappy.iloc[0, 3] = np.nan
    with pytest.raises(ValueError, match="fill_gaps"):
        cc.sigma_convergence(gappy)
    with pytest.raises(ValueError, match="at least"):
        cc.sigma_convergence(gdp.iloc[:, :2])


def test_plot(gdp, clubs):
    result = cc.sigma_convergence(gdp, clubs=clubs)
    ax = result.plot()
    lines = [ln for ln in ax.get_lines() if not ln.get_label().startswith("_")]
    assert [ln.get_label() for ln in lines] == list(result.series.columns)
    indexed = result.plot(index=True, groups=["all units", "club1"], theme="dark")
    first = [ln.get_ydata()[0] for ln in indexed.get_lines() if ln.get_label() == "club1"]
    assert first == [pytest.approx(1.0)]
    assert {t.get_text() for t in indexed.texts} == {"all units", "club1"}
    plt.close("all")
