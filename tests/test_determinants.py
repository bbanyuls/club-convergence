import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from statsmodels.miscmodels.ordinal_model import OrderedModel

import convergence_clubs as cc


@pytest.fixture(scope="module")
def clubs():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cc.ClubConvergenceWarning)
        lp = cc.load_productivity("lp")
        return cc.find_clubs(cc.hp_filter(np.log(lp)), convention="Stata", time_trim=0.333).merge(
            "PS"
        )


@pytest.fixture(scope="module")
def covariates():
    info = cc.productivity_info()
    return pd.DataFrame(
        {
            "log_h": np.log(cc.load_productivity("h")[1990]),
            "log_ky": np.log(cc.load_productivity("ky")[1990]),
            "log_tfp": np.log(cc.load_productivity("tfp")[1990]),
            "africa": (info["region"] == "Africa").astype(float),
        }
    )


@pytest.mark.parametrize("model", ["logit", "probit"])
def test_matches_statsmodels_ordered_model(clubs, covariates, model):
    result = cc.club_determinants(clubs, covariates, model=model)
    members = clubs.memberships()
    members = members[members != "divergent"]
    order = [c.name for c in reversed(clubs.clubs)]
    y = pd.Series(pd.Categorical(members, categories=order, ordered=True), index=members.index)
    ref = OrderedModel(y, covariates.loc[members.index], distr=model).fit(
        method="newton", disp=False, maxiter=200
    )
    k = covariates.shape[1]
    np.testing.assert_allclose(result.coefficients["coef"], ref.params[:k], rtol=1e-6)
    np.testing.assert_allclose(result.coefficients["std_err"], ref.bse[:k], rtol=1e-4)
    np.testing.assert_allclose(
        result.thresholds, ref.model.transform_threshold_params(ref.params)[1:-1], rtol=1e-6
    )
    assert result.log_likelihood == pytest.approx(ref.llf, abs=1e-8)
    assert result.converged and result.n == len(members)
    assert result.classes == tuple(order)
    assert set(result.excluded) == set(clubs.divergent)


def test_interpretation_and_tables(clubs, covariates):
    result = cc.club_determinants(clubs, covariates)
    coef = result.summary()
    # Initial human capital and TFP raise the odds of a higher club; Africa lowers them.
    assert coef.loc["log_h", "coef"] > 0 and coef.loc["log_tfp", "coef"] > 0
    assert coef.loc["africa", "coef"] < 0 and coef.loc["africa", "p_value"] < 0.05
    np.testing.assert_allclose(coef["odds_ratio"], np.exp(coef["coef"]))
    assert 0 < result.pseudo_r2 < 1
    assert list(result.thresholds.index) == ["club4|club3", "club3|club2", "club2|club1"]
    assert np.all(np.diff(result.thresholds.to_numpy()) > 0)
    assert "ordered logit" in repr(result)
    assert "odds_ratio" not in cc.club_determinants(clubs, covariates, model="probit").summary()


def test_predicted_probabilities(clubs, covariates):
    result = cc.club_determinants(clubs, covariates)
    proba = result.predict_proba(covariates)
    assert list(proba.columns) == list(result.classes)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert (proba.to_numpy() >= 0).all()
    # Raising initial human capital shifts probability towards the top club.
    better = covariates.assign(log_h=covariates["log_h"] + 0.5)
    assert (result.predict_proba(better)["club1"] >= proba["club1"] - 1e-12).all()


def test_errors(clubs, covariates):
    with pytest.raises(ValueError, match="model"):
        cc.club_determinants(clubs, covariates, model="poisson")
    with pytest.raises(ValueError, match="constant"):
        cc.club_determinants(clubs, covariates.assign(one=1.0))
    with pytest.raises(ValueError, match="collinear"):
        cc.club_determinants(clubs, covariates.assign(twice=2 * covariates["log_h"]))
    with pytest.raises(ValueError, match="covariates missing"):
        cc.club_determinants(clubs, covariates.iloc[5:])
    single = cc.find_clubs(cc.load_filtered_gdp().iloc[:2])
    with pytest.raises(ValueError, match="two clubs"):
        cc.club_determinants(single, covariates)


def test_plot(clubs, covariates):
    ax = cc.club_determinants(clubs, covariates).plot(theme="dark")
    assert [t.get_text() for t in ax.get_yticklabels()] == ["africa", "log_tfp", "log_ky", "log_h"]
    assert "ordered logit" in ax.get_title(loc="left")
    plt.close("all")
