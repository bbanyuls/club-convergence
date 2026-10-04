"""The bundled productivity panel of Mendez (2020) and its Stata results."""

import warnings

import numpy as np
import pandas as pd
import pytest

import convergence_clubs as cc

from .test_stata import MENDEZ, _labels, _stata

SAMPLES = {"hiYes": "high_income", "hiNo": "other"}
VARIABLES = {"log_lp": "lp", "log_ky": "ky", "log_h": "h", "log_tfp": "tfp"}


def test_shape_and_samples():
    lp = cc.load_productivity()
    assert lp.shape == (108, 25) and list(lp.columns) == list(range(1990, 2015))
    assert not lp.isna().to_numpy().any() and (lp.to_numpy() > 0).all()
    high, other = cc.load_productivity(sample="high_income"), cc.load_productivity(sample="other")
    assert (len(high), len(other)) == (26, 82)
    assert set(high.index) | set(other.index) == set(lp.index)
    info = cc.productivity_info()
    assert info.loc[high.index, "high_income_1990"].all()
    assert info["iso3"].is_unique and set(info["region"]) >= {"Africa", "Europe", "Asia"}
    assert cc.load_productivity("GDPpc").shape == (108, 25)  # case-insensitive


def test_errors():
    with pytest.raises(ValueError, match="variable"):
        cc.load_productivity("wages")
    with pytest.raises(ValueError, match="sample"):
        cc.load_productivity(sample="oecd")


@pytest.mark.parametrize("sample", SAMPLES)
@pytest.mark.parametrize("variable", VARIABLES)
def test_hp_trends_match_the_published_series(sample, variable):
    levels = cc.load_productivity(VARIABLES[variable], sample=SAMPLES[sample])
    published = cc.wide_panel(
        MENDEZ[MENDEZ["sample"] == sample], unit="country", time="year", value=variable
    )
    ours = cc.hp_filter(np.log(levels), lamb=400)
    pd.testing.assert_index_equal(ours.index, published.index)
    # The published series were stored as single-precision floats.
    np.testing.assert_allclose(ours.to_numpy(), published.to_numpy(), atol=2e-6)


@pytest.mark.parametrize("sample", SAMPLES)
@pytest.mark.parametrize("variable", VARIABLES)
def test_book_clubs_from_raw_levels(sample, variable):
    levels = cc.load_productivity(VARIABLES[variable], sample=SAMPLES[sample])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cc.ClubConvergenceWarning)
        clubs = cc.find_clubs(cc.hp_filter(np.log(levels)), convention="Stata", time_trim=0.333)
    data = MENDEZ[MENDEZ["sample"] == sample]
    last = data[data["year"] == 2014].set_index("country").reindex(levels.index)
    assert _labels(clubs) == _stata(last[f"club_{variable}"])
    assert _labels(clubs.merge("PS")) == _stata(last[f"finalclub_{variable}"])
