"""Results published in Sichera and Pizzuto (2019), The R Journal 11(2), 142-151."""

import numpy as np
import pytest

import convergence_clubs as cc


@pytest.fixture(scope="module")
def clubs(filtered_gdp):
    return cc.find_clubs(filtered_gdp, time_trim=1 / 3, cstar=0, hac_method="FQSB")


def test_whole_panel_diverges(filtered_gdp):
    H = cc.cross_sectional_variation(filtered_gdp.to_numpy())
    model = cc.log_t_test(H, time_trim=1 / 3, hac_method="FQSB")
    assert (round(model.beta, 3), round(model.std_err, 3), round(model.t_value, 3)) == (
        -0.875,
        0.005,
        -159.555,
    )
    assert not model.converges()


def test_clubs_summary(clubs):
    summary = clubs.summary()
    assert clubs.n_divergent == 0
    assert summary["n_units"].tolist() == [50, 30, 21, 24, 14, 11, 2]
    np.testing.assert_allclose(
        summary[["beta", "std_err", "t_value"]].round(3).to_numpy(),
        [
            [0.382, 0.041, 9.282],
            [0.240, 0.035, 6.904],
            [0.110, 0.032, 3.402],
            [0.131, 0.064, 2.055],
            [0.190, 0.111, 1.701],
            [1.003, 0.166, 6.024],
            [-0.470, 0.842, -0.559],
        ],
    )


def test_first_club_members(clubs):
    expected = (
        "United States, Norway, Bermuda, United Arab Emirates, Qatar, Luxembourg, Singapore, "
        "Switzerland, Hong Kong, Denmark, Ireland, Austria, Australia, Canada, Macao, "
        "Netherlands, Kuwait, Iceland, United Kingdom, Germany, France, Sweden, Belgium, Japan, "
        "Brunei, Finland, Italy, Cyprus, Puerto Rico, Israel, New Zealand, Taiwan, Spain, Malta, "
        "Korea, Republic of, Portugal, Oman, Mauritius, Antigua, St. Kitts & Nevis, Chile, "
        "Malaysia, Equatorial Guinea, Dominica, St.Vincent & Grenadines, Botswana, Thailand, "
        "Cape Verde, China, Maldives"
    )
    # The paper prints R's make.names() version of these names (spaces -> dots).
    normalize = lambda s: "".join(ch for ch in s if ch.isalnum())  # noqa: E731
    got = [normalize(u) for u in clubs.units("club1")]
    assert normalize(expected) == "".join(got)
    assert len(got) == 50


def test_ps_merge(clubs):
    summary = clubs.merge("PS").summary()
    assert summary["n_units"].tolist() == [50, 30, 21, 38, 11, 2]
    assert summary.loc["club4", "merged_clubs"] == "club4, club5"
    assert summary.loc["club4", ["beta", "std_err", "t_value"]].astype(float).round(3).tolist() == [
        -0.044,
        0.07,
        -0.636,
    ]
