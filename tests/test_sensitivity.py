import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import convergence_clubs as cc
from convergence_clubs.sensitivity import _ari


@pytest.fixture(scope="module")
def runs(filtered_gdp):
    return cc.sensitivity(filtered_gdp, time_trim=[1 / 3, 0.3], merge=[None, "PS"])


def test_grid_order_and_baseline(runs, filtered_gdp):
    assert len(runs) == 2 * 2 * 1 * 2  # time_trim x hac_method x cstar x merge
    params, first = runs.runs[0]
    assert params == {
        "time_trim": 1 / 3, "hac_method": "FQSB", "cstar": 0.0, "merge": None, "sort_fraction": 0.0,
    }  # fmt: skip
    assert runs.baseline is first
    assert [len(c) for c in first] == [50, 30, 21, 24, 14, 11, 2]


def test_table(runs):
    table = runs.table()
    assert len(table) == len(runs)
    assert table.loc[0, "ari_vs_baseline"] == 1.0
    assert table.loc[0, "sizes"] == "50-30-21-24-14-11-2"
    assert set(table["merge"]) == {"none", "PS"}
    assert table["ari_vs_baseline"].between(-1, 1).all()


def test_co_membership(runs):
    share = runs.co_membership()
    assert share.shape == (152, 152)
    np.testing.assert_allclose(share.to_numpy(), share.to_numpy().T)
    assert share.to_numpy().min() >= 0 and share.to_numpy().max() <= 1
    # Units that are always in the same club have share 1, e.g. within the baseline.
    first = list(runs.baseline.units("club1"))
    assert share.loc["United States", "Norway"] == pytest.approx(
        np.mean([r.memberships()["United States"] == r.memberships()["Norway"] for _, r in runs])
    )
    assert share.loc[first[0], first[0]] == 1.0


def test_stability(runs):
    st = runs.stability()
    assert list(st.columns) == ["baseline_club", "stability", "divergent_share"]
    assert st["stability"].between(0, 1).all()
    assert (st["baseline_club"] == runs.baseline.memberships()).all()


def test_ari_reference_values():
    # Hand-computed: 4/7 (scikit-learn's adjusted_rand_score gives the same).
    assert _ari(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 2])) == pytest.approx(4 / 7)
    labels = np.array([0, 0, 1, 1, 2])
    assert _ari(labels, labels + 5) == 1.0
    assert _ari(np.arange(4), np.arange(4)) == 1.0


def test_compare_clubs(filtered_gdp):
    fqsb = cc.find_clubs(filtered_gdp)
    merged = fqsb.merge("PS")
    table = cc.compare_clubs(fqsb, merged)
    assert table.to_numpy().sum() == len(filtered_gdp)
    assert table.loc["club4", "club4"] == 24 and table.loc["club5", "club4"] == 14
    assert cc.adjusted_rand_index(fqsb, fqsb) == 1.0
    with pytest.raises(ValueError, match="same units"):
        cc.compare_clubs(fqsb, cc.find_clubs(filtered_gdp.iloc[:50]))


def test_plot(runs):
    ax = runs.plot()
    assert ax.get_title(loc="left") == f"Co-membership across {len(runs)} runs"
    assert [t.get_text() for t in ax.get_yticklabels()][:2] == ["club1", "club2"]
    small = cc.sensitivity(cc.load_filtered_gdp().iloc[:30], hac_method=["FQSB"])
    assert len(small.plot(theme="dark").get_yticklabels()) == 30  # unit names
    plt.close("all")


def test_empty_setting_is_rejected(filtered_gdp):
    with pytest.raises(ValueError, match="at least one value"):
        cc.sensitivity(filtered_gdp, cstar=[])


def test_wide_panel(filtered_gdp):
    long = filtered_gdp.stack().rename("y").rename_axis(["country", "year"]).reset_index()
    wide = cc.wide_panel(
        long.sample(frac=1, random_state=0), unit="country", time="year", value="y"
    )
    pd.testing.assert_frame_equal(
        wide, filtered_gdp.rename_axis(index="country", columns=None), check_like=False
    )
    assert (
        cc.wide_panel(long.iloc[1:], unit="country", time="year", value="y", dropna=True).shape[0]
        == len(filtered_gdp) - 1
    )
    with pytest.raises(ValueError, match="several rows"):
        cc.wide_panel(pd.concat([long, long.head(1)]), unit="country", time="year", value="y")
    with pytest.raises(KeyError):
        cc.wide_panel(long, unit="region", time="year", value="y")


def test_sort_fraction(filtered_gdp):
    one_period = cc.find_clubs(filtered_gdp, sort_fraction=1 / 34)
    assert [len(c) for c in one_period] == [50, 30, 21, 24, 14, 11, 2]
    assert [len(c) for c in cc.find_clubs(filtered_gdp, sort_fraction=0.1)] != [
        len(c) for c in one_period
    ]
    with pytest.raises(ValueError, match="either"):
        cc.find_clubs(filtered_gdp, ref_col=2003, sort_fraction=0.2)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        cc.find_clubs(filtered_gdp, sort_fraction=2)


def test_input_errors_name_the_units(filtered_gdp):
    data = filtered_gdp.copy()
    data.iloc[3, 5] = np.nan
    with pytest.raises(ValueError, match="Argentina"):
        cc.find_clubs(data)
    with pytest.raises(ValueError, match="unique"):
        cc.find_clubs(pd.concat([filtered_gdp, filtered_gdp.head(1)]))


def test_summary_members(filtered_gdp):
    summary = cc.find_clubs(filtered_gdp).summary(members=True)
    assert summary.loc["club7", "members"] == "Congo, Dem. Rep.; Liberia"
    assert summary["members"].str.count("; ").add(1).tolist() == summary["n_units"].tolist()
