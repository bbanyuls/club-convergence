import numpy as np
import pandas as pd
import pytest

import convergence_clubs as cc
from convergence_clubs.cli import main


@pytest.fixture
def gappy():
    gdp = cc.load_gdp().copy()
    gdp.iloc[0, 5] = np.nan  # Afghanistan: one interior year
    gdp.iloc[1, 10:13] = np.nan  # Algeria: three interior years
    gdp.iloc[2, :2] = np.nan  # Antigua: two leading years
    gdp.iloc[3, -1] = np.nan  # Argentina: last year
    return gdp


def test_describe_missing(gappy):
    table = cc.describe_missing(gappy)
    assert list(table.index) == ["Afghanistan", "Algeria", "Antigua", "Argentina"]
    assert table.loc["Algeria"].tolist() == [3, 0, 0, 3, 3]
    assert table.loc["Antigua", "leading"] == 2 and table.loc["Argentina", "trailing"] == 1
    assert cc.describe_missing(cc.load_gdp()).empty


def test_fill_gaps_only_fills_short_interior_gaps(gappy):
    filled = cc.fill_gaps(gappy, max_gap=2)
    assert list(cc.describe_missing(filled).index) == ["Algeria", "Antigua", "Argentina"]
    row = gappy.iloc[0]
    assert filled.iloc[0, 5] == pytest.approx((row.iloc[4] + row.iloc[6]) / 2)
    geometric = cc.fill_gaps(gappy, max_gap=2, log=True)
    assert geometric.iloc[0, 5] == pytest.approx(np.sqrt(row.iloc[4] * row.iloc[6]))
    assert cc.describe_missing(cc.fill_gaps(gappy, max_gap=3)).index.tolist() == [
        "Antigua", "Argentina",
    ]  # fmt: skip
    pd.testing.assert_frame_equal(cc.fill_gaps(cc.load_gdp()), cc.load_gdp())


def test_fill_gaps_validation(gappy):
    with pytest.raises(ValueError, match="max_gap"):
        cc.fill_gaps(gappy, max_gap=0)
    with pytest.raises(ValueError, match="positive"):
        cc.fill_gaps(-gappy, log=True)


def test_balance(gappy):
    assert cc.balance(gappy).shape == (148, 34)
    trimmed = cc.balance(cc.fill_gaps(gappy, max_gap=3), "trim_periods")
    assert trimmed.shape == (152, 31) and list(trimmed.columns) == list(range(1972, 2003))
    with pytest.raises(ValueError, match="how"):
        cc.balance(gappy, "impute")
    all_missing = gappy.copy()
    all_missing.iloc[0, :] = np.nan
    with pytest.raises(ValueError, match="no period"):
        cc.balance(all_missing, "trim_periods")


def test_find_clubs_error_points_to_the_helpers(gappy):
    with pytest.raises(ValueError, match="fill_gaps"):
        cc.find_clubs(np.log(gappy))


def test_cli_fill_and_drop(gappy, tmp_path, capsys):
    csv = tmp_path / "gappy.csv"
    gappy.to_csv(csv)
    assert main([str(csv), "--log", "--hp-filter", "--fill-gaps", "2", "--drop-incomplete"]) == 0
    captured = capsys.readouterr()
    assert "dropped 3 incomplete units: Algeria; Antigua; Argentina" in captured.err
    assert main([str(csv), "--log", "--hp-filter"]) == 1
