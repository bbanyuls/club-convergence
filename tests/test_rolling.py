import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import convergence_clubs as cc
from convergence_clubs.results import Club
from convergence_clubs.rolling import RollingClubs

pytestmark = pytest.mark.filterwarnings("ignore::convergence_clubs.ClubConvergenceWarning")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def roll():
    return cc.rolling_clubs(np.log(cc.load_gdp()), window=20, step=7, filter_lambda=400)


def test_windows(roll):
    assert roll.windows == ("1970–1989", "1977–1996", "1984–2003")
    assert len(roll) == 3 and roll["1977–1996"] is roll[1]
    assert [r.data.shape[1] for r in roll.results] == [20, 20, 20]
    default = cc.rolling_clubs(cc.load_filtered_gdp(), window=20)
    assert default.windows == ("1970–1989", "1980–1999", "1984–2003")
    assert "3 windows" in repr(roll)


def test_each_window_is_clustered_on_its_own(roll):
    first = cc.find_clubs(cc.hp_filter(np.log(cc.load_gdp()).iloc[:, :20], lamb=400)).merge("PS")
    assert roll[0].memberships().equals(first.memberships())


def test_tables(roll):
    members = roll.memberships()
    assert members.shape == (152, 3) and list(members.columns) == list(roll.windows)
    table = roll.transitions()
    assert table.to_numpy().sum() == 152
    assert list(table.index) == [c.name for c in roll[0]] + ["divergent"]
    assert table.loc["club1", "club1"] == 41  # of the 45 in club1 at the end
    summary = roll.summary()
    assert summary["n_clubs"].tolist() == [5, 5, 8]
    assert np.isnan(summary["ari_vs_previous"].iloc[0])
    assert 0 < summary["ari_vs_previous"].iloc[1] < 1


def test_movers_follow_club_mates(roll):
    movers = roll.movers()
    members = roll.memberships()
    for a, b in zip(roll.windows[:-1], roll.windows[1:], strict=True):
        for club, mates in members.groupby(members[a])[b]:
            if club == "divergent":
                continue
            modal = mates.value_counts().index[0]
            moved = set(movers[(movers.from_window == a) & (movers.from_club == club)].unit)
            assert moved == set(mates.index[mates != modal])
    assert {"Brazil", "Hungary"} <= set(movers[movers.direction == "down"].unit)


def _result(base, clubs):
    """A result with the given clubs (lists of row positions) on the data of ``base``."""
    model = base.clubs[0].model
    made = tuple(Club(f"club{k + 1}", tuple(ids), model) for k, ids in enumerate(clubs))
    used = {i for ids in clubs for i in ids}
    rest = tuple(i for i in range(len(base.data)) if i not in used)
    return base._replace(clubs=made, divergent_ids=rest)


def test_movers_definition():
    base = cc.find_clubs(cc.load_filtered_gdp().iloc[:6])
    units = base.data.index
    # Window 1: {0,1,2} {3,4}, 5 divergent. Window 2: {3,4} {0,1} {2} {5}:
    # 0 and 1 move together (renumbered, not movers); 2 falls behind them;
    # 3 and 4 are renumbered to club1; 5 joins a club.
    w1 = _result(base, [[0, 1, 2], [3, 4]])
    w2 = _result(base, [[3, 4], [0, 1], [2], [5]])
    movers = RollingClubs(("a", "b"), (w1, w2)).movers().set_index("unit")
    assert set(movers.index) == {units[2], units[5]}
    assert movers.loc[units[2], "direction"] == "down"
    assert movers.loc[units[2], "mates_went_to"] == "club2"
    assert movers.loc[units[5], "direction"] == "from divergent"
    # 2 joins the club above the one its mates 0 and 1 go to.
    up = RollingClubs(("a", "c"), (w1, _result(base, [[2, 3, 4], [0, 1]]))).movers()
    assert up.set_index("unit").loc[units[2], "direction"] == "up"
    to_div = RollingClubs(("a", "c"), (w1, _result(base, [[0, 1, 5], [3, 4]]))).movers()
    assert to_div.set_index("unit").loc[units[2], "direction"] == "to divergent"
    # 0 and 1 become divergent, 2 stays in a club.
    alone = RollingClubs(("a", "d"), (w1, _result(base, [[3, 4], [2]]))).movers()
    assert alone.set_index("unit").loc[units[2], "direction"] == "mates diverged"


def test_parallel_and_validation():
    data = cc.load_filtered_gdp().iloc[:40]
    serial = cc.rolling_clubs(data, window=15, step=10)
    parallel = cc.rolling_clubs(data, window=15, step=10, n_jobs=2)
    pd.testing.assert_frame_equal(serial.memberships(), parallel.memberships())
    with pytest.raises(ValueError, match="window"):
        cc.rolling_clubs(data, window=4)
    with pytest.raises(ValueError, match="window"):
        cc.rolling_clubs(data, window=99)
    with pytest.raises(ValueError, match="step"):
        cc.rolling_clubs(data, window=15, step=0)


def test_plots(roll):
    ax = roll.plot()
    bars = [p for p in ax.patches if type(p).__name__ == "Rectangle"]
    groups = sum(r.n_clubs + (r.n_divergent > 0) for r in roll.results)
    assert len(bars) == groups
    flows = [p for p in ax.patches if type(p).__name__ == "PathPatch"]
    members = roll.memberships()
    expected = sum(
        len(members.groupby([members[a], members[b]]).size())
        for a, b in zip(roll.windows[:-1], roll.windows[1:], strict=True)
    )
    assert len(flows) == expected
    assert [t.get_text() for t in ax.get_xticklabels()] == list(roll.windows)
    assert "club1 (58)" in [t.get_text() for t in ax.texts]
    assert not roll.plot(labels=False, colors=["#000000"]).texts

    fig = roll.plot_interactive(theme="dark")
    assert sum(fig.data[0].link.value) == 152 * 2
    assert fig.layout.paper_bgcolor == cc.DARK.surface
