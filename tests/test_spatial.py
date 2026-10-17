import matplotlib.pyplot as plt
import numpy as np
import pytest

import convergence_clubs as cc
from convergence_clubs.results import Club

pytestmark = pytest.mark.filterwarnings("ignore::convergence_clubs.ClubConvergenceWarning")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def base():
    return cc.find_clubs(cc.load_filtered_gdp().iloc[:16])


def _with_clubs(base, clubs, n=None):
    """A result on the first ``n`` units of ``base`` with the given clubs (row positions)."""
    model = base.clubs[0].model
    n = n or len(base.data)
    data = base.data.iloc[:n]
    made = tuple(Club(f"club{k + 1}", tuple(ids), model) for k, ids in enumerate(clubs))
    used = {i for ids in clubs for i in ids}
    rest = tuple(i for i in range(n) if i not in used)
    return base._replace(data=data, clubs=made, divergent_ids=rest)


def _grid(units, side=4):
    """Rook neighbours on a side x side grid, units in row-major order."""
    out = {u: [] for u in units}
    for k, u in enumerate(units):
        r, c = divmod(k, side)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if 0 <= r + dr < side and 0 <= c + dc < side:
                out[u].append(units[(r + dr) * side + c + dc])
    return out


def test_hand_computed_path(base):
    result = _with_clubs(base, [[0, 1], [2, 3]], n=4)
    a, b, c, d = result.data.index
    spatial = cc.spatial_clustering(result, {a: [b], b: [a, c], c: [b, d], d: [c]}, seed=0)
    summary = spatial.summary()
    # 3 joins (a-b, b-c, c-d); a-b and c-d are within clubs.
    assert summary.loc["club1", "joins_within"] == 1 and summary.loc["club2", "joins_within"] == 1
    assert summary.loc["all clubs", "joins_within"] == 2
    # Random labels: P(both in club1) = 2*1 / (4*3) per join.
    assert summary.loc["club1", "expected"] == pytest.approx(3 * 2 / 12)
    assert summary.loc["all clubs", "expected"] == pytest.approx(3 * 4 / 12)
    assert spatial.n_isolated == 0
    local = spatial.local()
    assert local.loc[b, "n_neighbours"] == 2 and local.loc[b, "same_club_share"] == 0.5


def test_blocks_and_checkerboard(base):
    units = list(base.data.index)
    grid = _grid(units)
    blocks = _with_clubs(base, [[0, 1, 4, 5, 2, 3, 6, 7], [8, 9, 12, 13, 10, 11, 14, 15]])
    clustered = cc.spatial_clustering(blocks, grid, permutations=499, seed=1).summary()
    assert clustered.loc["all clubs", "joins_within"] == 20  # all but the 4 joins across halves
    assert clustered.loc["all clubs", "p_value"] < 0.01
    assert clustered.loc["all clubs", "ratio"] > 1.5

    black = [k for k in range(16) if (k // 4 + k % 4) % 2 == 0]
    white = [k for k in range(16) if k not in black]
    checker = cc.spatial_clustering(_with_clubs(base, [black, white]), grid, seed=1).summary()
    assert checker.loc["all clubs", "joins_within"] == 0
    assert checker.loc["all clubs", "p_value"] == 1.0


def test_expectation_matches_permutations(base):
    result = _with_clubs(base, [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11]])
    spatial = cc.spatial_clustering(
        result, _grid(list(result.data.index)), permutations=4000, seed=2
    )
    summary = spatial.summary()
    means = spatial.permutations.mean(axis=0)
    np.testing.assert_allclose(means, summary["expected"].to_numpy(), atol=0.12)
    assert list(summary.index) == ["club1", "club2", "club3", "divergent", "all clubs"]


def test_isolated_units_and_errors(base):
    result = _with_clubs(base, [[0, 1], [2, 3]], n=5)
    a, b, c, d, e = result.data.index
    spatial = cc.spatial_clustering(result, {a: [b], c: [d], "Atlantis": [a]}, seed=0)
    assert spatial.n_isolated == 1 and len(spatial.joins) == 2
    assert "2 joins" in repr(spatial)
    with pytest.raises(ValueError, match="neighbour_map or geodata"):
        cc.spatial_clustering(result)
    with pytest.raises(ValueError, match="permutations"):
        cc.spatial_clustering(result, {a: [b]}, permutations=0)
    with pytest.raises(ValueError, match="no two units"):
        cc.spatial_clustering(result, {a: [], e: ["Atlantis"]})


def test_neighbours_from_geodata(base):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    units = list(base.data.index[:9])
    squares = [box(k % 3, k // 3, k % 3 + 1, k // 3 + 1) for k in range(9)]
    geo = gpd.GeoDataFrame({"name": units}, geometry=squares)
    queen = cc.neighbours(geo, units, key="name")
    rook = cc.neighbours(geo, units, key="name", contiguity="rook")
    centre = units[4]
    assert len(queen[centre]) == 8 and len(rook[centre]) == 4
    assert set(rook[units[0]]) == {units[1], units[3]}

    # A small gap between regions: no contact unless the tolerance closes it.
    apart = gpd.GeoDataFrame({"name": units[:2]}, geometry=[box(0, 0, 1, 1), box(1.01, 0, 2, 1)])
    assert cc.neighbours(apart, units[:2], key="name")[units[0]] == []
    assert cc.neighbours(apart, units[:2], key="name", tolerance=0.01)[units[0]] == [units[1]]

    named = gpd.GeoDataFrame({"code": ["A", "B"]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)])
    mapped = cc.neighbours(named, units[:3], key="code", names={units[0]: "A", units[1]: "B"})
    assert mapped == {units[0]: [units[1]], units[1]: [units[0]], units[2]: []}
    with pytest.raises(ValueError, match="contiguity"):
        cc.neighbours(geo, units, key="name", contiguity="bishop")

    result = _with_clubs(base, [[0, 1, 3, 4], [2, 5, 6, 7, 8]], n=9)
    from_geo = cc.spatial_clustering(result, geodata=geo, key="name", seed=0)
    assert len(from_geo.joins) == 20  # queen contiguity on a 3 x 3 grid


def test_plot(base):
    result = _with_clubs(base, [[0, 1, 4, 5, 2, 3, 6, 7], [8, 9, 12, 13], [10, 11]])
    spatial = cc.spatial_clustering(result, _grid(list(result.data.index)), seed=0)
    ax = spatial.plot()
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels[-1] == "all clubs" and "club3" not in labels  # expected < 0.5 joins: left out
    assert labels[0] == "club1"
    assert "joins" in ax.get_title(loc="left")
