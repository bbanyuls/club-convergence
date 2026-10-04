import numpy as np
import pandas as pd
import pytest

import convergence_clubs as cc
from convergence_clubs.sensitivity import _ari, _partition


def _clear_clubs(seed=0, n_per=15, t=40, limits=(1.6, 1.0, 0.4), noise=0.001):
    """Three well separated groups converging to different limits."""
    rng = np.random.default_rng(seed)
    tt = np.arange(1, t + 1)
    rows, truth = [], []
    for group, limit in enumerate(limits):
        for _ in range(n_per):
            start, speed = rng.uniform(0.3, 1.9), rng.uniform(0.1, 0.3)
            path = limit + (start - limit) * np.exp(-speed * tt)
            rows.append(10 + path + np.cumsum(rng.normal(0, noise, t)))
            truth.append(group)
    index = [f"u{i:02d}" for i in range(len(rows))]
    return pd.DataFrame(rows, index=index, columns=tt), np.array(truth)


@pytest.fixture(scope="module")
def clear():
    return _clear_clubs()


def test_known_clubs_are_recovered_with_full_confidence(clear):
    data, truth = clear
    assert _ari(_partition(cc.find_clubs(data)), truth) == 1.0
    for method, kwargs in [("units", {}), ("cycle", {"filter_lambda": 400}), ("endpoint", {})]:
        runs = cc.bootstrap(data, 20, method=method, seed=0, **kwargs)
        assert runs.stability()["stability"].min() > 0.95, method
        assert runs.uncertain().empty


def test_units_subsampling(filtered_gdp):
    runs = cc.bootstrap(filtered_gdp, 15, fraction=0.7, seed=3)
    assert len(runs) == 16
    table = runs.table()
    assert table.loc[0, "n_units"] == 152 and (table["n_units"][1:] == 106).all()
    assert table.loc[0, "sizes"] == "50-30-21-24-14-11-2"  # full-sample baseline
    st = runs.stability()
    assert (st["n_runs"] < 16).any() and (st["n_runs"] >= 1).all()
    share = runs.co_membership().to_numpy()
    finite = share[np.isfinite(share)]
    assert finite.min() >= 0 and finite.max() <= 1


def test_reproducible_with_seed(filtered_gdp):
    a = cc.bootstrap(filtered_gdp, 5, seed=42).table()
    b = cc.bootstrap(filtered_gdp, 5, seed=42).table()
    pd.testing.assert_frame_equal(a, b)
    c = cc.bootstrap(filtered_gdp, 5, seed=43).table()
    assert not a["sizes"].equals(c["sizes"])


def test_cycle_bootstrap_uses_unfiltered_data():
    log_gdp = np.log(cc.load_gdp())
    runs = cc.bootstrap(log_gdp, 5, method="cycle", filter_lambda=400, seed=0)
    # The full-sample run is the HP-filtered data, i.e. the paper's clubs.
    assert runs.table().loc[0, "sizes"] == "50-30-21-24-14-11-2"
    assert (runs.stability()["n_runs"] == 6).all()


def test_endpoint(filtered_gdp):
    runs = cc.bootstrap(filtered_gdp, method="endpoint", max_drop=3)
    table = runs.table()
    assert len(runs) == 4 and table["dropped_periods"].tolist()[1:] == [1, 2, 3]
    assert [r.data.shape[1] for _, r in runs] == [34, 33, 32, 31]


def test_merge_options_are_applied(filtered_gdp):
    runs = cc.bootstrap(filtered_gdp, 2, merge="PS", seed=0)
    assert runs.baseline.is_merged
    assert runs.table().loc[0, "sizes"] == "50-30-21-38-11-2"


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"method": "jackknife"}, "method"),
        ({"method": "cycle"}, "filter_lambda"),
        ({"fraction": 0}, "fraction"),
        ({"n": 0}, "n must"),
        ({"method": "endpoint", "max_drop": 40}, "max_drop"),
        ({"method": "cycle", "filter_lambda": 400, "block_length": 99}, "block_length"),
    ],
)
def test_invalid_arguments(filtered_gdp, kwargs, match):
    with pytest.raises(ValueError, match=match):
        cc.bootstrap(filtered_gdp, **kwargs)


def test_stability_plot(filtered_gdp):
    import matplotlib.pyplot as plt

    runs = cc.bootstrap(filtered_gdp, 10, seed=0)
    ax = runs.plot_stability(label=8)
    assert [t.get_text() for t in ax.get_yticklabels()] == [f"club{i}" for i in range(1, 8)]
    assert len(ax.collections) == 7  # one set of dots per club
    names = [t.get_text() for t in ax.texts if t.get_text() != "0.8"]
    assert 1 <= len(names) <= 8
    least = runs.stability()["stability"].idxmin()
    assert least in names  # the least stable unit is always labelled
    renderer = ax.figure.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in ax.texts if t.get_text() != "0.8"]
    assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1 :])
    assert runs.plot_stability(threshold=None, theme="dark").get_title(loc="left")
    plt.close("all")
