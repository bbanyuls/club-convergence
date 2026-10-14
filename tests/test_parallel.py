import os
import warnings

import numpy as np
import pandas as pd
import pytest

import convergence_clubs as cc
from convergence_clubs._parallel import n_workers, parallel_map

pytestmark = pytest.mark.filterwarnings("ignore::convergence_clubs.ClubConvergenceWarning")


def test_n_workers():
    cpus = os.cpu_count() or 1
    assert n_workers(None) == n_workers(1) == 1
    assert n_workers(3) == 3
    assert n_workers(-1) == cpus
    assert n_workers(-2) == max(1, cpus - 1)
    assert n_workers(-10_000) == 1
    for bad in (0, 1.5, True):
        with pytest.raises(ValueError, match="n_jobs"):
            n_workers(bad)


def test_parallel_map_keeps_order():
    assert parallel_map(abs, [-3, 2, -1], n_jobs=2) == [3, 2, 1]
    assert parallel_map(abs, [], n_jobs=4) == []


@pytest.fixture(scope="module")
def data():
    return cc.load_filtered_gdp().iloc[:60]


def test_results_do_not_depend_on_n_jobs(data):
    serial = cc.bootstrap(data, 6, seed=3, merge="PS")
    parallel = cc.bootstrap(data, 6, seed=3, merge="PS", n_jobs=2)
    pd.testing.assert_frame_equal(serial.stability(), parallel.stability())
    pd.testing.assert_frame_equal(serial.table(), parallel.table())

    grid = {"time_trim": [1 / 3, 0.3], "hac_method": ["FQSB", "AQSB"]}
    pd.testing.assert_frame_equal(
        cc.sensitivity(data, **grid).table(), cc.sensitivity(data, **grid, n_jobs=2).table()
    )

    gdp = cc.load_gdp().iloc[:25]
    a = cc.pairwise_convergence(gdp, kpss=True)
    b = cc.pairwise_convergence(gdp, kpss=True, n_jobs=3)
    pd.testing.assert_frame_equal(a.pairs, b.pairs)
    assert list(a.pairs.columns[:2]) == ["unit_i", "unit_j"]
    assert a.pairs.iloc[0]["unit_j"] == gdp.index[1]


def test_lag_search_matches_separate_fits():
    from convergence_clubs.pairwise import _design, _fit, _nested_ssr

    rng = np.random.default_rng(0)
    x = rng.normal(size=60).cumsum()
    y, X = _design(x, 6, 6, "ct")
    expected = [_fit(y, X[:, :p])[2] for p in range(3, X.shape[1] + 1)]
    np.testing.assert_allclose(_nested_ssr(y, X, 3), expected, rtol=1e-10)
    # A constant column twice: rank deficient, so the separate fits are used.
    collinear = np.column_stack([X, X[:, 0]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert len(_nested_ssr(y, collinear, 3)) == collinear.shape[1] - 2
