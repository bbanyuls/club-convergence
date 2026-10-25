"""Replicates every number in the manuscript.

Run from the repository root:

    uv run --extra plot python paper/jss/replication.py

It stops with an AssertionError if a number differs from the published one, and
prints the tables the manuscript quotes. It takes a few seconds.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import convergence_clubs as cc

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "mendez2020_stata.csv"


def section(title):
    print(f"\n{'=' * 8} {title}")


def r_journal():
    """Sichera and Pizzuto (2019), The R Journal 11(2): clubs of the GDP panel."""
    section("1. GDP panel: the R package's published example")
    data = cc.load_filtered_gdp()
    model = cc.log_t_test(cc.cross_sectional_variation(data.to_numpy()), time_trim=1 / 3)
    print(model)
    assert (round(model.beta, 3), round(model.std_err, 3), round(model.t_value, 3)) == (
        -0.875,
        0.005,
        -159.555,
    )
    clubs = cc.find_clubs(data)
    print(clubs.summary())
    assert clubs.summary()["n_units"].tolist() == [50, 30, 21, 24, 14, 11, 2]
    merged = clubs.merge("PS")
    print(merged.summary())
    assert len(merged) == 6
    return data


def stata():
    """Mendez (2020): the eight psecta/imergeclub classifications."""
    section("2. Eight published Stata classifications (Mendez, 2020)")
    mendez = pd.read_csv(FIXTURE)
    rows = []
    for sample in ["hiYes", "hiNo"]:
        for variable in ["log_lp", "log_ky", "log_h", "log_tfp"]:
            part = mendez[mendez["sample"] == sample]
            wide = cc.wide_panel(part, unit="country", time="year", value=variable)
            last = part[part["year"] == part["year"].max()].set_index("country").reindex(wide.index)
            clubs = cc.find_clubs(wide, convention="Stata", time_trim=0.333)

            def labels(result):
                return [None if c == "divergent" else int(c[4:]) for c in result.memberships()]

            def stata_col(name):
                return [None if pd.isna(v) else int(v) for v in last[name]]

            same_initial = labels(clubs) == stata_col(f"club_{variable}")
            same_final = labels(clubs.merge("PS")) == stata_col(f"finalclub_{variable}")
            rows.append((sample, variable, len(wide), same_initial, same_final))
    table = pd.DataFrame(rows, columns=["sample", "variable", "units", "clubs", "merged"])
    print(table.to_string(index=False))
    assert table[["clubs", "merged"]].all().all()


def robustness(data):
    section("3. Sensitivity to the settings")
    runs = cc.sensitivity(
        data, time_trim=[1 / 3, 0.25, 0.3, 0.4], hac_method=["FQSB", "AQSB"], merge=[None, "PS"]
    )
    table = runs.table()
    print(table.head(8).to_string())
    print(f"{len(table)} runs; clubs found: {sorted(int(n) for n in table["n_clubs"].unique())}")
    stability = runs.stability()
    print(stability.describe().round(3))

    section("4. Bootstrap confidence in each membership")
    boot = cc.bootstrap(data, 200, merge="PS", seed=0)
    uncertain = boot.uncertain(0.8)
    print(f"{len(uncertain)} of {data.shape[0]} units below 80% confidence")
    print(uncertain.head(10))


def main():
    data = r_journal()
    stata()
    robustness(data)
    section("done: every check passed")


if __name__ == "__main__":
    np.random.seed(0)
    main()
