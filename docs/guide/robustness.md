# Robustness: sensitivity and bootstrap

How stable are the clubs when the settings or the data change?

[← Back to the README](../../README.md)

## Sensitivity analysis

Club memberships depend on the trimming fraction, the HAC estimator, `c*`, merging and how
units are ordered. On the GDP data, for example, FQSB gives 7 clubs and AQSB gives 5 clubs
plus 3 divergent units. `sensitivity()` reruns the procedure over a grid of settings. The
first value of each setting forms the baseline run.

```python
runs = cc.sensitivity(
    data,
    time_trim=[1/3, 0.25, 0.3, 0.4],   # defaults
    hac_method=["FQSB", "AQSB"],       # defaults
    cstar=[0, -1],
    merge=[None, "PS"],
)
runs.table()          # one row per run: settings, clubs, sizes, agreement with the baseline (ARI)
runs.stability()      # per unit: share of runs spent with its baseline club mates
runs.co_membership()  # units x units: share of runs in the same club
runs.plot()           # heatmap below

cc.compare_clubs(a, b)          # crosstab of two results' memberships
cc.adjusted_rand_index(a, b)    # 1 = identical memberships, 0 = chance agreement
```

In the heatmap, a club that stays together in every run shows as a solid dark block. Lighter
stripes are units that change clubs between runs.

![Co-membership across 32 runs](../co_membership.png)

## Bootstrap confidence in club membership

`sensitivity()` varies the *settings*. `bootstrap()` perturbs the *data*, and tells you how
confident you can be that a unit belongs to its club. There are three resampling schemes:

| `method=` | What it varies | Question it answers |
|---|---|---|
| `"units"` (default) | Cluster random 80% subsets of the units | Does a unit's club depend on which other units are in the sample? |
| `"cycle"` | Block-resample the HP cyclical component and filter again (needs the *unfiltered* data and `filter_lambda=`) | Is the club robust to business-cycle noise? |
| `"endpoint"` | Drop the last 1…`max_drop` periods | Does the club hinge on the last years, where the HP filter is least reliable? |

Years are never resampled independently: the log t regression depends on the time index, so
shuffling periods would be meaningless. The cycle bootstrap uses the same blocks for every
unit, so shocks that hit all units at once stay common.

```python
runs = cc.bootstrap(data, 200, method="units", seed=0)   # ~3 s on the GDP data
runs.stability()          # each unit's confidence in its full-sample club, and runs included
runs.uncertain(0.8)       # units below 80% confidence, least confident first
runs.plot_stability()     # figure below
runs.plot()               # co-membership heatmap, as for sensitivity()

raw = np.log(cc.load_gdp())
cc.bootstrap(raw, 200, method="cycle", filter_lambda=400, merge="PS", seed=0)
```

![Membership confidence across 200 bootstrap resamples](../stability.png)

On the GDP data, Somalia, Burkina Faso and Vanuatu are among the 8 least confident units
under all three schemes. Club 1 is solid when units are resampled (every member ≥ 0.94) and
fairly solid under cycle resampling (≥ 0.80). Dropping the last years, however, moves some of
its members (down to 0.45), so the end of the sample matters for the top club.
