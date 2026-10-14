# Choosing the settings

Which values to use for the HP filter, `time_trim`, `c*`, the HAC method and the merging
rule, and how much each choice matters. The numbers are from the bundled GDP data
(152 countries, 1970–2003); `sensitivity()` and `bootstrap()` give the same information for
your data.

[← Back to the README](../../README.md)

## Summary

| Setting | Default | Usual choice | How much it matters on the GDP data |
|---|---|---|---|
| HP filter λ | — (you filter) | 400 for annual data, as Phillips and Sul (2009) | **a lot**: λ = 100 gives 3 clubs instead of 6 |
| `time_trim` (r) | 1/3 (R), 0.3 (Stata) | 0.3 for T ≤ 50, 0.2 for T ≥ 100 (Phillips and Sul, 2007) | moderate: always 6 clubs after merging, but memberships shift (ARI 0.65–0.66) |
| `hac_method` | `"FQSB"` | `"FQSB"`, as Phillips and Sul's own code | a lot: AQSB gives 4–5 clubs, one of which fails its own test |
| `cstar` | 0 | 0 for short panels; −1.65 is allowed for long ones | moderate: −1.65 gives 6 larger clubs instead of 7 |
| `cstar_method` | `"fixed"` (R), `"incremental"` (Stata) | `"incremental"` if a club fails its own test | removes failing clubs |
| `merge` | — | `"PS"`; `"vLT"` is more conservative | here both merge the same two clubs |
| `convention` | `"R"` | the implementation you compare with | 6 of 8 published Stata results agree under the R rules |

Whatever you choose, report a sensitivity analysis and the bootstrap stability of the
memberships ([Robustness](robustness.md)): clubs are an estimate, and readers should see how
firm they are.

## The HP filter

Phillips and Sul cluster the trend of the log series, so that business cycles do not drive
the clubs; they use λ = 400 for annual data. The choice of λ changes the clubs more than any
other setting:

| λ | Club sizes (merged with PS) | Agreement with λ = 400 (ARI) |
|---|---|---|
| 6.25 (Ravn and Uhlig, 2002) | 60, 49, 29, 12, 2 | 0.61 |
| 100 | 106, 44, 2 | 0.30 |
| **400** | 50, 30, 21, 38, 11, 2 | 1 |
| 1600 | 66, 31, 21, 32, and 2 divergent | 0.55 |

Use the value of the literature you compare with, state it, and show the others in an
appendix: `bootstrap(raw_logs, method="cycle", filter_lambda=400)` resamples the cyclical
component and tells you which memberships depend on it. Do not filter the data you pass to
β-convergence, σ-convergence, distribution dynamics or the pairwise test: they use the levels.

## `time_trim`

The log t regression drops the first r·T periods, where the variation H(t) mostly reflects
initial conditions. Phillips and Sul (2007) recommend r = 0.3 for samples of up to about 50
periods and r = 0.2 for 100 periods or more. The R package uses 1/3 and Stata 0.3.

On the GDP data (T = 34) the number of clubs *before* merging changes with r (7 clubs at 1/3,
9 at 0.3 and 0.25, 10 at 0.2). After merging there are 6 clubs for every value, but not the
same ones: the first two clubs have 50 and 30 countries with r = 1/3 and 55 and 43 with
r = 0.3, and the agreement with r = 1/3 is an ARI of 0.65–0.66. A stable number of clubs does
not mean stable memberships; `runs.uncertain()` lists the countries that move.

## The HAC standard errors

`"FQSB"` (the default) is the Quadratic Spectral kernel with the fixed bandwidth of Phillips
and Sul's GAUSS code; Stata uses the same estimator. `"AQSB"` uses Andrews' data-driven
bandwidth, as `sandwich::vcovHAC` in R. On the GDP data AQSB gives 4–5 clubs instead of 6, and
its club 1 fails its own test (t = −4.98) with the fixed `c*`: a sign that the bandwidth
matters here. If you use AQSB, combine it with `cstar_method="incremental"`, which removes the
failing club (sizes 79, 20, 23, 26, 2 and 2 divergent units).

## `c*` and the incremental method

`c*` is the threshold for adding a unit to a club. Phillips and Sul (2007) use c* = 0, which is
conservative when T is small; with a long panel the 5% critical value −1.65 is also reasonable.
Lower values make larger clubs (on the GDP data: −1.65 gives 57, 38, 27, 17, 11, 2 instead of
50, 30, 21, 24, 14, 11, 2).

With the fixed method a club can be accepted even though it fails the log t test as a whole;
`find_clubs` warns when that happens. `cstar_method="incremental"` raises `c*` for that club
until it converges, as Stata does; `cstar_increment` and `cstar_cap` control the steps.

## Merging

The clustering tends to find too many clubs, so adjacent clubs are tested for convergence
together:
- `merge("PS")`, Phillips and Sul (2009), merges two adjacent clubs when they converge together;
- `merge("vLT")`, von Lyncker and Thoennissen (2017), only merges when the pair also converges
  more strongly than the next pair, so it merges less;
- `iterate=True` repeats the merging until nothing changes, as Stata's `imore` option.

`merge_divergent()` then assigns divergent units to a club they converge with. Rename the clubs
after merging, since merging numbers them again.

## R or Stata

Use the convention of the implementation whose results you compare with or extend:
`convention="Stata"` reproduces `psecta` and `imergeclub`, `convention="R"` the R package. For
a new analysis either is defensible; Stata's rules follow the paper more closely for the core
group and raise `c*` automatically. See [R and Stata conventions](clubs.md#r-and-stata-conventions).

## Checking the choices

```python
runs = cc.sensitivity(
    data, time_trim=[1/3, 0.3, 0.2], hac_method=["FQSB", "AQSB"], merge=["PS"], n_jobs=-1
)
runs.table()          # clubs, divergent units, failing clubs and agreement for every run
runs.uncertain(0.8)   # units whose club depends on the settings

cc.bootstrap(np.log(gdp), 200, method="cycle", filter_lambda=400, merge="PS", seed=0).uncertain()
```
