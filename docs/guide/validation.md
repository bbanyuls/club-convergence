# Validation against R and Stata

[← Back to the README](../../README.md)

## API correspondence with the R package

| R (`ConvergenceClubs`) | Python (`convergence_clubs`) |
|---|---|
| `computeH(X, "h")` / `computeH(X, "H")` | `relative_transition(X)` / `cross_sectional_variation(X)` |
| `estimateMod(H, time_trim, HACmethod)` | `log_t_test(H, time_trim, hac_method)` |
| `findClubs(X, dataCols, unit_names, refCol, ...)` | `find_clubs(df, data_cols=, ref_col=, ...)` |
| `mergeClubs(clubs, mergeMethod, mergeDivergent, estar)` | `clubs.merge(method, merge_divergent=, estar=)` |
| `mergeDivergent(clubs, estar)` | `clubs.merge_divergent(estar=)` |
| `transition_paths(clubs)` | `clubs.transition_paths()` |
| `summary()`, `print()`, `dim()` | `.summary()`, `print()`, `.n_clubs` / `.n_divergent` |
| `plot(clubs)`, `plot(clubs, clubs=NULL)` | `.plot()`, `.plot(kind="average")` |
| `data(GDP)`, `data(filteredGDP)` | `load_gdp()`, `load_filtered_gdp()` |
| `mFilter::hpfilter(x, 400, type="lambda")` | `hp_filter(x, lamb=400)` |

## Validation

`tests/fixtures/r_reference.json` is produced by running the R package
(`Rscript scripts/generate_r_fixtures.R`). The test suite checks that Python reproduces it:

- the log t test on several subsets, for both HAC methods and three trimming values;
- `findClubs` with 6 configurations (FQSB/AQSB, trimming, `c*`, incremental `c*`) on the
  GDP data and on three simulated panels. These include cases with divergent units and
  with `c*` reaching its cap.
- `mergeClubs` with PS/vLT, with and without `mergeDivergent`, for every one of those runs;
- the HP filter (`mFilter::hpfilter`), and the results published in the R Journal paper.

Club memberships match exactly, and the statistics match to a relative tolerance of 1e-8.
The reference file records the R and package versions that produced it. A weekly CI job
(`.github/workflows/r-parity.yml`) regenerates it with the current CRAN release and reruns
these tests, so changes upstream are caught.

**Stata.** `tests/fixtures/mendez2020_stata.csv` holds the club memberships found with Stata's
`psecta` and `imergeclub` in the replication files of Mendez (2020). They cover 4 variables
(labor productivity, capital-output ratio, human capital, TFP), each in 2 samples (26
high-income and 82 other countries), for 1990–2014. With `convention="Stata"` all 8 initial
and all 8 merged classifications are reproduced exactly. The fixture is rebuilt with
`scripts/build_mendez_fixture.R`. The Stata rules above come from the source of `psecta` 2.0
(Stata Journal, st0503_1).

### Known differences from the R package

- With `cstar_method="incremental"`, if `c*` exceeds `cstar_cap`, the club falls back to
  its core group. R then reports the regression of the core *plus the rejected candidates*.
  This Python port reports the core group's own regression. Memberships are the same.
- `merge("vLT")` runs its second test (the club against the next one) with the clubs' HAC
  method. R 2.2.5 omits `HACmethod` there, so it always uses FQSB. On the GDP data the
  merges are the same either way
  ([upstream report](https://github.com/rhobis/ConvergenceClubs/issues/6)).
- If the Andrews bandwidth is zero (residual scores with no autocorrelation), `sandwich`
  raises an error. This port falls back to White's estimator.

- `merge_divergent(joint_first=True)` adds the first step of the von Lyncker and
  Thoennissen (2017) procedure: test all divergent units together, and make them a new club
  if they converge. The R package skips this step, so it is off by default.

Another quirk is kept on purpose so that results stay identical to R: the fixed-bandwidth
HAC estimator follows Phillips and Sul's GAUSS code.
