<img src="assets/icon.svg" width="64" height="64" alt="">

# Club convergence

Club convergence analysis in Python, following Phillips and Sul (2007, 2009).

This is a port of the R package
[ConvergenceClubs](https://github.com/rhobis/ConvergenceClubs) (Sichera and Pizzuto, 2019).
It implements:

- the **log t convergence test**, with HAC standard errors (Quadratic Spectral kernel,
  fixed or Andrews adaptive bandwidth);
- the **club clustering algorithm** (core group formation, sieving with `c*`, fixed or
  incremental);
- **club merging**, using either the Phillips and Sul (2009) rule or the von Lyncker and
  Thoennissen (2017) rule, plus merging of divergent units;
- **HP filtering**, relative transition paths and plots.

Results match the R package to within 1e-8 (see [Validation](#validation)).

## Installation

```bash
pip install "convergence-clubs[plot] @ git+https://github.com/bbanyuls/club-convergence"
```

The core package needs only `numpy` and `pandas`. The `plot` extra adds `matplotlib`.

## Quick start

```python
import numpy as np
import convergence_clubs as cc

gdp = cc.load_gdp()                         # 152 countries x 1970-2003
data = cc.hp_filter(np.log(gdp), lamb=400)  # same as cc.load_filtered_gdp()

# Does the whole panel converge?
H = cc.cross_sectional_variation(data.to_numpy())
cc.log_t_test(H, time_trim=1/3)
# LogTResult(beta=-0.8748..., std_err=0.0054..., t_value=-159.55..., p_value=0.0)

clubs = cc.find_clubs(data, time_trim=1/3, cstar=0, hac_method="FQSB")
clubs.summary()
#        n_units   beta  std_err  t_value  p_value  cstar
# club
# club1       50  0.382    0.041    9.282    1.000    0.0
# club2       30  0.240    0.035    6.904    1.000    0.0
# club3       21  0.110    0.032    3.402    1.000    0.0
# club4       24  0.131    0.064    2.055    0.980    0.0
# club5       14  0.190    0.111    1.701    0.956    0.0
# club6       11  1.003    0.166    6.024    1.000    0.0
# club7        2 -0.470    0.842   -0.559    0.288    0.0

merged = clubs.merge("PS")       # or "vLT"; merge_divergent=True is also available
merged.summary()                 # club4 + club5 -> one club of 38 countries
merged.memberships()             # pandas Series: unit -> club
merged.units("club1")            # names of the units in a club
print(merged)                    # detailed report, like R's print()
merged.plot()                    # transition paths
```

![Transition paths of the merged clubs](docs/transition_paths.png)

### Input data

Pass a `pandas.DataFrame` with **units in rows** (the index holds the unit names) and
**time periods in columns**. Use `data_cols=` to restrict the columns used and `ref_col=`
to choose the column units are sorted by (the last period by default). Data must not
contain missing values. As in Phillips and Sul, you will usually HP-filter the log of the
variable first.

## API correspondence with the R package

| R (`ConvergenceClubs`) | Python (`convergence_clubs`) |
|---|---|
| `computeH(X, "h")` / `computeH(X, "H")` | `relative_transition(X)` / `cross_sectional_variation(X)` |
| `estimateMod(H, time_trim, HACmethod)` | `log_t_test(H, time_trim, hac_method)` |
| `findClubs(X, dataCols, unit_names, refCol, ...)` | `find_clubs(df, data_cols=, ref_col=, ...)` |
| `mergeClubs(clubs, mergeMethod, mergeDivergent, estar)` | `clubs.merge(method, merge_divergent=, estar=)` |
| `mergeDivergent(clubs, estar)` | `clubs.merge_divergent(estar=)` |
| `transition_paths(clubs)` | `clubs.transition_paths()` |
| `summary()`, `print()`, `dim()`, `plot()` | `.summary()`, `print()`, `.n_clubs` / `.n_divergent`, `.plot()` |
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

### Known differences from the R package

- With `cstar_method="incremental"`, if `c*` exceeds `cstar_cap`, the club falls back to
  its core group. R then reports the regression of the core *plus the rejected candidates*.
  This Python port reports the core group's own regression. Memberships are the same.
- If the Andrews bandwidth is zero (residual scores with no autocorrelation), `sandwich`
  raises an error. This port falls back to White's estimator.

Some other quirks are kept on purpose so that results stay identical to R:
- the fixed-bandwidth HAC estimator follows Phillips and Sul's GAUSS code;
- the vLT comparison test always uses the FQSB estimator.

## Development

```bash
uv sync                              # create the environment
uv run pytest                        # run the tests
uv run ruff check . && uv run ruff format --check .
Rscript scripts/generate_r_fixtures.R  # regenerate the R reference (needs R)
```

## References

- Phillips, P. C. B. and Sul, D. (2007). Transition modeling and econometric convergence
  tests. *Econometrica*, 75(6), 1771-1855.
- Phillips, P. C. B. and Sul, D. (2009). Economic transition and growth. *Journal of
  Applied Econometrics*, 24(7), 1153-1185.
- von Lyncker, K. and Thoennissen, R. (2017). Regional club convergence in the EU:
  evidence from a panel data analysis. *Empirical Economics*, 52(2), 525-553.
- Sichera, R. and Pizzuto, P. (2019). ConvergenceClubs: A package for performing the
  Phillips and Sul's club convergence clustering procedure. *The R Journal*, 11(2),
  142-151.
- Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation consistent covariance
  matrix estimation. *Econometrica*, 59(3), 817-858.

## License

GPL-3.0-or-later. This package is a derivative of the R package ConvergenceClubs
(© Roberto Sichera and Pietro Pizzuto, GPL-3), and the bundled GDP datasets come from it.
