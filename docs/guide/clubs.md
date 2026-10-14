# Finding clubs: data, options and conventions

How to prepare data, which options change the clubs, and how the R and Stata implementations differ.

[← Back to the README](../../README.md)

Which values to choose for these options, and how much each matters, is in
[Choosing the settings](choosing-settings.md).

## Bundled datasets

| Loader | Data |
|---|---|
| `load_gdp()`, `load_filtered_gdp()` | GDP per capita of 152 countries, 1970–2003 (Phillips and Sul, 2009), from the R package. `country_codes()` gives their ISO3 codes. |
| `load_productivity(variable, sample=)` | Penn World Table 9.0 for 108 countries, 1990–2014 (Mendez, 2020): labor productivity `"lp"`, capital-output ratio `"ky"`, human capital `"h"`, TFP `"tfp"` and GDP per capita `"gdppc"`, in levels. `sample` is `"all"`, `"high_income"` (26 countries) or `"other"` (82). `productivity_info()` gives the ISO3 codes, regions and 1990 income groups. |

Both are published with their club results, so they work as worked examples. Applying `hp_filter` and
`convention="Stata"` to the productivity data reproduces the book's Stata clubs exactly, for all four
variables in both samples:

```python
lp = cc.load_productivity("lp", sample="other")                  # 82 countries, levels
clubs = cc.find_clubs(cc.hp_filter(np.log(lp)), convention="Stata", time_trim=0.333)
[len(c) for c in clubs]                # [31, 20, 8, 15, 5, 2] and 1 divergent, as psecta
[len(c) for c in clubs.merge("PS")]    # [51, 8, 15, 5, 2], as imergeclub
```

## Input data

Pass a `pandas.DataFrame` with **units in rows** (the index holds unique unit names) and
**time periods in columns**. Long data (one row per unit and year) can be converted with
`cc.wide_panel(df, unit="region", time="year", value="gdp_pc")`. As in Phillips and Sul,
you will usually HP-filter the log of the variable first.

**Missing values.** The procedure needs a complete panel, and the error message lists the
units that have gaps. To deal with them:
- `cc.describe_missing(df)` shows the gaps of each unit (leading, trailing, interior);
- `cc.fill_gaps(df, max_gap=2, log=True)` interpolates *interior* gaps of up to 2 periods,
  in logs;
- `cc.balance(df)` drops units that are still incomplete, and
  `cc.balance(df, "trim_periods")` instead keeps the longest run of periods that every unit
  has.

Missing values at the start or end of a series are never filled: that would invent the
transition path the test examines. Filled values also feed the HP filter, so check how they
affect the clubs with `bootstrap()` or `sensitivity()`.

Units are sorted by the last period before clustering. To sort by a different column, use
`ref_col=`. To sort by the average of the last share of periods, use `sort_fraction=0.1`,
exactly as Stata's `psecta, fr()` does: it averages the last T − int((1 − fr)·T) periods.
This makes the ordering less sensitive to a single year.
`data_cols=` restricts which columns are used.

For a paper, `clubs.to_latex()` writes the usual table of clubs as LaTeX (booktabs): the size,
the log t coefficient and t statistic, and the members of each club, with a row for divergent
units and a note that states the settings. It needs no extra Python package; the document needs
`\usepackage{booktabs,array,longtable}`.

```python
merged.to_latex(path="clubs.tex", label="tab:clubs")   # longtable when the lists are long
merged.to_latex(members=False)                          # just N, beta and t
cc.to_latex(beta.by_club(), header={"beta": r"$\hat\beta$"})   # any results table
```

Special characters in unit names are escaped ("St. Kitts \& Nevis"), negative numbers get a
true minus sign, and the tests compile the tables with LaTeX. `summary(members=True)` gives the
same information as a DataFrame, e.g. for `.to_csv()`. From the command line: `--latex clubs.tex`.

To give the clubs meaningful names in every table, plot, chart, map and report, rename them
after merging (merging numbers the clubs again):

```python
named = merged.rename(["High", "Upper middle", "Middle", "Lower middle", "Low", "Bottom"])
named = merged.rename({"club1": "Frontier"})                     # only some clubs
named = merged.rename(lambda club: f"Club {club.name[4:]} (n = {len(club)})")
```

Names must be unique, and `"divergent"` is reserved. `cc.report(..., club_names=[...])` and
`--club-names "High; Middle; Low"` in the command line do the same.

## Clubs that fail their own test

With the default `cstar_method="fixed"` (as in R), the procedure can accept a club even
though the club as a whole fails the log t test (t ≤ −1.65). On the GDP data this happens to
club 1 with AQSB (t = −4.98). `find_clubs` warns when this happens (`ClubConvergenceWarning`),
`summary()` has a `converges` column, and `result.failing_clubs` lists the failing clubs.
Phillips and Sul (2007) suggest raising `c*` in that case, which is what
`cstar_method="incremental"` does. It is also the default of `convention="Stata"`.

## R and Stata conventions

The two established implementations, the R package ConvergenceClubs and the Stata package
`psecta` (Du, 2018), can give different clubs on the same data. Reading both sources shows
why:

| | `convention="R"` (default) | `convention="Stata"` |
|---|---|---|
| Core group | the candidate group with the largest \|t\| | the largest t, as in Phillips and Sul (2007); ties go to the larger group |
| Club that fails its own test | kept (`c*` fixed) | `c*` raised by 0.05 until the club converges, up to 50 |
| Default `time_trim` | 1/3 | 0.3 (`kq(0.3)`) |
| Rounding of [rT] | half to even | half up (35 × 0.3 = 10.5 drops 11 periods, not 10) |
| HAC standard errors | Phillips–Sul GAUSS estimator | the same estimator |

```python
cc.find_clubs(data, convention="Stata", time_trim=0.333)   # as psecta, kq(0.333)
clubs.merge("PS")                                           # as imergeclub
clubs.merge("PS", iterate=True)                             # as imergeclub, imore
clubs.merge("PS", merge_divergent=True, divergent_method="last_club")   # imergeclub, mdiv
```

Explicit arguments override a convention's defaults. For example,
`find_clubs(data, core_rule="max_t")` keeps R's other choices but uses the paper's core rule.
On the replication data of Mendez (2020), the R convention reproduces 6 of the 8 Stata
results. The other two differ because of the core rule and because `c*` had to rise to 4.85,
above R's default cap of 3.
