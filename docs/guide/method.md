# The method

What `find_clubs` and `.merge()` compute, step by step: the log t test of Phillips and Sul
(2007), the clustering algorithm that uses it, and the merging of clubs (Phillips and Sul,
2009; von Lyncker and Thoennissen, 2017).

[← Back to the README](../README.md)

## Relative transition paths

For a panel $X_{it}$ of $N$ units over $T$ periods (usually the logs of a series, filtered
to its trend), the relative transition path of unit $i$ is its value relative to the
cross-section mean:

$$h_{it} = \frac{X_{it}}{\frac1N\sum_{i}X_{it}}$$

If the units converge, every $h_{it}$ goes to 1 and the cross-sectional variation goes to 0:

$$H_t = \frac1N\sum_i (h_{it}-1)^2$$

`relative_transition` and `cross_sectional_variation` compute these two.

## The log t test

Under the null of convergence, $H_t$ decays like $H_1 / (L(t)^2\, t^{2\alpha})$, where
$L(t)=\log t$ corrects for slow variation. The log t regression turns this into a line:

$$\log\frac{H_1}{H_t} - 2\log\log t = a + b\log t + u_t,\qquad t=\lfloor rT\rfloor,\dots,T$$

The first $rT$ periods are dropped (`time_trim`, $r$ = 1/3 by default). Convergence means
$\alpha\ge 0$, which corresponds to $b = 2\alpha \ge 0$. The test is one-sided: the null is
rejected when the t-statistic of $\hat b$ is below the critical value
$-1.65$ (`CRITICAL_VALUE`). The standard error of $\hat b$ is heteroskedasticity and
autocorrelation consistent (HAC), with a Quadratic Spectral kernel and either a fixed
(`"FQSB"`) or an Andrews adaptive (`"AQSB"`) bandwidth. `log_t_test` runs it and returns
`beta`, `std_err`, `t_value` and `p_value`.

A test that does not reject does not prove convergence: it says the data are compatible
with it. Phillips and Sul use it as the building block of the algorithm below.

## Clustering

`find_clubs` applies the test repeatedly:

1. **Sort** the units by their last observation (`ref_col`; `sort_fraction` averages the last periods instead), from highest to lowest.
2. **Core group.** Take the first two units and increase the group one unit at a time,
   running the test on each group until it rejects. The core is the group with the largest
   t-statistic among the sizes tried. (R takes the largest absolute value and
   resolves ties differently: see `convention=`.) If no pair of adjacent units passes, there
   is no club and the remaining units are divergent.
3. **Sieve.** Add each of the other units to the core one at a time, and keep those whose
   t-statistic is above $c^*$ (`cstar`, 0 by default). With
   `cstar_method="incremental"`, $c^*$ is raised in steps until the club passes its own
   test, up to `cstar_cap`.
4. **Recurse.** Remove the club and repeat steps 2–3 on the units that are left. The units
   that never enter a club are the divergent ones.

## Merging

The algorithm often splits a club into neighbouring pieces. `.merge("PS")` joins adjacent
clubs when the test on their union does not reject at `threshold` (−1.65 by default). With
`.merge("vLT")` a club is joined to the next one only if that union scores a higher t-statistic than the
next club joined to the one after it, which is more conservative. `.merge_divergent()` then assigns divergent units to
the clubs they fit.

The result of each choice on real data is in [Choosing the settings](choosing-settings.md);
the correspondence with R and Stata, and where this port differs, is in
[Validation](validation.md).
