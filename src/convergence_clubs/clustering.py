"""Phillips and Sul (2007, 2009) club clustering algorithm."""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd

from .logt import (
    CRITICAL_VALUE,
    HAC_METHODS,
    HACMethod,
    LogTResult,
    Rounding,
    cross_sectional_variation,
    log_t_test,
    trimmed_periods,
)
from .results import Club, ConvergenceClubs

CStarMethod = Literal["fixed", "incremental"]
Convention = Literal["R", "Stata"]
CoreRule = Literal["max_abs_t", "max_t"]


@dataclass(frozen=True)
class _Rules:
    """Implementation details that differ between the R and Stata packages."""

    rounding: Rounding | str = "half_even"
    core_rule: CoreRule | str = "max_abs_t"
    #: R keeps raising c* while c* <= cap, Stata while c* < cap.
    cap_inclusive: bool = True


R_RULES = _Rules()


class _Defaults(NamedTuple):
    time_trim: float
    cstar_method: str
    cstar_increment: float
    cstar_cap: float
    rules: _Rules


CONVENTIONS: dict[str, _Defaults] = {
    # R package ConvergenceClubs 2.2.5
    "R": _Defaults(1 / 3, "fixed", 0.1, 3.0, _Rules("half_even", "max_abs_t", True)),
    # Stata package psecta (Du, 2018): kq(0.3) cr(0) incr(0.05) maxcr(50)
    "Stata": _Defaults(0.3, "incremental", 0.05, 50.0, _Rules("half_up", "max_t", False)),
}


def _test(
    X: np.ndarray, ids, time_trim: float, hac_method: str, rules: _Rules = R_RULES
) -> LogTResult:
    H = cross_sectional_variation(X[np.asarray(ids)])
    return log_t_test(H, time_trim, hac_method, rules.rounding)


def _core_group(
    X: np.ndarray,
    rows: np.ndarray,
    time_trim: float,
    hac_method: str,
    threshold: float = CRITICAL_VALUE,
    rules: _Rules = R_RULES,
) -> list[int] | None:
    """Step 2: find the core group among ``rows`` (sorted, decreasing).

    Returns positions into ``rows``, or ``None`` when no pair of adjacent units
    passes the test. Scanning ``k`` while ``t_k > threshold``, the core is the
    group of the first ``k`` units with the largest ``t_k`` (as in Phillips
    and Sul, 2007, and Stata; ties go to the larger group) or with the
    largest ``|t_k|`` (as in the R package; ties go to the smaller group).
    """
    n = rows.size
    for i in range(1, n):
        t = _test(X, rows[[i - 1, i]], time_trim, hac_method, rules).t_value
        if t > threshold:
            break
    else:
        return None

    units = [i - 1, i]
    groups, t_values = [list(units)], [t]
    for k in range(i + 1, n):
        units.append(k)
        t = _test(X, rows[units], time_trim, hac_method, rules).t_value
        if not t > threshold:
            break
        groups.append(list(units))
        t_values.append(t)
    if rules.core_rule == "max_t":
        best = max(range(len(t_values)), key=lambda j: (t_values[j], j))
    else:
        best = int(np.argmax(np.abs(t_values)))
    return groups[best]


def _sieve(
    X: np.ndarray,
    rows: np.ndarray,
    core: list[int],
    time_trim: float,
    hac_method: str,
    cstar: float,
    cstar_method: str,
    cstar_increment: float,
    cstar_cap: float,
    rules: _Rules = R_RULES,
) -> tuple[list[int], LogTResult, float]:
    """Step 3: add to the core every unit whose t-statistic with the core exceeds ``cstar``.

    With ``cstar_method="incremental"``, ``cstar`` is raised by
    ``cstar_increment`` until the whole club passes the test; once it exceeds
    ``cstar_cap`` the club is reduced to the core group.
    """
    if cstar_increment <= 0:
        cstar_method = "fixed"
    core_set = set(core)
    candidates = [r for r in range(rows.size) if r not in core_set]
    t_values = np.array(
        [_test(X, rows[core + [r]], time_trim, hac_method, rules).t_value for r in candidates]
    )

    while True:
        members = core + [r for r, t in zip(candidates, t_values, strict=True) if t > cstar]
        model = _test(X, rows[members], time_trim, hac_method, rules)
        if model.converges() or cstar_method == "fixed":
            return members, model, cstar
        if cstar <= cstar_cap if rules.cap_inclusive else cstar < cstar_cap:
            cstar += cstar_increment
        else:
            # The R package reports the stale core+candidates model here; we
            # report the model of the club actually returned.
            return list(core), _test(X, rows[core], time_trim, hac_method, rules), cstar


def find_clubs(
    data: pd.DataFrame,
    *,
    time_trim: float | None = None,
    hac_method: HACMethod | str = "FQSB",
    cstar: float = 0.0,
    cstar_method: CStarMethod | str | None = None,
    cstar_increment: float | None = None,
    cstar_cap: float | None = None,
    data_cols: Sequence[Hashable] | None = None,
    ref_col: Hashable | None = None,
    sort_fraction: float = 0.0,
    convention: Convention | str = "R",
    core_rule: CoreRule | str | None = None,
) -> ConvergenceClubs:
    """Find convergence clubs with the Phillips and Sul clustering algorithm.

    Parameters
    ----------
    data
        Units in rows (the index holds unit names), time periods in columns.
        Values should usually be filtered (e.g. :func:`hp_filter` of log GDP).
    time_trim
        Share of initial periods discarded by the log t regression
        (default 1/3 with ``convention="R"``, 0.3 with ``"Stata"``).
    hac_method
        ``"FQSB"`` (fixed QS bandwidth) or ``"AQSB"`` (adaptive QS bandwidth).
    cstar
        Sieve threshold ``c*`` used to add units to the core group.
    cstar_method
        ``"fixed"`` or ``"incremental"``: in the latter case ``cstar`` is
        increased by ``cstar_increment`` until the club converges, up to
        ``cstar_cap`` (defaults: fixed, 0.1, 3 for R; incremental, 0.05, 50
        for Stata, which is the procedure of Phillips and Sul, 2007).
    data_cols
        Columns of ``data`` holding the time series; all columns by default.
    ref_col
        Column used to sort units before clustering; defaults to the last
        data column (the last period).
    sort_fraction
        Sort units by their average over the last periods instead of a single
        column, exactly as the ``fr()`` option of the Stata command ``psecta``:
        the last ``T - int((1 - sort_fraction) T)`` periods are averaged.
        ``0`` sorts by ``ref_col``. Averaging makes the ordering less
        sensitive to the last observation.
    convention
        ``"R"`` reproduces the R package ConvergenceClubs (the default),
        ``"Stata"`` the Stata package ``psecta`` (Du, 2018). They differ in
        the core group rule, the default ``c*`` procedure and trimming, and
        in how ``[rT]`` is rounded; explicit arguments override the defaults.
    core_rule
        ``"max_t"`` (Phillips and Sul, 2007; Stata) or ``"max_abs_t"`` (R):
        how the core group size is chosen among the candidate groups.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame with units in rows and periods in columns")
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {sorted(CONVENTIONS)}")
    defaults = CONVENTIONS[convention]
    rules = defaults.rules
    time_trim = defaults.time_trim if time_trim is None else time_trim
    cstar_method = defaults.cstar_method if cstar_method is None else cstar_method
    cstar_increment = defaults.cstar_increment if cstar_increment is None else cstar_increment
    cstar_cap = defaults.cstar_cap if cstar_cap is None else cstar_cap
    if core_rule is not None:
        if core_rule not in ("max_t", "max_abs_t"):
            raise ValueError("core_rule must be 'max_t' or 'max_abs_t'")
        rules = _Rules(rules.rounding, core_rule, rules.cap_inclusive)
    if not data.index.is_unique:
        dupes = data.index[data.index.duplicated()].unique().tolist()
        raise ValueError(f"unit names (the index) must be unique; duplicated: {dupes[:5]}")
    data_cols = list(data.columns if data_cols is None else data_cols)
    if sort_fraction and ref_col is not None:
        raise ValueError("give either ref_col or sort_fraction, not both")
    if not 0 <= sort_fraction <= 1:
        raise ValueError("sort_fraction must be in [0, 1]")
    ref_col = data_cols[-1] if ref_col is None else ref_col
    if ref_col not in data.columns:
        raise KeyError(f"ref_col {ref_col!r} is not a column of data")

    values = data[data_cols]
    if not all(pd.api.types.is_numeric_dtype(dt) for dt in values.dtypes):
        raise TypeError("all data columns must be numeric")
    missing = values.isna().any(axis=1)
    if missing.any():
        units = missing.index[missing].tolist()
        raise ValueError(
            f"data must not contain missing values; {len(units)} units have some, "
            f"e.g. {units[:5]}; see describe_missing(), fill_gaps() and balance()"
        )
    _validate_time_trim(len(data_cols), time_trim, rules.rounding)
    if hac_method not in HAC_METHODS:
        raise ValueError(f"hac_method must be one of {HAC_METHODS}, got {hac_method!r}")
    if cstar_method not in ("fixed", "incremental"):
        raise ValueError("cstar_method must be 'fixed' or 'incremental'")
    if cstar_method == "incremental" and (cstar_increment < 0 or cstar_cap < 0):
        raise ValueError("cstar_increment and cstar_cap must be non-negative")

    cstar = float(cstar)
    X = values.to_numpy(dtype=float)
    if sort_fraction:
        n_periods = X.shape[1]
        n_last = max(1, n_periods - int((1 - sort_fraction) * n_periods))  # as Stata's fr()
        key = X[:, -n_last:].mean(axis=1)
    else:
        key = data[ref_col].to_numpy(dtype=float)
    order = np.argsort(-key, kind="stable")

    clubs: list[Club] = []
    divergent: tuple[int, ...] = ()
    remaining = order
    while remaining.size:
        if remaining.size == 1:
            divergent = tuple(remaining.tolist())
            break

        name = f"club{len(clubs) + 1}"
        model_all = _test(X, remaining, time_trim, hac_method, rules)
        if model_all.converges():
            clubs.append(Club(name, tuple(remaining.tolist()), model_all, cstar=cstar))
            break

        core = _core_group(X, remaining, time_trim, hac_method, rules=rules)
        if core is None:
            divergent = tuple(remaining.tolist())
            break

        members, model, club_cstar = _sieve(
            X,
            remaining,
            core,
            time_trim,
            hac_method,
            cstar,
            cstar_method,
            cstar_increment,
            cstar_cap,
            rules,
        )
        clubs.append(Club(name, tuple(remaining[members].tolist()), model, cstar=club_cstar))
        remaining = np.delete(remaining, members)

    result = ConvergenceClubs(
        clubs=tuple(clubs),
        divergent_ids=divergent,
        data=values,
        time_trim=time_trim,
        hac_method=hac_method,
        params={
            "convention": convention,
            "core_rule": rules.core_rule,
            "rounding": rules.rounding,
            "sort_fraction": sort_fraction,
            "cstar": cstar,
            "cstar_method": cstar_method,
            "cstar_increment": cstar_increment,
            "cstar_cap": cstar_cap,
            "ref_col": ref_col,
        },
    )
    if result.failing_clubs:
        names = result.failing_clubs
        verb = "does" if len(names) == 1 else "do"
        warnings.warn(
            f"{', '.join(names)} {verb} not pass the log t test as a whole "
            f"(t <= {CRITICAL_VALUE}); consider cstar_method='incremental' or a higher cstar. "
            "See ConvergenceClubs.failing_clubs.",
            ClubConvergenceWarning,
            stacklevel=2,
        )
    return result


class ClubConvergenceWarning(UserWarning):
    """A club found by :func:`find_clubs` fails its own log t test."""


def _validate_time_trim(
    n_periods: int, time_trim: float, rounding: Rounding | str = "half_even"
) -> None:
    if n_periods < 2:
        raise ValueError("at least two time periods are needed")
    if not 0 < time_trim <= 1:
        raise ValueError("time_trim must be in (0, 1]")
    if n_periods - trimmed_periods(n_periods, time_trim, rounding) < 2:
        raise ValueError("too few time periods for this value of time_trim")
