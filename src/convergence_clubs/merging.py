"""Club merging (Phillips and Sul, 2009; von Lyncker and Thoennissen, 2017)."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .logt import CRITICAL_VALUE, LogTResult, cross_sectional_variation, log_t_test
from .results import Club, ConvergenceClubs

MergeMethod = Literal["PS", "vLT"]
DivergentMethod = Literal["pairwise", "last_club"]


def _test(
    result: ConvergenceClubs, ids, time_trim: float, hac_method: str | None = None
) -> LogTResult:
    H = cross_sectional_variation(result.values[list(ids)])
    rounding = result.params.get("rounding", "half_even")
    return log_t_test(H, time_trim, hac_method or result.hac_method, rounding)


def _origins(club: Club) -> tuple[str, ...]:
    """Names of the original clubs that ``club`` is made of."""
    return club.merged_from if club.merged_from is not None else (club.name,)


def _merge_pass(
    result: ConvergenceClubs,
    clubs: tuple[Club, ...],
    method: str,
    threshold: float,
    time_trim: float,
) -> tuple[Club, ...]:
    """One pass over adjacent clubs (the R package's mergeClubs, Stata's icheckmerge)."""
    n = len(clubs)
    if n < 2:
        return clubs
    new: list[Club] = []
    i = 0
    while i < n - 1:
        ids = list(clubs[i].ids)
        names = list(_origins(clubs[i]))
        append_last = False
        for k in range(i + 1, n):
            t = _test(result, ids + list(clubs[k].ids), time_trim).t_value
            if t > threshold:
                if method == "vLT" and k < n - 1:
                    # The R package runs this comparison test with the default
                    # FQSB bandwidth whatever hac_method is; kept for parity.
                    nxt = list(clubs[k].ids) + list(clubs[k + 1].ids)
                    if not t > _test(result, nxt, time_trim, "FQSB").t_value:
                        break
                ids += clubs[k].ids
                names += _origins(clubs[k])
            else:
                append_last = k == n - 1
                break
        i = k
        new.append(
            Club(
                f"club{len(new) + 1}",
                tuple(ids),
                _test(result, ids, time_trim),
                merged_from=tuple(names),
            )
        )
        if append_last:
            last = clubs[-1]
            new.append(
                Club(f"club{len(new) + 1}", last.ids, last.model, merged_from=_origins(last))
            )
    return tuple(new)


def merge_clubs(
    result: ConvergenceClubs,
    method: MergeMethod = "PS",
    *,
    threshold: float = CRITICAL_VALUE,
    iterate: bool = False,
    merge_divergent: bool = False,
    divergent_method: DivergentMethod = "pairwise",
    estar: float = CRITICAL_VALUE,
    joint_first: bool = False,
    time_trim: float | None = None,
) -> ConvergenceClubs:
    """Merge adjacent convergence clubs.

    ``method="PS"`` (Phillips and Sul, 2009) grows each club with the
    following ones as long as the merged group passes the log t test.
    ``method="vLT"`` (von Lyncker and Thoennissen, 2017) also requires the
    t-statistic of the merge to beat the one of merging the next two clubs.

    Parameters
    ----------
    result
        Output of :func:`find_clubs`.
    threshold
        Critical value of the log t test used to merge clubs.
    iterate
        Repeat the merging pass until no more clubs merge (the ``imore``
        option of Stata's ``imergeclub``); one pass by default, as in R.
    merge_divergent
        Also try to assign divergent units to the new clubs.
    divergent_method
        ``"pairwise"`` (von Lyncker and Thoennissen, 2017; the R package):
        see :func:`merge_divergent`. ``"last_club"`` (the ``mdiv`` option of
        Stata's ``imergeclub``): all divergent units join the last club if
        they converge together with it.
    estar
        Critical value used when merging divergent units.
    joint_first
        With ``divergent_method="pairwise"``, first test the divergent units
        all together (see :func:`merge_divergent`).
    time_trim
        Overrides the ``time_trim`` used to find ``result``.
    """
    if method not in ("PS", "vLT"):
        raise ValueError("method must be 'PS' or 'vLT'")
    if divergent_method not in ("pairwise", "last_club"):
        raise ValueError("divergent_method must be 'pairwise' or 'last_club'")
    time_trim = result.time_trim if time_trim is None else time_trim
    if len(result.clubs) < 2 and not merge_divergent:
        return result

    clubs, divergent = result.clubs, result.divergent_ids
    while True:
        merged_clubs = _merge_pass(result, clubs, method, threshold, time_trim)
        if merge_divergent and divergent_method == "last_club" and divergent and merged_clubs:
            # Stata applies mdiv within every pass.
            last = merged_clubs[-1]
            model = _test(result, last.ids + divergent, time_trim)
            if model.t_value > estar:
                joined = Club(
                    last.name,
                    last.ids + divergent,
                    model,
                    merged_from=_origins(last) + ("divergent",),
                    notes=last.notes + ("joined by all divergent units",),
                )
                merged_clubs, divergent = (*merged_clubs[:-1], joined), ()
        if not iterate or len(merged_clubs) == len(clubs):
            break
        clubs = merged_clubs
    merged = result._replace(clubs=merged_clubs, divergent_ids=divergent, time_trim=time_trim)

    if merge_divergent and divergent_method == "pairwise":
        return _merge_divergent(merged, time_trim, estar, joint_first)
    return merged


def merge_divergent(
    result: ConvergenceClubs,
    *,
    estar: float = CRITICAL_VALUE,
    joint_first: bool = False,
    time_trim: float | None = None,
) -> ConvergenceClubs:
    """Assign divergent units to clubs (von Lyncker and Thoennissen, 2017).

    Repeatedly tests every (club, divergent unit) pair and adds the unit with
    the highest t-statistic to its club, as long as that t-statistic is above
    ``estar``.

    With ``joint_first=True`` the divergent units are first tested all
    together and, if they converge (t > ``estar``), form a new club. This is
    step 1 of the procedure as described by von Lyncker and Thoennissen
    (2017); the R package skips it, so it is off by default.
    """
    time_trim = result.time_trim if time_trim is None else time_trim
    return _merge_divergent(result, time_trim, estar, joint_first)


def _merge_divergent(
    result: ConvergenceClubs, time_trim: float, estar: float, joint_first: bool = False
) -> ConvergenceClubs:
    clubs = list(result.clubs)
    divergent = list(result.divergent_ids)
    if joint_first and len(divergent) > 1:
        model = _test(result, divergent, time_trim)
        if model.t_value > estar:
            club = Club(
                f"club{len(clubs) + 1}",
                tuple(divergent),
                model,
                merged_from=("divergent",) if result.is_merged else None,
                notes=("formed by the divergent units, which converge together",),
            )
            return result._replace(clubs=(*clubs, club), divergent_ids=(), time_trim=time_trim)
    if not clubs:
        return result

    while divergent:
        t = np.array(
            [
                [_test(result, club.ids + (d,), time_trim).t_value for d in divergent]
                for club in clubs
            ]
        )
        if not t.max() > estar:
            break
        # First maximum in column-major order, as R's which(..., arr.ind = TRUE).
        flat = int(np.argmax(t.ravel(order="F")))
        ci, di = np.unravel_index(flat, t.shape, order="F")
        club, unit = clubs[ci], divergent.pop(di)
        ids = club.ids + (unit,)
        note = f"merged with divergent unit {result.data.index[unit]!s} (id {unit})"
        clubs[ci] = Club(
            club.name,
            ids,
            _test(result, ids, time_trim),
            cstar=club.cstar,
            merged_from=club.merged_from,
            notes=club.notes + (note,),
        )

    return result._replace(clubs=tuple(clubs), divergent_ids=tuple(divergent), time_trim=time_trim)
