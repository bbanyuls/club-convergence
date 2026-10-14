"""Publication tables in LaTeX (booktabs), without extra dependencies.

pandas' own ``to_latex`` needs jinja2; these tables need only the
``booktabs`` package in the document (and ``array`` for wrapped columns,
``longtable`` for tables that span pages).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .results import ConvergenceClubs

_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape(text: Any) -> str:
    """``text`` with LaTeX's special characters escaped."""
    return "".join(_SPECIAL.get(char, char) for char in str(text))


def _cell(value: Any, digits: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"$-${-value}" if value < 0 else str(value)
    if isinstance(value, (float, np.floating)):
        if math.isinf(value):
            return r"$-\infty$" if value < 0 else r"$\infty$"
        text = f"{abs(value):.{digits}f}"
        # A text-mode hyphen is too short for a minus sign.
        return f"$-${text}" if value < 0 and float(text) != 0 else text
    return escape(value)


def to_latex(
    table: pd.DataFrame,
    *,
    caption: str | None = None,
    label: str | None = None,
    digits: int | Mapping[str, int] = 3,
    index: bool = True,
    header: Mapping[Any, str] | None = None,
    wrap: Mapping[Any, str] | None = None,
    notes: str | None = None,
    longtable: bool = False,
    path: str | Path | None = None,
) -> str:
    """A DataFrame as a booktabs LaTeX table.

    Parameters
    ----------
    table
        Any table, e.g. ``result.summary()``.
    caption, label
        ``\\caption`` and ``\\label`` (the caption is LaTeX, not escaped).
    digits
        Decimals of float columns: one number, or a mapping column -> decimals.
    header
        LaTeX for some column headers, e.g. ``{"beta": r"$\\hat\\beta$"}``
        (not escaped); other headers are escaped.
    wrap
        Width of columns whose text wraps, e.g. ``{"members": r"0.55\\textwidth"}``.
    notes
        A note under the table (LaTeX, not escaped).
    longtable
        Use ``longtable``, which can break across pages, instead of
        ``table`` + ``tabular``.
    path
        Also write the table to this file.

    Cells are escaped, negative numbers get a proper minus sign, booleans
    become yes/no and missing values stay empty.
    """
    header = dict(header or {})
    wrap = dict(wrap or {})
    columns = list(table.columns)
    names = ([table.index.name] if index else []) + columns
    heads = [header.get(name, escape(name) if name is not None else "") for name in names]

    def decimals(column: Any) -> int:
        return digits.get(column, 3) if isinstance(digits, Mapping) else digits

    def spec(column: Any) -> str:
        if column in wrap:
            return rf">{{\raggedright\arraybackslash}}p{{{wrap[column]}}}"
        numeric = pd.api.types.is_numeric_dtype(table[column]) and not pd.api.types.is_bool_dtype(
            table[column]
        )
        return "r" if numeric else "l"

    column_format = ("l" if index else "") + "".join(spec(c) for c in columns)
    rows = []
    # Cell by cell: iterrows() would turn the integers of a mixed row into floats.
    for i, key in enumerate(table.index):
        cells = [escape(key)] if index else []
        cells += [_cell(table[c].iat[i], decimals(c)) for c in columns]
        rows.append(" & ".join(cells) + r" \\")
    if wrap:  # rows of wrapped text read better with some air between them
        rows = [rows[0], *(r"\addlinespace " + row for row in rows[1:])] if rows else rows

    packages = ["booktabs"] + (["array"] if wrap else []) + (["longtable"] if longtable else [])
    lines = [f"% Needs \\usepackage{{{','.join(packages)}}}"]
    head = [r"\toprule", " & ".join(heads) + r" \\", r"\midrule"]
    # The note is a paragraph after the table: as a full-width row it would
    # stretch the last column to the width of the page.
    note = [r"\par\smallskip{\footnotesize\raggedright " + notes + r"\par}"] if notes else []
    if longtable:
        lines.append(rf"\begin{{longtable}}{{{column_format}}}")
        if caption:
            lines.append(
                rf"\caption{{{caption}}}" + (rf"\label{{{label}}}" if label else "") + r" \\"
            )
        lines += head + [r"\endfirsthead"] + head + [r"\endhead"]
        lines += rows + [r"\bottomrule", r"\end{longtable}"] + note
    else:
        lines += [r"\begin{table}[htbp]", r"\centering"]
        if caption:
            lines.append(rf"\caption{{{caption}}}")
        if label:
            lines.append(rf"\label{{{label}}}")
        lines += [rf"\begin{{tabular}}{{{column_format}}}"] + head + rows + [r"\bottomrule"]
        lines += [r"\end{tabular}"] + note + [r"\end{table}"]
    text = "\n".join(lines) + "\n"
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


def clubs_table(
    result: ConvergenceClubs,
    *,
    members: bool = True,
    divergent: bool = True,
    caption: str | None = "auto",
    label: str | None = None,
    digits: int = 3,
    longtable: bool | None = None,
    members_width: str = r"0.55\textwidth",
    path: str | Path | None = None,
) -> str:
    """The usual table of convergence clubs: size, log t coefficient, t statistic and members.

    ``longtable`` defaults to True when the members are listed and there are
    more than 60 units, so long lists can break across pages. Divergent
    units get a row of their own. The note states the settings, so the
    table can be reproduced.
    """
    rows = []
    for club in result.clubs:
        row: dict[str, Any] = {
            "n": len(club),
            "beta": club.model.beta,
            "t": club.model.t_value,
        }
        if members:
            row["members"] = "; ".join(map(str, result.units(club.name)))
        rows.append(row)
    names = [c.name for c in result.clubs]
    if divergent and result.n_divergent:
        row = {"n": result.n_divergent, "beta": np.nan, "t": np.nan}
        if members:
            row["members"] = "; ".join(map(str, result.divergent))
        rows.append(row)
        names.append("Divergent")
    table = pd.DataFrame(rows, index=pd.Index(names, name="Club"))
    convention = result.params.get("convention", "R")
    notes = (
        r"Log $t$ test of Phillips and Sul (2007): convergence is rejected at 5\% when "
        rf"$t < -1.65$. {escape(result.hac_method)} standard errors, the first "
        rf"{result.time_trim:.3g} of the periods discarded, {escape(convention)} convention"
        + (", adjacent clubs merged." if result.is_merged else ".")
    )
    if caption == "auto":
        caption = "Convergence clubs" + (" after merging" if result.is_merged else "")
    return to_latex(
        table,
        caption=caption,
        label=label,
        digits={"beta": digits, "t": 2},
        header={
            "Club": "Club",
            "n": "$N$",
            "beta": r"$\hat\beta$",
            "t": "$t$",
            "members": "Members",
        },
        wrap={"members": members_width} if members else None,
        notes=notes,
        longtable=longtable if longtable is not None else members and len(result.data) > 60,
        path=path,
    )


__all__: Sequence[str] = ["clubs_table", "escape", "to_latex"]
