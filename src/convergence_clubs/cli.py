"""Command-line interface: ``convergence-clubs data.csv [options]``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .clustering import find_clubs
from .filters import hp_filter
from .logt import CRITICAL_VALUE


def _fraction(text: str) -> float:
    """Parse ``0.33`` or ``1/3``."""
    num, _, den = text.partition("/")
    return float(num) / float(den) if den else float(num)


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="convergence-clubs",
        description=(
            "Find convergence clubs (Phillips and Sul, 2007, 2009) in a CSV file with "
            "units in rows and time periods in columns."
        ),
    )
    parser.add_argument("csv", type=Path, help="input CSV file (wide format)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    data = parser.add_argument_group("data")
    data.add_argument("--index-col", default="0", help="column with unit names (name or position)")
    data.add_argument("--sep", default=",", help="CSV field separator (default ',')")
    data.add_argument("--log", action="store_true", help="take logs before filtering")
    data.add_argument(
        "--hp-filter",
        nargs="?",
        const=400.0,
        type=float,
        metavar="LAMBDA",
        help="HP-filter every unit (default lambda 400 when given without a value)",
    )

    method = parser.add_argument_group("clustering")
    method.add_argument("--time-trim", type=_fraction, default=1 / 3, help="e.g. 1/3 (default)")
    method.add_argument("--hac", choices=["FQSB", "AQSB"], default="FQSB")
    method.add_argument("--cstar", type=float, default=0.0)
    method.add_argument("--cstar-method", choices=["fixed", "incremental"], default="fixed")
    method.add_argument("--cstar-increment", type=float, default=0.1)
    method.add_argument("--cstar-cap", type=float, default=3.0)
    method.add_argument("--merge", choices=["PS", "vLT"], help="merge clubs with this method")
    method.add_argument(
        "--merge-divergent", action="store_true", help="also assign divergent units to clubs"
    )
    method.add_argument("--estar", type=float, default=CRITICAL_VALUE)

    out = parser.add_argument_group("output")
    out.add_argument("--memberships", type=Path, metavar="CSV", help="write unit -> club")
    out.add_argument("--summary", type=Path, metavar="CSV", help="write the club summary")
    out.add_argument("--plot", type=Path, metavar="FILE", help="save transition paths figure")
    out.add_argument("--theme", choices=["light", "dark"], default="light")
    out.add_argument("--units", choices=["lines", "band", "none"], default="lines")
    out.add_argument("--details", action="store_true", help="print the members of every club")
    return parser


def _read(args: argparse.Namespace) -> pd.DataFrame:
    index_col: int | str = int(args.index_col) if args.index_col.isdigit() else args.index_col
    data = pd.read_csv(args.csv, sep=args.sep, index_col=index_col)
    try:
        data.columns = pd.Index([int(c) for c in data.columns])
    except ValueError:
        pass
    if args.log:
        data = np.log(data)
    if args.hp_filter is not None:
        data = hp_filter(data, lamb=args.hp_filter)
    return data


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        data = _read(args)
        result = find_clubs(
            data,
            time_trim=args.time_trim,
            hac_method=args.hac,
            cstar=args.cstar,
            cstar_method=args.cstar_method,
            cstar_increment=args.cstar_increment,
            cstar_cap=args.cstar_cap,
        )
        if args.merge:
            result = result.merge(
                args.merge, merge_divergent=args.merge_divergent, estar=args.estar
            )
        elif args.merge_divergent:
            result = result.merge_divergent(estar=args.estar)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"convergence-clubs: error: {exc}", file=sys.stderr)
        return 1

    print(
        f"{_plural(result.n_clubs, 'convergence club')}, "
        f"{_plural(result.n_divergent, 'divergent unit')}\n"
    )
    if result.n_clubs:
        print(result.summary().to_string(float_format=lambda v: f"{v:.3f}"))
    if result.n_divergent:
        print(f"\ndivergent: {', '.join(map(str, result.divergent))}")
    if args.details:
        print("\n" + str(result))

    if args.memberships:
        result.memberships().to_csv(args.memberships)
    if args.summary:
        result.summary().to_csv(args.summary)
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        fig = result.plot(theme=args.theme, units=args.units, ncols=4)
        fig.savefig(args.plot, dpi=150)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
