"""Command-line interface: ``convergence-clubs data.csv [options]``."""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .beta import beta_convergence
from .bootstrap import bootstrap
from .clustering import ClubConvergenceWarning, find_clubs
from .dynamics import distribution_dynamics
from .filters import hp_filter
from .logt import CRITICAL_VALUE
from .panel import balance, fill_gaps, wide_panel
from .sensitivity import sensitivity
from .sigma import sigma_convergence


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
    data.add_argument(
        "--long",
        nargs=3,
        metavar=("UNIT", "TIME", "VALUE"),
        help="the CSV is in long format (one row per unit and period) with these columns",
    )
    data.add_argument("--log", action="store_true", help="take logs before filtering")
    data.add_argument(
        "--fill-gaps",
        type=int,
        metavar="N",
        help="interpolate interior gaps of up to N periods (in logs with --log)",
    )
    data.add_argument(
        "--drop-incomplete", action="store_true", help="drop units that still have missing values"
    )
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
    method.add_argument(
        "--joint-divergent",
        action="store_true",
        help="with --merge-divergent, first test all divergent units together (vLT step 1)",
    )
    method.add_argument(
        "--sort-fraction",
        type=_fraction,
        default=0.0,
        help="sort units by their mean over this last share of periods (default: last period)",
    )

    out = parser.add_argument_group("output")
    out.add_argument("--memberships", type=Path, metavar="CSV", help="write unit -> club")
    out.add_argument("--summary", type=Path, metavar="CSV", help="write the club summary")
    out.add_argument("--plot", type=Path, metavar="FILE", help="save transition paths figure")
    out.add_argument("--theme", choices=["light", "dark"], default="light")
    out.add_argument("--units", choices=["lines", "band", "none"], default="lines")
    out.add_argument(
        "--html", type=Path, metavar="FILE", help="save interactive transition paths (plotly)"
    )
    out.add_argument("--map", type=Path, metavar="FILE", help="save a map of the clubs")
    out.add_argument(
        "--geodata", type=Path, metavar="PATH", help="region boundaries for --map (any GIS file)"
    )
    out.add_argument(
        "--geo-key", metavar="COLUMN", help="column of --geodata holding the unit names"
    )
    out.add_argument("--details", action="store_true", help="print the members of every club")
    out.add_argument(
        "--members", action="store_true", help="add the members of each club to --summary"
    )

    robust = parser.add_argument_group("robustness")
    robust.add_argument(
        "--sensitivity",
        action="store_true",
        help="rerun with time_trim 1/3, 0.25, 0.3, 0.4 x FQSB/AQSB and report the agreement",
    )
    robust.add_argument(
        "--sensitivity-plot", type=Path, metavar="FILE", help="save the co-membership heatmap"
    )
    robust.add_argument(
        "--bootstrap", type=int, metavar="N", help="confidence in memberships from N resamples"
    )
    robust.add_argument(
        "--bootstrap-method",
        choices=["units", "cycle", "endpoint"],
        default="units",
        help="units: subsample units; cycle: block-resample the HP cycle (needs --hp-filter); "
        "endpoint: drop the last periods",
    )
    robust.add_argument("--seed", type=int, help="random seed for --bootstrap")

    classic = parser.add_argument_group("classical convergence")
    classic.add_argument(
        "--beta",
        action="store_true",
        help="β-convergence on the unfiltered data: absolute, conditional on clubs, by club",
    )
    classic.add_argument("--beta-plot", type=Path, metavar="FILE", help="save the β scatter")
    classic.add_argument(
        "--sigma",
        action="store_true",
        help="σ-convergence on the unfiltered data: dispersion trend, overall and within clubs",
    )
    classic.add_argument("--sigma-plot", type=Path, metavar="FILE", help="save the σ figure")
    classic.add_argument(
        "--dynamics",
        action="store_true",
        help="distribution dynamics: transition matrix of relative income classes",
    )
    classic.add_argument("--dynamics-classes", type=int, default=5, metavar="K")
    classic.add_argument("--dynamics-lag", type=int, default=5, metavar="L")
    classic.add_argument(
        "--dynamics-plot", type=Path, metavar="FILE", help="save the transition matrix figure"
    )
    return parser


def _read(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The data before HP filtering (but after logs) and the data to cluster."""
    if args.long:
        unit, time, value = args.long
        data = wide_panel(pd.read_csv(args.csv, sep=args.sep), unit=unit, time=time, value=value)
    else:
        index_col = int(args.index_col) if args.index_col.isdigit() else args.index_col
        data = pd.read_csv(args.csv, sep=args.sep, index_col=index_col)
    try:
        data.columns = pd.Index([int(c) for c in data.columns])
    except ValueError:
        pass
    if args.log:
        data = np.log(data)
    if args.fill_gaps:
        data = fill_gaps(data, max_gap=args.fill_gaps)
    if args.drop_incomplete:
        complete = balance(data)
        if len(complete) < len(data):
            dropped = data.index.difference(complete.index)
            print(
                f"convergence-clubs: dropped {_plural(len(dropped), 'incomplete unit')}: "
                f"{'; '.join(map(str, dropped[:10]))}{' ...' if len(dropped) > 10 else ''}",
                file=sys.stderr,
            )
        data = complete
    if args.hp_filter is not None:
        return data, hp_filter(data, lamb=args.hp_filter)
    return data, data


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        raw, data = _read(args)
        options = {
            "cstar_method": args.cstar_method,
            "cstar_increment": args.cstar_increment,
            "cstar_cap": args.cstar_cap,
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ClubConvergenceWarning)
            result = find_clubs(
                data,
                time_trim=args.time_trim,
                hac_method=args.hac,
                cstar=args.cstar,
                sort_fraction=args.sort_fraction,
                **options,
            )
        for w in caught:
            print(f"convergence-clubs: warning: {w.message}", file=sys.stderr)
        divergent_options = {"estar": args.estar, "joint_first": args.joint_divergent}
        if args.merge:
            result = result.merge(
                args.merge, merge_divergent=args.merge_divergent, **divergent_options
            )
        elif args.merge_divergent:
            result = result.merge_divergent(**divergent_options)
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
        print(f"\ndivergent: {'; '.join(map(str, result.divergent))}")
    if args.details:
        print("\n" + str(result))

    if args.memberships:
        result.memberships().to_csv(args.memberships)
    if args.summary:
        result.summary(members=args.members).to_csv(args.summary)
    if args.plot or args.sensitivity_plot:
        import matplotlib

        matplotlib.use("Agg")
    if args.html:
        try:
            result.plot.interactive(theme=args.theme).write_html(args.html)
        except ImportError as exc:
            print(f"convergence-clubs: error: {exc}", file=sys.stderr)
            return 1
    if args.map:
        if args.geodata is None:
            print("convergence-clubs: error: --map needs --geodata", file=sys.stderr)
            return 1
        try:
            import geopandas as gpd

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", UserWarning)
                ax = result.plot.map(
                    gpd.read_file(args.geodata), key=args.geo_key, theme=args.theme
                )
            for w in caught:
                print(f"convergence-clubs: warning: {w.message}", file=sys.stderr)
            ax.figure.savefig(args.map, dpi=150)
        except (ImportError, OSError, KeyError, ValueError) as exc:
            print(f"convergence-clubs: error in map: {exc}", file=sys.stderr)
            return 1
    if args.plot:
        result.plot(theme=args.theme, units=args.units, ncols=4).savefig(args.plot, dpi=150)

    if args.sensitivity or args.sensitivity_plot:
        try:
            _report_sensitivity(args, data, options)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(f"convergence-clubs: error in sensitivity analysis: {exc}", file=sys.stderr)
            return 1
    if args.beta or args.beta_plot:
        try:
            _report_beta(args, raw, result)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(f"convergence-clubs: error in β-convergence: {exc}", file=sys.stderr)
            return 1
    if args.sigma or args.sigma_plot:
        try:
            _report_sigma(args, raw, result)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(f"convergence-clubs: error in σ-convergence: {exc}", file=sys.stderr)
            return 1
    if args.dynamics or args.dynamics_plot:
        try:
            _report_dynamics(args, raw, result)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(f"convergence-clubs: error in distribution dynamics: {exc}", file=sys.stderr)
            return 1
    if args.bootstrap:
        try:
            _report_bootstrap(args, raw, options)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(f"convergence-clubs: error in bootstrap: {exc}", file=sys.stderr)
            return 1
    return 0


def _report_sensitivity(args: argparse.Namespace, data: pd.DataFrame, options: dict) -> None:
    runs = sensitivity(
        data,
        time_trim=_unique([args.time_trim, 1 / 3, 0.25, 0.3, 0.4]),
        hac_method=_unique([args.hac, "FQSB", "AQSB"]),
        cstar=[args.cstar],
        merge=[args.merge],
        sort_fraction=[args.sort_fraction],
        merge_divergent=args.merge_divergent,
        **options,
    )
    table = runs.table().drop(columns=["cstar", "merge", "sort_fraction"])
    print(f"\nsensitivity ({len(runs)} runs; the first is the run above)\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    least = runs.stability().dropna().sort_values("stability").head(10)
    print("\nleast stable units (share of runs with their baseline club mates)\n")
    print(least.to_string(float_format=lambda v: f"{v:.2f}"))
    if args.sensitivity_plot:
        ax = runs.plot(theme=args.theme)
        ax.figure.savefig(args.sensitivity_plot, dpi=150)


def _report_beta(args: argparse.Namespace, raw: pd.DataFrame, result) -> None:
    log = not args.log  # with --log the unfiltered data are already in logs
    absolute = beta_convergence(raw, log=log, clubs=result)
    conditional = beta_convergence(raw, log=log, clubs=result, club_effects=True)
    print(f"\nβ-convergence {absolute.start}-{absolute.end} (HC1 standard errors)\n")
    for name, fit in [("absolute", absolute), ("conditional on clubs", conditional)]:
        life = f"{fit.half_life:.1f}" if fit.converges() else "n/a"
        print(
            f"  {name:<22} beta {fit.beta: .5f}  s.e. {fit.std_err:.5f}  "
            f"t {fit.t_value: .2f}  p {fit.p_value:.3g}  half-life {life}"
        )
    print("\nwithin clubs\n")
    print(absolute.by_club().to_string(float_format=lambda v: f"{v:.4f}"))
    if args.beta_plot:
        import matplotlib

        matplotlib.use("Agg")
        conditional.plot(theme=args.theme).figure.savefig(args.beta_plot, dpi=150)


def _report_sigma(args: argparse.Namespace, raw: pd.DataFrame, result) -> None:
    sigma = sigma_convergence(raw, log=not args.log, clubs=result)
    trend = sigma.trend()[["n_units", "first", "last", "change", "slope", "t_value", "p_value"]]
    print(f"\nσ-convergence (sd of log, Newey-West with {sigma.hac_lags} lags)\n")
    print(trend.to_string(float_format=lambda v: f"{v:.4f}"))
    if args.sigma_plot:
        import matplotlib

        matplotlib.use("Agg")
        sigma.plot(index=True, theme=args.theme).figure.savefig(args.sigma_plot, dpi=150)


def _report_dynamics(args: argparse.Namespace, raw: pd.DataFrame, result) -> None:
    levels = np.exp(raw) if args.log else raw
    dyn = distribution_dynamics(
        levels, n_classes=args.dynamics_classes, lag=args.dynamics_lag, clubs=result
    )
    s = dyn.summary()
    print(
        f"\ndistribution dynamics ({dyn.n_classes} classes of relative income, "
        f"{dyn.lag}-period transitions): Shorrocks mobility {s['shorrocks']:.3f}, "
        f"|λ2| {s['second_eigenvalue']:.3f}, half-life {s['half_life']:.1f}\n"
    )
    print(dyn.matrix.to_string(float_format=lambda v: f"{v:.3f}"))
    print()
    print(dyn.distributions().to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nclubs by class in the last period\n")
    print(dyn.composition().to_string())
    if args.dynamics_plot:
        import matplotlib

        matplotlib.use("Agg")
        dyn.plot(theme=args.theme).savefig(args.dynamics_plot, dpi=150)


def _report_bootstrap(args: argparse.Namespace, raw: pd.DataFrame, options: dict) -> None:
    runs = bootstrap(
        raw,
        args.bootstrap,
        method=args.bootstrap_method,
        filter_lambda=args.hp_filter,
        merge=args.merge,
        merge_divergent=args.merge_divergent,
        seed=args.seed,
        time_trim=args.time_trim,
        hac_method=args.hac,
        cstar=args.cstar,
        sort_fraction=args.sort_fraction,
        **options,
    )
    table = runs.table().iloc[1:]
    print(
        f"\nbootstrap ({args.bootstrap_method}, {len(table)} replicates): mean agreement with "
        f"the full-sample clubs (ARI) {table['ari_vs_baseline'].mean():.3f}"
    )
    uncertain = runs.uncertain(0.8)
    print(f"\n{_plural(len(uncertain), 'unit')} with confidence below 0.8\n")
    if len(uncertain):
        print(uncertain.head(20).to_string(float_format=lambda v: f"{v:.2f}"))


def _unique(values: list) -> list:
    """Drop repeated values (with float tolerance), keeping the first occurrence."""
    out: list = []
    for v in values:
        if not any(v == u or (isinstance(v, float) and abs(v - u) < 1e-9) for u in out):
            out.append(v)
    return out


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
