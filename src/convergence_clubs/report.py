"""One-call HTML report: a replication appendix with every analysis.

``cc.report(data).save("appendix.html")`` writes a single self-contained file
(tables as HTML, figures embedded as PNG in a light and a dark version).
"""

from __future__ import annotations

import base64
import datetime as _dt
import html
import io
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .logt import cross_sectional_variation, log_t_test
from .plotting import get_theme

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from .plotting import Theme
    from .results import Club


# -- document model -------------------------------------------------------------


@dataclass(frozen=True)
class Paragraph:
    text: str  # plain text; escaped when rendered


@dataclass(frozen=True)
class Table:
    frame: pd.DataFrame
    caption: str = ""
    digits: int = 3
    #: Tables longer than this are folded into an expandable block.
    fold_after: int = 15


@dataclass(frozen=True)
class Image:
    light: bytes
    dark: bytes | None
    caption: str
    alt: str


@dataclass
class Section:
    title: str
    blocks: list[Paragraph | Table | Image] = field(default_factory=list)


@dataclass
class Report:
    """A report made of sections; render with :meth:`to_html` or :meth:`save`."""

    title: str
    subtitle: str
    sections: list[Section]

    def to_html(self) -> str:
        body = "\n".join(_render_section(s) for s in self.sections)
        toc = "".join(
            f'<li><a href="#{_anchor(s.title)}">{html.escape(s.title)}</a></li>'
            for s in self.sections
        )
        return _PAGE.format(
            title=html.escape(self.title),
            subtitle=html.escape(self.subtitle),
            toc=toc,
            body=body,
            css=_CSS,
        )

    def save(self, path: str | Path) -> Path:
        """Write the self-contained HTML file and return its path."""
        path = Path(path)
        path.write_text(self.to_html(), encoding="utf-8")
        return path

    def _repr_html_(self) -> str:
        return self.to_html()


# -- rendering --------------------------------------------------------------------


def _anchor(title: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in title.lower()).strip("-")


def _render_section(section: Section) -> str:
    parts = [f'<section id="{_anchor(section.title)}"><h2>{html.escape(section.title)}</h2>']
    for block in section.blocks:
        if isinstance(block, Paragraph):
            parts.append(f"<p>{html.escape(block.text)}</p>")
        elif isinstance(block, Table):
            parts.append(_render_table(block))
        else:
            parts.append(_render_image(block))
    parts.append("</section>")
    return "\n".join(parts)


def _cell(value: Any, digits: int) -> tuple[str, bool]:
    """Formatted, escaped cell text and whether it is a number (right-aligned)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–", True
    if isinstance(value, (bool, np.bool_)):
        return ("yes" if value else "no"), False
    if isinstance(value, (int, np.integer)):
        return f"{value:d}", True
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}", True
    return html.escape(str(value)), False


def _render_table(table: Table) -> str:
    frame = table.frame
    # The index becomes the first column (named after the index, if it has a name).
    if isinstance(frame.index, pd.RangeIndex):
        frame = frame.reset_index(drop=True)
        columns = list(frame.columns)
    else:
        name = frame.index.name or ""
        frame = frame.reset_index()
        frame.columns = [name, *list(frame.columns[1:])]
        columns = list(frame.columns)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in columns)
    rows = []
    for values in frame.itertuples(index=False):
        cells = []
        for value in values:
            text, numeric = _cell(value, table.digits)
            cells.append(f'<td class="num">{text}</td>' if numeric else f"<td>{text}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    rendered = (
        f'<table class="data"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )
    caption = html.escape(table.caption)
    if len(frame) > table.fold_after:
        label = f"{caption} ({len(frame)} rows)" if caption else f"{len(frame)} rows"
        return (
            f'<details class="table"><summary>{label}</summary>'
            f'<div class="scroll">{rendered}</div></details>'
        )
    figcaption = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f'<figure class="table">{figcaption}<div class="scroll">{rendered}</div></figure>'


def _render_image(image: Image) -> str:
    light = base64.b64encode(image.light).decode("ascii")
    source = ""
    if image.dark is not None:
        dark = base64.b64encode(image.dark).decode("ascii")
        source = (
            f'<source srcset="data:image/png;base64,{dark}" media="(prefers-color-scheme: dark)">'
        )
    return (
        f"<figure><picture>{source}"
        f'<img src="data:image/png;base64,{light}" alt="{html.escape(image.alt)}">'
        f"</picture><figcaption>{html.escape(image.caption)}</figcaption></figure>"
    )


def _png(fig: Figure, dpi: int) -> bytes:
    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    plt.close(fig)
    return buffer.getvalue()


def _image(
    make: Callable[[Any], Any],
    caption: str,
    alt: str,
    *,
    theme: str | Theme | None,
    dark: str | Theme | None,
    dpi: int,
) -> Image:
    """Render ``make(theme)`` (a Figure or Axes) in ``theme`` and, optionally, a dark theme."""

    def render(th: str | Theme | None) -> bytes:
        out = make(get_theme(th))
        return _png(out if hasattr(out, "savefig") else out.figure, dpi)

    return Image(render(theme), render(dark) if dark is not None else None, caption, alt)


# -- the report -------------------------------------------------------------------


def report(
    data: pd.DataFrame,
    *,
    log: bool = True,
    filter_lambda: float | None = 400.0,
    convention: str = "R",
    time_trim: float | None = None,
    hac_method: str = "FQSB",
    merge: str | None = "PS",
    sensitivity: bool = True,
    bootstrap: int = 200,
    seed: int | None = 0,
    beta: bool = True,
    sigma: bool = True,
    dynamics: bool = True,
    dynamics_lag: int = 5,
    covariates: pd.DataFrame | None = None,
    club_names: Mapping[str, str] | Sequence[str] | Callable[[Club], str] | None = None,
    title: str = "Convergence analysis",
    theme: str | Theme | None = None,
    dark_mode: bool | str | Theme = True,
    dpi: int = 110,
    **find_kwargs,
) -> Report:
    """Run the standard convergence analysis and collect it in a :class:`Report`.

    Parameters
    ----------
    data
        Units x periods in **levels** (e.g. GDP per capita); with ``log=False``
        the values are taken as logs already.
    filter_lambda
        HP filter applied to the log series before clustering (``None`` to
        cluster the logs as they are).
    convention, time_trim, hac_method, **find_kwargs
        Passed to :func:`find_clubs`.
    merge
        ``"PS"``, ``"vLT"`` or ``None``.
    sensitivity, bootstrap
        Include the sensitivity grid (time_trim x HAC method) and a bootstrap
        of ``bootstrap`` unit subsamples (0 to skip).
    beta, sigma, dynamics
        Include the classical convergence sections (computed on the
        unfiltered data).
    covariates
        Unit-level covariates (e.g. initial conditions); adds an ordered
        logit of club membership (see :func:`club_determinants`).
    theme
        Theme of the figures (default: the one chosen with :func:`set_theme`).
    club_names
        Names of the final clubs (see :meth:`ConvergenceClubs.rename`).
    dark_mode
        Also embed dark-theme figures, shown when the reader's system uses a
        dark colour scheme; a theme (or its name) replaces the built-in dark one.
    """
    import matplotlib

    from . import __version__
    from .beta import beta_convergence
    from .bootstrap import bootstrap as run_bootstrap
    from .clustering import ClubConvergenceWarning, find_clubs
    from .dynamics import distribution_dynamics
    from .filters import hp_filter
    from .sensitivity import sensitivity as run_sensitivity
    from .sigma import sigma_convergence

    if not matplotlib.is_interactive():
        matplotlib.use("Agg", force=False)
    # These are lists in sensitivity(), so they are passed explicitly everywhere.
    cstar = find_kwargs.pop("cstar", 0.0)
    sort_fraction = find_kwargs.pop("sort_fraction", 0.0)

    logs = data.astype(float).transform(np.log) if log else data.astype(float)
    levels = data if log else data.astype(float).transform(np.exp)
    clustered = hp_filter(logs, lamb=filter_lambda) if filter_lambda is not None else logs
    dark = "dark" if dark_mode is True else (dark_mode or None)
    image: dict[str, Any] = {"theme": theme, "dark": dark, "dpi": dpi}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ClubConvergenceWarning)
        clubs = find_clubs(
            clustered,
            convention=convention,
            time_trim=time_trim,
            hac_method=hac_method,
            cstar=cstar,
            sort_fraction=sort_fraction,
            **find_kwargs,
        )
    final = clubs.merge(merge) if merge else clubs
    if club_names is not None:
        final = final.rename(club_names)
        if not merge:
            clubs = final
    trim = clubs.time_trim
    periods = list(data.columns)

    # 1. Data and settings
    transform = ("log, " if log else "") + (
        f"Hodrick-Prescott trend (lambda = {filter_lambda:g})" if filter_lambda else "no filter"
    )
    settings = pd.DataFrame(
        {
            "value": [
                str(len(data)),
                f"{len(periods)} ({periods[0]}–{periods[-1]})",
                transform,
                convention,
                f"{trim:.3f}",
                hac_method,
                f"{clubs.params['cstar']:g} ({clubs.params['cstar_method']})",
                clubs.params["core_rule"],
                merge or "none",
                __version__,
            ]
        },
        index=pd.Index(
            [
                "units",
                "periods",
                "series clustered",
                "convention",
                "time_trim",
                "HAC method",
                "c*",
                "core group rule",
                "club merging",
                "convergence-clubs version",
            ],
            name="setting",
        ),
    )
    sections = [
        Section(
            "Data and settings",
            [
                Paragraph(
                    "Clubs are found with the Phillips and Sul (2007, 2009) log t test and "
                    "clustering algorithm on the series below; β-convergence, σ-convergence "
                    "and distribution dynamics use the unfiltered data."
                ),
                Table(settings),
            ],
        )
    ]

    # 2. Whole panel
    overall = log_t_test(cross_sectional_variation(clustered.to_numpy()), trim, hac_method)
    verdict = "is not rejected" if overall.converges() else "is rejected"
    sections.append(
        Section(
            "Convergence of the whole panel",
            [
                Paragraph(
                    f"log t test on all {len(data)} units: beta = {overall.beta:.3f} "
                    f"(s.e. {overall.std_err:.3f}), t = {overall.t_value:.2f}. "
                    f"The null hypothesis of convergence {verdict} at the 5% level "
                    "(critical value −1.65)."
                ),
                _image(
                    lambda th: clubs.plot.log_t(theme=th),
                    "log t regression for all units",
                    "Scatter of the log t regression for all units with the fitted line",
                    **image,
                ),
            ],
        )
    )

    # 3. Clubs
    club_blocks: list[Paragraph | Table | Image] = [
        Paragraph(
            f"The clustering algorithm finds {clubs.n_clubs} clubs and {clubs.n_divergent} "
            "divergent units."
        ),
        Table(clubs.summary(), "Initial clubs"),
    ]
    for w in caught:
        club_blocks.append(Paragraph(f"Warning: {w.message}"))
    if merge:
        club_blocks += [
            Paragraph(f"After merging adjacent clubs ({merge}) there are {final.n_clubs} clubs."),
            Table(final.summary(), f"Clubs after {merge} merging"),
        ]
    members = final.summary(members=True)[["n_units", "members"]]
    if final.n_divergent:
        divergent_row = pd.DataFrame(
            {"n_units": [final.n_divergent], "members": ["; ".join(map(str, final.divergent))]},
            index=pd.Index(["divergent"], name=members.index.name),
        )
        members = pd.concat([members, divergent_row])
    club_blocks += [
        Table(members, "Members of each club"),
        _image(
            lambda th: final.plot(ncols=4, theme=th),
            "Relative transition paths of the units of each club and club averages",
            "Small multiples of relative transition paths, one panel per club",
            **image,
        ),
        _image(
            lambda th: final.plot.variation(theme=th),
            "Cross-sectional variation H(t) within each club (log scale)",
            "Line chart of cross-sectional variation over time for each club",
            **image,
        ),
    ]
    sections.append(Section("Convergence clubs", club_blocks))

    # 4. Robustness
    robust: list[Paragraph | Table | Image] = []
    if sensitivity:
        runs = run_sensitivity(
            clustered,
            time_trim=_unique([trim, 1 / 3, 0.25, 0.3, 0.4]),
            hac_method=_unique([hac_method, "FQSB", "AQSB"]),
            cstar=[cstar],
            merge=[merge],
            sort_fraction=[sort_fraction],
            convention=convention,
            **find_kwargs,
        )
        table = runs.table().drop(columns=["cstar", "merge", "sort_fraction"])
        robust += [
            Paragraph(
                "Sensitivity: the clustering is repeated for other trimming fractions and HAC "
                "methods. ari_vs_baseline is the adjusted Rand index of agreement with the "
                "first run (1 = identical clubs, 0 = chance)."
            ),
            Table(table, "Sensitivity of the clubs to time_trim and the HAC method"),
        ]
    if bootstrap:
        boot = run_bootstrap(
            clustered,
            bootstrap,
            seed=seed,
            merge=merge,
            convention=convention,
            time_trim=trim,
            hac_method=hac_method,
            cstar=cstar,
            sort_fraction=sort_fraction,
            **find_kwargs,
        )
        uncertain = boot.uncertain(0.8)
        robust += [
            Paragraph(
                f"Bootstrap: {bootstrap} random subsamples of 80% of the units. A unit's "
                "stability is the share of subsamples in which it stays with its full-sample "
                f"club mates; {len(uncertain)} units are below 0.8."
            ),
            Table(uncertain.drop(columns="n_runs"), "Units with stability below 0.8"),
            _image(
                lambda th: boot.plot_stability(theme=th),
                "Membership confidence of every unit, by club",
                "Dot plot of bootstrap stability per unit, one row per club",
                **image,
            ),
        ]
    if robust:
        sections.append(Section("Robustness", robust))

    # 5-7. Classical convergence
    if beta:
        absolute = beta_convergence(levels, clubs=final)
        conditional = beta_convergence(levels, clubs=final, club_effects=True)
        rows = pd.DataFrame(
            [
                {
                    "model": name,
                    "beta": fit.beta,
                    "std_err": fit.std_err,
                    "t_value": fit.t_value,
                    "p_value": fit.p_value,
                    "half_life": fit.half_life if fit.converges() else np.nan,
                }
                for name, fit in [("absolute", absolute), ("conditional on clubs", conditional)]
            ]
        ).set_index("model")
        sections.append(
            Section(
                "β-convergence",
                [
                    Paragraph(
                        "Average growth regressed on the initial log level "
                        f"({absolute.start}–{absolute.end}, HC1 standard errors). A negative "
                        "and significant beta means that poorer units grow faster; the "
                        "half-life is reported only when beta is significantly negative."
                    ),
                    Table(rows, "Whole panel", digits=4),
                    Table(absolute.by_club().rename_axis("club"), "Within each club", digits=4),
                    _image(
                        lambda th: conditional.plot(theme=th),
                        "Growth against initial level, with club intercepts",
                        "Scatter of average growth against initial log level by club",
                        **image,
                    ),
                ],
            )
        )
    if sigma:
        sig = sigma_convergence(levels, clubs=final)
        sections.append(
            Section(
                "σ-convergence",
                [
                    Paragraph(
                        "Standard deviation of log levels across units in every period, and "
                        f"its linear trend (Newey-West standard errors, {sig.hac_lags} lags). "
                        "Clubs with fewer than 3 units are left out."
                    ),
                    Table(sig.trend(), "Trend of dispersion", digits=4),
                    _image(
                        lambda th: sig.plot(index=True, theme=th),
                        "Dispersion over time, indexed to the first period",
                        "Line chart of dispersion over time for all units and each club",
                        **image,
                    ),
                ],
            )
        )
    if dynamics:
        dyn = distribution_dynamics(levels, lag=dynamics_lag, clubs=final)
        s = dyn.summary()
        summary = pd.DataFrame(
            {
                "value": [
                    int(s["classes"]),
                    int(s["lag"]),
                    int(s["transitions"]),
                    s["shorrocks"],
                    s["second_eigenvalue"],
                    s["half_life"],
                ]
            },
            index=pd.Index(
                [
                    "classes",
                    "lag (periods)",
                    "transitions observed",
                    "Shorrocks mobility index",
                    "second eigenvalue |λ2|",
                    "half-life to the ergodic distribution",
                ],
                name="statistic",
            ),
            dtype=object,
        )
        sections.append(
            Section(
                "Distribution dynamics",
                [
                    Paragraph(
                        "Income relative to the cross-sectional mean is grouped into quintile "
                        f"classes; the transition matrix gives the probability of moving "
                        f"between classes over {dynamics_lag} periods, and the ergodic "
                        "distribution is the long-run distribution it implies."
                    ),
                    Table(summary, "Mobility"),
                    Table(dyn.matrix, "Transition probabilities"),
                    Table(dyn.distributions(), "Class distributions"),
                    Table(dyn.composition(), "Clubs by class in the last period", digits=0),
                    _image(
                        lambda th: dyn.plot_densities(theme=th),
                        "Densities of relative income in the first and last period",
                        "Kernel densities of relative income with a rug of club members",
                        **image,
                    ),
                ],
            )
        )

    if covariates is not None:
        from .determinants import club_determinants

        fit = club_determinants(final, covariates)
        sections.append(
            Section(
                "Determinants of club membership",
                [
                    Paragraph(
                        f"Ordered logit of the club rank (higher = a club with a higher "
                        f"relative level) on {covariates.shape[1]} covariates, n = {fit.n} "
                        f"(divergent units excluded), McFadden pseudo-R² = "
                        f"{fit.pseudo_r2:.3f}. A positive coefficient raises the probability "
                        "of belonging to a higher club."
                    ),
                    Table(fit.summary(), "Coefficients", digits=4),
                    _image(
                        lambda th: fit.plot(theme=th),
                        "Coefficients with 95% intervals",
                        "Coefficient plot of the ordered logit of club membership",
                        **image,
                    ),
                ],
            )
        )

    sections.append(_methods_section(__version__))
    subtitle = (
        f"{len(data)} units, {periods[0]}–{periods[-1]} · generated "
        f"{_dt.date.today().isoformat()} with convergence-clubs {__version__}"
    )
    return Report(title=title, subtitle=subtitle, sections=sections)


def _unique(values: list) -> list:
    out: list = []
    for v in values:
        if not any(v == u or (isinstance(v, float) and abs(v - u) < 1e-9) for u in out):
            out.append(v)
    return out


def _methods_section(version: str) -> Section:
    return Section(
        "Methods and references",
        [
            Paragraph(
                "Club convergence: Phillips and Sul (2007, 2009) log t test with HAC standard "
                "errors (Quadratic Spectral kernel), clustering algorithm and club merging; "
                "von Lyncker and Thoennissen (2017) for alternative merging rules. "
                "Implementation: convergence-clubs " + version + ", a Python port of the R "
                "package ConvergenceClubs (Sichera and Pizzuto, 2019), validated against it "
                "and against the Stata package psecta (Du, 2018)."
            ),
            Paragraph(
                "β-convergence: Barro and Sala-i-Martin (1992). σ-convergence: dispersion of "
                "log levels. Distribution dynamics: Quah (1993, 1996); mobility index of "
                "Shorrocks (1978)."
            ),
            Paragraph(
                "Phillips, P. C. B. and Sul, D. (2007). Transition modeling and econometric "
                "convergence tests. Econometrica 75(6), 1771-1855. — Phillips, P. C. B. and "
                "Sul, D. (2009). Economic transition and growth. Journal of Applied "
                "Econometrics 24(7), 1153-1185. — Sichera, R. and Pizzuto, P. (2019). "
                "ConvergenceClubs. The R Journal 11(2), 142-151. — Du, K. (2017). Econometric "
                "convergence test and club clustering using Stata. The Stata Journal 17(4), "
                "882-900."
            ),
        ],
    )


_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<main>
<header><h1>{title}</h1><p class="subtitle">{subtitle}</p></header>
<nav><ol>{toc}</ol></nav>
{body}
</main>
</body>
</html>
"""

_CSS = """
:root { color-scheme: light; --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b;
  --ink2: #52514e; --muted: #898781; --grid: #e1e0d9; --accent: #2a78d6; }
@media (prefers-color-scheme: dark) {
  :root { color-scheme: dark; --surface: #1a1a19; --page: #0d0d0d; --ink: #ffffff;
    --ink2: #c3c2b7; --muted: #898781; --grid: #2c2c2a; --accent: #3987e5; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1120px; margin: 0 auto; padding: 32px 24px 64px; background: var(--surface); }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 19px; margin: 40px 0 12px; padding-top: 12px; border-top: 1px solid var(--grid); }
.subtitle { color: var(--ink2); margin: 0 0 16px; }
nav ol { color: var(--ink2); padding-left: 20px; columns: 2; }
a { color: var(--accent); }
p { color: var(--ink2); max-width: 78ch; }
figure { margin: 20px 0; }
figcaption { color: var(--ink2); font-size: 13px; margin: 6px 0; }
img { max-width: 100%; height: auto; display: block; }
.scroll { overflow-x: auto; }
table.data { border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
table.data th, table.data td { padding: 4px 10px; border-bottom: 1px solid var(--grid);
  text-align: left; vertical-align: top; }
table.data th { color: var(--ink2); font-weight: 600; border-bottom-color: var(--muted); }
table.data td.num { text-align: right; white-space: nowrap; }
details.table { margin: 16px 0; }
details.table summary { color: var(--ink2); font-size: 13px; cursor: pointer; }
@media print { main { max-width: none; } nav { display: none; } }
"""
