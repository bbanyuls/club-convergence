"""Are convergence clubs spatially clustered? Join counts on a contiguity graph."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .results import ConvergenceClubs

DIVERGENT = "divergent"
ALL = "all clubs"


def neighbours(
    geodata: Any,
    units: Sequence[Hashable] | pd.Index,
    *,
    key: Hashable | None = None,
    names: Mapping | pd.Series | None = None,
    contiguity: Literal["queen", "rook"] = "queen",
    tolerance: float = 0.0,
) -> dict[Hashable, list[Hashable]]:
    """Contiguity neighbours of ``units`` from region boundaries (needs geopandas).

    Parameters
    ----------
    geodata
        A ``geopandas.GeoDataFrame``; ``key`` and ``names`` match units to
        regions as in :func:`plot_map`.
    contiguity
        ``"queen"``: regions that share any boundary point; ``"rook"``: only
        those that share a stretch of boundary.
    tolerance
        Grow every region by this distance (in the units of the CRS) before
        testing contact, to close small gaps between boundaries drawn at
        different resolutions.

    Units without a region, or without neighbours (islands), get an empty
    list.
    """
    if contiguity not in ("queen", "rook"):
        raise ValueError("contiguity must be 'queen' or 'rook'")
    mapping = pd.Series(names) if names is not None else pd.Series(dtype=object)
    region_of = {u: mapping.get(u, u) for u in units}
    keys = pd.Index(geodata[key] if key is not None else geodata.index)
    position = {k: i for i, k in enumerate(keys)}
    unit_of = {region: unit for unit, region in region_of.items() if region in position}
    geometry = geodata.geometry.reset_index(drop=True)
    grown = geometry.buffer(tolerance) if tolerance > 0 else geometry
    out: dict[Hashable, list[Hashable]] = {u: [] for u in units}
    for unit, region in region_of.items():
        if region not in position:
            continue
        i = position[region]
        for j in grown.sindex.query(grown.iloc[i], predicate="intersects"):
            other = keys[int(j)]
            if j == i or other not in unit_of:
                continue
            if contiguity == "rook":
                shared = grown.iloc[i].boundary.intersection(grown.iloc[int(j)].boundary)
                if shared.length <= 0:
                    continue
            out[unit].append(unit_of[other])
    return out


@dataclass(frozen=True)
class SpatialResult:
    """Join counts of club membership on a contiguity graph (see :func:`spatial_clustering`)."""

    joins: pd.DataFrame
    labels: pd.Series
    permutations: np.ndarray = field(repr=False)
    groups: tuple[str, ...]
    n_isolated: int

    def summary(self) -> pd.DataFrame:
        """Observed and expected joins within each club, and for all clubs together.

        ``joins_within`` counts pairs of neighbours in the same club;
        ``expected`` is its mean when club labels are assigned at random
        (keeping the club sizes); ``ratio`` is observed / expected;
        ``p_value`` is the one-sided permutation p-value of seeing at least
        as many joins, and ``z_value`` uses the permutation standard
        deviation.
        """
        n = len(self.labels)
        total = len(self.joins)
        sizes = self.labels.value_counts()
        observed = self._counts(self.labels.to_numpy())
        rows = {}
        for k, group in enumerate([*self.groups, ALL]):
            draws = self.permutations[:, k]
            if group == ALL:
                share = sum(s * (s - 1) for s in sizes) / (n * (n - 1))
                n_units = n
            else:
                share = sizes.get(group, 0) * (sizes.get(group, 0) - 1) / (n * (n - 1))
                n_units = int(sizes.get(group, 0))
            expected = total * share
            sd = draws.std(ddof=1) if len(draws) > 1 else np.nan
            rows[group] = {
                "n_units": n_units,
                "joins_within": int(observed[k]),
                "expected": expected,
                "ratio": observed[k] / expected if expected > 0 else np.nan,
                "z_value": (observed[k] - expected) / sd if sd and sd > 0 else np.nan,
                "p_value": (1 + np.sum(draws >= observed[k])) / (1 + len(draws)),
            }
        return pd.DataFrame.from_dict(rows, orient="index").rename_axis("club")

    def local(self) -> pd.DataFrame:
        """Each unit's neighbours and the share of them in its own club."""
        pairs = pd.concat(
            [self.joins, self.joins.rename(columns={"unit_i": "unit_j", "unit_j": "unit_i"})]
        )
        pairs["same"] = (
            pairs["unit_i"].map(self.labels).to_numpy()
            == pairs["unit_j"].map(self.labels).to_numpy()
        )
        grouped = pairs.groupby("unit_i")["same"]
        table = pd.DataFrame(
            {
                "club": self.labels,
                "n_neighbours": grouped.size().reindex(self.labels.index, fill_value=0),
                "same_club_share": grouped.mean().reindex(self.labels.index),
            }
        )
        return table.rename_axis("unit")

    def _counts(self, labels: np.ndarray) -> np.ndarray:
        index = {u: i for i, u in enumerate(self.labels.index)}
        i = np.array([index[u] for u in self.joins["unit_i"]], dtype=int)
        j = np.array([index[u] for u in self.joins["unit_j"]], dtype=int)
        same = labels[i] == labels[j]
        counts = [int(np.sum(same & (labels[i] == g))) for g in self.groups]
        return np.array([*counts, int(same.sum())])

    def plot(self, **kwargs) -> Axes:
        """Observed vs randomly expected joins per club; see :func:`plot_spatial`."""
        from .plotting import plot_spatial

        return plot_spatial(self, **kwargs)

    def __repr__(self) -> str:
        row = self.summary().loc[ALL]
        return (
            f"<SpatialResult: {len(self.joins)} joins, {int(row['joins_within'])} within clubs "
            f"({row['ratio']:.2f} x random, p = {row['p_value']:.3g})>"
        )


def spatial_clustering(
    result: ConvergenceClubs,
    neighbour_map: Mapping[Hashable, Sequence[Hashable]] | None = None,
    *,
    geodata: Any = None,
    key: Hashable | None = None,
    names: Mapping | pd.Series | None = None,
    contiguity: Literal["queen", "rook"] = "queen",
    tolerance: float = 0.0,
    permutations: int = 999,
    seed: int | np.random.Generator | None = None,
) -> SpatialResult:
    """Do units of the same club tend to be neighbours? Join counts with a permutation test.

    Counts the pairs of neighbours that share a club, overall and club by
    club, and compares them with random assignments of the same club sizes
    to the same map (``permutations`` of them).

    Parameters
    ----------
    neighbour_map
        Unit -> list of neighbouring units (e.g. from :func:`neighbours`).
        Either this or ``geodata`` is needed.
    geodata, key, names, contiguity, tolerance
        Build the neighbours from region boundaries; see :func:`neighbours`.

    Units with no neighbours (islands, or no region) do not enter any join,
    but keep their label in the permutations. Divergent units count as a
    group of their own.
    """
    if neighbour_map is None:
        if geodata is None:
            raise ValueError("pass neighbour_map or geodata")
        neighbour_map = neighbours(
            geodata, result.data.index, key=key, names=names,
            contiguity=contiguity, tolerance=tolerance,
        )  # fmt: skip
    if permutations < 1:
        raise ValueError("permutations must be at least 1")
    labels = result.memberships()
    known = set(labels.index)
    joins = sorted(
        {
            tuple(sorted((a, b), key=str))
            for a, others in neighbour_map.items()
            if a in known
            for b in others
            if b in known and b != a
        },
        key=lambda pair: (str(pair[0]), str(pair[1])),
    )
    if not joins:
        raise ValueError("no two units are neighbours")
    table = pd.DataFrame(joins, columns=["unit_i", "unit_j"])
    groups = tuple([c.name for c in result.clubs] + ([DIVERGENT] if result.n_divergent else []))
    connected = set(table["unit_i"]) | set(table["unit_j"])
    shell = SpatialResult(table, labels, np.empty((0, len(groups) + 1)), groups, 0)
    rng = np.random.default_rng(seed)
    values = labels.to_numpy()
    draws = np.array([shell._counts(rng.permutation(values)) for _ in range(permutations)])
    return SpatialResult(table, labels, draws, groups, len(known - connected))
