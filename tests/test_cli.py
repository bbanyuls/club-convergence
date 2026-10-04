import pandas as pd
import pytest

import convergence_clubs as cc
from convergence_clubs.cli import main


def test_raw_gdp_to_merged_clubs(tmp_path, capsys):
    csv = tmp_path / "gdp.csv"
    cc.load_gdp().to_csv(csv)
    members, summary, figure = tmp_path / "m.csv", tmp_path / "s.csv", tmp_path / "p.png"

    code = main([
        str(csv), "--log", "--hp-filter", "--merge", "PS",
        "--memberships", str(members), "--summary", str(summary), "--plot", str(figure),
    ])  # fmt: skip

    assert code == 0
    out = capsys.readouterr().out
    assert out.startswith("6 convergence clubs, 0 divergent units")
    assert pd.read_csv(summary)["n_units"].tolist() == [50, 30, 21, 38, 11, 2]
    assert pd.read_csv(members, index_col=0)["club"]["United States"] == "club1"
    assert figure.stat().st_size > 0


def test_options_and_divergent_units(tmp_path, capsys):
    csv = tmp_path / "filtered.csv"
    cc.load_filtered_gdp().to_csv(csv, sep=";")
    assert main([str(csv), "--sep", ";", "--hac", "AQSB", "--time-trim", "1/3", "--details"]) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("5 convergence clubs, 3 divergent units")
    assert "divergent:" in captured.out and "beta:" in captured.out
    assert "warning: club1 does not pass the log t test" in captured.err

    assert main([str(csv), "--sep", ";", "--hac", "AQSB", "--merge-divergent"]) == 0
    with pytest.warns(cc.ClubConvergenceWarning):
        expected = cc.find_clubs(cc.load_filtered_gdp(), hac_method="AQSB").merge_divergent()
    assert expected.n_divergent == 1
    assert "5 convergence clubs, 1 divergent unit\n" in capsys.readouterr().out


def test_errors_exit_with_status_1(tmp_path, capsys):
    assert main([str(tmp_path / "missing.csv")]) == 1
    assert "error" in capsys.readouterr().err


def test_long_input_and_sensitivity(tmp_path, capsys):
    long = cc.load_gdp().stack().rename("gdp").rename_axis(["country", "year"]).reset_index()
    csv, heatmap, summary = tmp_path / "long.csv", tmp_path / "co.png", tmp_path / "s.csv"
    long.to_csv(csv, index=False)

    code = main([
        str(csv), "--long", "country", "year", "gdp", "--log", "--hp-filter",
        "--summary", str(summary), "--members",
        "--sensitivity", "--sensitivity-plot", str(heatmap),
    ])  # fmt: skip

    assert code == 0
    out = capsys.readouterr().out
    assert out.startswith("7 convergence clubs, 0 divergent units")
    assert "sensitivity (8 runs" in out and "least stable units" in out
    assert pd.read_csv(summary)["members"].iloc[-1] == "Congo, Dem. Rep.; Liberia"
    assert heatmap.stat().st_size > 0


def test_sensitivity_errors_are_reported(tmp_path, capsys, monkeypatch):
    import convergence_clubs.cli as cli

    def fail(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(cli, "sensitivity", fail)
    csv = tmp_path / "f.csv"
    cc.load_filtered_gdp().to_csv(csv)
    assert main([str(csv), "--sensitivity"]) == 1
    assert "error in sensitivity analysis: boom" in capsys.readouterr().err


def test_sort_fraction_option(tmp_path, capsys):
    csv = tmp_path / "f.csv"
    cc.load_filtered_gdp().to_csv(csv)
    assert main([str(csv), "--sort-fraction", "1/34"]) == 0
    assert capsys.readouterr().out.startswith("7 convergence clubs")


def test_bootstrap_option(tmp_path, capsys):
    csv = tmp_path / "gdp.csv"
    cc.load_gdp().to_csv(csv)
    args = [str(csv), "--log", "--hp-filter", "--bootstrap", "5", "--seed", "1"]
    assert main([*args, "--bootstrap-method", "cycle"]) == 0
    out = capsys.readouterr().out
    assert "bootstrap (cycle, 5 replicates)" in out and "confidence below 0.8" in out


def test_joint_divergent_option(tmp_path, capsys):
    csv = tmp_path / "f.csv"
    cc.load_filtered_gdp().to_csv(csv)
    assert main([str(csv), "--hac", "AQSB", "--merge-divergent", "--joint-divergent"]) == 0
    assert "1 divergent unit" in capsys.readouterr().out


def test_beta_option(tmp_path, capsys):
    csv, figure = tmp_path / "gdp.csv", tmp_path / "beta.png"
    cc.load_gdp().to_csv(csv)
    assert (
        main([str(csv), "--log", "--hp-filter", "--merge", "PS", "--beta-plot", str(figure)]) == 0
    )
    out = capsys.readouterr().out
    assert "β-convergence 1970-2003" in out and "conditional on clubs" in out
    assert "within clubs" in out and figure.stat().st_size > 0


def test_sigma_option(tmp_path, capsys):
    csv, figure = tmp_path / "gdp.csv", tmp_path / "sigma.png"
    cc.load_gdp().to_csv(csv)
    assert (
        main([str(csv), "--log", "--hp-filter", "--merge", "PS", "--sigma-plot", str(figure)]) == 0
    )
    out = capsys.readouterr().out
    assert "σ-convergence (sd of log" in out and "all units" in out and "club5" in out
    assert figure.stat().st_size > 0


def test_dynamics_option(tmp_path, capsys):
    csv, figure = tmp_path / "gdp.csv", tmp_path / "dyn.png"
    cc.load_gdp().to_csv(csv)
    args = [str(csv), "--log", "--hp-filter", "--merge", "PS", "--dynamics-plot", str(figure)]
    assert main([*args, "--dynamics-lag", "5", "--dynamics-classes", "4"]) == 0
    out = capsys.readouterr().out
    assert "4 classes of relative income, 5-period transitions" in out
    assert "ergodic" in out and "clubs by class" in out and figure.stat().st_size > 0


def test_map_option(tmp_path, capsys):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    names = list(cc.load_filtered_gdp().index[:-1])  # leave one country out
    boxes = [box(i % 13, i // 13, i % 13 + 1, i // 13 + 1) for i in range(len(names))]
    geo = tmp_path / "regions.gpkg"
    gpd.GeoDataFrame({"name": names}, geometry=boxes, crs="EPSG:4326").to_file(geo)
    csv, figure = tmp_path / "f.csv", tmp_path / "map.png"
    cc.load_filtered_gdp().to_csv(csv)
    args = [str(csv), "--map", str(figure), "--geodata", str(geo), "--geo-key", "name"]
    assert main(args) == 0
    assert "1 units have no region" in capsys.readouterr().err and figure.stat().st_size > 0
    assert main([str(csv), "--map", str(figure)]) == 1


def test_html_option(tmp_path, capsys):
    csv, page = tmp_path / "f.csv", tmp_path / "clubs.html"
    cc.load_filtered_gdp().to_csv(csv)
    assert main([str(csv), "--html", str(page)]) == 0
    assert page.stat().st_size > 10_000
