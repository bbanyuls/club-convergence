import pandas as pd

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
    out = capsys.readouterr().out
    assert out.startswith("5 convergence clubs, 3 divergent units")
    assert "divergent:" in out and "beta:" in out

    assert main([str(csv), "--sep", ";", "--hac", "AQSB", "--merge-divergent"]) == 0
    expected = cc.find_clubs(cc.load_filtered_gdp(), hac_method="AQSB").merge_divergent()
    assert expected.n_divergent == 1
    assert "5 convergence clubs, 1 divergent unit\n" in capsys.readouterr().out


def test_errors_exit_with_status_1(tmp_path, capsys):
    assert main([str(tmp_path / "missing.csv")]) == 1
    assert "error" in capsys.readouterr().err
