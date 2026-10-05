import re

import numpy as np
import pandas as pd
import pytest

import convergence_clubs as cc
from convergence_clubs.report import Paragraph, Report, Section, Table, _render_table

pytestmark = pytest.mark.filterwarnings("ignore::convergence_clubs.ClubConvergenceWarning")


@pytest.fixture(scope="module")
def built():
    return cc.report(cc.load_gdp(), bootstrap=5, dark_mode=False, dpi=40, title="GDP & <clubs>")


def test_sections_and_numbers(built):
    titles = [s.title for s in built.sections]
    assert titles == [
        "Data and settings",
        "Convergence of the whole panel",
        "Convergence clubs",
        "Robustness",
        "β-convergence",
        "σ-convergence",
        "Distribution dynamics",
        "Methods and references",
    ]
    html = built.to_html()
    assert "t = -159.56" in html  # the whole-panel log t test (Sichera and Pizzuto, 2019)
    assert "finds 7 clubs and 0 divergent units" in html
    assert "there are 6 clubs" in html
    clubs = built.sections[2]
    tables = [b for b in clubs.blocks if isinstance(b, Table)]
    assert tables[1].frame["n_units"].tolist() == [50, 30, 21, 38, 11, 2]


def test_html_is_self_contained_and_escaped(built):
    html = built.to_html()
    assert html.startswith("<!DOCTYPE html>") and "</html>" in html
    assert "GDP &amp; &lt;clubs&gt;" in html and "<clubs>" not in html
    assert "St. Kitts &amp; Nevis" in html  # unit names are escaped
    assert not re.search(r'src="(?!data:)', html)  # no external resources
    images = re.findall(r'<img src="data:image/png;base64,', html)
    assert len(images) == 7
    assert "prefers-color-scheme: dark)" not in html.split("<style>")[1].split("</style>")[1]


def test_dark_variants_and_saving(tmp_path):
    rep = cc.report(
        cc.load_gdp().iloc[:40], bootstrap=0, sensitivity=False, beta=False, sigma=False,
        dynamics=False, dpi=30,
    )  # fmt: skip
    assert [s.title for s in rep.sections] == [
        "Data and settings", "Convergence of the whole panel", "Convergence clubs",
        "Methods and references",
    ]  # fmt: skip
    html = rep.to_html()
    assert html.count('media="(prefers-color-scheme: dark)"') == html.count("<img ")
    path = rep.save(tmp_path / "r.html")
    assert path.read_text(encoding="utf-8") == html
    assert rep._repr_html_() == html


def test_logs_and_stata_convention():
    logs = np.log(cc.load_gdp())
    rep = cc.report(
        logs, log=False, convention="Stata", bootstrap=0, sensitivity=False, dark_mode=False,
        dpi=30,
    )  # fmt: skip
    settings = rep.sections[0].blocks[1].frame["value"]
    assert settings["convention"] == "Stata" and settings["time_trim"] == "0.300"
    assert settings["core group rule"] == "max_t"


def test_table_rendering():
    import pandas as pd

    frame = pd.DataFrame(
        {"n": [1, 2], "x": [0.5, np.nan], "ok": [True, False], "name": ["a<b", "c"]},
        index=pd.Index(["r1", "r2"], name="row"),
    )
    out = _render_table(Table(frame, "cap", digits=2))
    assert "<th>row</th><th>n</th><th>x</th>" in out
    assert '<td class="num">1</td>' in out and '<td class="num">0.50</td>' in out
    assert '<td class="num">–</td>' in out and "<td>yes</td>" in out and "a&lt;b" in out
    long = _render_table(Table(pd.DataFrame({"v": range(20)}), "many"))
    assert long.startswith("<details") and "many (20 rows)" in long
    custom = Report("t", "s", [Section("A b", [Paragraph("x < y")])]).to_html()
    assert 'id="a-b"' in custom and "x &lt; y" in custom


def test_clustering_options_reach_every_analysis():
    rep = cc.report(
        cc.load_gdp(), cstar=-1, sort_fraction=0.1, bootstrap=3, dark_mode=False, dpi=30,
        beta=False, sigma=False, dynamics=False,
    )  # fmt: skip
    assert rep.sections[0].blocks[1].frame["value"]["c*"] == "-1 (fixed)"
    sensitivity = rep.sections[3].blocks[1].frame
    with __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        expected = cc.find_clubs(
            cc.hp_filter(np.log(cc.load_gdp())), cstar=-1, sort_fraction=0.1
        ).merge("PS")
    assert sensitivity["sizes"].iloc[0] == "-".join(str(len(c)) for c in expected)


def test_determinants_section():
    covariates = pd.DataFrame({"log_h": np.log(cc.load_productivity("h")[1990])})
    rep = cc.report(
        cc.load_productivity("lp"), convention="Stata", time_trim=0.333, covariates=covariates,
        bootstrap=0, sensitivity=False, beta=False, sigma=False, dynamics=False,
        dark_mode=False, dpi=30,
    )  # fmt: skip
    assert [s.title for s in rep.sections][-2] == "Determinants of club membership"
    assert "<td>log_h</td>" in rep.to_html()
