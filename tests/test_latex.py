import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

import convergence_clubs as cc
from convergence_clubs.latex import escape

pytestmark = pytest.mark.filterwarnings("ignore::convergence_clubs.ClubConvergenceWarning")


@pytest.fixture(scope="module")
def merged(filtered_gdp):
    return cc.find_clubs(filtered_gdp).merge("PS")


def test_escape():
    assert escape("St. Kitts & Nevis") == r"St. Kitts \& Nevis"
    assert escape("50%_{x}$#~^\\") == (
        r"50\%\_\{x\}\$\#\textasciitilde{}\textasciicircum{}\textbackslash{}"
    )


def test_cells():
    table = pd.DataFrame(
        {
            "n": [50, -3],
            "x": [-0.0435, np.nan],
            "ok": [True, False],
            "big": [np.inf, -np.inf],
            "tiny": [-0.0001, 1.0],
        },
        index=pd.Index(["a_1", "b"], name="unit"),
    )
    text = cc.to_latex(table, digits={"x": 2, "tiny": 2})
    assert r"a\_1 & 50 & $-$0.04 & yes & $\infty$ & 0.00 \\" in text  # -0.0001 rounds to 0.00
    assert r"b & $-$3 &  & no & $-\infty$ & 1.00 \\" in text
    assert r"\begin{tabular}{lrrlrr}" in text  # booleans are text, not numbers
    assert "unit & n & x & ok & big & tiny" in text
    assert cc.to_latex(table, index=False).count(" & ") == cc.to_latex(table).count(" & ") - 3


def test_layout_options(tmp_path):
    table = pd.DataFrame({"name": ["a long text"], "v": [1.5]})
    path = tmp_path / "t.tex"
    text = cc.to_latex(
        table, caption="Cap", label="tab:x", header={"v": r"$\nu$"},
        wrap={"name": "4cm"}, notes="A note.", path=path,
    )  # fmt: skip
    assert path.read_text(encoding="utf-8") == text
    assert text.startswith(r"% Needs \usepackage{booktabs,array}")
    assert r">{\raggedright\arraybackslash}p{4cm}" in text and r"$\nu$" in text
    assert text.index(r"\caption{Cap}") < text.index(r"\label{tab:x}") < text.index("tabular")
    assert text.index(r"\end{tabular}") < text.index("A note.") < text.index(r"\end{table}")

    long = cc.to_latex(table, caption="Cap", longtable=True, notes="N.")
    assert r"\begin{longtable}" in long and r"\endhead" in long and "tabular" not in long
    assert long.index(r"\end{longtable}") < long.index("N.")


def test_clubs_table(merged):
    text = merged.to_latex(label="tab:clubs")
    assert r"\begin{longtable}" in text  # 152 units with members: may span pages
    assert r"\caption{Convergence clubs after merging}\label{tab:clubs}" in text
    assert r"club1 & 50 & 0.382 & 9.28 & United States; Norway;" in text
    assert r"club4 & 38 & $-$0.044 & $-$0.64 &" in text
    assert r"St. Kitts \& Nevis" in text
    assert "FQSB standard errors" in text and "adjacent clubs merged" in text

    short = merged.rename(["High", "B", "C", "D", "E", "F"]).to_latex(members=False)
    assert r"\begin{table}" in short and "Members" not in short
    assert r"High & 50 & 0.382 & 9.28 \\" in short


def test_clubs_table_with_divergent_units(filtered_gdp):
    aqsb = cc.find_clubs(filtered_gdp, hac_method="AQSB")
    text = aqsb.to_latex(caption=None)
    assert r"Divergent & 3 &  &  & Somalia; Congo, Dem. Rep.; Liberia \\" in text
    assert r"\caption" not in text and "AQSB standard errors" in text
    assert "Divergent" not in aqsb.to_latex(divergent=False)


@pytest.mark.skipif(shutil.which("tectonic") is None, reason="needs tectonic to compile LaTeX")
def test_tables_compile(merged, tmp_path):
    beta = cc.beta_convergence(cc.load_gdp(), clubs=merged)
    body = "\n".join(
        [
            merged.to_latex(),
            merged.to_latex(members=False),
            cc.to_latex(beta.by_club(), header={"beta": r"$\hat\beta$"}),
        ]
    )
    document = (
        "\\documentclass{article}\n\\usepackage{booktabs,array,longtable}\n"
        f"\\begin{{document}}\n{body}\\end{{document}}\n"
    )
    (tmp_path / "t.tex").write_text(document, encoding="utf-8")
    run = subprocess.run(
        ["tectonic", "-X", "compile", "t.tex"], cwd=tmp_path, capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr[-2000:]
    assert (tmp_path / "t.pdf").stat().st_size > 0
