"""Run every cell of the notebooks in docs/ against the installed package."""

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

DOCS = Path(__file__).resolve().parent.parent / "docs"


def _execute(name: str, tmp_path: Path):
    notebook = nbformat.read(DOCS / name, as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    # Committed with outputs so they read on GitHub; scripts/run_notebooks.py refreshes them.
    executed = all(cell.execution_count is not None for cell in code)
    assert executed, f"run scripts/run_notebooks.py to save the outputs of {name}"
    for cell in code:
        cell.outputs = []
    client = nbclient.NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    )
    client.execute()
    errors = [out for cell in code for out in cell.outputs if out.output_type == "error"]
    assert not errors
    return "\n".join(
        str(out.get("data", {}).get("text/plain", "")) + out.get("text", "")
        for cell in code
        for out in cell.outputs
    )


def test_tutorial_runs(tmp_path):
    text = _execute("tutorial.ipynb", tmp_path)
    assert "([31, 20, 8, 15, 5, 2], [51, 8, 15, 5, 2])" in text  # the Stata replication
    assert (tmp_path / "report.html").exists()


def test_feature_tour_runs(tmp_path):
    text = _execute("feature-tour.ipynb", tmp_path)
    assert "t_value=-159.55" in text and "([31, 20, 8, 15, 5, 2], [51, 8, 15, 5, 2])" in text
    for written in ("report.html", "clubs.html", "clubs.png", "summary.csv"):
        assert (tmp_path / written).exists()
