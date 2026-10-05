"""Run every cell of docs/tutorial.ipynb against the installed package."""

import shutil
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

TUTORIAL = Path(__file__).resolve().parent.parent / "docs" / "tutorial.ipynb"


def test_tutorial_runs(tmp_path):
    notebook = nbformat.read(TUTORIAL, as_version=4)
    assert not any(c.get("outputs") for c in notebook.cells if c.cell_type == "code"), (
        "commit the tutorial without outputs"
    )
    shutil.copy(TUTORIAL, tmp_path / TUTORIAL.name)
    client = nbclient.NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    )
    client.execute()
    errors = [
        out
        for cell in notebook.cells
        if cell.cell_type == "code"
        for out in cell.outputs
        if out.output_type == "error"
    ]
    assert not errors
    text = "\n".join(
        str(out.get("data", {}).get("text/plain", "")) + out.get("text", "")
        for cell in notebook.cells
        if cell.cell_type == "code"
        for out in cell.outputs
    )
    assert "([31, 20, 8, 15, 5, 2], [51, 8, 15, 5, 2])" in text  # the Stata replication
    assert (tmp_path / "report.html").exists()
