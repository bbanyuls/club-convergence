"""Execute the notebooks in docs/ and save their outputs.

Run: uv run python scripts/run_notebooks.py [notebook ...]

The notebooks are committed with their outputs so that they read on GitHub without running
them; tests/test_notebooks.py runs them on every change. Files they write (report.html,
clubs.png, ...) go to a temporary directory.
"""

import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

DOCS = Path(__file__).resolve().parent.parent / "docs"
NOTEBOOKS = [DOCS / "tutorial.ipynb", DOCS / "feature-tour.ipynb"]


def run(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    with tempfile.TemporaryDirectory() as tmp:
        NotebookClient(
            notebook, timeout=900, kernel_name="python3", resources={"metadata": {"path": tmp}}
        ).execute()
    for cell in notebook.cells:  # stable diffs: no execution timestamps
        cell.get("metadata", {}).pop("execution", None)
    nbformat.write(notebook, path)
    print(f"wrote {path.name} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    for path in [Path(p) for p in sys.argv[1:]] or NOTEBOOKS:
        run(path)
