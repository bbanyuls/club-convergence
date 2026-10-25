# JSS manuscript

Draft for the *Journal of Statistical Software*.

- `convergence-clubs.tex` builds with `tectonic convergence-clubs.tex` (or `pdflatex` and `bibtex`). `jss.cls`, `jss.bst` and `jsslogo.jpg` are the journal's style files.
- `replication.py` reproduces every number in the manuscript and stops if one differs: `uv run --extra plot python paper/jss/replication.py` (a few seconds).
