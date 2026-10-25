# Contributing

Bug reports, questions and pull requests are welcome through
[GitHub issues](https://github.com/bbanyuls/club-convergence/issues).

## Setting up

```bash
git clone https://github.com/bbanyuls/club-convergence
cd club-convergence
uv sync --group dev
```

## Before opening a pull request

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest
```

The tests need to keep the coverage at its current level, and a change to a result
needs a test that fails without it. Changes that alter the clubs for a given input
should say so in `CHANGELOG.md` and, if they differ from the R package, in the
known-differences section of `docs/guide/validation.md`.

## Reporting a problem

Please include the version (`convergence-clubs --version`), a small input that shows
the problem and what you expected. For disagreements with R or Stata, the R or Stata
output for the same input helps most.
