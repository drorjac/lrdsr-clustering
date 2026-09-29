# Contributing

Contributions are welcome: bug reports, questions, new problems for the zoo,
new law engines or bases, new datasets, and corrections to the text.

## How

1. Open an issue describing the problem or the idea, or go straight to a pull request for small fixes.
2. Fork, create a branch, make the change, and open a pull request against `main`.

## Setting up

```bash
pip install -e ".[dev]"
pytest -q                     # all tests must pass
ruff check .                  # style
python notebooks/build.py     # only if you changed notebooks/sources.py
```

## The house rules

These keep every number in the repository checkable:

- **True labels reach the final metric only**, never a fit.
- **Tune on seeds {3, 7, 19}, report on {11, 23, 42}**, and never choose anything on a reported seed.
- **Write a prediction down before the run that tests it**, in the experiment's docstring, and report it whether it holds or fails.
- **No number is typed by hand.** `README.md` and `RESULTS.md` are generated from `results/` (`python -m analysis.report --write`).
- **Notebooks are generated**: edit `notebooks/sources.py`, never the `.ipynb`.
- **Negative results are results**, and go in the text, not a footnote.

Code style: match the surrounding code, keep functions documented the way the rest of `lrdsr/` is, and add a test for any new behaviour.
