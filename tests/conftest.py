import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

import convergence_clubs as cc

matplotlib.use("Agg")

FIXTURES = Path(__file__).parent / "fixtures" / "r_reference.json"


@pytest.fixture(scope="session")
def r_reference() -> dict:
    return json.loads(FIXTURES.read_text())


@pytest.fixture(scope="session")
def filtered_gdp() -> pd.DataFrame:
    return cc.load_filtered_gdp()


@pytest.fixture(scope="session")
def datasets(r_reference, filtered_gdp) -> dict[str, pd.DataFrame]:
    out = {"filtered_gdp": filtered_gdp}
    for name, d in r_reference["datasets"].items():
        values = np.asarray(d["values"], dtype=float)
        out[name] = pd.DataFrame(values, index=d["units"], columns=range(1, values.shape[1] + 1))
    return out
