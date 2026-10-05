"""Build src/convergence_clubs/data/productivity.csv from the files of Mendez (2020).

Source: C. Mendez (2020), "Convergence Clubs in Labor Productivity and its
Proximate Sources", Springer; data and code at
https://github.com/quarcs-lab/mendez2020-convergence-clubs-code-data (MIT
licence). The variables come from the Penn World Table 9.0 (Feenstra,
Inklaar and Timmer, 2015; CC BY 4.0).

Usage:
    git clone https://github.com/quarcs-lab/mendez2020-convergence-clubs-code-data /tmp/mendez
    uv run python scripts/build_productivity_data.py /tmp/mendez/assets/dat.csv
"""

import sys
from pathlib import Path

import pandas as pd

COLUMNS = {
    "country": "country",
    "isocode": "iso3",
    "region": "region",
    "hi1990": "high_income_1990",
    "year": "year",
    "lp": "lp",
    "ky": "ky",
    "h": "h",
    "TFP": "tfp",
    "GDPpc": "gdppc",
}


def main(source: str) -> None:
    data = pd.read_csv(source, dtype=str)[list(COLUMNS)].rename(columns=COLUMNS)
    data = data.sort_values(
        ["country", "year"], key=lambda s: s.str.lower() if s.name == "country" else s.astype(int)
    )
    out = Path(__file__).resolve().parent.parent / "src/convergence_clubs/data/productivity.csv"
    data.to_csv(out, index=False)  # values are copied as text: no rounding
    print(f"wrote {out} ({len(data)} rows, {data['country'].nunique()} countries)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/mendez/assets/dat.csv")
