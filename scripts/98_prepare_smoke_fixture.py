from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import shutil

import pandas as pd

from nimscale.io import ensure_dir
from nimscale.settings import load_config, project_root


def build_financial_rows() -> pd.DataFrame:
    quarter_dates = pd.date_range("2019-03-31", "2021-12-31", freq="QE-DEC")
    rows: list[dict[str, object]] = []
    for bank_num in range(1, 13):
        cert = 10_000 + bank_num
        base_asset = 80_000 + bank_num * 8_000
        for quarter_idx, repdte in enumerate(quarter_dates):
            asset = float(base_asset + quarter_idx * 2_400 + bank_num * 60)
            deposits = asset * (0.72 + 0.008 * ((bank_num + quarter_idx) % 4) + quarter_idx * 0.002)
            equity = asset * (0.09 + bank_num * 0.0005 + quarter_idx * 0.0015)
            loans = asset * (0.6 + bank_num * 0.002 + quarter_idx * 0.003)
            fedfunds = 0.35 + quarter_idx * 0.2
            nim = 4.35 - 0.07 * (bank_num / 12.0) - 0.05 * quarter_idx
            roa = 0.85 + bank_num * 0.015 - 0.015 * quarter_idx
            int_exp = 1.1 + 0.01 * bank_num + 0.05 * quarter_idx
            int_inc = nim + int_exp
            rows.append(
                {
                    "CERT": cert,
                    "NAME": f"Smoke Bank {bank_num}",
                    "REPDTE": repdte.date().isoformat(),
                    "ASSET": round(asset, 3),
                    "DEP": round(deposits, 3),
                    "EQ": round(equity, 3),
                    "LNLSNET": round(loans, 3),
                    "NIMY": round(nim, 4),
                    "INTINCY": round(int_inc, 4),
                    "INTEXPY": round(int_exp, 4),
                    "ROA": round(roa, 4),
                    "ROE": round(roa / 0.1, 4),
                    "OFFOA": 4 + (bank_num % 3),
                    "CITY": f"City {bank_num}",
                    "STALP": "NY" if bank_num <= 6 else "CA",
                    "REGAGNT": "FDIC" if bank_num % 2 else "OCC",
                    "BKCLASS": "SM",
                    "ACTIVE": 1,
                }
            )
    return pd.DataFrame(rows)


def build_rate_rows() -> pd.DataFrame:
    quarter_dates = pd.date_range("2019-03-31", "2021-12-31", freq="QE-DEC")
    rows = []
    for quarter_idx, repdte in enumerate(quarter_dates):
        rows.append(
            {
                "REPDTE": repdte,
                "FEDFUNDS": 0.35 + quarter_idx * 0.2,
                "SLOPE_10Y_3M": 1.1 - quarter_idx * 0.05 + (0.07 if quarter_idx % 2 else -0.03),
            }
        )
    return pd.DataFrame(rows)


def build_sod_rows(year: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bank_num in range(1, 13):
        cert = 10_000 + bank_num
        county = f"{36 if bank_num <= 6 else 6}{(bank_num % 3) + 1:03d}"
        for branch_idx in range(2):
            rows.append(
                {
                    "CERT": cert,
                    "YEAR": year,
                    "STCNTYBR": county,
                    "DEPSUMBR": 1_200 + bank_num * 75 + branch_idx * 30 + (year - 2019) * 25,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic smoke-test fixture data.")
    parser.add_argument("--config", default="config/project.smoke.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = project_root()
    smoke_root = root / ".smoke"
    if smoke_root.exists():
        shutil.rmtree(smoke_root)

    raw_financials = ensure_dir(root / cfg["paths"]["raw"] / "fdic_financials")
    raw_sod = ensure_dir(root / cfg["paths"]["raw"] / "fdic_sod")
    interim_dir = ensure_dir(root / cfg["paths"]["interim"])
    ensure_dir(root / cfg["paths"]["tables"])
    ensure_dir(root / cfg["paths"]["figures"])
    ensure_dir(root / cfg["paths"]["logs"])
    ensure_dir(root / cfg["paths"]["frontend"])

    build_financial_rows().to_csv(raw_financials / "financials_smoke.csv", index=False)
    build_rate_rows().to_parquet(interim_dir / "rates_quarterly.parquet", index=False)
    for year in (2018, 2019, 2020, 2021):
        build_sod_rows(year).to_csv(raw_sod / f"sod_{year}.csv", index=False)

    print(f"prepared smoke fixture under {smoke_root}")


if __name__ == "__main__":
    main()
