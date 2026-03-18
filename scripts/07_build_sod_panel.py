from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nimscale.bank_panel import map_sod_year_to_quarter, pick_first_existing, standardize_columns
from nimscale.io import ensure_dir
from nimscale.settings import load_config, project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate SOD and merge to the bank-quarter panel.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    sod_dir = project_root() / cfg["paths"]["raw"] / "fdic_sod"
    panel_path = project_root() / cfg["paths"]["interim"] / "bank_panel.parquet"
    out_dir = ensure_dir(project_root() / cfg["paths"]["interim"])

    if not panel_path.exists():
        raise FileNotFoundError(panel_path)

    files = sorted(sod_dir.glob("sod_*.csv"))
    if not files:
        raise FileNotFoundError(f"No SOD files found in {sod_dir}")

    frames = []
    cc = cfg["fdic"]["column_candidates"]
    for path in files:
        df = pd.read_csv(path, low_memory=False)
        df = standardize_columns(df)
        cert_col = pick_first_existing(df, cc["bank_id"])
        year_col = pick_first_existing(df, cc["sod_year"])
        dep_col = pick_first_existing(df, cc["sod_deposits"])

        branch_id_col = None
        for candidate in cc["sod_branch_id"]:
            if candidate.upper() in df.columns:
                branch_id_col = candidate.upper()
                break

        agg = (
            df.groupby([cert_col, year_col], dropna=False)
            .agg(
                BRANCH_COUNT=(branch_id_col or dep_col, "nunique" if branch_id_col else "size"),
                SOD_DEP_TOTAL=(dep_col, "sum"),
            )
            .reset_index()
        )
        agg.columns = ["CERT", "SOD_YEAR", "BRANCH_COUNT", "SOD_DEP_TOTAL"]
        frames.append(agg)

    sod = pd.concat(frames, ignore_index=True)
    sod = sod.groupby(["CERT", "SOD_YEAR"], as_index=False).agg(
        BRANCH_COUNT=("BRANCH_COUNT", "max"),
        SOD_DEP_TOTAL=("SOD_DEP_TOTAL", "sum"),
    )
    sod["DEP_PER_BRANCH"] = np.where(sod["BRANCH_COUNT"] > 0, sod["SOD_DEP_TOTAL"] / sod["BRANCH_COUNT"], np.nan)

    panel = pd.read_parquet(panel_path)
    panel["SOD_YEAR"] = map_sod_year_to_quarter(panel["REPDTE"])
    panel["CERT"] = panel["CERT"].astype(str)
    sod["CERT"] = sod["CERT"].astype(str)

    merged = panel.merge(sod, on=["CERT", "SOD_YEAR"], how="left")
    out_path = out_dir / "bank_panel_sod.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
