from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
from pathlib import Path

import pandas as pd

from nimscale.fdic_api import FDICClient
from nimscale.io import ensure_dir
from nimscale.settings import load_config, project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Download annual FDIC Summary of Deposits.")
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--all-fields", action="store_true", help="SOD defaults to all fields if no custom field list is used.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = ensure_dir(project_root() / cfg["paths"]["raw"] / "fdic_sod")
    log_dir = ensure_dir(project_root() / cfg["paths"]["logs"])

    client = FDICClient(
        base_url=cfg["fdic"]["base_url"],
        api_key_env=cfg["fdic"]["api_key_env"],
        pause_seconds=0.15,
    )

    manifest = []

    for year in range(args.start_year, args.end_year + 1):
        filt = f"YEAR:{year}"
        params = {
            "filters": filt,
            "sort_by": "CERT",
            "sort_order": "ASC",
        }

        print(f"downloading SOD {year}")
        try:
            df = client.get_csv_paged(
                endpoint=cfg["fdic"]["endpoints"]["sod"],
                params=params,
                page_size=cfg["fdic"]["page_size"],
            )
        except Exception as exc:
            raise RuntimeError(f"SOD API failed for {year}; aborting scripted pipeline") from exc

        out_path = raw_dir / f"sod_{year}.csv"
        df.to_csv(out_path, index=False)
        manifest.append({"year": year, "rows": len(df), "path": str(out_path)})
        print(f"saved {out_path} rows={len(df):,}")

    if manifest:
        pd.DataFrame(manifest).to_csv(log_dir / "fdic_sod_download_manifest.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
