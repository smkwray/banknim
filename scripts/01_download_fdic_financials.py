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

from nimscale.bank_panel import quarter_ends
from nimscale.fdic_api import FDICClient
from nimscale.io import ensure_dir, read_text_list
from nimscale.settings import load_config, project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FDIC financials quarter by quarter.")
    parser.add_argument("--start", required=True, help="Quarter start date, e.g. 2010-03-31")
    parser.add_argument("--end", required=True, help="Quarter end date, e.g. 2025-12-31")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--fields-file",
        default=str(project_root() / "config" / "fdic_financial_fields_core.txt"),
        help="Optional text file of FDIC fields to request.",
    )
    parser.add_argument("--all-fields", action="store_true", help="Request all fields instead of a narrow field list.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = ensure_dir(project_root() / cfg["paths"]["raw"] / "fdic_financials")
    log_dir = ensure_dir(project_root() / cfg["paths"]["logs"])

    client = FDICClient(
        base_url=cfg["fdic"]["base_url"],
        api_key_env=cfg["fdic"]["api_key_env"],
        pause_seconds=0.15,
    )

    dates = quarter_ends(args.start, args.end)
    fields = None if args.all_fields else read_text_list(args.fields_file)

    manifest = []

    for dt in dates:
        repdte = dt.strftime("%Y%m%d")
        filt = f'{cfg["fdic"]["default_filters"]["financials"]} AND REPDTE:{repdte}'
        params = {
            "filters": filt,
            "sort_by": "CERT",
            "sort_order": "ASC",
        }
        if fields:
            params["fields"] = ",".join(fields)

        print(f"downloading {repdte}")
        df = client.get_csv_paged(
            endpoint=cfg["fdic"]["endpoints"]["financials"],
            params=params,
            page_size=cfg["fdic"]["page_size"],
        )
        out_path = raw_dir / f"financials_{repdte}.csv"
        df.to_csv(out_path, index=False)
        manifest.append({"repdte": repdte, "rows": len(df), "path": str(out_path)})
        print(f"saved {out_path} rows={len(df):,}")

    pd.DataFrame(manifest).to_csv(log_dir / "fdic_financial_download_manifest.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
