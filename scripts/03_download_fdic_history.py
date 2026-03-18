from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

import pandas as pd

from nimscale.fdic_api import FDICClient
from nimscale.io import ensure_dir
from nimscale.settings import load_config, project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FDIC history / structure-change data and institutions roster.")
    parser.add_argument("--start-date", required=True, help="e.g. 2010-01-01")
    parser.add_argument("--end-date", required=True, help="e.g. 2025-12-31")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = ensure_dir(project_root() / cfg["paths"]["raw"] / "fdic_history")
    log_dir = ensure_dir(project_root() / cfg["paths"]["logs"])

    client = FDICClient(
        base_url=cfg["fdic"]["base_url"],
        api_key_env=cfg["fdic"]["api_key_env"],
        pause_seconds=0.15,
    )

    manifest = []

    history_filter = f'EFFDATE:[{args.start_date} TO {args.end_date}]'
    for endpoint, params, fname in [
        (
            cfg["fdic"]["endpoints"]["history"],
            {"filters": history_filter, "sort_by": "EFFDATE", "sort_order": "ASC"},
            "history_events.csv",
        ),
        (
            cfg["fdic"]["endpoints"]["institutions"],
            {"sort_by": "CERT", "sort_order": "ASC"},
            "institutions_current.csv",
        ),
    ]:
        try:
            df = client.get_csv_paged(endpoint=endpoint, params=params, page_size=cfg["fdic"]["page_size"])
        except Exception as exc:
            print(f"warning: failed endpoint {endpoint}: {exc}")
            continue

        out_path = raw_dir / fname
        df.to_csv(out_path, index=False)
        manifest.append({"endpoint": endpoint, "rows": len(df), "path": str(out_path)})
        print(f"saved {out_path} rows={len(df):,}")

    if manifest:
        pd.DataFrame(manifest).to_csv(log_dir / "fdic_history_download_manifest.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
