from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import shutil
from pathlib import Path

import pandas as pd

from nimscale.io import ensure_dir
from nimscale.settings import load_config, project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage a manually downloaded FFIEC bulk file.")
    parser.add_argument("--input", required=True, help="Path to a manually downloaded FFIEC tab-delimited or csv file.")
    parser.add_argument("--name", default="ffiec_bulk_staged")
    parser.add_argument("--config", default=None)
    parser.add_argument("--to-parquet", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = ensure_dir(project_root() / cfg["paths"]["raw"] / "ffiec")
    interim_dir = ensure_dir(project_root() / cfg["paths"]["interim"] / "ffiec")

    src = Path(args.input)
    if not src.exists():
        raise FileNotFoundError(src)

    dst = raw_dir / src.name
    shutil.copy2(src, dst)
    print(f"copied to {dst}")

    if args.to_parquet:
        sep = "\t" if src.suffix.lower() in {".txt", ".tsv"} else ","
        df = pd.read_csv(dst, sep=sep, low_memory=False)
        out_path = interim_dir / f"{args.name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
