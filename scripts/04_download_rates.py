from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from nimscale.io import ensure_dir
from nimscale.settings import load_config, project_root


def to_quarterly(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    out = df.copy()
    out.columns = ["DATE", value_name]
    out["DATE"] = pd.to_datetime(out["DATE"], errors="coerce")
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
    out = out.dropna(subset=["DATE"])
    out["REPDTE"] = out["DATE"].dt.to_period("Q").dt.end_time.dt.normalize()
    return out.groupby("REPDTE", as_index=False)[value_name].mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download macro rate series and build a quarterly file.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = ensure_dir(project_root() / cfg["paths"]["raw"] / "rates")
    interim_dir = ensure_dir(project_root() / cfg["paths"]["interim"])

    frames = []
    for series_name, url in cfg["rates"]["series"].items():
        print(f"downloading {series_name}")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        raw_path = raw_dir / f"{series_name}.csv"
        raw_path.write_bytes(resp.content)

        df = pd.read_csv(StringIO(resp.text))
        q = to_quarterly(df, series_name)
        frames.append(q)

    merged = None
    for q in frames:
        merged = q if merged is None else merged.merge(q, on="REPDTE", how="outer")

    merged = merged.sort_values("REPDTE").reset_index(drop=True)
    if {"DGS10", "DGS3MO"}.issubset(set(merged.columns)):
        merged["SLOPE_10Y_3M"] = merged["DGS10"] - merged["DGS3MO"]
    if {"DGS10", "DGS2"}.issubset(set(merged.columns)):
        merged["SLOPE_10Y_2Y"] = merged["DGS10"] - merged["DGS2"]

    out_path = interim_dir / "rates_quarterly.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
