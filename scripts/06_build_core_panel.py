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

from nimscale.bank_panel import (
    add_core_features,
    coerce_numeric,
    infer_main_columns,
    merge_rates,
    parse_fdic_quarter_date,
    standardize_columns,
)
from nimscale.io import ensure_dir
from nimscale.settings import load_config, project_root
from nimscale.validation import ValidationError, assert_merge_coverage, assert_nonempty_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the core bank-quarter panel.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = project_root() / cfg["paths"]["raw"] / "fdic_financials"
    interim_dir = ensure_dir(project_root() / cfg["paths"]["interim"])
    table_dir = ensure_dir(project_root() / cfg["paths"]["tables"])

    files = sorted(raw_dir.glob("financials_*.csv"))
    if not files:
        raise FileNotFoundError(f"No financial files found in {raw_dir}")

    frames = []
    for path in files:
        df = pd.read_csv(path, low_memory=False)
        df = standardize_columns(df)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df = coerce_numeric(df, exclude=["NAME", "CITY", "STALP", "REGAGNT", "BKCLASS"])

    cols = infer_main_columns(df, cfg)
    df[cols["date"]] = parse_fdic_quarter_date(df[cols["date"]])
    required_cols = [cols["bank_id"], cols["date"], cols["assets"]]
    missing_required = {
        col: int(df[col].isna().sum())
        for col in required_cols
        if int(df[col].isna().sum()) > 0
    }
    if missing_required:
        raise ValidationError(f"core panel build: required parsed fields contain missing values {missing_required}")

    df = df.sort_values([cols["bank_id"], cols["date"]])

    dupes = df[df.duplicated(subset=[cols["bank_id"], cols["date"]], keep=False)].copy()
    duplicate_key_count = len(dupes[[cols["bank_id"], cols["date"]]].drop_duplicates()) if not dupes.empty else 0
    dupes.to_csv(table_dir / "duplicate_keys.csv", index=False)

    before = len(df)
    df = df.drop_duplicates(subset=[cols["bank_id"], cols["date"]], keep="first").copy()
    after = len(df)

    panel = add_core_features(
        df=df,
        bank_id_col=cols["bank_id"],
        date_col=cols["date"],
        assets_col=cols["assets"],
        deposits_col=cols.get("deposits"),
        equity_col=cols.get("equity"),
        loans_col=cols.get("loans"),
        nim_col=cols.get("nim"),
    )

    rates_path = project_root() / cfg["paths"]["interim"] / "rates_quarterly.parquet"
    if rates_path.exists():
        rates = pd.read_parquet(rates_path)
        panel = merge_rates(panel, rates, date_col=cols["date"])
        assert_merge_coverage(panel, ["FEDFUNDS", "SLOPE_10Y_3M"], "core panel rate merge")

    assert_nonempty_sample(panel, "core panel build", min_rows=1, entity_col=cols["bank_id"], min_entities=1)

    out_path = interim_dir / "bank_panel.parquet"
    panel.to_parquet(out_path, index=False)

    summary = pd.DataFrame(
        [
            {"metric": "rows_before_dedup", "value": before},
            {"metric": "rows_after_dedup", "value": after},
            {"metric": "duplicate_row_count", "value": len(dupes)},
            {"metric": "duplicate_key_count", "value": duplicate_key_count},
            {"metric": "unique_banks", "value": panel[cols["bank_id"]].nunique()},
            {"metric": "unique_quarters", "value": panel[cols["date"]].nunique()},
            {"metric": "min_date", "value": str(panel[cols["date"]].min().date())},
            {"metric": "max_date", "value": str(panel[cols["date"]].max().date())},
        ]
    )
    summary.to_csv(table_dir / "core_sample_summary.csv", index=False)

    missing = panel.isna().mean().reset_index()
    missing.columns = ["column", "missing_share"]
    missing.to_csv(table_dir / "missingness_report.csv", index=False)

    print(f"saved {out_path}")
    print(f"banks={panel[cols['bank_id']].nunique():,} rows={len(panel):,}")


if __name__ == "__main__":
    main()
