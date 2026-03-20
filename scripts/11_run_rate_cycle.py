from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

import pandas as pd

from nimscale.io import ensure_dir
from nimscale.regression import fit_panel_fe, tidy_linearmodels
from nimscale.settings import load_config, project_root
from nimscale.validation import (
    assert_merge_coverage,
    assert_nonempty_sample,
    require_columns,
    winsorize_required,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="M5: Rate-cycle heterogeneity regressions.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    panel_path = project_root() / cfg["paths"]["interim"] / "bank_panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(panel_path)

    table_dir = ensure_dir(project_root() / cfg["paths"]["tables"])

    df = pd.read_parquet(panel_path).copy()
    df["CERT"] = df["CERT"].astype(str)
    df["REPDTE"] = pd.to_datetime(df["REPDTE"])

    require_columns(
        df,
        ["CERT", "REPDTE", "NIM", "LN_ASSETS", "EQ_RATIO", "LOANS_SHARE", "FEDFUNDS", "SLOPE_10Y_3M"],
        "rate-cycle models",
    )
    wp = cfg["project"]["winsor_pct"]
    df["NIM_W"] = winsorize_required(df, "NIM", p=wp, context="rate-cycle models")
    df["EQ_RATIO_W"] = winsorize_required(df, "EQ_RATIO", p=wp, context="rate-cycle models")
    df["LOANS_SHARE_W"] = winsorize_required(df, "LOANS_SHARE", p=wp, context="rate-cycle models")

    # Rate main effects are absorbed by quarter FE, but their
    # interactions with bank-varying variables are identified.
    assert_merge_coverage(df, ["FEDFUNDS", "SLOPE_10Y_3M"], "rate-cycle models")

    panel_df = df.dropna(subset=["NIM_W", "LN_ASSETS", "FEDFUNDS", "SLOPE_10Y_3M"]).copy()
    assert_nonempty_sample(panel_df, "rate-cycle sample", min_rows=1, entity_col="CERT", min_entities=2)

    # Build interactions
    panel_df["LN_ASSETS_x_FEDFUNDS"] = panel_df["LN_ASSETS"] * panel_df["FEDFUNDS"]
    panel_df["LN_ASSETS_x_SLOPE"] = panel_df["LN_ASSETS"] * panel_df["SLOPE_10Y_3M"]

    print(f"M5 panel: {len(panel_df):,} obs, {panel_df['CERT'].nunique():,} banks")

    results = []

    # M5a: size * fedfunds
    f5a = (
        "NIM_W ~ 1 + LN_ASSETS + LN_ASSETS_x_FEDFUNDS + "
        "EQ_RATIO_W + LOANS_SHARE_W + EntityEffects + TimeEffects"
    )
    res5a = fit_panel_fe(panel_df, formula=f5a, entity_col="CERT", time_col="REPDTE")
    results.append(tidy_linearmodels(res5a, "rate_cycle_fedfunds"))

    # M5b: size * slope
    f5b = (
        "NIM_W ~ 1 + LN_ASSETS + LN_ASSETS_x_SLOPE + "
        "EQ_RATIO_W + LOANS_SHARE_W + EntityEffects + TimeEffects"
    )
    res5b = fit_panel_fe(panel_df, formula=f5b, entity_col="CERT", time_col="REPDTE")
    results.append(tidy_linearmodels(res5b, "rate_cycle_slope"))

    # M5c: both interactions
    f5c = (
        "NIM_W ~ 1 + LN_ASSETS + LN_ASSETS_x_FEDFUNDS + LN_ASSETS_x_SLOPE + "
        "EQ_RATIO_W + LOANS_SHARE_W + EntityEffects + TimeEffects"
    )
    res5c = fit_panel_fe(panel_df, formula=f5c, entity_col="CERT", time_col="REPDTE")
    results.append(tidy_linearmodels(res5c, "rate_cycle_both"))

    out = pd.concat(results, ignore_index=True)
    assert_nonempty_sample(out, "rate-cycle result export")
    out.to_csv(table_dir / "rate_cycle_results.csv", index=False)
    print(f"saved {table_dir / 'rate_cycle_results.csv'}")

    # Print key coefficients
    for model in out["model"].unique():
        sub = out[out["model"] == model]
        interaction_rows = sub[sub["term"].str.startswith("LN_ASSETS_x_")]
        for _, row in interaction_rows.iterrows():
            sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
            print(f"  {model} | {row['term']}: {row['coef']:.6f} (se={row['std_err']:.6f}) {sig}")


if __name__ == "__main__":
    main()
