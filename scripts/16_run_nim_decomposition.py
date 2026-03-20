from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

import pandas as pd

from nimscale.bank_panel import winsorize_series
from nimscale.io import ensure_dir
from nimscale.regression import fit_panel_fe, tidy_linearmodels
from nimscale.settings import load_config, project_root
from nimscale.validation import assert_nonempty_sample, require_columns, winsorize_required


def prep_panel(cfg: dict) -> pd.DataFrame:
    panel_path = project_root() / cfg["paths"]["interim"] / "bank_panel.parquet"
    df = pd.read_parquet(panel_path).copy()
    df["CERT"] = df["CERT"].astype(str)
    df["REPDTE"] = pd.to_datetime(df["REPDTE"])
    wp = cfg["project"]["winsor_pct"]
    require_columns(
        df,
        ["EQ_RATIO", "LOANS_SHARE", "NIMY", "INTINCY", "INTEXPY", "ASSET", "LN_ASSETS"],
        "nim decomposition prep",
    )

    df["EQ_RATIO_W"] = winsorize_required(df, "EQ_RATIO", p=wp, context="nim decomposition prep")
    df["LOANS_SHARE_W"] = winsorize_required(df, "LOANS_SHARE", p=wp, context="nim decomposition prep")

    # FDIC financials sometimes expose interest income/expense already scaled
    # to average assets. Detect that case using the NIM identity to avoid
    # dividing a ratio by assets a second time.
    nim_identity_gap = (df["NIMY"] - (df["INTINCY"] - df["INTEXPY"])).abs().median()
    asset_scaled_gap = (df["NIMY"] - ((df["INTINCY"] - df["INTEXPY"]) / df["ASSET"])).abs().median()
    use_pre_scaled_components = pd.notna(nim_identity_gap) and nim_identity_gap <= asset_scaled_gap

    if use_pre_scaled_components:
        df["INT_INCOME_RATIO"] = pd.to_numeric(df["INTINCY"], errors="coerce")
        df["INT_EXPENSE_RATIO"] = pd.to_numeric(df["INTEXPY"], errors="coerce")
    else:
        df["INT_INCOME_RATIO"] = pd.to_numeric(df["INTINCY"], errors="coerce") / pd.to_numeric(df["ASSET"], errors="coerce")
        df["INT_EXPENSE_RATIO"] = pd.to_numeric(df["INTEXPY"], errors="coerce") / pd.to_numeric(df["ASSET"], errors="coerce")

    df["INT_INCOME_RATIO_W"] = winsorize_series(df["INT_INCOME_RATIO"], p=wp)
    df["INT_EXPENSE_RATIO_W"] = winsorize_series(df["INT_EXPENSE_RATIO"], p=wp)
    return df


def run_model(df: pd.DataFrame, dep_var: str, model_name: str) -> pd.DataFrame:
    sub = df.dropna(subset=[dep_var, "LN_ASSETS", "EQ_RATIO_W", "LOANS_SHARE_W"]).copy()
    assert_nonempty_sample(sub, model_name, min_rows=1, entity_col="CERT", min_entities=2)
    formula = f"{dep_var} ~ 1 + LN_ASSETS + EQ_RATIO_W + LOANS_SHARE_W + EntityEffects + TimeEffects"
    res = fit_panel_fe(sub, formula=formula, entity_col="CERT", time_col="REPDTE")
    out = tidy_linearmodels(res, model_name)
    print(
        f"{model_name}: LN_ASSETS={res.params['LN_ASSETS']:.6f} "
        f"(p={res.pvalues['LN_ASSETS']:.4f}), nobs={res.nobs}"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NIM decomposition fixed-effects models.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    table_dir = ensure_dir(project_root() / cfg["paths"]["tables"])
    df = prep_panel(cfg)

    results = [
        run_model(df, "INT_INCOME_RATIO_W", "int_income_fe"),
        run_model(df, "INT_EXPENSE_RATIO_W", "int_expense_fe"),
    ]
    out = pd.concat(results, ignore_index=True)
    assert_nonempty_sample(out, "nim decomposition result export")
    out.to_csv(table_dir / "nim_decomposition_results.csv", index=False)
    print(f"saved {table_dir / 'nim_decomposition_results.csv'}")


if __name__ == "__main__":
    main()
