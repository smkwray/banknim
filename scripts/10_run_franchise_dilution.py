from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

import pandas as pd

from nimscale.bank_panel import add_sod_franchise_features, winsorize_series
from nimscale.io import ensure_dir
from nimscale.regression import fit_panel_fe, tidy_linearmodels
from nimscale.settings import load_config, project_root


def prepare_panel(df: pd.DataFrame, winsor_pct: float) -> pd.DataFrame:
    out = df.copy()
    out["CERT"] = out["CERT"].astype(str)
    out["REPDTE"] = pd.to_datetime(out["REPDTE"])

    out = add_sod_franchise_features(out, bank_id_col="CERT", sod_year_col="SOD_YEAR")

    for col in ["NIM", "ASSET_GROWTH_QOQ", "EQ_RATIO", "LOANS_SHARE"]:
        if col not in out.columns:
            raise KeyError(f"Missing required column: {col}")

    out["NIM_W"] = winsorize_series(pd.to_numeric(out["NIM"], errors="coerce"), p=winsor_pct)
    out["ASSET_GROWTH_QOQ_W"] = winsorize_series(pd.to_numeric(out["ASSET_GROWTH_QOQ"], errors="coerce"), p=winsor_pct)
    out["EQ_RATIO_W"] = winsorize_series(pd.to_numeric(out["EQ_RATIO"], errors="coerce"), p=winsor_pct)
    out["LOANS_SHARE_W"] = winsorize_series(pd.to_numeric(out["LOANS_SHARE"], errors="coerce"), p=winsor_pct)

    for col in ["BRANCH_COUNT_GROWTH_YOY", "SOD_DEP_TOTAL_GROWTH_YOY", "DEP_PER_BRANCH_GROWTH_YOY"]:
        out[f"{col}_W"] = winsorize_series(pd.to_numeric(out[col], errors="coerce"), p=winsor_pct)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SOD-based franchise-dilution regressions.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    panel_path = project_root() / cfg["paths"]["interim"] / "bank_panel_sod.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(panel_path)

    table_dir = ensure_dir(project_root() / cfg["paths"]["tables"])

    df = pd.read_parquet(panel_path)
    df = prepare_panel(df, winsor_pct=cfg["project"]["winsor_pct"])

    formulas = {
        "franchise_branch_fe": (
            "NIM_W ~ 1 + ASSET_GROWTH_QOQ_W + BRANCH_COUNT_GROWTH_YOY_W + "
            "ASSET_GROWTH_QOQ_W:BRANCH_COUNT_GROWTH_YOY_W + EQ_RATIO_W + LOANS_SHARE_W + "
            "EntityEffects + TimeEffects"
        ),
        "franchise_sod_dep_fe": (
            "NIM_W ~ 1 + ASSET_GROWTH_QOQ_W + SOD_DEP_TOTAL_GROWTH_YOY_W + "
            "ASSET_GROWTH_QOQ_W:SOD_DEP_TOTAL_GROWTH_YOY_W + EQ_RATIO_W + LOANS_SHARE_W + "
            "EntityEffects + TimeEffects"
        ),
        "franchise_dep_per_branch_fe": (
            "NIM_W ~ 1 + ASSET_GROWTH_QOQ_W + DEP_PER_BRANCH_GROWTH_YOY_W + "
            "ASSET_GROWTH_QOQ_W:DEP_PER_BRANCH_GROWTH_YOY_W + EQ_RATIO_W + LOANS_SHARE_W + "
            "EntityEffects + TimeEffects"
        ),
    }

    results = []
    summary_rows = []
    for model_name, formula in formulas.items():
        res = fit_panel_fe(df, formula=formula, entity_col="CERT", time_col="REPDTE")
        results.append(tidy_linearmodels(res, model_name))
        summary_rows.append(
            {
                "model": model_name,
                "nobs": float(res.nobs),
                "r2_within": getattr(res, "rsquared_within", float("nan")),
                "r2_between": getattr(res, "rsquared_between", float("nan")),
                "r2_overall": getattr(res, "rsquared_overall", float("nan")),
            }
        )

    pd.concat(results, ignore_index=True).to_csv(table_dir / "franchise_dilution_results.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(table_dir / "franchise_dilution_model_summary.csv", index=False)
    print(f"saved {table_dir / 'franchise_dilution_results.csv'}")


if __name__ == "__main__":
    main()
