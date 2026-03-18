from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from nimscale.bank_panel import winsorize_series
from nimscale.io import ensure_dir
from nimscale.regression import fit_between_ols, fit_panel_fe, tidy_linearmodels, tidy_statsmodels
from nimscale.settings import load_config, project_root


def make_size_quantile_figure(df: pd.DataFrame, fig_dir: Path) -> None:
    plot_df = df.dropna(subset=["NIM", "LN_ASSETS"]).copy()
    plot_df["SIZE_Q"] = pd.qcut(plot_df["LN_ASSETS"], 10, labels=False, duplicates="drop") + 1
    agg = plot_df.groupby("SIZE_Q", as_index=False)["NIM"].mean()

    plt.figure(figsize=(7, 4))
    plt.plot(agg["SIZE_Q"], agg["NIM"], marker="o")
    plt.xlabel("Asset size decile")
    plt.ylabel("Average NIM")
    plt.title("Average NIM by asset size decile")
    plt.tight_layout()
    plt.savefig(fig_dir / "nim_by_size_quantile.png", dpi=160)
    plt.close()


def make_binscatter_figure(df: pd.DataFrame, fig_dir: Path) -> None:
    plot_df = (
        df.groupby("CERT", as_index=False)[["NIM", "LN_ASSETS"]]
        .mean(numeric_only=True)
        .dropna()
        .sort_values("LN_ASSETS")
    )
    plot_df["BIN"] = pd.qcut(plot_df["LN_ASSETS"], 20, labels=False, duplicates="drop") + 1
    agg = plot_df.groupby("BIN", as_index=False)[["NIM", "LN_ASSETS"]].mean()

    plt.figure(figsize=(7, 4))
    plt.scatter(agg["LN_ASSETS"], agg["NIM"])
    plt.xlabel("Average log assets")
    plt.ylabel("Average NIM")
    plt.title("Average NIM vs average size (20 bins)")
    plt.tight_layout()
    plt.savefig(fig_dir / "nim_size_bins.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline NIM regressions.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    panel_path = project_root() / cfg["paths"]["interim"] / "bank_panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(panel_path)

    table_dir = ensure_dir(project_root() / cfg["paths"]["tables"])
    fig_dir = ensure_dir(project_root() / cfg["paths"]["figures"])

    df = pd.read_parquet(panel_path).copy()
    df["CERT"] = df["CERT"].astype(str)
    df["REPDTE"] = pd.to_datetime(df["REPDTE"])

    # Minimal required columns
    need = ["CERT", "REPDTE", "NIM", "LN_ASSETS"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for baseline models: {missing}")

    if "EQ_RATIO" in df.columns:
        df["EQ_RATIO_W"] = winsorize_series(df["EQ_RATIO"], p=cfg["project"]["winsor_pct"])
    else:
        df["EQ_RATIO_W"] = 0.0

    if "LOANS_SHARE" in df.columns:
        df["LOANS_SHARE_W"] = winsorize_series(df["LOANS_SHARE"], p=cfg["project"]["winsor_pct"])
    else:
        df["LOANS_SHARE_W"] = 0.0

    if "ASSET_GROWTH_QOQ" in df.columns:
        df["ASSET_GROWTH_QOQ_W"] = winsorize_series(df["ASSET_GROWTH_QOQ"], p=cfg["project"]["winsor_pct"])
    else:
        df["ASSET_GROWTH_QOQ_W"] = pd.NA

    if "DEP_GROWTH_QOQ" in df.columns:
        df["DEP_GROWTH_QOQ_W"] = winsorize_series(df["DEP_GROWTH_QOQ"], p=cfg["project"]["winsor_pct"])
    else:
        df["DEP_GROWTH_QOQ_W"] = pd.NA

    df["NIM_W"] = winsorize_series(df["NIM"], p=cfg["project"]["winsor_pct"])

    if "FEDFUNDS" not in df.columns:
        df["FEDFUNDS"] = 0.0
    if "SLOPE_10Y_3M" not in df.columns:
        df["SLOPE_10Y_3M"] = 0.0

    panel_df = df.dropna(subset=["NIM_W", "LN_ASSETS"]).copy()
    between_df = (
        panel_df.groupby("CERT", as_index=False)[
            ["NIM_W", "LN_ASSETS", "EQ_RATIO_W", "LOANS_SHARE_W", "FEDFUNDS", "SLOPE_10Y_3M"]
        ]
        .mean(numeric_only=True)
        .rename(
            columns={
                "NIM_W": "AVG_NIM",
                "LN_ASSETS": "AVG_LN_ASSETS",
                "EQ_RATIO_W": "AVG_EQ_RATIO",
                "LOANS_SHARE_W": "AVG_LOANS_SHARE",
                "FEDFUNDS": "AVG_FEDFUNDS",
                "SLOPE_10Y_3M": "AVG_SLOPE_10Y_3M",
            }
        )
    )

    results = []

    # Rate main effects are absorbed by quarter FE — drop them from FE specs.
    fe_formula = "NIM_W ~ 1 + LN_ASSETS + EQ_RATIO_W + LOANS_SHARE_W + EntityEffects + TimeEffects"
    fe_res = fit_panel_fe(panel_df, formula=fe_formula, entity_col="CERT", time_col="REPDTE")
    results.append(tidy_linearmodels(fe_res, "within_fe_size"))

    between_formula = "AVG_NIM ~ AVG_LN_ASSETS + AVG_EQ_RATIO + AVG_LOANS_SHARE"
    between_res = fit_between_ols(between_df, formula=between_formula)
    results.append(tidy_statsmodels(between_res, "between_bank_means"))

    growth_df = panel_df.dropna(subset=["ASSET_GROWTH_QOQ_W"]).copy()
    if "DEP_GROWTH_QOQ_W" in growth_df.columns:
        growth_df["DEP_GROWTH_QOQ_W"] = growth_df["DEP_GROWTH_QOQ_W"].fillna(0.0)
    else:
        growth_df["DEP_GROWTH_QOQ_W"] = 0.0

    growth_formula = (
        "NIM_W ~ 1 + ASSET_GROWTH_QOQ_W + DEP_GROWTH_QOQ_W + "
        "ASSET_GROWTH_QOQ_W:DEP_GROWTH_QOQ_W + EQ_RATIO_W + LOANS_SHARE_W + "
        "EntityEffects + TimeEffects"
    )
    growth_res = fit_panel_fe(growth_df, formula=growth_formula, entity_col="CERT", time_col="REPDTE")
    results.append(tidy_linearmodels(growth_res, "growth_fe"))

    out = pd.concat(results, ignore_index=True)
    out.to_csv(table_dir / "regression_results.csv", index=False)

    make_size_quantile_figure(panel_df, fig_dir)
    make_binscatter_figure(panel_df, fig_dir)

    print(f"saved {table_dir / 'regression_results.csv'}")


if __name__ == "__main__":
    main()
