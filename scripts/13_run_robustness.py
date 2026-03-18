from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nimscale.bank_panel import winsorize_series
from nimscale.io import ensure_dir
from nimscale.regression import fit_between_ols, fit_panel_fe, tidy_linearmodels, tidy_statsmodels
from nimscale.settings import load_config, project_root


def prep_panel(cfg: dict) -> pd.DataFrame:
    panel_path = project_root() / cfg["paths"]["interim"] / "bank_panel.parquet"
    df = pd.read_parquet(panel_path).copy()
    df["CERT"] = df["CERT"].astype(str)
    df["REPDTE"] = pd.to_datetime(df["REPDTE"])
    wp = cfg["project"]["winsor_pct"]
    df["NIM_W"] = winsorize_series(df["NIM"], p=wp)
    df["EQ_RATIO_W"] = winsorize_series(df["EQ_RATIO"], p=wp) if "EQ_RATIO" in df.columns else 0.0
    df["LOANS_SHARE_W"] = winsorize_series(df["LOANS_SHARE"], p=wp) if "LOANS_SHARE" in df.columns else 0.0
    df["ASSET_GROWTH_QOQ_W"] = winsorize_series(df["ASSET_GROWTH_QOQ"], p=wp) if "ASSET_GROWTH_QOQ" in df.columns else np.nan
    df["DEP_GROWTH_QOQ_W"] = winsorize_series(df["DEP_GROWTH_QOQ"], p=wp) if "DEP_GROWTH_QOQ" in df.columns else np.nan
    return df


# ── Summary statistics (Table 1) ──────────────────────────────────────────────

def make_summary_stats(df: pd.DataFrame, table_dir: Path) -> None:
    cols = {
        "NIM": "NIM (%)",
        "LN_ASSETS": "Log assets",
        "ASSET": "Total assets ($000s)",
        "DEP": "Total deposits ($000s)",
        "EQ_RATIO": "Equity / assets",
        "LOANS_SHARE": "Loans / assets",
        "ASSET_GROWTH_QOQ": "Asset growth (q/q)",
        "DEP_GROWTH_QOQ": "Deposit growth (q/q)",
        "FEDFUNDS": "Fed funds rate",
        "SLOPE_10Y_3M": "Yield slope (10Y-3M)",
    }
    present = {k: v for k, v in cols.items() if k in df.columns}
    stats = df[list(present.keys())].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    stats.index = [present[c] for c in stats.index]
    stats = stats[["count", "mean", "std", "1%", "25%", "50%", "75%", "99%"]]
    stats.to_csv(table_dir / "summary_statistics.csv", float_format="%.4f")
    print(f"  saved summary_statistics.csv ({len(stats)} variables)")


# ── QA distribution figures ───────────────────────────────────────────────────

def make_distribution_figures(df: pd.DataFrame, fig_dir: Path) -> None:
    # Assets distribution (log scale)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    assets = df["ASSET"].dropna()
    axes[0].hist(np.log10(assets[assets > 0]), bins=80, edgecolor="white", linewidth=0.3)
    axes[0].set_xlabel("Log₁₀ assets ($000s)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of bank assets")
    # NIM distribution
    nim = df["NIM"].dropna()
    axes[1].hist(nim, bins=80, edgecolor="white", linewidth=0.3, range=(nim.quantile(0.005), nim.quantile(0.995)))
    axes[1].set_xlabel("NIM (%)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distribution of NIM")
    plt.tight_layout()
    plt.savefig(fig_dir / "assets_dist.png", dpi=160)
    plt.savefig(fig_dir / "nim_dist.png", dpi=160)
    plt.close()

    # Asset growth vs deposit growth
    g = df.dropna(subset=["ASSET_GROWTH_QOQ", "DEP_GROWTH_QOQ"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hexbin(
        g["ASSET_GROWTH_QOQ"].clip(-0.5, 0.5),
        g["DEP_GROWTH_QOQ"].clip(-0.5, 0.5),
        gridsize=40, mincnt=1, cmap="Blues",
    )
    ax.plot([-0.5, 0.5], [-0.5, 0.5], "r--", linewidth=1, label="45° line")
    ax.set_xlabel("Asset growth (q/q)")
    ax.set_ylabel("Deposit growth (q/q)")
    ax.set_title("Asset growth vs deposit growth")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "growth_asset_vs_deposit.png", dpi=160)
    plt.close()
    print("  saved assets_dist.png, nim_dist.png, growth_asset_vs_deposit.png")


# ── Robustness: raw NIM (no winsorization) ────────────────────────────────────

def robustness_raw_nim(df: pd.DataFrame) -> list[pd.DataFrame]:
    panel_df = df.dropna(subset=["NIM", "LN_ASSETS"]).copy()
    results = []

    formula = "NIM ~ 1 + LN_ASSETS + EQ_RATIO_W + LOANS_SHARE_W + EntityEffects + TimeEffects"
    res = fit_panel_fe(panel_df, formula=formula, entity_col="CERT", time_col="REPDTE")
    results.append(tidy_linearmodels(res, "rob_raw_nim_fe_size"))
    print(f"  raw NIM FE: LN_ASSETS={res.params['LN_ASSETS']:.6f}, nobs={res.nobs}")
    return results


# ── Robustness: exclude banks < 12 quarters ───────────────────────────────────

def robustness_min_quarters(df: pd.DataFrame, min_q: int = 12) -> list[pd.DataFrame]:
    counts = df.groupby("CERT")["REPDTE"].nunique()
    keep = counts[counts >= min_q].index
    sub = df[df["CERT"].isin(keep)].copy()
    panel_df = sub.dropna(subset=["NIM_W", "LN_ASSETS"]).copy()

    results = []
    formula = "NIM_W ~ 1 + LN_ASSETS + EQ_RATIO_W + LOANS_SHARE_W + EntityEffects + TimeEffects"
    res = fit_panel_fe(panel_df, formula=formula, entity_col="CERT", time_col="REPDTE")
    results.append(tidy_linearmodels(res, f"rob_min{min_q}q_fe_size"))
    print(f"  min {min_q}q FE: LN_ASSETS={res.params['LN_ASSETS']:.6f}, nobs={res.nobs}, banks={panel_df['CERT'].nunique()}")
    return results


# ── Robustness: log deposits as alternative size measure ──────────────────────

def robustness_log_deposits(df: pd.DataFrame) -> list[pd.DataFrame]:
    if "LN_DEP" not in df.columns:
        print("  LN_DEP not available, skipping")
        return []
    panel_df = df.dropna(subset=["NIM_W", "LN_DEP"]).copy()
    results = []
    formula = "NIM_W ~ 1 + LN_DEP + EQ_RATIO_W + LOANS_SHARE_W + EntityEffects + TimeEffects"
    res = fit_panel_fe(panel_df, formula=formula, entity_col="CERT", time_col="REPDTE")
    results.append(tidy_linearmodels(res, "rob_log_deposits_fe"))
    print(f"  log deposits FE: LN_DEP={res.params['LN_DEP']:.6f}, nobs={res.nobs}")
    return results


# ── Robustness: first differences (removes bank-specific linear trends) ───────

def robustness_first_diff(df: pd.DataFrame) -> list[pd.DataFrame]:
    panel_df = df.dropna(subset=["NIM_W", "LN_ASSETS"]).copy()
    panel_df = panel_df.sort_values(["CERT", "REPDTE"])

    # First-difference within bank
    for col in ["NIM_W", "LN_ASSETS", "EQ_RATIO_W", "LOANS_SHARE_W"]:
        panel_df[f"D_{col}"] = panel_df.groupby("CERT")[col].diff()

    fd = panel_df.dropna(subset=["D_NIM_W", "D_LN_ASSETS"]).copy()

    results = []
    formula = "D_NIM_W ~ 1 + D_LN_ASSETS + D_EQ_RATIO_W + D_LOANS_SHARE_W + EntityEffects + TimeEffects"
    res = fit_panel_fe(fd, formula=formula, entity_col="CERT", time_col="REPDTE")
    results.append(tidy_linearmodels(res, "rob_first_diff_fe"))
    print(f"  first-diff FE: D_LN_ASSETS={res.params['D_LN_ASSETS']:.6f}, nobs={res.nobs}")
    return results


# ── Robustness: growth model with winsorized panel ────────────────────────────

def robustness_growth_winsorized(df: pd.DataFrame) -> list[pd.DataFrame]:
    panel_df = df.dropna(subset=["NIM_W", "ASSET_GROWTH_QOQ_W"]).copy()
    if "DEP_GROWTH_QOQ_W" in panel_df.columns:
        panel_df["DEP_GROWTH_QOQ_W"] = panel_df["DEP_GROWTH_QOQ_W"].fillna(0.0)
    else:
        panel_df["DEP_GROWTH_QOQ_W"] = 0.0

    results = []
    formula = (
        "NIM_W ~ 1 + ASSET_GROWTH_QOQ_W + DEP_GROWTH_QOQ_W + "
        "ASSET_GROWTH_QOQ_W:DEP_GROWTH_QOQ_W + EQ_RATIO_W + LOANS_SHARE_W + "
        "EntityEffects + TimeEffects"
    )
    res = fit_panel_fe(panel_df, formula=formula, entity_col="CERT", time_col="REPDTE")
    results.append(tidy_linearmodels(res, "rob_growth_winsorized"))
    print(f"  growth winsorized FE: ASSET_GROWTH={res.params['ASSET_GROWTH_QOQ_W']:.6f}, nobs={res.nobs}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness checks for baseline models.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    table_dir = ensure_dir(project_root() / cfg["paths"]["tables"])
    fig_dir = ensure_dir(project_root() / cfg["paths"]["figures"])

    df = prep_panel(cfg)

    print("Summary statistics:")
    make_summary_stats(df, table_dir)

    print("Distribution figures:")
    make_distribution_figures(df, fig_dir)

    print("Robustness checks:")
    all_results = []

    print("  [1/5] Raw NIM (no winsorization)...")
    all_results.extend(robustness_raw_nim(df))

    print("  [2/5] Exclude banks < 12 quarters...")
    all_results.extend(robustness_min_quarters(df, min_q=12))

    print("  [3/5] Log deposits as size measure...")
    all_results.extend(robustness_log_deposits(df))

    print("  [4/5] First differences (bank-specific trends)...")
    all_results.extend(robustness_first_diff(df))

    print("  [5/5] Growth model winsorized...")
    all_results.extend(robustness_growth_winsorized(df))

    out = pd.concat(all_results, ignore_index=True)
    out.to_csv(table_dir / "robustness_results.csv", index=False)
    print(f"\nsaved {table_dir / 'robustness_results.csv'} ({len(out)} rows, {out['model'].nunique()} models)")


if __name__ == "__main__":
    main()
