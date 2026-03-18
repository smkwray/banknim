from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import glob as globmod

import numpy as np
import pandas as pd

from nimscale.bank_panel import map_sod_year_to_quarter, winsorize_series
from nimscale.io import ensure_dir
from nimscale.regression import fit_panel_fe, tidy_linearmodels
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
    return df


# ── 1. NIM volatility ─────────────────────────────────────────────────────────

def run_nim_volatility(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Rolling 8-quarter SD of NIM within bank, then regress on size."""
    print("\n=== NIM Volatility ===")
    panel = df.sort_values(["CERT", "REPDTE"]).copy()
    panel["NIM_VOL_8Q"] = (
        panel.groupby("CERT")["NIM"]
        .transform(lambda x: x.rolling(8, min_periods=6).std())
    )
    panel["NIM_VOL_8Q_W"] = winsorize_series(panel["NIM_VOL_8Q"], p=0.01)
    sub = panel.dropna(subset=["NIM_VOL_8Q_W", "LN_ASSETS"]).copy()
    print(f"  obs with nim_vol_8q: {len(sub):,}, banks: {sub['CERT'].nunique():,}")
    print(f"  nim_vol_8q mean={sub['NIM_VOL_8Q_W'].mean():.4f}, median={sub['NIM_VOL_8Q_W'].median():.4f}")

    results = []
    # Level: does size predict NIM volatility?
    formula = "NIM_VOL_8Q_W ~ 1 + LN_ASSETS + EQ_RATIO_W + LOANS_SHARE_W + EntityEffects + TimeEffects"
    res = fit_panel_fe(sub, formula=formula, entity_col="CERT", time_col="REPDTE")
    results.append(tidy_linearmodels(res, "nim_vol_fe_size"))
    print(f"  FE size->vol: LN_ASSETS={res.params['LN_ASSETS']:.6f} (p={res.pvalues['LN_ASSETS']:.4f})")

    return results


# ── 2. Fee income / expense offset (H6) ──────────────────────────────────────

def run_h6_fee_offset(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Test whether large banks offset lower NIM with better ROA or different
    interest income/expense composition."""
    print("\n=== H6: Fee / Expense Offset ===")
    results = []
    wp = 0.01

    # ROA: does size predict ROA the same way it predicts NIM?
    if "ROA" in df.columns:
        df["ROA_W"] = winsorize_series(df["ROA"], p=wp)
        sub = df.dropna(subset=["ROA_W", "LN_ASSETS"]).copy()
        formula = "ROA_W ~ 1 + LN_ASSETS + EQ_RATIO_W + LOANS_SHARE_W + EntityEffects + TimeEffects"
        res = fit_panel_fe(sub, formula=formula, entity_col="CERT", time_col="REPDTE")
        results.append(tidy_linearmodels(res, "h6_roa_fe"))
        print(f"  ROA FE: LN_ASSETS={res.params['LN_ASSETS']:.6f} (p={res.pvalues['LN_ASSETS']:.4f}), nobs={res.nobs}")

    # Interest expense: do large banks pay more for funding?
    if "INTEXPY" in df.columns:
        df["INTEXPY_W"] = winsorize_series(df["INTEXPY"], p=wp)
        sub = df.dropna(subset=["INTEXPY_W", "LN_ASSETS"]).copy()
        formula = "INTEXPY_W ~ 1 + LN_ASSETS + EQ_RATIO_W + LOANS_SHARE_W + EntityEffects + TimeEffects"
        res = fit_panel_fe(sub, formula=formula, entity_col="CERT", time_col="REPDTE")
        results.append(tidy_linearmodels(res, "h6_intexp_fe"))
        print(f"  INTEXP FE: LN_ASSETS={res.params['LN_ASSETS']:.6f} (p={res.pvalues['LN_ASSETS']:.4f}), nobs={res.nobs}")

    # Non-interest margin proxy: ROA - (INTINCY - INTEXPY) ≈ noninterest income - noninterest expense, scaled
    if all(c in df.columns for c in ["ROA", "INTINCY", "INTEXPY"]):
        df["NONINT_MARGIN"] = df["ROA"] - (df["INTINCY"] - df["INTEXPY"])
        df["NONINT_MARGIN_W"] = winsorize_series(df["NONINT_MARGIN"], p=wp)
        sub = df.dropna(subset=["NONINT_MARGIN_W", "LN_ASSETS"]).copy()
        formula = "NONINT_MARGIN_W ~ 1 + LN_ASSETS + EQ_RATIO_W + LOANS_SHARE_W + EntityEffects + TimeEffects"
        res = fit_panel_fe(sub, formula=formula, entity_col="CERT", time_col="REPDTE")
        results.append(tidy_linearmodels(res, "h6_nonint_margin_fe"))
        print(f"  NONINT_MARGIN FE: LN_ASSETS={res.params['LN_ASSETS']:.6f} (p={res.pvalues['LN_ASSETS']:.4f}), nobs={res.nobs}")

    return results


# ── 3. Market-power channel (local deposit HHI from SOD) ─────────────────────

def compute_bank_year_hhi(cfg: dict) -> pd.DataFrame:
    """Read branch-level SOD, compute county deposit HHI, return bank-year
    weighted-average local HHI."""
    print("\n=== Market-Power Channel: Computing HHI from SOD ===")
    sod_dir = project_root() / cfg["paths"]["raw"] / "fdic_sod"
    sod_files = sorted(globmod.glob(str(sod_dir / "sod_*.csv")))
    if not sod_files:
        print("  no SOD files found")
        return pd.DataFrame()

    frames = []
    for f in sod_files:
        chunk = pd.read_csv(f, usecols=lambda c: c.upper() in {"CERT", "YEAR", "STCNTYBR", "DEPSUMBR"}, low_memory=False)
        chunk.columns = [c.upper() for c in chunk.columns]
        frames.append(chunk)
    sod = pd.concat(frames, ignore_index=True)
    sod["CERT"] = pd.to_numeric(sod["CERT"], errors="coerce")
    sod["DEPSUMBR"] = pd.to_numeric(sod["DEPSUMBR"], errors="coerce").fillna(0)
    sod["STCNTYBR"] = sod["STCNTYBR"].astype(str)
    sod["YEAR"] = pd.to_numeric(sod["YEAR"], errors="coerce")
    sod = sod.dropna(subset=["CERT", "YEAR"])
    print(f"  SOD branches loaded: {len(sod):,}")

    # Aggregate to bank-county-year
    bcy = sod.groupby(["CERT", "STCNTYBR", "YEAR"], as_index=False)["DEPSUMBR"].sum()
    bcy.rename(columns={"DEPSUMBR": "DEP_BCY"}, inplace=True)

    # County totals
    county_totals = bcy.groupby(["STCNTYBR", "YEAR"], as_index=False)["DEP_BCY"].sum()
    county_totals.rename(columns={"DEP_BCY": "DEP_COUNTY"}, inplace=True)

    bcy = bcy.merge(county_totals, on=["STCNTYBR", "YEAR"])
    bcy["SHARE"] = np.where(bcy["DEP_COUNTY"] > 0, bcy["DEP_BCY"] / bcy["DEP_COUNTY"], 0)
    bcy["SHARE_SQ"] = bcy["SHARE"] ** 2

    # County HHI
    county_hhi = bcy.groupby(["STCNTYBR", "YEAR"], as_index=False)["SHARE_SQ"].sum()
    county_hhi.rename(columns={"SHARE_SQ": "HHI_COUNTY"}, inplace=True)

    # Bank-year weighted average local HHI (weighted by bank's deposits in each county)
    bcy = bcy.merge(county_hhi, on=["STCNTYBR", "YEAR"])
    bcy["WEIGHTED_HHI"] = bcy["HHI_COUNTY"] * bcy["DEP_BCY"]
    bank_year = bcy.groupby(["CERT", "YEAR"], as_index=False).agg(
        TOTAL_DEP_BCY=("DEP_BCY", "sum"),
        SUM_WEIGHTED_HHI=("WEIGHTED_HHI", "sum"),
    )
    bank_year["LOCAL_HHI"] = np.where(
        bank_year["TOTAL_DEP_BCY"] > 0,
        bank_year["SUM_WEIGHTED_HHI"] / bank_year["TOTAL_DEP_BCY"],
        np.nan,
    )
    bank_year = bank_year[["CERT", "YEAR", "LOCAL_HHI"]].copy()
    bank_year["CERT"] = bank_year["CERT"].astype(int).astype(str)
    bank_year.rename(columns={"YEAR": "SOD_YEAR"}, inplace=True)
    print(f"  bank-year HHI computed: {len(bank_year):,} rows, mean HHI={bank_year['LOCAL_HHI'].mean():.4f}")
    return bank_year


def run_market_power(df: pd.DataFrame, hhi: pd.DataFrame) -> list[pd.DataFrame]:
    """Merge local HHI to panel and test whether market power protects NIM."""
    print("\n=== Market-Power Regressions ===")
    if hhi.empty:
        return []

    # Map quarter dates to SOD years
    df["SOD_YEAR"] = map_sod_year_to_quarter(df["REPDTE"]).astype("Int64")
    hhi["SOD_YEAR"] = hhi["SOD_YEAR"].astype("Int64")
    merged = df.merge(hhi, on=["CERT", "SOD_YEAR"], how="left")
    merged["LOCAL_HHI_W"] = winsorize_series(merged["LOCAL_HHI"], p=0.01)
    merged["LN_ASSETS_x_HHI"] = merged["LN_ASSETS"] * merged["LOCAL_HHI_W"]

    sub = merged.dropna(subset=["NIM_W", "LN_ASSETS", "LOCAL_HHI_W"]).copy()
    print(f"  merged panel: {len(sub):,} obs, {sub['CERT'].nunique():,} banks")
    print(f"  LOCAL_HHI mean={sub['LOCAL_HHI_W'].mean():.4f}, median={sub['LOCAL_HHI_W'].median():.4f}")

    results = []

    # HHI main effect
    f1 = "NIM_W ~ 1 + LN_ASSETS + LOCAL_HHI_W + EQ_RATIO_W + LOANS_SHARE_W + EntityEffects + TimeEffects"
    res1 = fit_panel_fe(sub, formula=f1, entity_col="CERT", time_col="REPDTE")
    results.append(tidy_linearmodels(res1, "market_power_hhi"))
    print(f"  HHI main: LOCAL_HHI={res1.params['LOCAL_HHI_W']:.6f} (p={res1.pvalues['LOCAL_HHI_W']:.4f})")

    # HHI × size interaction
    f2 = "NIM_W ~ 1 + LN_ASSETS + LOCAL_HHI_W + LN_ASSETS_x_HHI + EQ_RATIO_W + LOANS_SHARE_W + EntityEffects + TimeEffects"
    res2 = fit_panel_fe(sub, formula=f2, entity_col="CERT", time_col="REPDTE")
    results.append(tidy_linearmodels(res2, "market_power_hhi_x_size"))
    print(f"  HHI×size: LN_ASSETS_x_HHI={res2.params['LN_ASSETS_x_HHI']:.6f} (p={res2.pvalues['LN_ASSETS_x_HHI']:.4f})")

    return results


# ── 4. Lagged balance-sheet controls ─────────────────────────────────────────

def run_lagged_controls(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Re-run M1 with one-quarter-lagged EQ_RATIO and LOANS_SHARE."""
    print("\n=== Lagged Controls Robustness ===")
    panel = df.sort_values(["CERT", "REPDTE"]).copy()
    panel["EQ_RATIO_LAG1"] = panel.groupby("CERT")["EQ_RATIO_W"].shift(1)
    panel["LOANS_SHARE_LAG1"] = panel.groupby("CERT")["LOANS_SHARE_W"].shift(1)
    sub = panel.dropna(subset=["NIM_W", "LN_ASSETS", "EQ_RATIO_LAG1", "LOANS_SHARE_LAG1"]).copy()
    print(f"  obs with lagged controls: {len(sub):,}")

    results = []
    formula = "NIM_W ~ 1 + LN_ASSETS + EQ_RATIO_LAG1 + LOANS_SHARE_LAG1 + EntityEffects + TimeEffects"
    res = fit_panel_fe(sub, formula=formula, entity_col="CERT", time_col="REPDTE")
    results.append(tidy_linearmodels(res, "rob_lagged_controls_fe"))
    print(f"  Lagged FE: LN_ASSETS={res.params['LN_ASSETS']:.6f} (p={res.pvalues['LN_ASSETS']:.4f}), nobs={res.nobs}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Extensions: NIM vol, H6, market power, lagged controls.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    table_dir = ensure_dir(project_root() / cfg["paths"]["tables"])
    df = prep_panel(cfg)

    all_results = []

    all_results.extend(run_nim_volatility(df))
    all_results.extend(run_h6_fee_offset(df))

    hhi = compute_bank_year_hhi(cfg)
    all_results.extend(run_market_power(df, hhi))

    all_results.extend(run_lagged_controls(df))

    out = pd.concat(all_results, ignore_index=True)
    out.to_csv(table_dir / "extension_results.csv", index=False)
    print(f"\nsaved {table_dir / 'extension_results.csv'} ({len(out)} rows, {out['model'].nunique()} models)")

    # Print summary
    print("\n" + "=" * 70)
    print("EXTENSION SUMMARY")
    print("=" * 70)
    for model in out["model"].unique():
        sub = out[out["model"] == model]
        key_terms = ["LN_ASSETS", "LOCAL_HHI_W", "LN_ASSETS_x_HHI",
                     "EQ_RATIO_LAG1", "LOANS_SHARE_LAG1"]
        key = sub[sub["term"].isin(key_terms)]
        for _, r in key.iterrows():
            sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
            print(f"  {model:30s} {r['term']:20s} {r['coef']:10.6f} (se={r['std_err']:.6f}) {sig}")


if __name__ == "__main__":
    main()
