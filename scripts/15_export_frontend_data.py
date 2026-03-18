"""Export all model results and chart data as JSON for the static frontend."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import json

import numpy as np
import pandas as pd

from nimscale.bank_panel import winsorize_series
from nimscale.io import ensure_dir
from nimscale.settings import load_config, project_root


def json_safe(obj):
    """Convert numpy/pandas types to JSON-safe Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else round(float(obj), 6)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.ndarray,)):
        return [json_safe(x) for x in obj]
    return obj


def write_json(data, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, default=json_safe, indent=2)
    print(f"  {path.name} ({path.stat().st_size / 1024:.0f} KB)")


# ── 1. Consolidated model results ─────────────────────────────────────────────

def export_model_results(table_dir: Path, out_dir: Path) -> None:
    """Merge all result CSVs into one tidy JSON with uniform schema."""
    files = {
        "regression_results.csv": "baseline",
        "nim_decomposition_results.csv": "nim_decomposition",
        "franchise_dilution_results.csv": "franchise_dilution",
        "rate_cycle_results.csv": "rate_cycle",
        "event_study_results.csv": "event_study",
        "threshold_crossing_results.csv": "threshold",
        "robustness_results.csv": "robustness",
        "extension_results.csv": "extension",
    }
    all_rows = []
    for fname, group in files.items():
        path = table_dir / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            entry = {
                "group": group,
                "model": row.get("model", ""),
                "term": row.get("term", ""),
                "coef": row.get("coef"),
                "std_err": row.get("std_err"),
                "p_value": row.get("p_value"),
                "ci_low": row.get("ci_low"),
                "ci_high": row.get("ci_high"),
                "nobs": row.get("nobs"),
            }
            # Grab whichever R² columns exist
            for r2col in ["r2", "r2_within", "r2_between", "r2_overall"]:
                if r2col in row.index and pd.notna(row[r2col]):
                    entry[r2col] = row[r2col]
            all_rows.append(entry)

    write_json(all_rows, out_dir / "model_results.json")


# ── 2. Summary statistics ─────────────────────────────────────────────────────

def export_summary_stats(table_dir: Path, out_dir: Path) -> None:
    path = table_dir / "summary_statistics.csv"
    if not path.exists():
        return
    df = pd.read_csv(path, index_col=0)
    rows = []
    for var_name, row in df.iterrows():
        rows.append({
            "variable": var_name,
            **{k: round(float(v), 4) if pd.notna(v) else None for k, v in row.items()},
        })
    write_json(rows, out_dir / "summary_statistics.json")


# ── 3. NIM trend over time by size decile ─────────────────────────────────────

def export_nim_trend_by_size(panel: pd.DataFrame, out_dir: Path) -> None:
    """Quarterly mean NIM by asset-size decile — the main time-series chart."""
    df = panel.dropna(subset=["NIM", "LN_ASSETS"]).copy()
    # Assign size decile based on within-quarter ranking (so deciles shift over time)
    df["QUARTER"] = df["REPDTE"].dt.to_period("Q").astype(str)
    df["SIZE_DECILE"] = df.groupby("QUARTER")["LN_ASSETS"].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop") + 1
    )
    agg = (
        df.groupby(["QUARTER", "SIZE_DECILE"], as_index=False)
        .agg(
            nim_mean=("NIM", "mean"),
            nim_median=("NIM", "median"),
            n_banks=("CERT", "nunique"),
        )
    )
    agg = agg.sort_values(["QUARTER", "SIZE_DECILE"])
    write_json(agg.to_dict(orient="records"), out_dir / "nim_trend_by_size.json")


# ── 4. Event study coefficients ───────────────────────────────────────────────

def export_event_study(table_dir: Path, out_dir: Path) -> None:
    path = table_dir / "event_study_results.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    et = df[df["term"].str.startswith("ET_")].copy()

    def parse_et(t):
        x = t.replace("ET_", "")
        return -int(x[1:]) if x.startswith("m") else int(x[1:])

    et["event_time"] = et["term"].map(parse_et)
    # Add the omitted reference period
    ref = pd.DataFrame([{
        "term": "ET_m1", "event_time": -1, "coef": 0, "std_err": 0,
        "ci_low": 0, "ci_high": 0, "p_value": None,
    }])
    et = pd.concat([et, ref], ignore_index=True).sort_values("event_time")
    rows = et[["event_time", "coef", "std_err", "ci_low", "ci_high", "p_value"]].to_dict(orient="records")
    nobs = int(df["nobs"].iloc[0])
    write_json({"nobs": nobs, "model": "acquirer_event_study", "coefficients": rows}, out_dir / "event_study.json")


# ── 5. Threshold crossing coefficients ────────────────────────────────────────

def export_threshold_crossing(table_dir: Path, out_dir: Path) -> None:
    path = table_dir / "threshold_crossing_results.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    out = {}
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        et = sub[sub["term"].str.startswith("ET_")].copy()

        def parse_et(t):
            x = t.replace("ET_", "")
            return -int(x[1:]) if x.startswith("m") else int(x[1:])

        et["event_time"] = et["term"].map(parse_et)
        ref = pd.DataFrame([{
            "term": "ET_m1", "event_time": -1, "coef": 0, "std_err": 0,
            "ci_low": 0, "ci_high": 0, "p_value": None,
        }])
        et = pd.concat([et, ref], ignore_index=True).sort_values("event_time")
        out[model] = {
            "nobs": int(sub["nobs"].iloc[0]),
            "coefficients": et[["event_time", "coef", "std_err", "ci_low", "ci_high", "p_value"]].to_dict(orient="records"),
        }
    write_json(out, out_dir / "threshold_crossing.json")


# ── 6. Distribution data (binned histograms) ─────────────────────────────────

def export_distributions(panel: pd.DataFrame, out_dir: Path) -> None:
    out = {}

    # Assets (log10)
    assets = panel["ASSET"].dropna()
    log_assets = np.log10(assets[assets > 0])
    counts, edges = np.histogram(log_assets, bins=60)
    out["assets_log10"] = {
        "bin_edges": [round(float(e), 3) for e in edges],
        "counts": [int(c) for c in counts],
        "xlabel": "Log₁₀ assets ($000s)",
    }

    # NIM
    nim = panel["NIM"].dropna()
    lo, hi = nim.quantile(0.005), nim.quantile(0.995)
    counts, edges = np.histogram(nim.clip(lo, hi), bins=60)
    out["nim"] = {
        "bin_edges": [round(float(e), 4) for e in edges],
        "counts": [int(c) for c in counts],
        "xlabel": "NIM (%)",
    }

    # Asset growth vs deposit growth (2D hex → binned grid)
    g = panel.dropna(subset=["ASSET_GROWTH_QOQ", "DEP_GROWTH_QOQ"]).copy()
    ag = g["ASSET_GROWTH_QOQ"].clip(-0.3, 0.3)
    dg = g["DEP_GROWTH_QOQ"].clip(-0.3, 0.3)
    H, xedges, yedges = np.histogram2d(ag, dg, bins=40)
    out["growth_scatter"] = {
        "x_edges": [round(float(e), 4) for e in xedges],
        "y_edges": [round(float(e), 4) for e in yedges],
        "counts": [[int(c) for c in row] for row in H],
        "xlabel": "Asset growth (q/q)",
        "ylabel": "Deposit growth (q/q)",
    }

    write_json(out, out_dir / "distributions.json")


# ── 7. Binscatter and size-decile cross-section ──────────────────────────────

def export_cross_section(panel: pd.DataFrame, out_dir: Path) -> None:
    df = panel.dropna(subset=["NIM", "LN_ASSETS"]).copy()

    # Binscatter: bank means → 20 bins
    bank_means = df.groupby("CERT")[["NIM", "LN_ASSETS"]].mean().dropna().sort_values("LN_ASSETS")
    bank_means["BIN"] = pd.qcut(bank_means["LN_ASSETS"], 20, labels=False, duplicates="drop") + 1
    bins = bank_means.groupby("BIN")[["NIM", "LN_ASSETS"]].mean().reset_index()
    binscatter = bins.rename(columns={"NIM": "avg_nim", "LN_ASSETS": "avg_ln_assets"}).to_dict(orient="records")

    # Size decile means
    df["SIZE_DECILE"] = pd.qcut(df["LN_ASSETS"], 10, labels=False, duplicates="drop") + 1
    deciles = df.groupby("SIZE_DECILE").agg(
        avg_nim=("NIM", "mean"),
        avg_roa=("ROA", "mean") if "ROA" in df.columns else ("NIM", "count"),
        avg_ln_assets=("LN_ASSETS", "mean"),
        n_banks=("CERT", "nunique"),
        n_obs=("NIM", "count"),
    ).reset_index()
    # Recompute ROA properly if it was a count placeholder
    if "ROA" in df.columns:
        deciles_roa = df.groupby("SIZE_DECILE")["ROA"].mean().reset_index()
        deciles["avg_roa"] = deciles_roa["ROA"]
    else:
        deciles["avg_roa"] = None

    write_json({
        "binscatter": binscatter,
        "deciles": deciles.to_dict(orient="records"),
    }, out_dir / "cross_section.json")


# ── 8. Panel metadata ────────────────────────────────────────────────────────

def export_metadata(panel: pd.DataFrame, cfg: dict, out_dir: Path) -> None:
    meta = {
        "project": cfg["project"]["name"],
        "sample_start": str(panel["REPDTE"].min().date()),
        "sample_end": str(panel["REPDTE"].max().date()),
        "n_obs": int(len(panel)),
        "n_banks": int(panel["CERT"].nunique()),
        "n_quarters": int(panel["REPDTE"].nunique()),
        "nim_mean": round(float(panel["NIM"].mean()), 4),
        "nim_median": round(float(panel["NIM"].median()), 4),
        "avg_assets_millions": round(float(panel["ASSET"].mean() / 1000), 1),
        "median_assets_millions": round(float(panel["ASSET"].median() / 1000), 1),
        "models_fitted": [
            "M1: Within-bank FE size",
            "M2: Between-bank means",
            "M3: Growth",
            "M4: Franchise dilution",
            "M5: Rate-cycle heterogeneity",
            "M6: Acquisition event study",
            "M7: Threshold crossing",
        ],
        "headline_results": {
            "within_bank_size_nim": {"coef": -0.090, "p": "<0.001", "interpretation": "When a bank grows, its NIM falls"},
            "between_bank_size_nim": {"coef": -0.085, "p": "<0.001", "interpretation": "Banks with larger average size have lower average NIM"},
            "roa_offsets_nim": {"coef": 0.205, "p": "<0.001", "interpretation": "Large banks offset lower NIM with higher overall ROA"},
            "rate_cycle": {"coef": -0.012, "p": "<0.001", "interpretation": "Size penalty on NIM strengthens when rates rise"},
            "lagged_size_effect": {"coef": -0.119, "p": "<0.001", "interpretation": "Size effect strengthens with lagged controls"},
        },
    }
    write_json(meta, out_dir / "metadata.json")


# ── 9. Robustness comparison ─────────────────────────────────────────────────

def export_robustness_comparison(table_dir: Path, out_dir: Path) -> None:
    """Extract the key size coefficient from each model for a forest-plot style comparison."""
    specs = []

    # Baseline
    base = pd.read_csv(table_dir / "regression_results.csv")
    fe = base[(base["model"] == "within_fe_size") & (base["term"] == "LN_ASSETS")]
    if len(fe):
        r = fe.iloc[0]
        specs.append({"label": "Baseline (winsorized)", "coef": r["coef"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "nobs": r["nobs"]})

    # Robustness
    rob = pd.read_csv(table_dir / "robustness_results.csv")
    rob_map = {
        "rob_raw_nim_fe_size": ("Raw NIM (no winsorize)", "LN_ASSETS"),
        "rob_min12q_fe_size": ("Exclude <12 quarters", "LN_ASSETS"),
        "rob_log_deposits_fe": ("Log deposits as size", "LN_DEP"),
        "rob_first_diff_fe": ("First differences", "D_LN_ASSETS"),
    }
    for model, (label, term) in rob_map.items():
        row = rob[(rob["model"] == model) & (rob["term"] == term)]
        if len(row):
            r = row.iloc[0]
            specs.append({"label": label, "coef": r["coef"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "nobs": r["nobs"]})

    # Extensions
    ext = pd.read_csv(table_dir / "extension_results.csv")
    lag = ext[(ext["model"] == "rob_lagged_controls_fe") & (ext["term"] == "LN_ASSETS")]
    if len(lag):
        r = lag.iloc[0]
        specs.append({"label": "Lagged controls", "coef": r["coef"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "nobs": r["nobs"]})

    write_json(specs, out_dir / "robustness_forest.json")


# ── 10. Rolling coefficient path ─────────────────────────────────────────────

def export_rolling_coefficients(table_dir: Path, out_dir: Path) -> None:
    path = table_dir / "rolling_coefficients_results.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    sub = df[(df["model"] == "rolling_within_fe_size") & (df["term"] == "LN_ASSETS")].copy()
    if sub.empty:
        return
    for col in ["window_start", "window_end", "window_mid"]:
        sub[col] = pd.to_datetime(sub[col]).dt.date.astype(str)
    rows = sub[
        [
            "window_start",
            "window_end",
            "window_mid",
            "window_quarters",
            "coef",
            "std_err",
            "p_value",
            "ci_low",
            "ci_high",
            "nobs",
            "n_banks",
            "r2_within",
            "r2_between",
            "r2_overall",
        ]
    ].to_dict(orient="records")
    write_json(
        {
            "model": "rolling_within_fe_size",
            "term": "LN_ASSETS",
            "rows": rows,
        },
        out_dir / "rolling_coefficients.json",
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    table_dir = project_root() / cfg["paths"]["tables"]
    out_dir = ensure_dir(project_root() / "output" / "frontend")

    print("Exporting frontend data:")

    panel = pd.read_parquet(project_root() / cfg["paths"]["interim"] / "bank_panel.parquet")
    panel["CERT"] = panel["CERT"].astype(str)
    panel["REPDTE"] = pd.to_datetime(panel["REPDTE"])

    export_model_results(table_dir, out_dir)
    export_summary_stats(table_dir, out_dir)
    export_nim_trend_by_size(panel, out_dir)
    export_event_study(table_dir, out_dir)
    export_threshold_crossing(table_dir, out_dir)
    export_distributions(panel, out_dir)
    export_cross_section(panel, out_dir)
    export_metadata(panel, cfg, out_dir)
    export_robustness_comparison(table_dir, out_dir)
    export_rolling_coefficients(table_dir, out_dir)

    print(f"\nAll frontend data exported to {out_dir}/")
    print(f"Files: {len(list(out_dir.glob('*.json')))}")


if __name__ == "__main__":
    main()
