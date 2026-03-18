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


WINDOW_QUARTERS = 20


def prep_panel(cfg: dict) -> pd.DataFrame:
    panel_path = project_root() / cfg["paths"]["interim"] / "bank_panel.parquet"
    df = pd.read_parquet(panel_path).copy()
    df["CERT"] = df["CERT"].astype(str)
    df["REPDTE"] = pd.to_datetime(df["REPDTE"])
    wp = cfg["project"]["winsor_pct"]
    df["NIM_W"] = winsorize_series(df["NIM"], p=wp)
    df["EQ_RATIO_W"] = winsorize_series(df["EQ_RATIO"], p=wp) if "EQ_RATIO" in df.columns else 0.0
    df["LOANS_SHARE_W"] = winsorize_series(df["LOANS_SHARE"], p=wp) if "LOANS_SHARE" in df.columns else 0.0
    return df.dropna(subset=["NIM_W", "LN_ASSETS", "EQ_RATIO_W", "LOANS_SHARE_W"]).copy()


def run_rolling_fe(panel: pd.DataFrame, window_quarters: int) -> pd.DataFrame:
    quarters = sorted(panel["REPDTE"].drop_duplicates())
    if len(quarters) < window_quarters:
        raise ValueError(f"Need at least {window_quarters} quarters, found {len(quarters)}")

    rows: list[pd.DataFrame] = []
    formula = "NIM_W ~ 1 + LN_ASSETS + EQ_RATIO_W + LOANS_SHARE_W + EntityEffects + TimeEffects"

    for i in range(len(quarters) - window_quarters + 1):
        window = quarters[i : i + window_quarters]
        start = window[0]
        end = window[-1]
        sub = panel[panel["REPDTE"].isin(window)].copy()
        res = fit_panel_fe(sub, formula=formula, entity_col="CERT", time_col="REPDTE")
        tidy = tidy_linearmodels(res, "rolling_within_fe_size")
        tidy["window_start"] = start
        tidy["window_end"] = end
        tidy["window_mid"] = window[len(window) // 2]
        tidy["window_quarters"] = window_quarters
        tidy["n_banks"] = sub["CERT"].nunique()
        rows.append(tidy)
        print(
            f"{start.date()} to {end.date()}: "
            f"LN_ASSETS={res.params['LN_ASSETS']:.6f} "
            f"(p={res.pvalues['LN_ASSETS']:.4f}), nobs={res.nobs}"
        )

    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling-window within-bank FE coefficients.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--window-quarters", type=int, default=WINDOW_QUARTERS)
    args = parser.parse_args()

    cfg = load_config(args.config)
    table_dir = ensure_dir(project_root() / cfg["paths"]["tables"])

    panel = prep_panel(cfg)
    out = run_rolling_fe(panel, window_quarters=args.window_quarters)
    path = table_dir / "rolling_coefficients_results.csv"
    out.to_csv(path, index=False)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
