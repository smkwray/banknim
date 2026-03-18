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


def prep_panel(cfg: dict) -> pd.DataFrame:
    panel_path = project_root() / cfg["paths"]["interim"] / "bank_panel.parquet"
    df = pd.read_parquet(panel_path).copy()
    df["CERT"] = df["CERT"].astype(str)
    df["REPDTE"] = pd.to_datetime(df["REPDTE"])
    wp = cfg["project"]["winsor_pct"]

    df["EQ_RATIO_DEP_W"] = winsorize_series(df["EQ_RATIO"], p=wp)
    df["LOANS_SHARE_W"] = winsorize_series(df["LOANS_SHARE"], p=wp) if "LOANS_SHARE" in df.columns else 0.0
    return df


def upsert_extension_results(table_dir: Path, new_rows: pd.DataFrame) -> None:
    path = table_dir / "extension_results.csv"
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[~existing["model"].isin(new_rows["model"].unique())].copy()
        out = pd.concat([existing, new_rows], ignore_index=True)
    else:
        out = new_rows.copy()
    out.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run equity-ratio-as-outcome fixed-effects model.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    table_dir = ensure_dir(project_root() / cfg["paths"]["tables"])
    df = prep_panel(cfg)
    sub = df.dropna(subset=["EQ_RATIO_DEP_W", "LN_ASSETS", "LOANS_SHARE_W"]).copy()

    formula = "EQ_RATIO_DEP_W ~ 1 + LN_ASSETS + LOANS_SHARE_W + EntityEffects + TimeEffects"
    res = fit_panel_fe(sub, formula=formula, entity_col="CERT", time_col="REPDTE")
    out = tidy_linearmodels(res, "equity_ratio_fe")
    upsert_extension_results(table_dir, out)

    print(
        f"equity_ratio_fe: LN_ASSETS={res.params['LN_ASSETS']:.6f} "
        f"(p={res.pvalues['LN_ASSETS']:.4f}), nobs={res.nobs}"
    )
    print(f"updated {table_dir / 'extension_results.csv'}")


if __name__ == "__main__":
    main()
