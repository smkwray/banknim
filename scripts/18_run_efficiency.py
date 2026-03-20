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
from nimscale.validation import assert_nonempty_sample, require_columns, winsorize_required


def prep_panel(cfg: dict) -> pd.DataFrame:
    panel_path = project_root() / cfg["paths"]["interim"] / "bank_panel.parquet"
    df = pd.read_parquet(panel_path).copy()
    df["CERT"] = df["CERT"].astype(str)
    df["REPDTE"] = pd.to_datetime(df["REPDTE"])
    wp = cfg["project"]["winsor_pct"]
    require_columns(df, ["ROA", "NIM", "EQ_RATIO", "LOANS_SHARE", "LN_ASSETS"], "efficiency prep")

    df["ROA_W"] = winsorize_required(df, "ROA", p=wp, context="efficiency prep")
    df["NIM_W"] = winsorize_required(df, "NIM", p=wp, context="efficiency prep")
    df["EQ_RATIO_W"] = winsorize_required(df, "EQ_RATIO", p=wp, context="efficiency prep")
    df["LOANS_SHARE_W"] = winsorize_required(df, "LOANS_SHARE", p=wp, context="efficiency prep")
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
    parser = argparse.ArgumentParser(description="Run efficiency pass-through fixed-effects model.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    table_dir = ensure_dir(project_root() / cfg["paths"]["tables"])
    df = prep_panel(cfg)
    sub = df.dropna(subset=["ROA_W", "NIM_W", "LN_ASSETS", "EQ_RATIO_W", "LOANS_SHARE_W"]).copy()
    assert_nonempty_sample(sub, "efficiency sample", min_rows=1, entity_col="CERT", min_entities=2)

    formula = "ROA_W ~ 1 + LN_ASSETS + NIM_W + EQ_RATIO_W + LOANS_SHARE_W + EntityEffects + TimeEffects"
    res = fit_panel_fe(sub, formula=formula, entity_col="CERT", time_col="REPDTE")
    out = tidy_linearmodels(res, "efficiency_fe")
    assert_nonempty_sample(out, "efficiency result export")
    upsert_extension_results(table_dir, out)

    print(
        f"efficiency_fe: NIM_W={res.params['NIM_W']:.6f} "
        f"(p={res.pvalues['NIM_W']:.4f}), LN_ASSETS={res.params['LN_ASSETS']:.6f}, nobs={res.nobs}"
    )
    print(f"updated {table_dir / 'extension_results.csv'}")


if __name__ == "__main__":
    main()
