from __future__ import annotations

from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

import numpy as np
import pandas as pd

from nimscale.bank_panel import winsorize_series
from nimscale.io import ensure_dir
from nimscale.regression import fit_panel_fe, tidy_linearmodels
from nimscale.settings import load_config, project_root


RCCI_MEMBER_RE = re.compile(r"FFIEC CDR Call Schedule RCCI (\d{8})\.txt$")
POR_MEMBER_RE = re.compile(r"FFIEC CDR Call Bulk POR (\d{8})\.txt$")


def coalesce_pair(df: pd.DataFrame, total_code: str | None, domestic_code: str | None, foreign_code: str | None = None) -> pd.Series:
    total = pd.to_numeric(df[total_code], errors="coerce") if total_code and total_code in df.columns else pd.Series(np.nan, index=df.index, dtype="float64")
    domestic = pd.to_numeric(df[domestic_code], errors="coerce") if domestic_code and domestic_code in df.columns else pd.Series(np.nan, index=df.index, dtype="float64")
    if foreign_code and foreign_code in df.columns:
        foreign = pd.to_numeric(df[foreign_code], errors="coerce")
        parts = domestic.fillna(0) + foreign.fillna(0)
        parts = parts.where(domestic.notna() | foreign.notna())
    else:
        parts = domestic
    return total.combine_first(parts)


def load_ffiec_loan_panel(cfg: dict) -> pd.DataFrame:
    ffiec_dir = project_root() / cfg["paths"]["raw"] / "ffiec"
    zip_paths = sorted(ffiec_dir.glob("ffiec_call_*.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"No FFIEC call-report zips found in {ffiec_dir}")

    sample_start = pd.to_datetime(cfg["project"]["sample_start"])
    sample_end = pd.to_datetime(cfg["project"]["sample_end"])
    frames: list[pd.DataFrame] = []
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as zf:
            members = [name for name in zf.namelist() if RCCI_MEMBER_RE.search(Path(name).name)]
            por_members = [name for name in zf.namelist() if POR_MEMBER_RE.search(Path(name).name)]
            if not members or not por_members:
                continue
            member = members[0]
            repdte = pd.to_datetime(RCCI_MEMBER_RE.search(Path(member).name).group(1), format="%m%d%Y")
            if repdte < sample_start or repdte > sample_end:
                continue
            with zf.open(member) as f:
                df = pd.read_csv(f, sep="\t", skiprows=[1], low_memory=False)
            with zf.open(por_members[0]) as f:
                por = pd.read_csv(f, sep="\t", low_memory=False)
            por["IDRSSD"] = pd.to_numeric(por["IDRSSD"], errors="coerce")
            por["CERT"] = pd.to_numeric(por["FDIC Certificate Number"], errors="coerce")

            ci_loans = coalesce_pair(df, "RCON1766", "RCFD1763", "RCFD1764")
            residential_loans = (
                coalesce_pair(df, None, "RCON1797", "RCFD1797").fillna(0)
                + coalesce_pair(df, None, "RCON5367", "RCFD5367").fillna(0)
                + coalesce_pair(df, None, "RCON5368", "RCFD5368").fillna(0)
                + coalesce_pair(df, None, "RCONF158", "RCFDF158").fillna(0)
            )
            cre_loans = (
                coalesce_pair(df, None, "RCON1460", "RCFD1460").fillna(0)
                + coalesce_pair(df, None, "RCONF159", "RCFDF159").fillna(0)
                + coalesce_pair(df, None, "RCONF160", "RCFDF160").fillna(0)
                + coalesce_pair(df, None, "RCONF161", "RCFDF161").fillna(0)
            )
            consumer_loans = (
                coalesce_pair(df, None, "RCONB538", "RCFDB538").fillna(0)
                + coalesce_pair(df, None, "RCONB539", "RCFDB539").fillna(0)
                + coalesce_pair(df, None, "RCONK137", "RCFDK137").fillna(0)
            )

            loan_frame = pd.DataFrame(
                {
                    "IDRSSD": pd.to_numeric(df["IDRSSD"], errors="coerce"),
                    "REPDTE": repdte,
                    "CI_LOANS": ci_loans,
                    "RESIDENTIAL_LOANS": residential_loans,
                    "CRE_LOANS": cre_loans,
                    "CONSUMER_LOANS": consumer_loans,
                }
            )
            loan_frame = loan_frame.merge(por[["IDRSSD", "CERT"]], on="IDRSSD", how="left")
            frames.append(loan_frame[["CERT", "REPDTE", "CI_LOANS", "RESIDENTIAL_LOANS", "CRE_LOANS", "CONSUMER_LOANS"]].copy())

    if not frames:
        raise ValueError(f"No RCCI schedule files found in {ffiec_dir}")

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["CERT"]).copy()
    out["CERT"] = out["CERT"].astype(int).astype(str)
    out = out.drop_duplicates(subset=["CERT", "REPDTE"], keep="last")
    return out


def prep_panel(cfg: dict) -> pd.DataFrame:
    panel_path = project_root() / cfg["paths"]["interim"] / "bank_panel.parquet"
    df = pd.read_parquet(panel_path).copy()
    df["CERT"] = pd.to_numeric(df["CERT"], errors="coerce").astype("Int64").astype(str)
    df["REPDTE"] = pd.to_datetime(df["REPDTE"])
    wp = cfg["project"]["winsor_pct"]

    df["NIM_W"] = winsorize_series(df["NIM"], p=wp)
    df["EQ_RATIO_W"] = winsorize_series(df["EQ_RATIO"], p=wp) if "EQ_RATIO" in df.columns else 0.0
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
    parser = argparse.ArgumentParser(description="Run FFIEC loan-composition extension model.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    table_dir = ensure_dir(project_root() / cfg["paths"]["tables"])

    panel = prep_panel(cfg)
    ffiec = load_ffiec_loan_panel(cfg)
    merged = panel.merge(ffiec, on=["CERT", "REPDTE"], how="inner")

    for loan_col, share_col in [
        ("CI_LOANS", "CI_SHARE"),
        ("RESIDENTIAL_LOANS", "RESIDENTIAL_SHARE"),
        ("CRE_LOANS", "CRE_SHARE"),
        ("CONSUMER_LOANS", "CONSUMER_SHARE"),
    ]:
        merged[share_col] = pd.to_numeric(merged[loan_col], errors="coerce") / pd.to_numeric(merged["ASSET"], errors="coerce")
        merged[f"{share_col}_W"] = winsorize_series(merged[share_col], p=cfg["project"]["winsor_pct"])

    sub = merged.dropna(
        subset=[
            "NIM_W",
            "LN_ASSETS",
            "EQ_RATIO_W",
            "CI_SHARE_W",
            "RESIDENTIAL_SHARE_W",
            "CRE_SHARE_W",
            "CONSUMER_SHARE_W",
        ]
    ).copy()
    formula = (
        "NIM_W ~ 1 + LN_ASSETS + EQ_RATIO_W + "
        "CI_SHARE_W + RESIDENTIAL_SHARE_W + CRE_SHARE_W + CONSUMER_SHARE_W + "
        "EntityEffects + TimeEffects"
    )
    res = fit_panel_fe(sub, formula=formula, entity_col="CERT", time_col="REPDTE")
    out = tidy_linearmodels(res, "loan_mix_fe")
    upsert_extension_results(table_dir, out)

    print(
        f"loan_mix_fe: LN_ASSETS={res.params['LN_ASSETS']:.6f} "
        f"(p={res.pvalues['LN_ASSETS']:.4f}), nobs={res.nobs}, banks={sub['CERT'].nunique()}"
    )
    print(f"updated {table_dir / 'extension_results.csv'}")


if __name__ == "__main__":
    main()
