from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().upper() for c in out.columns]
    return out


def pick_first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    cols = {c.upper(): c for c in df.columns}
    for c in candidates:
        if c.upper() in cols:
            return cols[c.upper()]
    raise KeyError(f"None of the candidate columns exist: {list(candidates)}")


def parse_fdic_quarter_date(values: pd.Series) -> pd.Series:
    s = values.astype(str).str.strip()
    # handles 20241231, 2024-12-31, or 2024/12/31
    s = s.str.replace("/", "-", regex=False)
    s = s.where(s.str.contains("-"), s.str.replace(r"^(\d{4})(\d{2})(\d{2})$", r"\1-\2-\3", regex=True))
    out = pd.to_datetime(s, errors="coerce")
    return out


def quarter_ends(start: str, end: str) -> list[pd.Timestamp]:
    idx = pd.period_range(start=start, end=end, freq="Q")
    return [p.end_time.normalize() for p in idx]


def coerce_numeric(df: pd.DataFrame, exclude: Iterable[str] | None = None) -> pd.DataFrame:
    exclude_set = {c.upper() for c in (exclude or [])}
    out = df.copy()
    for col in out.columns:
        if col.upper() in exclude_set:
            continue
        converted = pd.to_numeric(out[col], errors="coerce")
        raw = out[col]
        nonempty = raw.notna() & raw.astype(str).str.strip().ne("")
        if converted[nonempty].notna().all():
            out[col] = converted
    return out


def winsorize_series(s: pd.Series, p: float = 0.01) -> pd.Series:
    if s.dropna().empty:
        return s
    lo = s.quantile(p)
    hi = s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)


def add_core_features(
    df: pd.DataFrame,
    bank_id_col: str,
    date_col: str,
    assets_col: str,
    deposits_col: str | None = None,
    equity_col: str | None = None,
    loans_col: str | None = None,
    nim_col: str | None = None,
) -> pd.DataFrame:
    out = df.copy().sort_values([bank_id_col, date_col]).reset_index(drop=True)

    out["LN_ASSETS"] = np.where(out[assets_col] > 0, np.log(out[assets_col]), np.nan)
    out["AVG_LN_ASSETS_BANK"] = out.groupby(bank_id_col)["LN_ASSETS"].transform("mean")
    out["LN_ASSETS_WITHIN"] = out["LN_ASSETS"] - out["AVG_LN_ASSETS_BANK"]
    out["ASSET_GROWTH_QOQ"] = out.groupby(bank_id_col)["LN_ASSETS"].diff()

    if deposits_col and deposits_col in out.columns:
        out["LN_DEP"] = np.where(out[deposits_col] > 0, np.log(out[deposits_col]), np.nan)
        out["DEP_GROWTH_QOQ"] = out.groupby(bank_id_col)["LN_DEP"].diff()

    if equity_col and equity_col in out.columns:
        out["EQ_RATIO"] = np.where(out[assets_col] != 0, out[equity_col] / out[assets_col], np.nan)

    if loans_col and loans_col in out.columns:
        out["LOANS_SHARE"] = np.where(out[assets_col] != 0, out[loans_col] / out[assets_col], np.nan)

    if nim_col and nim_col in out.columns:
        out["NIM"] = pd.to_numeric(out[nim_col], errors="coerce")

    # FDIC assets are in thousands; thresholds in thousands accordingly.
    out["GT_10B"] = (out[assets_col] >= 10_000_000).astype("Int64")
    out["GT_50B"] = (out[assets_col] >= 50_000_000).astype("Int64")
    out["GT_100B"] = (out[assets_col] >= 100_000_000).astype("Int64")

    return out


def map_sod_year_to_quarter(dates: pd.Series) -> pd.Series:
    dt = pd.to_datetime(dates)
    # SOD is measured at June 30. Use year t for Q3/Q4 of t and Q1/Q2 of t+1.
    year = dt.dt.year
    month = dt.dt.month
    sod_year = np.where(month.isin([9, 12]), year, year - 1)
    return pd.Series(sod_year, index=dates.index, name="SOD_YEAR")


def make_quarter_key(dates: pd.Series) -> pd.Series:
    dt = pd.to_datetime(dates)
    return dt.dt.to_period("Q").astype(str)


def merge_rates(panel: pd.DataFrame, rates: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = panel.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    rates = rates.copy()
    rates["REPDTE"] = pd.to_datetime(rates["REPDTE"])
    return out.merge(rates, left_on=date_col, right_on="REPDTE", how="left", suffixes=("", "_RATE"))


def build_bank_year_means(df: pd.DataFrame, bank_id_col: str, cols: list[str]) -> pd.DataFrame:
    return df.groupby(bank_id_col, dropna=False)[cols].mean(numeric_only=True).reset_index()


def add_sod_franchise_features(
    df: pd.DataFrame,
    bank_id_col: str,
    sod_year_col: str,
    branch_count_col: str = "BRANCH_COUNT",
    sod_dep_total_col: str = "SOD_DEP_TOTAL",
    dep_per_branch_col: str = "DEP_PER_BRANCH",
) -> pd.DataFrame:
    out = df.copy()
    feature_specs = [
        ("BRANCH_COUNT", branch_count_col),
        ("SOD_DEP_TOTAL", sod_dep_total_col),
        ("DEP_PER_BRANCH", dep_per_branch_col),
    ]

    yearly = out[[bank_id_col, sod_year_col] + [col for _, col in feature_specs]].drop_duplicates(
        subset=[bank_id_col, sod_year_col]
    )
    yearly = yearly.sort_values([bank_id_col, sod_year_col]).reset_index(drop=True)

    for prefix, col in feature_specs:
        yearly[col] = pd.to_numeric(yearly[col], errors="coerce")
        yearly[f"LN_{prefix}"] = np.where(yearly[col] > 0, np.log(yearly[col]), np.nan)
        yearly[f"{prefix}_GROWTH_YOY"] = yearly.groupby(bank_id_col)[f"LN_{prefix}"].diff()

    merge_cols = [bank_id_col, sod_year_col]
    for prefix, _col in feature_specs:
        merge_cols.extend([f"LN_{prefix}", f"{prefix}_GROWTH_YOY"])
    return out.merge(yearly[merge_cols], on=[bank_id_col, sod_year_col], how="left")


def infer_main_columns(df: pd.DataFrame, cfg: dict) -> dict[str, str]:
    cc = cfg["fdic"]["column_candidates"]
    out = {
        "bank_id": pick_first_existing(df, cc["bank_id"]),
        "date": pick_first_existing(df, cc["quarter_date"]),
        "assets": pick_first_existing(df, cc["total_assets"]),
    }
    for key, cfg_key in [
        ("deposits", "total_deposits"),
        ("equity", "equity"),
        ("loans", "loans_net"),
        ("nim", "nim"),
        ("roa", "roa"),
        ("roe", "roe"),
        ("offices", "office_count"),
    ]:
        try:
            out[key] = pick_first_existing(df, cc[cfg_key])
        except KeyError:
            pass
    return out
