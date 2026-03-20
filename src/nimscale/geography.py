from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .bank_panel import map_sod_year_to_quarter, standardize_columns


STATE_NAMES = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "DC": "District of Columbia",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    "AS": "American Samoa",
    "FM": "Federated States of Micronesia",
    "GU": "Guam",
    "MP": "Northern Mariana Islands",
    "PR": "Puerto Rico",
    "VI": "U.S. Virgin Islands",
}


def normalize_cert_ids(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype("Int64")
    return numeric.astype("string")


def normalize_county_fips(values: pd.Series) -> pd.Series:
    digits = values.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
    digits = digits.str.replace(r"\D", "", regex=True)
    digits = digits.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    normalized = pd.to_numeric(digits, errors="coerce").astype("Int64")
    return normalized.astype("string").str.zfill(5)


def load_cbsa_crosswalk(path: Path, *, header_row: int = 2) -> pd.DataFrame:
    df = pd.read_excel(path, header=header_row, dtype=str)
    df = standardize_columns(df)
    rename_map = {
        "CBSA CODE": "CBSA_CODE",
        "CBSA TITLE": "CBSA_TITLE",
        "METROPOLITAN/MICROPOLITAN STATISTICAL AREA": "CBSA_TYPE",
        "COUNTY/COUNTY EQUIVALENT": "COUNTY_NAME",
        "STATE NAME": "STATE_NAME",
        "FIPS STATE CODE": "FIPS_STATE_CODE",
        "FIPS COUNTY CODE": "FIPS_COUNTY_CODE",
    }
    df = df.rename(columns=rename_map)
    keep = list(rename_map.values())
    out = df[keep].copy()
    out = out.dropna(subset=["CBSA_CODE", "FIPS_STATE_CODE", "FIPS_COUNTY_CODE"])
    out["county_fips"] = out["FIPS_STATE_CODE"].str.zfill(2) + out["FIPS_COUNTY_CODE"].str.zfill(3)
    out["cbsa_code"] = out["CBSA_CODE"].str.replace(r"\.0$", "", regex=True).str.zfill(5)
    out["cbsa_title"] = out["CBSA_TITLE"].astype(str).str.strip()
    out["cbsa_type"] = out["CBSA_TYPE"].astype(str).str.replace(" Statistical Area", "", regex=False).str.strip()
    out["county_name"] = out["COUNTY_NAME"].astype(str).str.strip()
    out["state_name"] = out["STATE_NAME"].astype(str).str.strip()
    return out[["county_fips", "cbsa_code", "cbsa_title", "cbsa_type", "county_name", "state_name"]].drop_duplicates()


def build_state_summary(panel: pd.DataFrame) -> list[dict[str, object]]:
    df = panel.dropna(subset=["STALP", "NIM"]).copy()
    agg_spec = {
        "avg_nim": ("NIM", "mean"),
        "avg_ln_assets": ("LN_ASSETS", "mean"),
        "avg_assets_millions": ("ASSET", "mean"),
        "n_banks": ("CERT", "nunique"),
        "n_obs": ("NIM", "count"),
    }
    if "ROA" in df.columns:
        agg_spec["avg_roa"] = ("ROA", "mean")
    agg = df.groupby("STALP", as_index=False).agg(**agg_spec).sort_values("STALP")
    agg["avg_assets_millions"] = agg["avg_assets_millions"] / 1000.0
    if "avg_roa" not in agg.columns:
        agg["avg_roa"] = np.nan
    agg["state_name"] = agg["STALP"].map(lambda code: STATE_NAMES.get(code, code))
    return agg.rename(columns={"STALP": "state_code"}).to_dict(orient="records")


def load_sod_cbsa_exposure(sod_dir: Path, crosswalk: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    files = sorted(sod_dir.glob("sod_*.csv"))
    if not files:
        raise FileNotFoundError(f"No SOD files found in {sod_dir}")

    frames: list[pd.DataFrame] = []
    for path in files:
        chunk = pd.read_csv(
            path,
            usecols=lambda c: c.upper() in {"CERT", "YEAR", "STCNTYBR", "DEPSUMBR"},
            low_memory=False,
        )
        chunk.columns = [c.upper() for c in chunk.columns]
        frames.append(chunk)

    sod = pd.concat(frames, ignore_index=True)
    sod["CERT"] = pd.to_numeric(sod["CERT"], errors="coerce").astype("Int64")
    sod["SOD_YEAR"] = pd.to_numeric(sod["YEAR"], errors="coerce").astype("Int64")
    sod["DEPSUMBR"] = pd.to_numeric(sod["DEPSUMBR"], errors="coerce").fillna(0.0)
    sod["county_fips"] = normalize_county_fips(sod["STCNTYBR"])
    sod = sod.dropna(subset=["CERT", "SOD_YEAR", "county_fips"]).copy()

    total_dep = (
        sod.groupby(["CERT", "SOD_YEAR"], as_index=False)["DEPSUMBR"]
        .sum()
        .rename(columns={"DEPSUMBR": "total_sod_dep"})
    )
    mapped = sod.merge(crosswalk, on="county_fips", how="left")
    matched = mapped.dropna(subset=["cbsa_code"]).copy()
    exposure = (
        matched.groupby(["CERT", "SOD_YEAR", "cbsa_code", "cbsa_title", "cbsa_type"], as_index=False)
        .agg(dep_cbsa=("DEPSUMBR", "sum"), county_count=("county_fips", "nunique"))
    )
    exposure = exposure.merge(total_dep, on=["CERT", "SOD_YEAR"], how="left")
    exposure["matched_dep_share"] = np.where(
        exposure["total_sod_dep"] > 0,
        exposure["dep_cbsa"] / exposure["total_sod_dep"],
        np.nan,
    )
    exposure["CERT"] = normalize_cert_ids(exposure["CERT"])

    coverage = {
        "sod_rows": int(len(sod)),
        "mapped_sod_rows": int(len(matched)),
        "mapped_row_share": float(len(matched) / len(sod)) if len(sod) else 0.0,
        "total_sod_dep": float(total_dep["total_sod_dep"].sum()),
        "mapped_sod_dep": float(matched["DEPSUMBR"].sum()),
    }
    return exposure, coverage


def build_msa_summary(panel: pd.DataFrame, sod_dir: Path, crosswalk: pd.DataFrame) -> tuple[list[dict[str, object]], dict[str, object]]:
    exposure, coverage = load_sod_cbsa_exposure(sod_dir, crosswalk)
    msa_panel = panel.copy()
    msa_panel["CERT"] = normalize_cert_ids(msa_panel["CERT"])
    msa_panel["SOD_YEAR"] = map_sod_year_to_quarter(msa_panel["REPDTE"]).astype("Int64")
    msa_panel = msa_panel.merge(exposure, on=["CERT", "SOD_YEAR"], how="inner")
    msa_panel = msa_panel.dropna(subset=["NIM", "dep_cbsa", "cbsa_code"]).copy()
    msa_panel = msa_panel[msa_panel["dep_cbsa"] > 0].copy()

    rows: list[dict[str, object]] = []
    for (cbsa_code, cbsa_title, cbsa_type), sub in msa_panel.groupby(["cbsa_code", "cbsa_title", "cbsa_type"], sort=True):
        weights = sub["dep_cbsa"].astype(float)
        if float(weights.sum()) <= 0:
            continue

        def weighted_average(column: str) -> float | None:
            vals = pd.to_numeric(sub[column], errors="coerce")
            mask = vals.notna() & weights.notna()
            if not mask.any():
                return None
            return float(np.average(vals[mask], weights=weights[mask]))

        rows.append(
            {
                "cbsa_code": cbsa_code,
                "cbsa_title": cbsa_title,
                "cbsa_type": cbsa_type,
                "avg_nim": weighted_average("NIM"),
                "avg_roa": weighted_average("ROA"),
                "avg_assets_millions": weighted_average("ASSET") / 1000.0 if weighted_average("ASSET") is not None else None,
                "avg_ln_assets": weighted_average("LN_ASSETS"),
                "matched_dep_share": float(sub["matched_dep_share"].mean()),
                "n_banks": int(sub["CERT"].nunique()),
                "n_obs": int(len(sub)),
                "n_counties": int(sub["county_count"].max()),
                "latest_sod_year": int(sub["SOD_YEAR"].max()),
            }
        )

    rows = sorted(rows, key=lambda row: (row["avg_nim"] is None, -(row["avg_nim"] or 0.0), row["cbsa_title"]))
    coverage.update(
        {
            "msa_count": len(rows),
            "matched_panel_rows": int(len(msa_panel)),
            "matched_panel_bank_count": int(msa_panel["CERT"].nunique()) if not msa_panel.empty else 0,
        }
    )
    return rows, coverage


def build_geography_payload(panel: pd.DataFrame, cfg: dict, root: Path) -> dict[str, object]:
    geo_cfg = cfg["geography"]
    crosswalk_path = root / geo_cfg["cbsa_crosswalk"]
    crosswalk = load_cbsa_crosswalk(crosswalk_path, header_row=geo_cfg.get("cbsa_header_row", 2))
    state_rows = build_state_summary(panel)
    msa_rows, coverage = build_msa_summary(panel, root / cfg["paths"]["raw"] / "fdic_sod", crosswalk)
    return {
        "metadata": {
            "state_method": "HQ state from the bank-quarter panel",
            "msa_method": "Branch-deposit-weighted CBSA exposure using annual SOD carried forward by SOD year",
            "crosswalk_source": geo_cfg["cbsa_source_url"],
            "crosswalk_vintage": geo_cfg["cbsa_vintage"],
            "default_cbsa_filter": {
                "scope": "screened_metros",
                "cbsa_type": "Metropolitan",
                "min_banks": 20,
                "min_matched_dep_share": 0.10,
                "default_sort": "n_banks",
            },
            "state_count": len(state_rows),
            "msa_count": len(msa_rows),
            **coverage,
        },
        "states": state_rows,
        "msas": msa_rows,
    }
