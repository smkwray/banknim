from __future__ import annotations

from pathlib import Path

import pandas as pd

from nimscale.geography import (
    build_geography_payload,
    build_msa_summary,
    load_cbsa_crosswalk,
    normalize_county_fips,
)


def write_crosswalk(path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "CBSA Code": "11111",
                "CBSA Title": "Alpha, AL",
                "Metropolitan/Micropolitan Statistical Area": "Metropolitan Statistical Area",
                "County/County Equivalent": "Autauga County",
                "State Name": "Alabama",
                "FIPS State Code": "01",
                "FIPS County Code": "001",
            },
            {
                "CBSA Code": "11111",
                "CBSA Title": "Alpha, AL",
                "Metropolitan/Micropolitan Statistical Area": "Metropolitan Statistical Area",
                "County/County Equivalent": "Autauga County",
                "State Name": "Alabama",
                "FIPS State Code": "01",
                "FIPS County Code": "001",
            },
            {
                "CBSA Code": "22222",
                "CBSA Title": "Beta, AL",
                "Metropolitan/Micropolitan Statistical Area": "Micropolitan Statistical Area",
                "County/County Equivalent": "Baldwin County",
                "State Name": "Alabama",
                "FIPS State Code": "01",
                "FIPS County Code": "003",
            },
            {
                "CBSA Code": "33333",
                "CBSA Title": "Gamma, GA",
                "Metropolitan/Micropolitan Statistical Area": "Metropolitan Statistical Area",
                "County/County Equivalent": "Appling County",
                "State Name": "Georgia",
                "FIPS State Code": "13",
                "FIPS County Code": "001",
            },
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, startrow=2)


def write_sod_fixture(root: Path) -> None:
    sod_dir = root / "data" / "raw" / "fdic_sod"
    sod_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"CERT": 101, "YEAR": 2019, "STCNTYBR": "01001", "DEPSUMBR": 100.0},
            {"CERT": 202, "YEAR": 2019, "STCNTYBR": "13001", "DEPSUMBR": 80.0},
        ]
    ).to_csv(sod_dir / "sod_2019.csv", index=False)
    pd.DataFrame(
        [
            {"CERT": 101, "YEAR": 2020, "STCNTYBR": "01003", "DEPSUMBR": 120.0},
            {"CERT": 202, "YEAR": 2020, "STCNTYBR": "13001", "DEPSUMBR": 90.0},
        ]
    ).to_csv(sod_dir / "sod_2020.csv", index=False)


def panel_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "CERT": 101,
                "REPDTE": pd.Timestamp("2020-03-31"),
                "STALP": "AL",
                "NIM": 3.1,
                "ROA": 0.7,
                "LN_ASSETS": 12.0,
                "ASSET": 150000.0,
            },
            {
                "CERT": 101,
                "REPDTE": pd.Timestamp("2020-09-30"),
                "STALP": "AL",
                "NIM": 3.4,
                "ROA": 0.9,
                "LN_ASSETS": 12.2,
                "ASSET": 170000.0,
            },
            {
                "CERT": 202,
                "REPDTE": pd.Timestamp("2020-06-30"),
                "STALP": "GA",
                "NIM": 2.8,
                "ROA": 0.6,
                "LN_ASSETS": 11.3,
                "ASSET": 90000.0,
            },
            {
                "CERT": 202,
                "REPDTE": pd.Timestamp("2020-12-31"),
                "STALP": "GA",
                "NIM": 2.9,
                "ROA": 0.65,
                "LN_ASSETS": 11.5,
                "ASSET": 95000.0,
            },
        ]
    )


def test_normalize_county_fips_handles_mixed_formats():
    out = normalize_county_fips(pd.Series([36061, "36061.0", "36-061", "", None]))
    assert out.fillna("NA").tolist() == ["36061", "36061", "36061", "NA", "NA"]


def test_load_cbsa_crosswalk_normalizes_and_deduplicates(tmp_path):
    crosswalk_path = tmp_path / "config" / "reference" / "list1_2023.xlsx"
    write_crosswalk(crosswalk_path)

    crosswalk = load_cbsa_crosswalk(crosswalk_path, header_row=2)

    assert len(crosswalk) == 3
    assert set(crosswalk.columns) == {
        "county_fips",
        "cbsa_code",
        "cbsa_title",
        "cbsa_type",
        "county_name",
        "state_name",
    }
    alpha = crosswalk.loc[crosswalk["cbsa_code"] == "11111"].iloc[0]
    assert alpha["county_fips"] == "01001"
    assert alpha["cbsa_title"] == "Alpha, AL"
    assert alpha["cbsa_type"] == "Metropolitan"


def test_build_msa_summary_handles_int_panel_cert_and_sod_timing(tmp_path):
    crosswalk_path = tmp_path / "config" / "reference" / "list1_2023.xlsx"
    write_crosswalk(crosswalk_path)
    write_sod_fixture(tmp_path)

    crosswalk = load_cbsa_crosswalk(crosswalk_path, header_row=2)
    rows, coverage = build_msa_summary(panel_fixture(), tmp_path / "data" / "raw" / "fdic_sod", crosswalk)

    assert rows
    assert coverage["matched_panel_rows"] == 4
    assert coverage["matched_panel_bank_count"] == 2

    by_code = {row["cbsa_code"]: row for row in rows}
    assert {"11111", "22222", "33333"} == set(by_code)
    assert by_code["11111"]["n_obs"] == 1
    assert by_code["11111"]["latest_sod_year"] == 2019
    assert by_code["22222"]["n_obs"] == 1
    assert by_code["22222"]["latest_sod_year"] == 2020
    assert by_code["33333"]["n_obs"] == 2
    assert by_code["33333"]["latest_sod_year"] == 2020


def test_build_geography_payload_returns_non_empty_state_and_msa_summaries(tmp_path):
    crosswalk_path = tmp_path / "config" / "reference" / "list1_2023.xlsx"
    write_crosswalk(crosswalk_path)
    write_sod_fixture(tmp_path)

    cfg = {
        "paths": {"raw": "data/raw"},
        "geography": {
            "cbsa_crosswalk": "config/reference/list1_2023.xlsx",
            "cbsa_header_row": 2,
            "cbsa_vintage": "Jul. 2023 delineation file",
            "cbsa_source_url": "https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2023/delineation-files/list1_2023.xlsx",
        },
    }

    payload = build_geography_payload(panel_fixture(), cfg, tmp_path)

    assert set(payload) == {"metadata", "states", "msas"}
    assert payload["states"]
    assert payload["msas"]

    metadata = payload["metadata"]
    assert metadata["state_count"] == len(payload["states"])
    assert metadata["msa_count"] == len(payload["msas"])
    assert metadata["matched_panel_rows"] == 4
    assert metadata["matched_panel_bank_count"] == 2
    assert metadata["mapped_sod_rows"] == 4
    assert metadata["mapped_row_share"] == 1.0
    assert metadata["default_cbsa_filter"] == {
        "scope": "screened_metros",
        "cbsa_type": "Metropolitan",
        "min_banks": 20,
        "min_matched_dep_share": 0.10,
        "default_sort": "n_banks",
    }

    state_row = payload["states"][0]
    assert {
        "state_code",
        "state_name",
        "avg_nim",
        "avg_ln_assets",
        "avg_assets_millions",
        "n_banks",
        "n_obs",
    }.issubset(state_row)

    msa_row = payload["msas"][0]
    assert {
        "cbsa_code",
        "cbsa_title",
        "cbsa_type",
        "avg_nim",
        "avg_ln_assets",
        "avg_assets_millions",
        "matched_dep_share",
        "n_banks",
        "n_obs",
        "n_counties",
        "latest_sod_year",
    }.issubset(msa_row)
