from __future__ import annotations

import math

import pandas as pd

from nimscale.bank_panel import (
    add_core_features,
    add_sod_franchise_features,
    coerce_numeric,
    map_sod_year_to_quarter,
    parse_fdic_quarter_date,
)


def test_parse_fdic_quarter_date_handles_multiple_formats():
    s = pd.Series(["20241231", "2024-09-30", "2024/06/30"])
    out = parse_fdic_quarter_date(s)
    assert out.dt.year.tolist() == [2024, 2024, 2024]
    assert out.dt.month.tolist() == [12, 9, 6]


def test_within_between_decomposition_reconstructs_ln_assets():
    df = pd.DataFrame(
        {
            "CERT": ["1", "1", "2", "2"],
            "REPDTE": pd.to_datetime(["2024-03-31", "2024-06-30", "2024-03-31", "2024-06-30"]),
            "ASSET": [100.0, 200.0, 400.0, 800.0],
            "DEP": [90.0, 180.0, 350.0, 700.0],
            "EQ": [10.0, 20.0, 50.0, 90.0],
            "LNLSNET": [60.0, 120.0, 250.0, 500.0],
            "NIMY": [3.2, 3.1, 2.8, 2.7],
        }
    )
    out = add_core_features(
        df=df,
        bank_id_col="CERT",
        date_col="REPDTE",
        assets_col="ASSET",
        deposits_col="DEP",
        equity_col="EQ",
        loans_col="LNLSNET",
        nim_col="NIMY",
    )
    lhs = out["LN_ASSETS"]
    rhs = out["AVG_LN_ASSETS_BANK"] + out["LN_ASSETS_WITHIN"]
    assert ((lhs - rhs).abs() < 1e-12).all()


def test_sod_year_mapping():
    dates = pd.Series(pd.to_datetime(["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31", "2025-03-31"]))
    sod_year = map_sod_year_to_quarter(dates)
    assert sod_year.tolist() == [2023, 2023, 2024, 2024, 2024]


def test_coerce_numeric_converts_numeric_like_columns_only():
    df = pd.DataFrame(
        {
            "CERT": ["1", "2"],
            "ASSET": ["100.5", ""],
            "NAME": ["Bank A", "Bank B"],
            "MIXED": ["10", "oops"],
        }
    )

    out = coerce_numeric(df, exclude=["NAME"])

    assert out["CERT"].tolist() == [1, 2]
    assert out["ASSET"].iloc[0] == 100.5
    assert pd.isna(out["ASSET"].iloc[1])
    assert out["NAME"].tolist() == ["Bank A", "Bank B"]
    assert out["MIXED"].tolist() == ["10", "oops"]


def test_add_sod_franchise_features_builds_bank_year_growth_once():
    df = pd.DataFrame(
        {
            "CERT": ["1", "1", "1", "1"],
            "SOD_YEAR": [2020, 2020, 2021, 2021],
            "BRANCH_COUNT": [10, 10, 12, 12],
            "SOD_DEP_TOTAL": [100.0, 100.0, 121.0, 121.0],
            "DEP_PER_BRANCH": [10.0, 10.0, 10.0833333333, 10.0833333333],
        }
    )

    out = add_sod_franchise_features(df, bank_id_col="CERT", sod_year_col="SOD_YEAR")

    assert pd.isna(out.loc[0, "BRANCH_COUNT_GROWTH_YOY"])
    assert pd.isna(out.loc[1, "SOD_DEP_TOTAL_GROWTH_YOY"])
    assert abs(out.loc[2, "BRANCH_COUNT_GROWTH_YOY"] - (math.log(12.0) - math.log(10.0))) < 1e-12
    assert abs(out.loc[3, "SOD_DEP_TOTAL_GROWTH_YOY"] - (math.log(121.0) - math.log(100.0))) < 1e-12
