from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
DOCS_DATA = ROOT / "docs" / "data"
DATA_PAGE = ROOT / "docs" / "data.html"


def load_json(name: str):
    return json.loads((DOCS_DATA / name).read_text(encoding="utf-8"))


def load_export_module():
    path = ROOT / "scripts" / "15_export_frontend_data.py"
    spec = importlib.util.spec_from_file_location("export_frontend_data_mod", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_model_results_json_has_expected_schema_and_coverage():
    rows = load_json("model_results.json")
    assert isinstance(rows, list)
    assert len(rows) >= 170

    required = {"group", "model", "term", "coef", "std_err", "p_value", "ci_low", "ci_high", "nobs"}
    for row in rows:
        assert required.issubset(row)

    groups = {row["group"] for row in rows}
    assert {
        "baseline",
        "nim_decomposition",
        "franchise_dilution",
        "rate_cycle",
        "event_study",
        "threshold",
        "robustness",
        "extension",
    }.issubset(groups)

    model_terms = {(row["model"], row["term"]) for row in rows}
    assert ("within_fe_size", "LN_ASSETS") in model_terms
    assert ("int_income_fe", "LN_ASSETS") in model_terms
    assert ("int_expense_fe", "LN_ASSETS") in model_terms
    assert ("rate_cycle_fedfunds", "LN_ASSETS_x_FEDFUNDS") in model_terms


def test_rolling_coefficients_json_has_ordered_windows_and_required_fields():
    payload = load_json("rolling_coefficients.json")
    assert payload["model"] == "rolling_within_fe_size"
    assert payload["term"] == "LN_ASSETS"

    rows = payload["rows"]
    assert len(rows) >= 40

    required = {
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
    }
    for row in rows:
        assert required.issubset(row)

    mids = pd.to_datetime([row["window_mid"] for row in rows])
    starts = pd.to_datetime([row["window_start"] for row in rows])
    ends = pd.to_datetime([row["window_end"] for row in rows])

    assert mids.is_monotonic_increasing
    assert starts.is_monotonic_increasing
    assert ends.is_monotonic_increasing
    assert {row["window_quarters"] for row in rows} == {20}
    assert all(start <= mid <= end for start, mid, end in zip(starts, mids, ends))
    assert all(row["nobs"] > 0 for row in rows)
    assert all(row["n_banks"] > 0 for row in rows)


def test_metadata_and_chart_exports_cover_expected_public_artifacts():
    metadata = load_json("metadata.json")
    assert metadata["project"] == "NimScale"
    assert metadata["n_banks"] >= 8000
    assert metadata["n_quarters"] == 64
    assert {
        "within_bank_size_nim",
        "between_bank_size_nim",
        "roa_offsets_nim",
        "rate_cycle",
        "lagged_size_effect",
    }.issubset(metadata["headline_results"])

    event = load_json("event_study.json")
    assert event["model"] == "acquirer_event_study"
    assert any(row["event_time"] == -1 for row in event["coefficients"])

    threshold = load_json("threshold_crossing.json")
    assert {"threshold_cross_10b", "threshold_cross_50b", "threshold_cross_100b"} == set(threshold)

    cross_section = load_json("cross_section.json")
    assert len(cross_section["binscatter"]) == 20
    assert len(cross_section["deciles"]) == 10

    geography = load_json("geography.json")
    assert {"metadata", "states", "msas"} == set(geography)


def test_geography_json_has_expected_schema_and_coverage():
    geography = load_json("geography.json")
    metadata = geography["metadata"]
    states = geography["states"]
    msas = geography["msas"]

    assert metadata["crosswalk_source"].endswith("list1_2023.xlsx")
    assert metadata["state_count"] == len(states)
    assert metadata["msa_count"] == len(msas)
    assert metadata["state_count"] >= 50
    assert metadata["msa_count"] >= 500
    assert 0.75 <= metadata["mapped_row_share"] <= 1.0
    assert metadata["matched_panel_bank_count"] > 0

    state_required = {
        "state_code",
        "state_name",
        "avg_nim",
        "avg_roa",
        "avg_ln_assets",
        "avg_assets_millions",
        "n_banks",
        "n_obs",
    }
    for row in states:
        assert state_required.issubset(row)
        assert len(row["state_code"]) == 2
        assert row["n_banks"] > 0
        assert row["n_obs"] > 0

    msa_required = {
        "cbsa_code",
        "cbsa_title",
        "cbsa_type",
        "avg_nim",
        "avg_roa",
        "avg_ln_assets",
        "avg_assets_millions",
        "matched_dep_share",
        "n_banks",
        "n_obs",
        "n_counties",
        "latest_sod_year",
    }
    for row in msas[:100]:
        assert msa_required.issubset(row)
        assert len(row["cbsa_code"]) == 5
        assert row["n_banks"] > 0
        assert row["n_obs"] > 0
        assert row["n_counties"] > 0
        assert 0 <= row["matched_dep_share"] <= 1


def test_headline_export_logic_is_table_driven(tmp_path):
    module = load_export_module()

    pd.DataFrame(
        [
            {"model": "within_fe_size", "term": "LN_ASSETS", "coef": -0.09, "p_value": 0.0002},
            {"model": "between_bank_means", "term": "AVG_LN_ASSETS", "coef": -0.085, "p_value": 0.0004},
        ]
    ).to_csv(tmp_path / "regression_results.csv", index=False)
    pd.DataFrame(
        [
            {"model": "h6_roa_fe", "term": "LN_ASSETS", "coef": 0.205, "p_value": 0.0006},
            {"model": "rob_lagged_controls_fe", "term": "LN_ASSETS", "coef": -0.119, "p_value": 0.0123},
        ]
    ).to_csv(tmp_path / "extension_results.csv", index=False)
    pd.DataFrame(
        [{"model": "rate_cycle_fedfunds", "term": "LN_ASSETS_x_FEDFUNDS", "coef": -0.012, "p_value": 0.0008}]
    ).to_csv(tmp_path / "rate_cycle_results.csv", index=False)

    headlines = module.build_headline_results(tmp_path)

    assert headlines["within_bank_size_nim"]["coef"] == -0.09
    assert headlines["within_bank_size_nim"]["p"] == "<0.001"
    assert headlines["between_bank_size_nim"]["coef"] == -0.085
    assert headlines["roa_offsets_nim"]["coef"] == 0.205
    assert headlines["rate_cycle"]["coef"] == -0.012
    assert headlines["lagged_size_effect"]["p"] == "0.012"


def test_published_frontend_data_matches_output_frontend_when_present():
    if os.environ.get("BANKNIM_VERIFY_PUBLISHED_FRONTEND") != "1":
        pytest.skip("set BANKNIM_VERIFY_PUBLISHED_FRONTEND=1 to verify publish sync")

    output_dir = ROOT / "output" / "frontend"
    if not output_dir.exists():
        pytest.skip("output/frontend does not exist in this checkout")

    output_files = sorted(path.name for path in output_dir.glob("*.json"))
    if not output_files:
        pytest.skip("output/frontend has no JSON exports to compare")

    docs_files = sorted(path.name for path in DOCS_DATA.glob("*.json"))
    assert output_files == docs_files

    for name in output_files:
        output_payload = json.loads((output_dir / name).read_text(encoding="utf-8"))
        docs_payload = load_json(name)
        assert output_payload == docs_payload


def test_data_page_includes_geography_section_and_hooks():
    html = DATA_PAGE.read_text(encoding="utf-8")
    assert 'id="geography"' in html
    assert "fetch('data/geography.json')" in html
    assert 'id="stateTileGrid"' in html
    assert 'id="msaTableBody"' in html
    assert 'id="geoMetricToggles"' in html
