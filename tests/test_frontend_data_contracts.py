from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DOCS_DATA = ROOT / "docs" / "data"


def load_json(name: str):
    return json.loads((DOCS_DATA / name).read_text(encoding="utf-8"))


def test_model_results_json_has_expected_schema_and_coverage():
    rows = load_json("model_results.json")
    assert isinstance(rows, list)
    assert len(rows) >= 250

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
    assert ("loan_mix_fe", "LN_ASSETS") in model_terms
    assert ("distress_logit", "NIM_W") in model_terms


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
