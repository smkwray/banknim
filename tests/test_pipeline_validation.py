from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest

from nimscale.validation import ValidationError


ROOT = Path(__file__).resolve().parents[1]


def load_script_module(script_name: str):
    path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", "_mod"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_test_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "project.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                '  name: "NimScale Test"',
                '  sample_start: "2020-03-31"',
                '  sample_end: "2021-12-31"',
                "  min_bank_quarters: 4",
                "  winsor_pct: 0.01",
                "paths:",
                '  raw: "data/raw"',
                '  external: "data/external"',
                '  interim: "data/interim"',
                '  final: "data/final"',
                '  figures: "output/figures"',
                '  tables: "output/tables"',
                '  logs: "output/logs"',
                '  frontend: "output/frontend"',
                "fdic:",
                '  base_url: "https://api.fdic.gov/banks"',
                '  api_key_env: "FDIC_API_KEY"',
                "  page_size: 10000",
                "  endpoints:",
                '    financials: "financials"',
                '    institutions: "institutions"',
                '    history: "history"',
                '    sod: "sod"',
                "  metadata_urls: {}",
                "  default_filters: {}",
                "  financial_fields_core: []",
                "  column_candidates:",
                "    bank_id: [CERT]",
                "    quarter_date: [REPDTE]",
                "    total_assets: [ASSET]",
                "    total_deposits: [DEP]",
                "    equity: [EQ]",
                "    loans_net: [LNLSNET]",
                "    nim: [NIMY]",
                "rates:",
                "  series: {}",
                "ffiec:",
                '  bulk_page: ""',
                '  user_guide_page: ""',
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def test_core_panel_duplicate_audit_is_written_before_dedup(tmp_path, monkeypatch):
    script = load_script_module("06_build_core_panel.py")
    config_path = write_test_config(tmp_path)

    raw_dir = tmp_path / "data" / "raw" / "fdic_financials"
    raw_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"CERT": 1, "REPDTE": "2020-03-31", "ASSET": 100.0, "DEP": 80.0, "EQ": 10.0, "LNLSNET": 70.0, "NIMY": 3.0},
            {"CERT": 1, "REPDTE": "2020-03-31", "ASSET": 100.0, "DEP": 80.0, "EQ": 10.0, "LNLSNET": 70.0, "NIMY": 3.0},
            {"CERT": 1, "REPDTE": "2020-06-30", "ASSET": 110.0, "DEP": 88.0, "EQ": 11.0, "LNLSNET": 75.0, "NIMY": 2.9},
        ]
    ).to_csv(raw_dir / "financials_fixture.csv", index=False)

    monkeypatch.setattr(script, "project_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["06_build_core_panel.py", "--config", str(config_path)])

    script.main()

    dupes = pd.read_csv(tmp_path / "output" / "tables" / "duplicate_keys.csv")
    summary = pd.read_csv(tmp_path / "output" / "tables" / "core_sample_summary.csv")

    assert len(dupes) == 2
    assert int(summary.loc[summary["metric"] == "duplicate_row_count", "value"].iloc[0]) == 2
    assert int(summary.loc[summary["metric"] == "duplicate_key_count", "value"].iloc[0]) == 1


def test_baseline_models_raise_when_required_controls_are_missing(tmp_path, monkeypatch):
    script = load_script_module("08_run_baseline_regressions.py")
    config_path = write_test_config(tmp_path)

    interim_dir = tmp_path / "data" / "interim"
    interim_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "CERT": ["1", "1", "2", "2"],
            "REPDTE": pd.to_datetime(["2020-03-31", "2020-06-30", "2020-03-31", "2020-06-30"]),
            "NIM": [3.0, 2.9, 3.1, 3.0],
            "LN_ASSETS": [4.6, 4.7, 4.8, 4.9],
            "ASSET_GROWTH_QOQ": [0.1, 0.1, 0.1, 0.1],
            "DEP_GROWTH_QOQ": [0.1, 0.1, 0.1, 0.1],
            "FEDFUNDS": [1.0, 1.1, 1.0, 1.1],
            "SLOPE_10Y_3M": [0.5, 0.6, 0.5, 0.6],
        }
    ).to_parquet(interim_dir / "bank_panel.parquet", index=False)

    monkeypatch.setattr(script, "project_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["08_run_baseline_regressions.py", "--config", str(config_path)])

    with pytest.raises(ValidationError):
        script.main()


def test_rate_cycle_models_raise_when_rate_series_are_missing(tmp_path, monkeypatch):
    script = load_script_module("11_run_rate_cycle.py")
    config_path = write_test_config(tmp_path)

    interim_dir = tmp_path / "data" / "interim"
    interim_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "CERT": ["1", "1", "2", "2"],
            "REPDTE": pd.to_datetime(["2020-03-31", "2020-06-30", "2020-03-31", "2020-06-30"]),
            "NIM": [3.0, 2.9, 3.1, 3.0],
            "LN_ASSETS": [4.6, 4.7, 4.8, 4.9],
            "EQ_RATIO": [0.1, 0.1, 0.11, 0.11],
            "LOANS_SHARE": [0.7, 0.71, 0.72, 0.73],
            "FEDFUNDS": [1.0, None, 1.0, None],
            "SLOPE_10Y_3M": [0.5, None, 0.5, None],
        }
    ).to_parquet(interim_dir / "bank_panel.parquet", index=False)

    monkeypatch.setattr(script, "project_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["11_run_rate_cycle.py", "--config", str(config_path)])

    with pytest.raises(ValidationError):
        script.main()


def test_extension_optional_blocks_skip_explicitly(capsys):
    script = load_script_module("14_run_extensions.py")
    df = pd.DataFrame(
        {
            "CERT": ["1", "1", "2", "2"],
            "REPDTE": pd.to_datetime(["2020-03-31", "2020-06-30", "2020-03-31", "2020-06-30"]),
            "NIM": [3.0, 2.9, 3.1, 3.0],
            "NIM_W": [3.0, 2.9, 3.1, 3.0],
            "LN_ASSETS": [4.6, 4.7, 4.8, 4.9],
            "EQ_RATIO_W": [0.1, 0.1, 0.11, 0.11],
            "LOANS_SHARE_W": [0.7, 0.71, 0.72, 0.73],
        }
    )

    result = script.run_h6_fee_offset(df)
    captured = capsys.readouterr()

    assert result == []
    assert "skipping h6_roa_fe" in captured.out
    assert "skipping h6_intexp_fe" in captured.out
    assert "skipping h6_nonint_margin_fe" in captured.out
