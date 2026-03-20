PYTHON ?= python3
SMOKE_CONFIG ?= config/project.smoke.yaml
PYTEST_ADDOPTS ?= -o cache_dir=/tmp/banknim-pytest-cache

.PHONY: setup metadata download panel core sod history robustness \
	extensions tier1 tier2 tier3 frontend smoke-fixture smoke test all

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .[test]

metadata:
	$(PYTHON) scripts/00_fetch_metadata.py

download:
	$(PYTHON) scripts/01_download_fdic_financials.py --start 2010-03-31 --end 2025-12-31
	$(PYTHON) scripts/02_download_fdic_sod.py --start-year 2010 --end-year 2025
	$(PYTHON) scripts/03_download_fdic_history.py --start-date 2010-01-01 --end-date 2025-12-31
	$(PYTHON) scripts/04_download_rates.py

panel:
	$(PYTHON) scripts/06_build_core_panel.py
	$(PYTHON) scripts/07_build_sod_panel.py

core:
	$(PYTHON) scripts/08_run_baseline_regressions.py
	$(PYTHON) scripts/10_run_franchise_dilution.py
	$(PYTHON) scripts/11_run_rate_cycle.py
	$(PYTHON) scripts/12_run_threshold_crossing.py

history:
	$(PYTHON) scripts/09_event_study_acquirers.py

robustness:
	$(PYTHON) scripts/13_run_robustness.py

extensions:
	$(PYTHON) scripts/14_run_extensions.py

tier1:
	$(PYTHON) scripts/16_run_nim_decomposition.py
	$(PYTHON) scripts/17_run_equity_as_outcome.py
	$(PYTHON) scripts/18_run_efficiency.py

tier2:
	$(PYTHON) scripts/19_run_loan_composition.py
	$(PYTHON) scripts/20_run_failure_prediction.py

tier3:
	$(PYTHON) scripts/21_run_rolling_coefficients.py

frontend:
	$(PYTHON) scripts/15_export_frontend_data.py
	cp output/frontend/*.json docs/data/

smoke-fixture:
	$(PYTHON) scripts/98_prepare_smoke_fixture.py --config $(SMOKE_CONFIG)

smoke: smoke-fixture
	$(PYTHON) scripts/06_build_core_panel.py --config $(SMOKE_CONFIG)
	$(PYTHON) scripts/08_run_baseline_regressions.py --config $(SMOKE_CONFIG)
	$(PYTHON) scripts/11_run_rate_cycle.py --config $(SMOKE_CONFIG)
	$(PYTHON) scripts/13_run_robustness.py --config $(SMOKE_CONFIG)
	$(PYTHON) scripts/14_run_extensions.py --config $(SMOKE_CONFIG)
	$(PYTHON) scripts/15_export_frontend_data.py --config $(SMOKE_CONFIG)

test:
	PYTEST_ADDOPTS='$(PYTEST_ADDOPTS)' $(PYTHON) -m pytest -q

all: metadata download panel core history robustness extensions tier1 tier2 tier3 frontend
