PYTHON ?= python

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

metadata:
	$(PYTHON) scripts/00_fetch_metadata.py

core:
	$(PYTHON) scripts/01_download_fdic_financials.py --start 2010-03-31 --end 2025-12-31
	$(PYTHON) scripts/04_download_rates.py
	$(PYTHON) scripts/06_build_core_panel.py
	$(PYTHON) scripts/08_run_baseline_regressions.py

sod:
	$(PYTHON) scripts/02_download_fdic_sod.py --start-year 2010 --end-year 2025
	$(PYTHON) scripts/07_build_sod_panel.py

history:
	$(PYTHON) scripts/03_download_fdic_history.py --start-date 2010-01-01 --end-date 2025-12-31
	$(PYTHON) scripts/09_event_study_acquirers.py

test:
	$(PYTHON) -m pytest -q
