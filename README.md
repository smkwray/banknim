# Bank Size, Growth, and Net Interest Margin

[![CI](https://github.com/smkwray/banknim/actions/workflows/ci.yml/badge.svg)](https://github.com/smkwray/banknim/actions/workflows/ci.yml)

**https://smkwray.github.io/banknim/**

A scripted public-data pipeline studying the relationship between bank size, asset growth, and net interest margin (NIM) across 8,032 FDIC-insured commercial banks over 64 quarters (2010–2025).

---

## What this project does

This project estimates how bank size is associated with net interest margins, and whether the relationship operates through being big (cross-sectional), becoming bigger (within-bank growth), franchise dilution, rate-cycle amplification, regulatory threshold crossing, or time-varying size effects. Every data source is free, the pipeline is scripted end to end, and full reruns depend on external-data availability plus the documented setup prerequisites.

## Research question

**What explains changes in bank net interest margin, and is the lower NIM of large banks mainly a fact about being big or becoming bigger?**

---

## Findings

Results are organized by hypothesis. All coefficient estimates come from panel models on 373,220 bank-quarter observations unless otherwise noted.

### Headline caveat

- **Raw NIM (no winsorization):** size effect is insignificant (−0.01, p=0.87)
- The most reassuring evidence comes from first-difference (−0.11, p<0.001) and lagged-controls (−0.12, p<0.001) specifications
- The baseline winsorized result is informative, but it should not be read in isolation

### H1: Between-bank size penalty

- **Coefficient:** −0.085 (p<0.001)
- Banks 10× larger run ~20 bp lower NIM in the cross-section
- OLS on 8,023 bank-level means, R² = 0.23

### H2: Within-bank size penalty

- **Coefficient:** −0.090 (p<0.001)
- Bank fixed effects absorb all time-invariant bank characteristics
- Survives first-differencing (−0.11, p<0.001) and lagged controls (−0.12, p<0.001)

### H3: Franchise dilution

- **Branch-growth interaction:** −0.58 (p<0.01)
- Growth without franchise deepening hurts margins
- Rapid branch expansion dilutes the deposit franchise

### H4: Acquisitions

- **Event study:** NIM V-shape around deals
- Pre-event decline from +0.11 (t−8) to +0.02 (t−2) relative to t−1
- Post-event rebound peaking at t+2 (+0.12) before fading
- Acquirers may be buying when their own margins are under pressure

### H5: Rate cycles

- **Size×FedFunds:** −0.012 (p<0.001)
- The size penalty on NIM worsens during rate-hiking cycles
- A steeper yield curve partially offsets (+0.015)

### H6: Fee offset

- **ROA rises with size:** +0.205 (p<0.001)
- **Noninterest margin:** +0.31 (p<0.001)
- Large banks offset lower NIM with higher fee income — a different earnings model, not a worse one

### NIM decomposition

- **Interest income FE:** +0.154 (p<0.001)
- **Interest expense FE:** +0.245 (p<0.001)
- The NIM penalty appears to come more from the funding-cost side than from lower asset yields

### Equity, efficiency, and loan mix extensions

- **Equity ratio FE:** −0.0177 (p<0.001), consistent with larger banks running thinner capital buffers
- **Efficiency pass-through:** each 1 pp of NIM translates into about +0.514 pp of ROA after controls
- **Loan mix FE:** adding detailed CRE, residential, C&I, and consumer shares attenuates the size coefficient from −0.090 to −0.070, but does not eliminate it

### Time-varying coefficients

- **Rolling 20-quarter within-bank FE:** size coefficient ranges from about −0.033 to −0.144
- Weakest around the mid-2010s, strongest in 2019–2024, and weaker again in the latest window
- The size penalty is persistent, but not constant over time

### Distress extension (exploratory)

- **Failure-only 4-quarter-ahead logit:** higher NIM and larger size predict the event label, while higher ROA lowers it
- This is not a clean “low NIM predicts failure” result, so it should be treated as exploratory rather than headline evidence

### Null results

- **NIM volatility:** size coefficient insignificant (p=0.19)
- **Market power:** local deposit HHI does not predict NIM (p=0.70)

### Robustness flag

- The baseline result depends on trimming extreme NIM outliers
- First-difference (−0.11) and lagged-controls (−0.12) specifications are the most reassuring

---

## Data sources

All data are from official government agencies. No proprietary, scraped, or paywalled data are used.

| Source | Agency | Frequency | Coverage |
|--------|--------|-----------|----------|
| BankFind Financials | FDIC | Quarterly | 2010–2025 |
| Summary of Deposits (SOD) | FDIC | Annual | 2010–2025 |
| Institution History | FDIC | Event-based | Merger/acquisition events |
| FRED Rates | Federal Reserve | Daily/Quarterly | Fed funds, prime, treasury yields |

**373,220** bank-quarter observations. **8,032** unique FDIC-insured commercial banks.

---

## Data QC / sample construction

The review-facing sample flow is explicit rather than implicit:

- Raw FDIC financial rows loaded: 373,220
- Rows with required bank/date/assets parsed successfully: 373,220
- Rows after bank-quarter deduplication: 373,220
- Final core panel rows: 373,220 across 8,032 banks from 2010-03-31 through 2025-12-31

The current public build records zero duplicate bank-quarter keys in `duplicate_keys.csv`. Column-level missingness is exported separately in `missingness_report.csv`, and the site publishes the same staged flow as `sample_selection.json`.

---

## Model inventory

| Model | Description | Key variable | N |
|-------|-------------|-------------|---|
| Within-bank FE | Bank fixed effects, size on NIM | Log assets | 372,644 |
| Between-bank means | OLS on bank-level averages | Avg log assets | 8,023 |
| Growth dynamics | Asset and deposit growth | Asset growth | 364,621 |
| Franchise dilution | Branch-growth interaction | Branch growth × size | 283,636 |
| Rate cycle | Size × fed funds, size × yield slope | Log assets × fed funds | 372,644 |
| Acquisition event study | Event-time dummies around deals | t−8 through t+7 | 23,837 |
| Threshold crossing | Event study at $10B / $50B / $100B | Crossing dummies | 2,217 / 619 / 426 |
| NIM decomposition | Interest income and expense FE outcomes | Log assets | 372,644 |
| Rolling coefficients | Rolling 20-quarter within-bank FE | Log assets | 45 windows |

**Robustness specifications:** Baseline (winsorized), raw NIM (no winsorize), exclude <12 quarters, log deposits as size, first differences, lagged controls.

**Extensions:** NIM volatility, fee offset (ROA, interest expense, noninterest margin), market power (HHI), lagged controls, equity ratio, efficiency pass-through, loan mix, and exploratory distress prediction.

---

## Repository structure

```
banknim/
  config/           Project settings and source inventory
  data/             Raw inputs and processed panels (gitignored)
  docs/             Static site (GitHub Pages)
  output/           Model results, figures, frontend JSON (gitignored)
  scripts/          Reproducible command-line entry points
  src/              Python package: data parsers, model estimation, utilities
  tests/            Automated test suite
```

## Reproducing the results

```bash
# set up the environment
python3 -m venv "$HOME/venvs/banknim"
source "$HOME/venvs/banknim/bin/activate"
pip install -e .[test]
pip install -r requirements.txt
source .env

# download raw data
python3 -B scripts/00_fetch_metadata.py
python3 -B scripts/01_download_fdic_financials.py --start 2010-03-31 --end 2025-12-31
python3 -B scripts/02_download_fdic_sod.py --start-year 2010 --end-year 2025
python3 -B scripts/03_download_fdic_history.py --start-date 2010-01-01 --end-date 2025-12-31
python3 -B scripts/04_download_rates.py

# build panels
python3 -B scripts/06_build_core_panel.py
python3 -B scripts/07_build_sod_panel.py

# run models
python3 -B scripts/08_run_baseline_regressions.py
python3 -B scripts/09_event_study_acquirers.py
python3 -B scripts/10_run_franchise_dilution.py
python3 -B scripts/11_run_rate_cycle.py
python3 -B scripts/12_run_threshold_crossing.py
python3 -B scripts/13_run_robustness.py
python3 -B scripts/14_run_extensions.py
python3 -B scripts/16_run_nim_decomposition.py
python3 -B scripts/17_run_equity_as_outcome.py
python3 -B scripts/18_run_efficiency.py
python3 -B scripts/19_run_loan_composition.py
python3 -B scripts/20_run_failure_prediction.py
python3 -B scripts/21_run_rolling_coefficients.py

# export frontend data for the site
python3 -B scripts/15_export_frontend_data.py
cp output/frontend/*.json docs/data/

# run tests
python3 -B -m pytest -q
```

Or use the Make targets:

```bash
make setup
source .env
make all
make test
```

## License

See [LICENSE](LICENSE).
