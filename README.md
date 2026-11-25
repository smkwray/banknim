# Bank Size, Growth, and Net Interest Margin

**[Browse the interactive results site](https://smkwray.github.io/banknim/)**

banknim asks whether the lower net interest margin (NIM) of larger U.S. banks comes from being big or from becoming bigger. It is a scripted panel study of 8,032 FDIC-insured commercial banks and 373,220 bank-quarters from 2010 to 2025, built entirely from free public data.

Larger banks earn lower margins both across banks and as a given bank grows, mostly because they pay more for funding, and they make up for it with fee income. The main caveat is that the baseline size effect depends on trimming extreme NIM values: on raw, unwinsorized NIM it is insignificant (−0.01, p=0.87), so the first-difference and lagged-controls results carry the most weight.

## Headline findings

- Size and NIM: banks 10× larger run about 20 basis points lower NIM in the cross-section (−0.085, p<0.001, using the 8,023 banks with complete bank-level averages), and NIM falls as a bank grows (−0.090 with bank fixed effects, p<0.001). The within-bank result holds in first differences (−0.11, p<0.001) and with lagged controls (−0.12, p<0.001).
- Funding cost: with bank fixed effects, size raises interest expense (+0.245) more than interest income (+0.154), both p<0.001.
- Fee offset: return on assets (ROA) rises with size (+0.205, p<0.001), as does noninterest margin (+0.31, p<0.001).
- Franchise dilution: asset growth combined with fast growth in branch count lowers margins (interaction −0.58, p<0.01).
- Rate cycles: the size penalty deepens as the federal funds rate rises (−0.012, p<0.001) and is partly offset by a steeper yield curve (+0.015).
- Null results: size does not predict NIM volatility (p=0.19), and local deposit concentration (HHI) does not predict NIM (p=0.70).

The site also covers acquisitions, crossings of the $10 billion, $50 billion and $100 billion asset thresholds, equity, efficiency, loan mix, rolling 20-quarter coefficients, and an exploratory failure model.

## Method

The main models regress NIM on log assets with equity and loan-share controls. Within-bank models use bank and quarter fixed effects with standard errors clustered by bank; the between-bank model is ordinary least squares on each bank's averages. Extensions add asset and deposit growth, interact asset growth with branch and deposit growth, interact size with the federal funds rate and yield-curve slope, and run event studies around acquisitions and asset-threshold crossings. Robustness checks use raw NIM, first differences, lagged controls, log deposits as the size measure, and a sample excluding banks with fewer than 12 quarters.

## Data sources

| Source | Provider | Frequency | Coverage |
|---|---|---|---|
| BankFind financials | FDIC | Quarterly | 2010–2025 |
| Summary of Deposits (SOD) | FDIC | Annual | 2010–2025 branch deposits |
| Institution history | FDIC | Event-based | Mergers and acquisitions |
| FRED interest rates | Federal Reserve Bank of St. Louis | Daily and monthly | Fed funds, prime rate, Treasury yields |

No API keys are required. The FDIC downloader sends `FDIC_API_KEY` if it is set in the environment.

## Reproduce

Requires Python 3.11 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
make setup     # install pinned dependencies and the package
make all       # download data, build panels, run every model, export site data to docs/data/
make test
make smoke     # quick end-to-end run on a small fixture
```

Each step is also a numbered script in `scripts/`, and the Makefile shows the order. To preview the site, run `python3 -m http.server 8000` in `docs/`.

## Outputs and layout

```
config/    Sample window, data sources, and field lists
scripts/   Numbered pipeline steps
src/       Python package (nimscale): data parsing and model estimation
data/      Raw downloads and processed panels (not tracked)
output/    Model tables, figures, quality checks, and site JSON (not tracked)
docs/      Static results site; docs/data/ holds the exported JSON
tests/     Test suite
```

Sample construction, duplicate checks, and column missingness are written to `output/tables/` and published on the site.

## Limitations

- The design is observational, so the size coefficients are associations, not causal effects.
- The baseline size effect depends on winsorizing NIM outliers.
- The failure model is exploratory: higher NIM and larger size predict the failure label, while higher ROA lowers it, which is not a clean "low NIM predicts failure" result.
- The size penalty varies over time (rolling coefficients range from about −0.033 to −0.144).
- Full reruns depend on FDIC and FRED services being available.

## License

MIT; see [LICENSE](LICENSE).
