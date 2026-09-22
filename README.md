# Stata to Python: An Automated Panel Data Cleaning Pipeline

## Project Purpose

This project replicates, in Python, a panel-data cleaning and regression
workflow I originally completed in Stata for my undergraduate Econometrics
coursework. The goal was to migrate from a Stata-based empirical workflow
to an open-source, reproducible Python pipeline using `pandas` and
`statsmodels`.

The core challenge this project solves is not the regression itself, but
the data cleaning that precedes it: Chinese provincial statistics bureau
exports are notoriously inconsistent in encoding (UTF-8 vs. GBK), delimiter,
and header structure across different indicators and years. Rather than
manually reformatting each file by hand (as is typically done before
importing into Stata), I wrote a single pipeline that auto-detects these
inconsistencies and produces a clean, analysis-ready panel dataset.

## What the Pipeline Does

`scripts/data_detective.py` implements a `DataDetective` class that:

1. **Scans** a folder of raw CSVs from the National Bureau of Statistics
   (or provincial-level bureaus)
2. **Auto-detects** file encoding (UTF-8 / GBK / GB18030) and delimiter,
   since different bureaus export in different formats
3. **Identifies** which economic indicator each file represents, by parsing
   footnotes or falling back to filename matching
4. **Reshapes** each file from wide format (one column per year) to long
   panel format (entity-year rows), and standardizes the region column name
   across files so they can be merged
5. **Merges** all indicators into a single entity-year panel
6. **Cleans** the merged panel:
   - removes duplicate rows
   - flags and removes statistical outliers using a 3-sigma rule per column
   - validates domain constraints (e.g. GDP and population must be positive)
   - drops any indicator column with >50% missingness
   - linearly interpolates remaining missing values within each entity
7. **Estimates** an OLS regression and **exports** results and diagnostic
   plots

## Data

This pipeline uses province-year panel data from China's National Bureau of
Statistics (国家统计局, data.stats.gov.cn): GDP and resident population for
all 31 provinces/municipalities/autonomous regions, 2016–2025.

- `data/raw/国民生产总值.csv` — raw GDP data, unmodified from source
- `data/raw/总人口.csv` — raw population data, unmodified from source
- `data/processed/panel_data.csv` — cleaned and merged panel dataset (301 observations after cleaning)

Place additional raw CSV files in `data/raw/` to extend the panel with more
indicators (e.g. fiscal revenue, R&D expenditure) — the pipeline will
auto-detect and merge them as long as they follow the same export format.

## Key Findings

Regressing GDP on population across all 31 provinces, 2016–2025 (OLS, n=301
after cleaning):

- A 1万-person increase in population is associated with a 7.15亿元 increase
  in GDP, holding other factors constant (coefficient = 7.15, p < 0.001).
- The model explains 67.8% of the variance in provincial GDP (R² = 0.678).
- 9 outlier observations were removed via the 3-sigma rule before estimation
  (see `output/cleaning_report.txt` for the full audit trail).

Full regression output is in `output/regression_results.txt`; trend and
coefficient plots are in `output/panel_analysis.png`.

## How to Run

```bash
pip install -r requirements.txt
python scripts/data_detective.py
```

This will read every CSV in `data/raw/`, clean and merge them, run an OLS
regression on the first two resulting indicator columns, and write:

- `data/processed/panel_data.csv` — the cleaned panel dataset
- `output/cleaning_report.txt` — a log of what was removed and why
- `output/regression_results.txt` — full OLS regression output
- `output/panel_analysis.png` — trend lines and a regression coefficient plot

## Notes on Debugging

While generalizing this script for reuse on different export formats, I
found and fixed two issues in the original version:

1. **Region column name collision.** The cleaning and regression steps
   assumed every merged file's region column was literally named "地区".
   This worked for files that already used that exact header, but broke
   silently if a source file used a different label (e.g. "省份"). I fixed
   this by standardizing the region column name immediately after reshaping
   each file, before the merge step.

2. **Header row assumption.** National Bureau of Statistics exports
   (data.stats.gov.cn) include two metadata lines — a database title and a
   "时间：" line — before the real header row. The original script hardcoded
   this as `i >= 2` when scanning lines. I made this a configurable
   parameter (`header_row_index`) so the same pipeline can also handle
   single-header-row exports from other sources, while keeping `2` as the
   default since it matches the bureau format this project was built around.

## Why This Project

This was originally a Stata-only econometrics assignment. I rebuilt the
workflow in Python to (1) understand the mechanics of data cleaning and
panel construction more deeply by implementing them outside of Stata's
built-in import routines, and (2) build fluency in the tools most commonly
used in applied economic research and data analysis roles.

## Project Timeline

- **Jan 2026 – Feb 2026** — Initial development. Built the core
  pipeline over the winter break, working locally in a Jupyter
  Notebook. The pipeline was fully functional at this stage: it could
  read raw National Bureau of Statistics exports, detect encoding and
  delimiter, reshape wide files into a panel, clean and merge
  indicators, and run an OLS regression.

- **Jun 2026** — Refactored the code into a reusable Python package,
  added this documentation and usage examples, and open-sourced the
  project on GitHub. The commit history begins here because the code
  was developed locally before being uploaded.

## Tools Used

- Python 3.11
- pandas, numpy
- statsmodels (OLS regression)
- matplotlib (visualization)

