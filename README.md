# Bitcoin Fear/Greed Sentiment vs Hyperliquid Trader Performance

This project analyzes how Bitcoin market sentiment relates to Hyperliquid trader behavior and performance. It loads raw CSV files, cleans and merges them by date, engineers trader-level features, runs EDA and statistical tests, segments traders, and generates a final Markdown report.

## Project Structure

```text
project/
  data/                  Input CSV files, ignored by Git
  notebooks/             Optional notebooks
  output/                Generated reports, charts, and tables
  src/                   Reusable pipeline modules
  main.py                Full pipeline entry point
  preprocessing.py       Standalone preprocessing pipeline
  requirements.txt       Python package requirements
  RUN_PROJECT.md         Detailed setup and run guide
  DEPENDENCIES.txt       Dependency explanation
```

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

For detailed setup steps, see [RUN_PROJECT.md](RUN_PROJECT.md).

## Required Input Files

Place the input CSV files in `data/`.

Current expected files:

- `fear_greed_index.csv`
- `historical_data.csv`

The pipeline supports similar filenames and common column-name variations through automatic detection.

## Main Outputs

After running `python main.py`, outputs are generated under `output/`.

Important outputs:

- `output/final_summary.md`
- `output/tables/merged_dataset.csv`
- `output/tables/statistical_summary.csv`
- `output/tables/trader_segment_summary.csv`
- `output/charts/`

Most generated outputs are ignored by Git to keep the repository lightweight. Re-run the pipeline to regenerate them.

## Pipeline Stages

1. Data loading and validation
2. Data cleaning and timestamp normalization
3. Dataset merging by date
4. Feature engineering
5. Exploratory Data Analysis
6. Statistical analysis
7. Trader segmentation
8. Visualization generation
9. Final report generation

## Notes

- The current dataset does not contain leverage, margin, or collateral fields. Leverage outputs are marked unavailable rather than estimated.
- Raw datasets are ignored by Git because they may be large or private.
- Generated caches, logs, charts, tables, and virtual environments are ignored by `.gitignore`.
