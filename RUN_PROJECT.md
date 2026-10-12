# How to Run the Project

## Project Name

Bitcoin Fear/Greed Sentiment vs Hyperliquid Trader Performance

## 1. Prerequisites

- Python 3.11 or newer
- `pip`
- The two input CSV files inside the `data/` folder

Expected data folder structure:

```text
project/
  data/
    fear_greed_index.csv
    historical_data.csv
```

The code automatically detects the CSV files by filename keywords and detects common column-name variations.

## 2. Open the Project Folder

From PowerShell or Command Prompt, go to the project directory:

```powershell
cd "C:\Users\piyus\OneDrive\Desktop\data analyst intern task\project"
```

## 3. Create a Virtual Environment

```powershell
python -m venv .venv
```

## 4. Activate the Virtual Environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On Windows Command Prompt:

```cmd
.venv\Scripts\activate
```

## 5. Install Dependencies

```powershell
pip install -r requirements.txt
```

## 6. Run the Complete Pipeline

```powershell
python main.py
```

This runs the full project pipeline:

1. Data loading
2. Data validation
3. Data cleaning
4. Date normalization
5. Dataset merging
6. Feature engineering
7. Exploratory Data Analysis
8. Statistical analysis
9. Trader segmentation
10. Chart generation
11. Final report generation

## 7. Main Outputs

After running `python main.py`, check the `output/` folder.

### Tables

Saved in:

```text
output/tables/
```

Important files:

- `merged_dataset.csv`
- `daily_trader_stats.csv`
- `trader_features.csv`
- `eda_summary.csv`
- `sentiment_pnl_summary.csv`
- `statistical_summary.csv`
- `chi_square_results.csv`
- `correlation_matrix.csv`
- `ranked_correlations.csv`
- `trader_behavior_segments.csv`
- `trader_segment_summary.csv`
- `trader_segment_interpretations.csv`
- `trader_segments.csv`

### Charts

Saved in:

```text
output/charts/
```

This folder contains all EDA, statistical, and trader segmentation visualizations.

### Final Report

Saved as:

```text
output/final_summary.md
```

## 8. Optional: Run Only Preprocessing

To run the standalone preprocessing pipeline:

```powershell
python preprocessing.py
```

This creates a clean merged dataset and previews the processed data.

## 9. Notes

- Leverage fields are created in the outputs, but the current source dataset does not contain actual leverage, margin, or collateral values.
- Because of that, leverage analysis is marked as unavailable instead of using fabricated values.
- All charts, statistics, segment summaries, and report insights are generated from the actual data.

## 10. Troubleshooting

If dependencies are missing, rerun:

```powershell
pip install -r requirements.txt
```

If PowerShell blocks virtual environment activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If output files look old, delete the `output/` folder contents and rerun:

```powershell
python main.py
```
