"""
preprocessing.py
----------------
End-to-end preprocessing pipeline for the Bitcoin Fear/Greed vs
Hyperliquid Trader Performance project.

Responsibilities
----------------
1.  Load both raw CSV files (filenames auto-detected inside data/).
2.  Validate inputs  – shape, required columns, data types.
3.  Detect & report missing values.
4.  Remove duplicate rows.
5.  Convert & normalise timestamps → UTC date (YYYY-MM-DD, no time).
6.  Standardise column names to snake_case.
7.  Merge the two datasets on the shared date key (inner join).
8.  Final validation of the merged frame.
9.  Return a clean DataFrame ready for feature engineering.

Design decisions
----------------
- Column detection is done via ranked candidate lists so the module
  survives upstream renames without code changes.
- No rows are silently dropped; every removal is logged with a reason
  and a count so results are fully reproducible.
- Outliers are *flagged* (pnl_outlier column) but never removed here;
  that decision belongs to the analysis layer.
- The IST timestamp in the trades file is parsed with dayfirst=True
  because the format is DD-MM-YYYY HH:MM.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output" / "tables"

# ---------------------------------------------------------------------------
# Column candidate lists  (ordered: most-likely first)
# ---------------------------------------------------------------------------
FG_DATE_CANDIDATES   = ["date", "Date", "timestamp", "Timestamp", "time"]
FG_VALUE_CANDIDATES  = ["value", "Value", "score", "fg_value", "index"]
FG_CLASS_CANDIDATES  = ["classification", "value_classification",
                        "label", "sentiment", "Classification"]

TR_DATE_CANDIDATES   = ["Timestamp IST", "timestamp_ist", "date", "Date",
                        "timestamp", "Timestamp", "time", "closed_at"]
TR_ACCOUNT_CANDIDATES = ["Account", "account", "trader", "address",
                         "wallet", "account_address"]
TR_PNL_CANDIDATES    = ["Closed PnL", "closed_pnl", "pnl", "realized_pnl",
                        "profit", "realizedPnl", "PnL"]
TR_SIZE_CANDIDATES   = ["Size USD", "size_usd", "size", "volume",
                        "notional", "Size Tokens"]
TR_SIDE_CANDIDATES   = ["Side", "side", "direction", "Direction",
                        "trade_type", "buy_sell"]
TR_COIN_CANDIDATES   = ["Coin", "coin", "symbol", "asset", "market", "pair"]
TR_FEE_CANDIDATES    = ["Fee", "fee", "commission", "trading_fee"]
TR_PRICE_CANDIDATES  = ["Execution Price", "execution_price", "price",
                        "fill_price", "avg_price"]

# Outlier thresholds (percentile-based, applied to PnL)
OUTLIER_LOWER_PCT = 1.0
OUTLIER_UPPER_PCT = 99.0


# ===========================================================================
# Helper utilities
# ===========================================================================

def _detect_column(df: pd.DataFrame, candidates: list[str], label: str,
                   required: bool = True) -> Optional[str]:
    """
    Return the first candidate column name that exists in *df*.

    Matching is tried in order:
      1. Exact match
      2. Case-insensitive match

    Parameters
    ----------
    df        : DataFrame to search.
    candidates: Ordered list of possible column names.
    label     : Human-readable name used in log/error messages.
    required  : If True, raises KeyError when no match is found.

    Returns
    -------
    Matched column name, or None if not required and not found.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            logger.info("    Column '%s' detected as: '%s'", label, candidate)
            return candidate
        if candidate.lower() in cols_lower:
            matched = cols_lower[candidate.lower()]
            logger.info("    Column '%s' detected as: '%s' (case-insensitive)", label, matched)
            return matched
    if required:
        raise KeyError(
            f"Required column '{label}' not found.\n"
            f"  Tried   : {candidates}\n"
            f"  Available: {list(df.columns)}"
        )
    logger.warning("    Optional column '%s' not found – skipping.", label)
    return None


def _auto_find_csv(data_dir: Path, keywords: list[str]) -> Path:
    """
    Find a CSV file in *data_dir* whose name contains any of *keywords*.

    Parameters
    ----------
    data_dir : Directory to search.
    keywords : Substrings to match (case-insensitive).

    Returns
    -------
    Path to the matched file.

    Raises
    ------
    FileNotFoundError if no match is found.
    """
    csvs = list(data_dir.glob("*.csv"))
    for csv in csvs:
        name_lower = csv.name.lower()
        if any(kw.lower() in name_lower for kw in keywords):
            return csv
    raise FileNotFoundError(
        f"No CSV matching keywords {keywords} found in {data_dir}.\n"
        f"Files present: {[f.name for f in csvs]}"
    )


def _section(title: str) -> None:
    """Print a visible section header to the log."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("  %s", title)
    logger.info("=" * 60)


# ===========================================================================
# Step 1 – Load
# ===========================================================================

def load_fear_greed(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the Fear & Greed Index CSV from *filepath* or auto-detect it.

    Returns
    -------
    pd.DataFrame with original column names intact.
    """
    path = filepath or _auto_find_csv(DATA_DIR, ["fear", "greed", "fg"])
    logger.info("Loading Fear & Greed from: %s", path.name)
    df = pd.read_csv(path)
    logger.info("  Raw shape: %s", df.shape)
    return df


def load_trades(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the Hyperliquid trades CSV from *filepath* or auto-detect it.

    Returns
    -------
    pd.DataFrame with original column names intact.
    """
    path = filepath or _auto_find_csv(
        DATA_DIR, ["historical", "trades", "hyperliquid", "hl"]
    )
    logger.info("Loading Hyperliquid trades from: %s", path.name)
    df = pd.read_csv(path)
    logger.info("  Raw shape: %s", df.shape)
    return df


# ===========================================================================
# Step 2 – Validate
# ===========================================================================

def validate(df: pd.DataFrame, name: str) -> None:
    """
    Log a full validation report for *df*.

    Checks: shape, column names, dtypes, missing values, duplicates.
    Does NOT modify the DataFrame.
    """
    _section(f"Validation: {name}")
    logger.info("Shape      : %s rows × %s columns", *df.shape)
    logger.info("Columns    : %s", df.columns.tolist())
    logger.info("Dtypes     :\n%s", df.dtypes.to_string())

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_report = pd.DataFrame({"missing": missing, "pct": missing_pct})
    missing_report = missing_report[missing_report["missing"] > 0]
    if missing_report.empty:
        logger.info("Missing    : None")
    else:
        logger.warning("Missing values detected:\n%s", missing_report.to_string())

    # Duplicates
    n_dupes = df.duplicated().sum()
    if n_dupes == 0:
        logger.info("Duplicates : None")
    else:
        logger.warning("Duplicates : %d rows", n_dupes)


# ===========================================================================
# Step 3 – Clean Fear & Greed
# ===========================================================================

def clean_fear_greed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardise the Fear & Greed DataFrame.

    Steps
    -----
    1. Detect columns via candidate lists.
    2. Rename to snake_case standard names.
    3. Drop columns not needed for analysis.
    4. Parse date → datetime64[ns], normalise to midnight (date only).
    5. Coerce fg_value to int, clip to [0, 100].
    6. Derive fg_classification if missing.
    7. Remove duplicate dates (keep first occurrence).
    8. Sort ascending by date.

    Returns
    -------
    Cleaned DataFrame with columns: date, fg_value, fg_classification.
    """
    _section("Cleaning: Fear & Greed")
    df = df.copy()
    logger.info("Input shape: %s", df.shape)

    # --- 1. Detect columns ---
    date_col  = _detect_column(df, FG_DATE_CANDIDATES,  "date")
    value_col = _detect_column(df, FG_VALUE_CANDIDATES, "fg_value")
    class_col = _detect_column(df, FG_CLASS_CANDIDATES, "fg_classification",
                               required=False)

    # --- 2. Rename to standard names ---
    rename_map = {date_col: "date", value_col: "fg_value"}
    if class_col:
        rename_map[class_col] = "fg_classification"
    df = df.rename(columns=rename_map)

    # Keep only the columns we need
    keep = ["date", "fg_value"] + (["fg_classification"] if class_col else [])
    df = df[keep]
    logger.info("Columns kept: %s", keep)

    # --- 3. Parse date ---
    # The 'date' column is already in YYYY-MM-DD string format.
    # pd.to_datetime handles this natively; errors='coerce' turns
    # unparseable values into NaT so we can count and drop them cleanly.
    before = len(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    n_bad_dates = df["date"].isna().sum()
    if n_bad_dates:
        logger.warning("  Dropping %d rows with unparseable dates.", n_bad_dates)
        df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.normalize()   # strip any time component → midnight
    logger.info("  Date parsed. Bad dates dropped: %d", n_bad_dates)

    # --- 4. Coerce fg_value ---
    df["fg_value"] = pd.to_numeric(df["fg_value"], errors="coerce")
    n_bad_val = df["fg_value"].isna().sum()
    if n_bad_val:
        logger.warning("  Dropping %d rows with non-numeric fg_value.", n_bad_val)
        df = df.dropna(subset=["fg_value"])
    df["fg_value"] = df["fg_value"].clip(0, 100).astype(int)
    logger.info("  fg_value coerced to int, clipped to [0, 100].")

    # --- 5. Derive classification if absent ---
    if "fg_classification" not in df.columns:
        bins   = [0, 25, 45, 55, 75, 100]
        labels = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
        df["fg_classification"] = pd.cut(
            df["fg_value"], bins=bins, labels=labels, include_lowest=True
        ).astype(str)
        logger.info("  fg_classification derived from fg_value thresholds.")
    else:
        df["fg_classification"] = df["fg_classification"].astype(str).str.strip()

    # --- 6. Remove duplicate dates ---
    n_dupes = df.duplicated(subset=["date"]).sum()
    if n_dupes:
        logger.warning("  Dropping %d duplicate date rows (keeping first).", n_dupes)
        df = df.drop_duplicates(subset=["date"], keep="first")
    else:
        logger.info("  No duplicate dates found")

    # --- 7. Sort ---
    df = df.sort_values("date").reset_index(drop=True)

    logger.info("Output shape: %s", df.shape)
    logger.info("Date range  : %s to %s",
                df["date"].min().date(), df["date"].max().date())
    logger.info("fg_value    : min=%d, max=%d, mean=%.1f",
                df["fg_value"].min(), df["fg_value"].max(), df["fg_value"].mean())
    logger.info("Classifications:\n%s",
                df["fg_classification"].value_counts().to_string())
    return df


# ===========================================================================
# Step 4 – Clean Trades
# ===========================================================================

def clean_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardise the Hyperliquid trades DataFrame.

    Steps
    -----
    1. Detect columns via candidate lists.
    2. Rename to snake_case standard names.
    3. Parse 'Timestamp IST' (DD-MM-YYYY HH:MM) → datetime, normalise to date.
    4. Coerce pnl, size_usd, fee, execution_price to float.
    5. Add is_win boolean flag.
    6. Flag PnL outliers (1st / 99th percentile) without removing them.
    7. Remove exact duplicate rows.
    8. Sort ascending by date.

    Returns
    -------
    Cleaned DataFrame with standardised column names.
    """
    _section("Cleaning: Hyperliquid Trades")
    df = df.copy()
    logger.info("Input shape: %s", df.shape)

    # --- 1. Detect columns ---
    date_col    = _detect_column(df, TR_DATE_CANDIDATES,    "date")
    account_col = _detect_column(df, TR_ACCOUNT_CANDIDATES, "account",  required=False)
    pnl_col     = _detect_column(df, TR_PNL_CANDIDATES,     "pnl",      required=False)
    size_col    = _detect_column(df, TR_SIZE_CANDIDATES,    "size_usd", required=False)
    side_col    = _detect_column(df, TR_SIDE_CANDIDATES,    "side",     required=False)
    coin_col    = _detect_column(df, TR_COIN_CANDIDATES,    "coin",     required=False)
    fee_col     = _detect_column(df, TR_FEE_CANDIDATES,     "fee",      required=False)
    price_col   = _detect_column(df, TR_PRICE_CANDIDATES,   "execution_price", required=False)

    # --- 2. Rename to standard names ---
    rename_map = {date_col: "date"}
    optional = {
        account_col: "account",
        pnl_col:     "pnl",
        size_col:    "size_usd",
        side_col:    "side",
        coin_col:    "coin",
        fee_col:     "fee",
        price_col:   "execution_price",
    }
    for orig, std in optional.items():
        if orig and orig != std:
            rename_map[orig] = std
    df = df.rename(columns=rename_map)

    # --- 3. Parse timestamp ---
    # 'Timestamp IST' format is DD-MM-YYYY HH:MM  → dayfirst=True is critical.
    # We normalise to midnight so each trade maps to a single calendar date,
    # which is the granularity of the Fear & Greed index.
    before = len(df)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    n_bad = df["date"].isna().sum()
    if n_bad:
        logger.warning("  Dropping %d rows with unparseable timestamps.", n_bad)
        df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.normalize()   # strip HH:MM → midnight
    logger.info("  Timestamps parsed. Bad rows dropped: %d", n_bad)
    logger.info("  Date range: %s to %s",
                df["date"].min().date(), df["date"].max().date())

    # --- 4. Coerce numeric columns ---
    numeric_cols = {
        "pnl": "Closed PnL",
        "size_usd": "Size USD",
        "fee": "Fee",
        "execution_price": "Execution Price",
    }
    for std_name, display_name in numeric_cols.items():
        if std_name in df.columns:
            df[std_name] = pd.to_numeric(df[std_name], errors="coerce")
            n_coerce_fail = df[std_name].isna().sum()
            if n_coerce_fail:
                logger.warning("  %d non-numeric values in '%s' set to NaN.",
                               n_coerce_fail, display_name)

    # --- 5. Add is_win flag ---
    # A trade is a 'win' if its realised PnL is strictly positive.
    # Trades with PnL == 0 (e.g. open positions, fees-only) are not wins.
    if "pnl" in df.columns:
        df["is_win"] = df["pnl"] > 0
        win_rate = df["is_win"].mean() * 100
        logger.info("  is_win flag added. Overall win rate: %.1f%%", win_rate)

    # --- 6. Flag PnL outliers ---
    # We flag rather than remove so downstream analysis can choose to
    # include or exclude extreme trades transparently.
    if "pnl" in df.columns:
        pnl_clean = df["pnl"].dropna()
        lower = np.percentile(pnl_clean, OUTLIER_LOWER_PCT)
        upper = np.percentile(pnl_clean, OUTLIER_UPPER_PCT)
        df["pnl_outlier"] = (df["pnl"] < lower) | (df["pnl"] > upper)
        n_out = df["pnl_outlier"].sum()
        logger.info(
            "  PnL outlier flag: %d rows (%.2f%%) outside [%.2f, %.2f]",
            n_out, 100 * n_out / len(df), lower, upper
        )

    # --- 7. Remove exact duplicates ---
    n_dupes = df.duplicated().sum()
    if n_dupes:
        logger.warning("  Dropping %d exact duplicate rows.", n_dupes)
        df = df.drop_duplicates()
    else:
        logger.info("  No exact duplicate rows found")

    # --- 8. Sort ---
    df = df.sort_values("date").reset_index(drop=True)

    logger.info("Output shape: %s", df.shape)
    if "pnl" in df.columns:
        logger.info("PnL stats   : min=%.2f, max=%.2f, mean=%.2f, median=%.2f",
                    df["pnl"].min(), df["pnl"].max(),
                    df["pnl"].mean(), df["pnl"].median())
    if "account" in df.columns:
        logger.info("Unique accounts: %d", df["account"].nunique())
    if "coin" in df.columns:
        logger.info("Unique coins   : %d", df["coin"].nunique())
    return df


# ===========================================================================
# Step 5 – Merge
# ===========================================================================

def merge_datasets(fg: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Fear & Greed data with trades data on the shared 'date' column.

    Strategy
    --------
    - Inner join: only dates present in BOTH datasets are kept.
      This is intentional — a trade with no sentiment data (or vice versa)
      cannot contribute to the sentiment-performance analysis.
    - After merging, a 'sentiment_regime' column is added that maps
      fg_value to a human-readable 5-class label.
    - A 'is_fear' boolean column is added for binary fear/greed splits.

    Returns
    -------
    Merged DataFrame sorted by date, ready for feature engineering.
    """
    _section("Merging Datasets")
    logger.info("Fear & Greed shape : %s  (%s to %s)",
                fg.shape, fg["date"].min().date(), fg["date"].max().date())
    logger.info("Trades shape       : %s  (%s to %s)",
                trades.shape, trades["date"].min().date(), trades["date"].max().date())

    # Identify date overlap
    fg_dates     = set(fg["date"])
    trade_dates  = set(trades["date"])
    overlap      = fg_dates & trade_dates
    only_fg      = fg_dates - trade_dates
    only_trades  = trade_dates - fg_dates

    logger.info("Date overlap       : %d days", len(overlap))
    logger.info("Dates only in FG   : %d days (excluded by inner join)", len(only_fg))
    logger.info("Dates only in trades: %d days (excluded by inner join)", len(only_trades))

    merged = pd.merge(trades, fg, on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)

    logger.info("Merged shape       : %s", merged.shape)
    logger.info("Merged date range  : %s to %s",
                merged["date"].min().date(), merged["date"].max().date())

    # Add sentiment regime labels
    bins   = [0, 25, 45, 55, 75, 100]
    labels = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    merged["sentiment_regime"] = pd.cut(
        merged["fg_value"], bins=bins, labels=labels, include_lowest=True
    ).astype(str)

    # Binary fear / greed flag
    merged["is_fear"]  = merged["fg_value"] < 50
    merged["is_greed"] = merged["fg_value"] >= 50

    logger.info("Sentiment regime distribution:")
    regime_counts = merged["sentiment_regime"].value_counts()
    for regime, count in regime_counts.items():
        pct = 100 * count / len(merged)
        logger.info("  %-15s : %6d trades  (%.1f%%)", regime, count, pct)

    return merged


# ===========================================================================
# Step 6 – Final validation of merged frame
# ===========================================================================

def validate_merged(df: pd.DataFrame) -> None:
    """
    Run a final sanity check on the merged DataFrame and log a summary.
    Raises ValueError if critical columns are missing or the frame is empty.
    """
    _section("Final Validation: Merged Dataset")

    if df.empty:
        raise ValueError("Merged DataFrame is empty. Check date overlap between datasets.")

    critical = ["date", "fg_value", "fg_classification", "sentiment_regime"]
    missing_critical = [c for c in critical if c not in df.columns]
    if missing_critical:
        raise ValueError(f"Critical columns missing from merged frame: {missing_critical}")

    logger.info("Shape              : %s", df.shape)
    logger.info("Columns            : %s", df.columns.tolist())
    logger.info("Date range         : %s to %s",
                df["date"].min().date(), df["date"].max().date())
    logger.info("Unique dates       : %d", df["date"].nunique())

    remaining_missing = df.isnull().sum()
    remaining_missing = remaining_missing[remaining_missing > 0]
    if remaining_missing.empty:
        logger.info("Remaining NaNs     : None")
    else:
        logger.warning("Remaining NaNs:\n%s", remaining_missing.to_string())

    logger.info("All critical columns present")
    logger.info("Merged dataset is ready for analysis")


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_preprocessing(
    fg_path: Optional[Path] = None,
    trades_path: Optional[Path] = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Execute the full preprocessing pipeline.

    Parameters
    ----------
    fg_path     : Optional override path for the Fear & Greed CSV.
    trades_path : Optional override path for the trades CSV.
    save        : If True, saves the merged DataFrame to output/tables/.

    Returns
    -------
    Clean, merged pd.DataFrame ready for feature engineering.
    """
    _section("PREPROCESSING PIPELINE START")

    # Step 1 – Load
    fg_raw     = load_fear_greed(fg_path)
    trades_raw = load_trades(trades_path)

    # Step 2 – Validate raw inputs
    validate(fg_raw,     "Fear & Greed (raw)")
    validate(trades_raw, "Hyperliquid Trades (raw)")

    # Step 3 – Clean
    fg_clean     = clean_fear_greed(fg_raw)
    trades_clean = clean_trades(trades_raw)

    # Step 4 – Merge
    merged = merge_datasets(fg_clean, trades_clean)

    # Step 5 – Final validation
    validate_merged(merged)

    # Step 6 – Save
    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / "merged_dataset.csv"
        merged.to_csv(out_path, index=False)
        logger.info("Merged dataset saved to: %s", out_path)

    _section("PREPROCESSING PIPELINE COMPLETE")
    return merged


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    df = run_preprocessing()

    print("\n" + "=" * 60)
    print("PREVIEW OF MERGED DATASET")
    print("=" * 60)
    print(df.head(10).to_string())
    print("\nShape:", df.shape)
    print("Columns:", df.columns.tolist())
