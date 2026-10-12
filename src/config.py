"""
config.py
---------
Central configuration: file paths, constants, and column name mappings.
All other modules import from here to keep paths in one place.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"
TABLES_DIR = OUTPUT_DIR / "tables"

# ---------------------------------------------------------------------------
# Input file names — auto-detected by keyword so renaming is never needed
# ---------------------------------------------------------------------------
def _find_csv(keywords: list[str]) -> Path:
    """Return the first CSV in DATA_DIR whose name contains any keyword."""
    for csv in DATA_DIR.glob("*.csv"):
        if any(kw.lower() in csv.name.lower() for kw in keywords):
            return csv
    raise FileNotFoundError(
        f"No CSV matching {keywords} in {DATA_DIR}. "
        f"Files: {[f.name for f in DATA_DIR.glob('*.csv')]}"
    )

FEAR_GREED_FILE = _find_csv(["fear", "greed", "fg"])
TRADES_FILE     = _find_csv(["historical", "trades", "hyperliquid", "hl"])

# ---------------------------------------------------------------------------
# Output file names
# ---------------------------------------------------------------------------
MERGED_DATASET = TABLES_DIR / "merged_dataset.csv"
EDA_SUMMARY = TABLES_DIR / "eda_summary.csv"
TRADER_SEGMENTS = TABLES_DIR / "trader_segments.csv"
TRADER_BEHAVIOR_SEGMENTS = TABLES_DIR / "trader_behavior_segments.csv"
TRADER_SEGMENT_SUMMARY = TABLES_DIR / "trader_segment_summary.csv"
TRADER_SEGMENT_INTERPRETATIONS = TABLES_DIR / "trader_segment_interpretations.csv"
DAILY_TRADER_STATS = TABLES_DIR / "daily_trader_stats.csv"
TRADER_FEATURES = TABLES_DIR / "trader_features.csv"
EDA_CHART_NOTES = TABLES_DIR / "eda_chart_notes.csv"
STATISTICAL_SUMMARY = TABLES_DIR / "statistical_summary.csv"
CHI_SQUARE_RESULTS = TABLES_DIR / "chi_square_results.csv"
CORRELATION_MATRIX = TABLES_DIR / "correlation_matrix.csv"
RANKED_CORRELATIONS = TABLES_DIR / "ranked_correlations.csv"
FINAL_REPORT = OUTPUT_DIR / "final_summary.md"

# ---------------------------------------------------------------------------
# Analysis constants
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42
OUTLIER_LOWER_PERCENTILE: float = 1.0
OUTLIER_UPPER_PERCENTILE: float = 99.0
N_CLUSTERS: int = 4  # Number of trader segments

# Sentiment classification thresholds (standard Fear & Greed definitions)
SENTIMENT_BINS = [0, 25, 45, 55, 75, 100]
SENTIMENT_LABELS = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]

# ---------------------------------------------------------------------------
# Column name candidates (auto-detection fallback lists)
# ---------------------------------------------------------------------------
# Fear & Greed dataset
FG_DATE_CANDIDATES = ["date", "timestamp", "Date", "Timestamp", "time", "Time"]
FG_VALUE_CANDIDATES = ["value", "score", "fg_value", "fear_greed", "index"]
FG_CLASS_CANDIDATES = ["value_classification", "classification", "label", "sentiment"]

# Hyperliquid trades dataset
TR_DATE_CANDIDATES = [
    "Timestamp IST", "timestamp_ist", "date", "Date",
    "timestamp", "Timestamp", "time", "Time",
    "closed_at", "open_time", "trade_date",
]
TR_ACCOUNT_CANDIDATES = [
    "account", "trader", "address", "wallet", "user",
    "account_address", "trader_address",
]
TR_PNL_CANDIDATES = [
    "Closed PnL", "closed_pnl", "pnl", "realized_pnl",
    "profit", "profit_loss", "pnl_usd", "realizedPnl",
]
TR_SIZE_CANDIDATES = [
    "Size USD", "size_usd", "size", "volume", "notional",
    "trade_size", "position_size", "sz", "Size Tokens",
]
TR_SIDE_CANDIDATES = ["side", "direction", "type", "trade_type", "buy_sell"]
TR_COIN_CANDIDATES = ["coin", "symbol", "asset", "market", "pair", "ticker"]
