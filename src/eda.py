"""
eda.py
------
Exploratory Data Analysis: summary statistics and distribution checks.
Results are saved to output/tables/.
"""

import logging

import pandas as pd

from src.config import EDA_SUMMARY, TABLES_DIR

logger = logging.getLogger(__name__)


def summarise_merged(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics for the merged dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Merged and feature-engineered dataframe.

    Returns
    -------
    pd.DataFrame
        Descriptive statistics table with missing value counts appended.
    """
    summary = df.describe(include="all").T
    summary["missing"] = df.isnull().sum()
    summary["missing_pct"] = (df.isnull().sum() / len(df) * 100).round(2)
    logger.info("EDA summary computed. Shape: %s", summary.shape)
    return summary


def sentiment_pnl_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean, median, std, and win rate of PnL grouped by sentiment regime.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataframe with sentiment_regime, pnl, and is_win columns.

    Returns
    -------
    pd.DataFrame
        Summary table grouped by sentiment regime.
        Returns empty DataFrame if required columns are missing.
    """
    required = {"pnl", "sentiment_regime", "is_win"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        logger.warning(
            "Skipping sentiment_pnl_summary: missing columns %s.", missing_cols
        )
        return pd.DataFrame()

    summary = (
        df.groupby("sentiment_regime", observed=True)
        .agg(
            trade_count=("pnl", "count"),
            mean_pnl=("pnl", "mean"),
            median_pnl=("pnl", "median"),
            std_pnl=("pnl", "std"),
            win_rate=("is_win", "mean"),
            total_pnl=("pnl", "sum"),
        )
        .reset_index()
    )

    # Round for readability
    for col in ["mean_pnl", "median_pnl", "std_pnl", "win_rate", "total_pnl"]:
        summary[col] = summary[col].round(4)

    logger.info("Sentiment vs PnL summary:\n%s", summary.to_string())
    return summary


def save_eda_tables(
    summary: pd.DataFrame, sentiment_summary: pd.DataFrame
) -> None:
    """
    Save EDA tables to the output/tables/ directory.

    Parameters
    ----------
    summary : pd.DataFrame
        Full descriptive statistics table.
    sentiment_summary : pd.DataFrame
        Sentiment-grouped PnL summary.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    summary.to_csv(EDA_SUMMARY)
    logger.info("Saved EDA summary to: %s", EDA_SUMMARY)

    if not sentiment_summary.empty:
        sentiment_path = TABLES_DIR / "sentiment_pnl_summary.csv"
        sentiment_summary.to_csv(sentiment_path, index=False)
        logger.info("Saved sentiment PnL summary to: %s", sentiment_path)
