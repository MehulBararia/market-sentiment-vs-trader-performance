"""
merger.py
---------
Merges the cleaned Fear & Greed and trades datasets on a daily date key.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def merge_datasets(
    fg_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    how: str = "inner",
) -> pd.DataFrame:
    """
    Merge Fear & Greed data with trades data on a daily date key.

    Both dataframes must have been cleaned (dates normalised to midnight UTC).

    Parameters
    ----------
    fg_df : pd.DataFrame
        Cleaned Fear & Greed dataframe with column fg_date.
    trades_df : pd.DataFrame
        Cleaned trades dataframe with column trade_date.
    how : str
        Merge strategy passed to pd.merge. Default "inner" keeps only
        dates present in both datasets.

    Returns
    -------
    pd.DataFrame
        Merged dataframe with a unified date column.
    """
    logger.info("Merging datasets (how='%s').", how)

    fg = fg_df.copy().rename(columns={"fg_date": "date"})
    trades = trades_df.copy().rename(columns={"trade_date": "date"})

    merged = pd.merge(trades, fg, on="date", how=how)

    logger.info(
        "Merge complete. Trades: %d rows | FG: %d rows | Merged: %d rows.",
        len(trades),
        len(fg),
        len(merged),
    )

    if not merged.empty:
        date_range = f"{merged['date'].min().date()} to {merged['date'].max().date()}"
        logger.info("Date range in merged dataset: %s", date_range)

    return merged.sort_values("date").reset_index(drop=True)
