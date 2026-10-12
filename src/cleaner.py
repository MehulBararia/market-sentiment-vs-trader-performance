"""
cleaner.py
----------
Data cleaning routines for both datasets.
Handles type coercion, duplicates, missing values, and outlier flagging.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    OUTLIER_LOWER_PERCENTILE,
    OUTLIER_UPPER_PERCENTILE,
    SENTIMENT_BINS,
    SENTIMENT_LABELS,
)

logger = logging.getLogger(__name__)


def clean_fear_greed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Fear & Greed dataframe.

    Steps:
    1. Parse fg_date to datetime (date only).
    2. Coerce fg_value to numeric; drop rows where it is null.
    3. Clip fg_value to [0, 100].
    4. Derive fg_classification if missing.
    5. Drop duplicate dates (keep first).
    6. Sort by date ascending.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Fear & Greed dataframe (output of load_fear_greed).

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df = df.copy()

    # 1. Parse dates
    df["fg_date"] = pd.to_datetime(df["fg_date"], errors="coerce")
    n_bad_dates = df["fg_date"].isna().sum()
    if n_bad_dates:
        logger.warning("Dropping %d rows with unparseable dates in Fear & Greed.", n_bad_dates)
    df = df.dropna(subset=["fg_date"])
    df["fg_date"] = df["fg_date"].dt.normalize()  # strip time component

    # 2. Coerce fg_value
    df["fg_value"] = pd.to_numeric(df["fg_value"], errors="coerce")
    n_bad_val = df["fg_value"].isna().sum()
    if n_bad_val:
        logger.warning("Dropping %d rows with non-numeric fg_value.", n_bad_val)
    df = df.dropna(subset=["fg_value"])

    # 3. Clip to valid range
    df["fg_value"] = df["fg_value"].clip(0, 100)

    # 4. Derive classification if absent
    if "fg_classification" not in df.columns:
        df["fg_classification"] = pd.cut(
            df["fg_value"],
            bins=SENTIMENT_BINS,
            labels=SENTIMENT_LABELS,
            include_lowest=True,
        )
        logger.info("Derived fg_classification from fg_value.")
    else:
        df["fg_classification"] = df["fg_classification"].astype(str).str.strip()

    # 5. Drop duplicate dates
    n_dupes = df.duplicated(subset=["fg_date"]).sum()
    if n_dupes:
        logger.warning("Dropping %d duplicate dates in Fear & Greed.", n_dupes)
    df = df.drop_duplicates(subset=["fg_date"], keep="first")

    # 6. Sort
    df = df.sort_values("fg_date").reset_index(drop=True)
    logger.info("Fear & Greed cleaned. Shape: %s", df.shape)
    return df


def clean_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Hyperliquid trades dataframe.

    Steps:
    1. Parse trade_date to datetime (date only).
    2. Coerce pnl and size to numeric.
    3. Flag PnL outliers (beyond 1st/99th percentile).
    4. Drop exact duplicate rows.
    5. Sort by date ascending.

    Parameters
    ----------
    df : pd.DataFrame
        Raw trades dataframe (output of load_trades).

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with an added pnl_outlier boolean column.
    """
    df = df.copy()

    # 1. Parse dates
    df["trade_date"] = pd.to_datetime(
        df["trade_date"], dayfirst=True, errors="coerce"
    )
    n_bad = df["trade_date"].isna().sum()
    if n_bad:
        logger.warning("Dropping %d rows with unparseable trade_date.", n_bad)
    df = df.dropna(subset=["trade_date"])
    df["trade_date"] = df["trade_date"].dt.normalize()

    # 2. Coerce numeric columns
    for col in ["pnl", "size"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Flag PnL outliers (do NOT remove — flag only)
    if "pnl" in df.columns:
        df["is_win"] = df["pnl"] > 0
        pnl_clean = df["pnl"].dropna()
        if pnl_clean.empty:
            df["pnl_outlier"] = False
            logger.warning("No numeric PnL values available; outlier flag set to False.")
        else:
            lower = np.percentile(pnl_clean, OUTLIER_LOWER_PERCENTILE)
            upper = np.percentile(pnl_clean, OUTLIER_UPPER_PERCENTILE)
            df["pnl_outlier"] = (df["pnl"] < lower) | (df["pnl"] > upper)
            n_outliers = df["pnl_outlier"].sum()
            logger.info(
                "PnL outlier flag: %d rows (%.2f%%) outside [%.2f, %.2f].",
                n_outliers,
                100 * n_outliers / len(df),
                lower,
                upper,
            )

    # 4. Drop exact duplicates
    n_dupes = df.duplicated().sum()
    if n_dupes:
        logger.warning("Dropping %d exact duplicate rows in trades.", n_dupes)
    df = df.drop_duplicates()

    # 5. Sort
    df = df.sort_values("trade_date").reset_index(drop=True)
    logger.info("Trades cleaned. Shape: %s", df.shape)
    return df
