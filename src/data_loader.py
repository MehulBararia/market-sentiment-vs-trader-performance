"""
data_loader.py
--------------
Loads raw CSV datasets and performs initial validation.
Auto-detects column names using candidate lists from config.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import (
    FEAR_GREED_FILE,
    TRADES_FILE,
    FG_DATE_CANDIDATES,
    FG_VALUE_CANDIDATES,
    FG_CLASS_CANDIDATES,
    TR_DATE_CANDIDATES,
    TR_ACCOUNT_CANDIDATES,
    TR_PNL_CANDIDATES,
    TR_SIZE_CANDIDATES,
    TR_SIDE_CANDIDATES,
    TR_COIN_CANDIDATES,
)

logger = logging.getLogger(__name__)


def _detect_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    """
    Return the first column name from *candidates* that exists in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to search.
    candidates : list[str]
        Ordered list of possible column names.
    label : str
        Human-readable description used in error messages.

    Returns
    -------
    str
        Matched column name.

    Raises
    ------
    KeyError
        If none of the candidates are found.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in cols_lower:
            return cols_lower[candidate.lower()]
    raise KeyError(
        f"Could not detect '{label}' column. "
        f"Available columns: {list(df.columns)}. "
        f"Tried: {candidates}"
    )


def load_fear_greed(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the Bitcoin Fear & Greed Index CSV.

    Detects date, value, and classification columns automatically.
    Renames them to standardised names: fg_date, fg_value, fg_classification.

    Parameters
    ----------
    filepath : Path, optional
        Override the default path from config.

    Returns
    -------
    pd.DataFrame
        Raw Fear & Greed dataframe with standardised column names.
    """
    path = filepath or FEAR_GREED_FILE
    logger.info("Loading Fear & Greed data from: %s", path)

    df = pd.read_csv(path)
    logger.info("Shape: %s | Columns: %s", df.shape, list(df.columns))

    date_col = _detect_column(df, FG_DATE_CANDIDATES, "date")
    value_col = _detect_column(df, FG_VALUE_CANDIDATES, "fg_value")

    rename_map = {date_col: "fg_date", value_col: "fg_value"}

    # Classification column is optional
    try:
        class_col = _detect_column(df, FG_CLASS_CANDIDATES, "fg_classification")
        rename_map[class_col] = "fg_classification"
    except KeyError:
        logger.warning("No classification column found; will derive from fg_value.")

    df = df.rename(columns=rename_map)
    return df


def load_trades(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the Hyperliquid historical trades CSV.

    Detects date, account, PnL, size, side, and coin columns automatically.
    Renames them to standardised names.

    Parameters
    ----------
    filepath : Path, optional
        Override the default path from config.

    Returns
    -------
    pd.DataFrame
        Raw trades dataframe with standardised column names.
    """
    path = filepath or TRADES_FILE
    logger.info("Loading Hyperliquid trades from: %s", path)

    df = pd.read_csv(path)
    logger.info("Shape: %s | Columns: %s", df.shape, list(df.columns))

    date_col = _detect_column(df, TR_DATE_CANDIDATES, "date")
    rename_map = {date_col: "trade_date"}

    optional_cols = {
        "account": TR_ACCOUNT_CANDIDATES,
        "pnl": TR_PNL_CANDIDATES,
        "size": TR_SIZE_CANDIDATES,
        "side": TR_SIDE_CANDIDATES,
        "coin": TR_COIN_CANDIDATES,
    }

    for std_name, candidates in optional_cols.items():
        try:
            col = _detect_column(df, candidates, std_name)
            rename_map[col] = std_name
        except KeyError:
            logger.warning("Optional column '%s' not found; skipping.", std_name)

    df = df.rename(columns=rename_map)
    return df


def validate_dataframe(df: pd.DataFrame, name: str) -> None:
    """
    Log basic validation info for a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to validate.
    name : str
        Human-readable name for logging.
    """
    logger.info("--- Validating: %s ---", name)
    logger.info("Shape: %s", df.shape)
    logger.info("Columns: %s", list(df.columns))
    logger.info("Dtypes:\n%s", df.dtypes)
    missing = df.isnull().sum()
    if missing.any():
        logger.warning("Missing values:\n%s", missing[missing > 0])
    else:
        logger.info("No missing values detected.")
    logger.info("Duplicated rows: %d", df.duplicated().sum())
