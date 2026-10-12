"""
features.py
-----------
Feature engineering on the merged dataset.
Creates sentiment flags, rolling statistics, daily trader statistics, and
trader-level aggregates.
"""

import logging

import numpy as np
import pandas as pd

from src.config import SENTIMENT_BINS, SENTIMENT_LABELS

logger = logging.getLogger(__name__)


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first existing column from a candidate list."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series and return NaN where the denominator is zero."""
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def _add_trade_direction_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add is_long and is_short flags using the richest available direction column.

    Hyperliquid data usually has both side and Direction. Direction is preferred
    because it distinguishes Open Long/Open Short directly; side is used as a
    fallback for datasets that only contain BUY/SELL.
    """
    df = df.copy()
    direction_col = _first_existing(df, ["direction", "Direction"])
    side_col = _first_existing(df, ["side", "Side"])

    if direction_col:
        direction = df[direction_col].astype(str).str.lower()
        df["is_long"] = direction.str.contains("long", na=False)
        df["is_short"] = direction.str.contains("short", na=False)
    elif side_col:
        side = df[side_col].astype(str).str.lower()
        df["is_long"] = side.str.contains("buy|long", regex=True, na=False)
        df["is_short"] = side.str.contains("sell|short", regex=True, na=False)
    else:
        df["is_long"] = False
        df["is_short"] = False

    return df


def _add_leverage_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise leverage when the source dataset provides it.

    The provided Hyperliquid CSV does not include margin/collateral or leverage,
    so this function intentionally leaves leverage as NaN instead of inventing a
    misleading proxy.
    """
    df = df.copy()
    leverage_col = _first_existing(
        df,
        ["leverage", "Leverage", "avg_leverage", "Average Leverage"],
    )
    if leverage_col:
        df["leverage"] = pd.to_numeric(df[leverage_col], errors="coerce")
    else:
        df["leverage"] = np.nan
    return df


def add_sentiment_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary is_fear flag and a 5-class sentiment_regime column.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataframe containing fg_value and optionally fg_classification.

    Returns
    -------
    pd.DataFrame
        Dataframe with new columns: sentiment_regime, is_fear, is_greed.
    """
    df = df.copy()

    if "fg_classification" in df.columns:
        df["sentiment_regime"] = df["fg_classification"]
    else:
        df["sentiment_regime"] = pd.cut(
            df["fg_value"],
            bins=SENTIMENT_BINS,
            labels=SENTIMENT_LABELS,
            include_lowest=True,
        )

    df["is_fear"] = df["fg_value"] < 50
    df["is_greed"] = df["fg_value"] >= 50
    return df


def add_rolling_features(
    df: pd.DataFrame, windows: list[int] = [7, 14, 30]
) -> pd.DataFrame:
    """
    Add rolling mean and standard deviation of fg_value over multiple windows.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataframe sorted by date.
    windows : list[int]
        Rolling window sizes in days.

    Returns
    -------
    pd.DataFrame
        Dataframe with added rolling feature columns.
    """
    df = df.copy().sort_values("date")

    for w in windows:
        df[f"fg_rolling_mean_{w}d"] = (
            df["fg_value"].rolling(window=w, min_periods=1).mean()
        )
        df[f"fg_rolling_std_{w}d"] = (
            df["fg_value"].rolling(window=w, min_periods=1).std()
        )

    logger.info("Added rolling features for windows: %s", windows)
    return df


def add_pnl_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trade-level PnL features: win flag, absolute PnL, log-transformed size.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataframe with pnl column.

    Returns
    -------
    pd.DataFrame
        Dataframe with new columns: is_win, pnl_abs, log_size.
    """
    df = df.copy()

    if "pnl" in df.columns:
        df["is_win"] = df["pnl"] > 0
        df["pnl_abs"] = df["pnl"].abs()

    size_col = _first_existing(df, ["size", "size_usd"])
    if size_col:
        df["trade_size"] = pd.to_numeric(df[size_col], errors="coerce")
        df["log_size"] = np.log1p(df["trade_size"].clip(lower=0))
    else:
        df["trade_size"] = np.nan
        df["log_size"] = np.nan

    df = _add_trade_direction_flags(df)
    df = _add_leverage_feature(df)

    return df


def build_daily_trader_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build daily trader-level features.

    Creates one row per account/date with daily PnL, trade count, win rate,
    average trade size, average leverage, long/short ratio, cumulative PnL, and
    a drawdown proxy based on cumulative PnL versus its prior peak.
    """
    required = {"account", "date", "pnl"}
    missing = required - set(df.columns)
    if missing:
        logger.warning("Skipping daily trader stats: missing columns %s.", missing)
        return pd.DataFrame()

    work = add_pnl_features(df)
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work["pnl"] = pd.to_numeric(work["pnl"], errors="coerce")
    work = work.dropna(subset=["account", "date", "pnl"])

    agg = (
        work.groupby(["account", "date"], as_index=False)
        .agg(
            daily_pnl=("pnl", "sum"),
            daily_win_rate=("is_win", "mean"),
            daily_trade_count=("pnl", "count"),
            daily_avg_trade_size=("trade_size", "mean"),
            daily_avg_leverage=("leverage", "mean"),
            daily_long_trades=("is_long", "sum"),
            daily_short_trades=("is_short", "sum"),
        )
        .sort_values(["account", "date"])
        .reset_index(drop=True)
    )

    if "fg_value" in work.columns:
        sentiment = (
            work.groupby(["account", "date"], as_index=False)
            .agg(daily_avg_fg_value=("fg_value", "mean"))
        )
        sentiment["is_fear"] = sentiment["daily_avg_fg_value"] < 50
        agg = agg.merge(sentiment, on=["account", "date"], how="left")

    agg["daily_long_short_ratio"] = _safe_ratio(
        agg["daily_long_trades"], agg["daily_short_trades"]
    )
    agg["cumulative_pnl"] = agg.groupby("account")["daily_pnl"].cumsum()
    agg["running_peak_pnl"] = agg.groupby("account")["cumulative_pnl"].cummax()
    agg["drawdown_proxy"] = agg["cumulative_pnl"] - agg["running_peak_pnl"]

    logger.info("Daily trader stats built. Shape: %s", agg.shape)
    return agg


def build_trader_aggregates(
    df: pd.DataFrame, daily_stats: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Aggregate trade-level data to the trader (account) level.

    Computes total PnL, win rate, number of trades, average trade size,
    average leverage, long/short ratio, drawdown proxy, and performance broken
    down by sentiment regime.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataframe with account, pnl, is_win, sentiment_regime columns.

    Returns
    -------
    pd.DataFrame
        One row per trader with aggregated features.
        Returns empty DataFrame if account column is missing.
    """
    required = {"account", "pnl"}
    missing = required - set(df.columns)
    if missing:
        logger.warning("Skipping trader aggregation: missing columns %s.", missing)
        return pd.DataFrame()

    work = add_pnl_features(df)
    daily = daily_stats if daily_stats is not None else build_daily_trader_stats(work)

    agg = (
        work.groupby("account", as_index=False)
        .agg(
            total_pnl=("pnl", "sum"),
            mean_pnl=("pnl", "mean"),
            median_pnl=("pnl", "median"),
            trade_count=("pnl", "count"),
            win_rate=("is_win", "mean"),
            avg_trade_size=("trade_size", "mean"),
            total_volume=("trade_size", "sum"),
            avg_leverage=("leverage", "mean"),
            long_trades=("is_long", "sum"),
            short_trades=("is_short", "sum"),
        )
    )
    agg["long_short_ratio"] = _safe_ratio(agg["long_trades"], agg["short_trades"])

    if daily is not None and not daily.empty:
        daily_rollup = (
            daily.groupby("account", as_index=False)
            .agg(
                active_days=("date", "nunique"),
                avg_daily_pnl=("daily_pnl", "mean"),
                best_daily_pnl=("daily_pnl", "max"),
                worst_daily_pnl=("daily_pnl", "min"),
                max_drawdown_proxy=("drawdown_proxy", "min"),
            )
        )
        agg = agg.merge(daily_rollup, on="account", how="left")

    # Win rate during fear vs greed periods
    if "is_fear" in work.columns and "is_win" in work.columns:
        fear_wr = (
            work[work["is_fear"]]
            .groupby("account")["is_win"]
            .mean()
            .rename("win_rate_fear")
        )
        greed_wr = (
            work[~work["is_fear"]]
            .groupby("account")["is_win"]
            .mean()
            .rename("win_rate_greed")
        )
        agg = agg.merge(fear_wr, on="account", how="left")
        agg = agg.merge(greed_wr, on="account", how="left")

    logger.info("Trader aggregates built. Shape: %s", agg.shape)
    return agg
