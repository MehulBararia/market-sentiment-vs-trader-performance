"""
visualizations.py
-----------------
All chart generation. Saves figures to output/charts/.
Uses matplotlib/seaborn for static charts.
"""

import logging

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import CHARTS_DIR, EDA_CHART_NOTES

logger = logging.getLogger(__name__)

# Global style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150


def _save(fig: plt.Figure, filename: str) -> None:
    """Save a matplotlib figure to the charts directory."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CHARTS_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved chart: %s", path)


def _fear_greed_only(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows labelled as binary Fear or Greed."""
    if "is_fear" not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    work["market_sentiment"] = np.where(work["is_fear"], "Fear", "Greed")
    return work


def _numeric_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first numeric column available from a candidate list."""
    for col in candidates:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None


def _format_delta(fear_value: float, greed_value: float, unit: str = "") -> str:
    """Create a cautious comparison sentence for Fear vs Greed charts."""
    if pd.isna(fear_value) or pd.isna(greed_value):
        return "One or both groups do not have enough data for a comparison."
    if np.isclose(fear_value, greed_value):
        return f"Fear and Greed values are similar at about {fear_value:.2f}{unit}."
    higher = "Fear" if fear_value > greed_value else "Greed"
    return (
        f"{higher} has the higher observed value "
        f"(Fear={fear_value:.2f}{unit}, Greed={greed_value:.2f}{unit})."
    )


def _save_notes(notes: list[dict]) -> None:
    """Save chart explanations and data-based observations."""
    if not notes:
        return
    EDA_CHART_NOTES.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(notes).to_csv(EDA_CHART_NOTES, index=False)
    logger.info("Saved EDA chart notes to: %s", EDA_CHART_NOTES)


def _save_unavailable_chart(filename: str, title: str, message: str) -> None:
    """Save a simple figure explaining why a requested chart is unavailable."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        wrap=True,
        fontsize=11,
        transform=ax.transAxes,
    )
    _save(fig, filename)


def plot_fear_greed_pnl_comparison(df: pd.DataFrame) -> list[dict]:
    """Create Fear vs Greed PnL box and mean-bar charts."""
    work = _fear_greed_only(df)
    if work.empty or "pnl" not in work.columns:
        logger.warning("Skipping Fear vs Greed PnL comparison: missing columns.")
        return []

    mask = ~work.get("pnl_outlier", pd.Series(False, index=work.index))
    box_df = work[mask].dropna(subset=["pnl", "market_sentiment"])
    order = ["Fear", "Greed"]
    notes: list[dict] = []

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(
        data=box_df,
        x="market_sentiment",
        y="pnl",
        hue="market_sentiment",
        order=order,
        palette=["#C44E52", "#55A868"],
        legend=False,
        ax=ax,
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.9)
    ax.set_title("Fear vs Greed PnL Distribution")
    ax.set_xlabel("Market Sentiment")
    ax.set_ylabel("Closed PnL (USD, outliers excluded)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    _save(fig, "07_fear_vs_greed_pnl_boxplot.png")

    medians = work.groupby("market_sentiment")["pnl"].median()
    notes.append(
        {
            "filename": "07_fear_vs_greed_pnl_boxplot.png",
            "represents": "Distribution of trade-level Closed PnL on Fear versus Greed days, with flagged PnL outliers excluded for readability.",
            "observation": _format_delta(
                medians.get("Fear", np.nan), medians.get("Greed", np.nan), " median PnL"
            ),
        }
    )

    summary = (
        work.groupby("market_sentiment", as_index=False)
        .agg(mean_pnl=("pnl", "mean"), median_pnl=("pnl", "median"), trades=("pnl", "count"))
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=summary,
        x="market_sentiment",
        y="mean_pnl",
        hue="market_sentiment",
        order=order,
        palette=["#C44E52", "#55A868"],
        legend=False,
        ax=ax,
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.9)
    ax.set_title("Average PnL on Fear vs Greed Days")
    ax.set_xlabel("Market Sentiment")
    ax.set_ylabel("Mean Closed PnL (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    _save(fig, "08_fear_vs_greed_mean_pnl_bar.png")

    means = summary.set_index("market_sentiment")["mean_pnl"]
    notes.append(
        {
            "filename": "08_fear_vs_greed_mean_pnl_bar.png",
            "represents": "Mean trade-level Closed PnL grouped into binary Fear and Greed market days.",
            "observation": _format_delta(
                means.get("Fear", np.nan), means.get("Greed", np.nan), " mean PnL"
            ),
        }
    )
    return notes


def plot_fear_greed_metric_bar(
    df: pd.DataFrame,
    value_col: str,
    filename: str,
    title: str,
    ylabel: str,
    agg: str = "mean",
    percent: bool = False,
) -> list[dict]:
    """Plot a reusable Fear vs Greed comparison bar chart."""
    work = _fear_greed_only(df)
    if work.empty or value_col not in work.columns:
        logger.warning("Skipping %s: missing %s.", filename, value_col)
        return []

    grouped = getattr(work.groupby("market_sentiment")[value_col], agg)().reindex(["Fear", "Greed"])
    plot_df = grouped.reset_index(name=value_col).dropna()
    if plot_df.empty:
        message = f"{title} could not be plotted because {value_col} has no valid values."
        _save_unavailable_chart(filename, title, message)
        return [
            {
                "filename": filename,
                "represents": title,
                "observation": message,
            }
        ]

    y = plot_df[value_col] * 100 if percent else plot_df[value_col]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x=plot_df["market_sentiment"],
        y=y,
        hue=plot_df["market_sentiment"],
        palette=["#C44E52", "#55A868"],
        legend=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Market Sentiment")
    ax.set_ylabel(ylabel)
    if percent:
        ax.set_ylim(0, max(100, float(y.max()) * 1.15))
    _save(fig, filename)

    unit = "%" if percent else ""
    fear_val = grouped.get("Fear", np.nan) * (100 if percent else 1)
    greed_val = grouped.get("Greed", np.nan) * (100 if percent else 1)
    return [
        {
            "filename": filename,
            "represents": title,
            "observation": _format_delta(fear_val, greed_val, unit),
        }
    ]


def plot_distribution(
    df: pd.DataFrame, value_col: str, filename: str, title: str, xlabel: str
) -> list[dict]:
    """Plot a numeric distribution if the column is available."""
    if value_col not in df.columns:
        logger.warning("Skipping %s: missing %s.", filename, value_col)
        return []
    values = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if values.empty:
        logger.warning("Skipping %s: %s has no numeric values.", filename, value_col)
        message = f"{title} could not be plotted because {value_col} has no numeric values."
        _save_unavailable_chart(filename, title, message)
        return [
            {
                "filename": filename,
                "represents": title,
                "observation": message,
            }
        ]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(values, bins=50, kde=True, ax=ax, color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    _save(fig, filename)
    return [
        {
            "filename": filename,
            "represents": title,
            "observation": (
                f"The observed range is {values.min():.2f} to {values.max():.2f}; "
                f"median is {values.median():.2f}."
            ),
        }
    ]


def plot_long_short_ratio_comparison(df: pd.DataFrame) -> list[dict]:
    """Plot long/short ratio by Fear vs Greed."""
    work = _fear_greed_only(df)
    if work.empty or not {"is_long", "is_short"}.issubset(work.columns):
        logger.warning("Skipping long/short ratio comparison: missing direction flags.")
        return []
    grouped = work.groupby("market_sentiment").agg(
        long_trades=("is_long", "sum"), short_trades=("is_short", "sum")
    )
    grouped["long_short_ratio"] = grouped["long_trades"] / grouped["short_trades"].replace(0, np.nan)
    grouped = grouped.reindex(["Fear", "Greed"]).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=grouped,
        x="market_sentiment",
        y="long_short_ratio",
        hue="market_sentiment",
        palette=["#C44E52", "#55A868"],
        legend=False,
        ax=ax,
    )
    ax.set_title("Long/Short Ratio on Fear vs Greed Days")
    ax.set_xlabel("Market Sentiment")
    ax.set_ylabel("Long Trades / Short Trades")
    _save(fig, "12_long_short_ratio_fear_vs_greed.png")
    ratios = grouped.set_index("market_sentiment")["long_short_ratio"]
    return [
        {
            "filename": "12_long_short_ratio_fear_vs_greed.png",
            "represents": "Ratio of long-direction trades to short-direction trades on Fear and Greed days.",
            "observation": _format_delta(
                ratios.get("Fear", np.nan), ratios.get("Greed", np.nan), " long/short ratio"
            ),
        }
    ]


def plot_daily_trade_volume(df: pd.DataFrame) -> list[dict]:
    """Plot daily aggregate trade size over time."""
    size_col = _numeric_col(df, ["trade_size", "size", "size_usd"])
    if size_col is None or "date" not in df.columns:
        logger.warning("Skipping daily volume plot: missing date or size column.")
        return []
    daily = (
        df.assign(date=pd.to_datetime(df["date"], errors="coerce"))
        .dropna(subset=["date", size_col])
        .groupby("date", as_index=False)[size_col]
        .sum()
        .rename(columns={size_col: "daily_trade_volume"})
    )
    if daily.empty:
        return []
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily["date"], daily["daily_trade_volume"], color="#4C72B0", linewidth=1.1)
    ax.set_title("Daily Trade Volume Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Trade Size / Volume")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    _save(fig, "16_daily_trade_volume_over_time.png")
    return [
        {
            "filename": "16_daily_trade_volume_over_time.png",
            "represents": "Daily aggregate trade size over the merged analysis period.",
            "observation": (
                f"Daily observed volume ranges from {daily['daily_trade_volume'].min():,.2f} "
                f"to {daily['daily_trade_volume'].max():,.2f}."
            ),
        }
    ]


def generate_eda_visualizations(
    merged: pd.DataFrame,
    daily_trader_stats: pd.DataFrame | None = None,
    trader_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate all Stage 4 EDA visualizations and save chart notes."""
    notes: list[dict] = []
    notes.extend(plot_fear_greed_pnl_comparison(merged))
    notes.extend(
        plot_fear_greed_metric_bar(
            merged,
            "is_win",
            "09_win_rate_fear_vs_greed.png",
            "Win Rate on Fear vs Greed Days",
            "Win Rate (%)",
            percent=True,
        )
    )
    notes.extend(
        plot_fear_greed_metric_bar(
            merged,
            "pnl",
            "10_trade_frequency_fear_vs_greed.png",
            "Trade Frequency on Fear vs Greed Days",
            "Number of Trades",
            agg="count",
        )
    )
    notes.extend(
        plot_fear_greed_metric_bar(
            merged,
            "leverage",
            "11_average_leverage_fear_vs_greed.png",
            "Average Leverage on Fear vs Greed Days",
            "Average Leverage",
        )
    )
    notes.extend(plot_long_short_ratio_comparison(merged))
    notes.extend(
        plot_distribution(
            merged,
            "leverage",
            "17_leverage_distribution.png",
            "Leverage Distribution",
            "Leverage",
        )
    )
    plot_correlation_heatmap(merged)
    notes.append(
        {
            "filename": "04_correlation_heatmap.png",
            "represents": "Correlation heatmap across relevant numeric features in the merged dataset.",
            "observation": "Use the ranked correlation table for exact strongest positive and negative relationships.",
        }
    )
    notes.extend(
        plot_distribution(
            trader_features if trader_features is not None else merged,
            "total_pnl" if trader_features is not None else "pnl",
            "13_trader_profitability_distribution.png",
            "Trader Profitability Distribution",
            "Total PnL (USD)",
        )
    )
    notes.extend(
        plot_distribution(
            merged,
            _numeric_col(merged, ["trade_size", "size", "size_usd"]) or "trade_size",
            "14_average_trade_size_distribution.png",
            "Trade Size Distribution",
            "Trade Size",
        )
    )
    notes.extend(plot_daily_trade_volume(merged))
    if daily_trader_stats is not None and not daily_trader_stats.empty:
        notes.extend(
            plot_distribution(
                daily_trader_stats,
                "daily_pnl",
                "15_daily_trader_pnl_distribution.png",
                "Daily Trader PnL Distribution",
                "Daily Trader PnL (USD)",
            )
        )
    _save_notes(notes)
    return pd.DataFrame(notes)


def plot_fear_greed_over_time(df: pd.DataFrame) -> None:
    """Line chart of the Fear & Greed Index over time with sentiment shading."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["date"], df["fg_value"], linewidth=1.2, color="#2196F3", label="F&G Index")
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="Neutral (50)")
    ax.fill_between(
        df["date"], 0, df["fg_value"],
        where=df["fg_value"] < 50, alpha=0.15, color="red", label="Fear",
    )
    ax.fill_between(
        df["date"], 0, df["fg_value"],
        where=df["fg_value"] >= 50, alpha=0.15, color="green", label="Greed",
    )
    ax.set_title("Bitcoin Fear & Greed Index Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Index Value (0-100)")
    ax.legend()
    _save(fig, "01_fear_greed_over_time.png")


def plot_pnl_by_sentiment(df: pd.DataFrame) -> None:
    """Box plot of PnL distribution across sentiment regimes."""
    if "pnl" not in df.columns or "sentiment_regime" not in df.columns:
        logger.warning("Skipping plot_pnl_by_sentiment: missing columns.")
        return

    order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    present = [o for o in order if o in df["sentiment_regime"].unique()]

    # Exclude flagged outliers for cleaner visualisation
    mask = ~df.get("pnl_outlier", pd.Series(False, index=df.index))
    plot_df = df[mask]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=plot_df,
        x="sentiment_regime",
        y="pnl",
        hue="sentiment_regime",
        order=present,
        ax=ax,
        palette="RdYlGn",
        legend=False,
    )
    ax.set_title("PnL Distribution by Sentiment Regime (outliers excluded)")
    ax.set_xlabel("Sentiment Regime")
    ax.set_ylabel("PnL (USD)")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    plt.xticks(rotation=15)
    _save(fig, "02_pnl_by_sentiment.png")


def plot_win_rate_by_sentiment(df: pd.DataFrame) -> None:
    """Bar chart of win rate by sentiment regime."""
    if "is_win" not in df.columns or "sentiment_regime" not in df.columns:
        logger.warning("Skipping plot_win_rate_by_sentiment: missing columns.")
        return

    order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    wr = (
        df.groupby("sentiment_regime", observed=True)["is_win"]
        .mean()
        .reindex(order)
        .dropna()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = sns.color_palette("RdYlGn", len(wr))
    bars = ax.bar(wr["sentiment_regime"], wr["is_win"] * 100, color=palette)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="50% baseline")
    ax.set_title("Win Rate by Sentiment Regime")
    ax.set_xlabel("Sentiment Regime")
    ax.set_ylabel("Win Rate (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    for bar, val in zip(bars, wr["is_win"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    _save(fig, "03_win_rate_by_sentiment.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of correlations between numeric features."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        logger.warning("Skipping correlation heatmap: fewer than 2 numeric columns.")
        return

    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap")
    _save(fig, "04_correlation_heatmap.png")


def plot_trader_segments(trader_agg: pd.DataFrame) -> None:
    """Scatter plot of trader segments by total PnL vs win rate."""
    if "segment_label" not in trader_agg.columns:
        logger.warning("Skipping trader segment plot: segment_label missing.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    palette = sns.color_palette("tab10", trader_agg["segment_label"].nunique())
    for i, (label, grp) in enumerate(trader_agg.groupby("segment_label")):
        ax.scatter(
            grp["win_rate"] * 100,
            grp["total_pnl"],
            label=label,
            alpha=0.6,
            s=40,
            color=palette[i],
        )
    ax.set_title("Trader Segments: Win Rate vs Total PnL")
    ax.set_xlabel("Win Rate (%)")
    ax.set_ylabel("Total PnL (USD)")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    ax.legend(title="Segment")
    _save(fig, "05_trader_segments.png")


def plot_daily_pnl_vs_fg(df: pd.DataFrame) -> None:
    """Scatter plot of daily aggregate PnL vs Fear & Greed value."""
    if "pnl" not in df.columns or "fg_value" not in df.columns:
        logger.warning("Skipping daily PnL vs FG plot: missing columns.")
        return

    daily = (
        df.groupby(["date", "fg_value"])
        .agg(daily_pnl=("pnl", "sum"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        daily["fg_value"],
        daily["daily_pnl"],
        alpha=0.4,
        c=daily["fg_value"],
        cmap="RdYlGn",
        s=20,
    )
    plt.colorbar(sc, ax=ax, label="Fear & Greed Value")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(50, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Daily Aggregate PnL vs Fear & Greed Index")
    ax.set_xlabel("Fear & Greed Value")
    ax.set_ylabel("Daily Total PnL (USD)")
    _save(fig, "06_daily_pnl_vs_fg.png")
