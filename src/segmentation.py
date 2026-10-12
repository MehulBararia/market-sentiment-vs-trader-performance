"""
segmentation.py
---------------
Trader segmentation using both unsupervised clustering and interpretable
behavioral rules.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import (
    CHARTS_DIR,
    N_CLUSTERS,
    RANDOM_STATE,
    TRADER_BEHAVIOR_SEGMENTS,
    TRADER_SEGMENT_INTERPRETATIONS,
    TRADER_SEGMENT_SUMMARY,
)

logger = logging.getLogger(__name__)

SEGMENT_FEATURES = [
    "total_pnl",
    "win_rate",
    "trade_count",
    "mean_pnl",
    "avg_trade_size",
    "avg_leverage",
    "long_short_ratio",
    "max_drawdown_proxy",
    "win_rate_fear",
    "win_rate_greed",
]

SUMMARY_METRICS = {
    "trader_count": ("account", "count"),
    "average_pnl": ("total_pnl", "mean"),
    "median_pnl": ("total_pnl", "median"),
    "win_rate": ("win_rate", "mean"),
    "trade_frequency": ("trade_count", "mean"),
    "average_leverage": ("avg_leverage", "mean"),
    "average_trade_size": ("avg_trade_size", "mean"),
    "long_short_ratio": ("long_short_ratio", "mean"),
}


def _median_threshold(series: pd.Series) -> float | None:
    """Return the median of valid numeric values, or None when unavailable."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty or values.nunique() < 2:
        return None
    return float(values.median())


def _binary_segment(
    df: pd.DataFrame,
    value_col: str,
    output_col: str,
    high_label: str,
    low_label: str,
    unavailable_label: str,
) -> tuple[pd.DataFrame, str]:
    """Create a median-threshold binary segment for a numeric feature."""
    df = df.copy()
    threshold = _median_threshold(df[value_col]) if value_col in df.columns else None
    if threshold is None:
        df[output_col] = unavailable_label
        rule = f"{output_col}: unavailable because {value_col} has no valid variation."
        return df, rule

    df[output_col] = np.where(df[value_col] >= threshold, high_label, low_label)
    rule = f"{output_col}: {high_label} if {value_col} >= median ({threshold:.4f}); otherwise {low_label}."
    return df, rule


def add_behavior_segments(trader_agg: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Add interpretable trader behavior segments.

    Segments include leverage availability, trade frequency, consistency,
    position-size aggression, and long/short directional bias.
    """
    df = trader_agg.copy()
    rules: list[str] = []

    df, rule = _binary_segment(
        df,
        "avg_leverage",
        "leverage_segment",
        "High Leverage",
        "Low Leverage",
        "Leverage Unavailable",
    )
    rules.append(rule)

    df, rule = _binary_segment(
        df,
        "trade_count",
        "frequency_segment",
        "Frequent Traders",
        "Infrequent Traders",
        "Frequency Unavailable",
    )
    rules.append(rule)

    df, rule = _binary_segment(
        df,
        "avg_trade_size",
        "position_size_segment",
        "Large Position Traders",
        "Small Position Traders",
        "Position Size Unavailable",
    )
    rules.append(rule)
    df["aggression_segment"] = df["position_size_segment"].replace(
        {
            "Large Position Traders": "Aggressive Traders",
            "Small Position Traders": "Conservative Traders",
        }
    )
    rules.append("aggression_segment: derived from average trade size median split.")

    win_threshold = _median_threshold(df["win_rate"]) if "win_rate" in df.columns else None
    drawdown_threshold = (
        _median_threshold(df["max_drawdown_proxy"])
        if "max_drawdown_proxy" in df.columns
        else None
    )
    if win_threshold is None:
        df["consistency_segment"] = "Consistency Unavailable"
        rules.append("consistency_segment: unavailable because win_rate has no valid variation.")
    else:
        profitable = pd.to_numeric(df.get("mean_pnl"), errors="coerce") > 0
        good_win_rate = pd.to_numeric(df["win_rate"], errors="coerce") >= win_threshold
        if drawdown_threshold is not None:
            stable_drawdown = (
                pd.to_numeric(df["max_drawdown_proxy"], errors="coerce")
                >= drawdown_threshold
            )
            criteria = profitable & good_win_rate & stable_drawdown
            rules.append(
                "consistency_segment: Consistent Winners require positive mean_pnl, "
                f"win_rate >= median ({win_threshold:.4f}), and max_drawdown_proxy "
                f">= median ({drawdown_threshold:.4f})."
            )
        else:
            criteria = profitable & good_win_rate
            rules.append(
                "consistency_segment: Consistent Winners require positive mean_pnl "
                f"and win_rate >= median ({win_threshold:.4f})."
            )
        df["consistency_segment"] = np.where(
            criteria, "Consistent Winners", "Inconsistent Traders"
        )

    if "long_short_ratio" in df.columns:
        ratio = pd.to_numeric(df["long_short_ratio"], errors="coerce")
        df["direction_bias_segment"] = np.select(
            [ratio > 1.05, ratio < 0.95],
            ["Long-Biased Traders", "Short-Biased Traders"],
            default="Balanced Direction Traders",
        )
        df.loc[ratio.isna(), "direction_bias_segment"] = "Direction Unavailable"
        rules.append(
            "direction_bias_segment: long_short_ratio > 1.05 is Long-Biased; "
            "< 0.95 is Short-Biased; otherwise Balanced."
        )
    else:
        df["direction_bias_segment"] = "Direction Unavailable"
        rules.append("direction_bias_segment: unavailable because long_short_ratio is missing.")

    return df, rules


def summarize_segments(
    segmented: pd.DataFrame, segment_columns: list[str]
) -> pd.DataFrame:
    """Compute professional summary statistics for each segment column."""
    rows: list[pd.DataFrame] = []
    for segment_col in segment_columns:
        if segment_col not in segmented.columns:
            continue
        summary = (
            segmented.groupby(segment_col, dropna=False)
            .agg(**SUMMARY_METRICS)
            .reset_index()
            .rename(columns={segment_col: "segment"})
        )
        summary.insert(0, "segment_type", segment_col)
        rows.append(summary)

    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    numeric_cols = result.select_dtypes(include="number").columns
    result[numeric_cols] = result[numeric_cols].round(4)
    return result


def build_segment_interpretations(
    summary: pd.DataFrame, rules: list[str]
) -> pd.DataFrame:
    """Create concise data-based interpretations for each segment row."""
    if summary.empty:
        return pd.DataFrame(columns=["segment_type", "segment", "interpretation"])

    interpretations = []
    for _, row in summary.iterrows():
        leverage_text = (
            "average leverage is unavailable in the source data"
            if pd.isna(row.get("average_leverage"))
            else f"average leverage is {row['average_leverage']:.2f}"
        )
        interpretations.append(
            {
                "segment_type": row["segment_type"],
                "segment": row["segment"],
                "interpretation": (
                    f"{row['segment']} contains {int(row['trader_count'])} traders. "
                    f"Average PnL is {row['average_pnl']:.2f}, median PnL is "
                    f"{row['median_pnl']:.2f}, win rate is {row['win_rate']:.2%}, "
                    f"average trade frequency is {row['trade_frequency']:.2f}, "
                    f"{leverage_text}, average trade size is "
                    f"{row['average_trade_size']:.2f}, and long/short ratio is "
                    f"{row['long_short_ratio']:.4f}."
                ),
            }
        )

    interpretations.append(
        {
            "segment_type": "criteria",
            "segment": "Segmentation rules",
            "interpretation": " ".join(rules),
        }
    )
    return pd.DataFrame(interpretations)


def _save(fig: plt.Figure, filename: str) -> None:
    """Save a segmentation figure to output/charts."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CHARTS_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved segmentation chart: %s", path)


def _plot_segment_bar(
    summary: pd.DataFrame, segment_type: str, metric: str, filename: str
) -> None:
    """Create a bar chart for one segment metric."""
    data = summary[summary["segment_type"] == segment_type].dropna(subset=[metric])
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=data, x="segment", y=metric, hue="segment", legend=False, ax=ax)
    ax.set_title(f"{metric.replace('_', ' ').title()} by {segment_type.replace('_', ' ').title()}")
    ax.set_xlabel("")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=20)
    if "pnl" in metric:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    _save(fig, filename)


def _plot_segment_box(
    segmented: pd.DataFrame, segment_col: str, value_col: str, filename: str
) -> None:
    """Create a box plot for trader-level metric distribution by segment."""
    if segment_col not in segmented.columns or value_col not in segmented.columns:
        return
    data = segmented.dropna(subset=[segment_col, value_col])
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=data,
        x=segment_col,
        y=value_col,
        hue=segment_col,
        legend=False,
        ax=ax,
    )
    ax.set_title(f"{value_col.replace('_', ' ').title()} Distribution by {segment_col.replace('_', ' ').title()}")
    ax.set_xlabel("")
    ax.set_ylabel(value_col.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=20)
    if "pnl" in value_col:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    _save(fig, filename)


def _plot_segment_pie(segmented: pd.DataFrame, segment_col: str, filename: str) -> None:
    """Create a pie chart showing segment composition."""
    if segment_col not in segmented.columns:
        return
    counts = segmented[segment_col].value_counts(dropna=False)
    if counts.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
    ax.set_title(f"Trader Composition by {segment_col.replace('_', ' ').title()}")
    _save(fig, filename)


def _plot_histogram(
    segmented: pd.DataFrame, value_col: str, filename: str, title: str
) -> None:
    """Create a histogram for a trader-level metric."""
    if value_col not in segmented.columns:
        return
    values = pd.to_numeric(segmented[value_col], errors="coerce").dropna()
    if values.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(values, bins=20, kde=True, color="#4C72B0", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(value_col.replace("_", " ").title())
    ax.set_ylabel("Trader Count")
    _save(fig, filename)


def generate_segment_visualizations(
    segmented: pd.DataFrame, summary: pd.DataFrame
) -> None:
    """Generate bar, box, histogram, pie, and distribution segment charts."""
    chart_specs = [
        ("frequency_segment", "average_pnl", "18_frequency_segment_average_pnl.png"),
        ("frequency_segment", "win_rate", "19_frequency_segment_win_rate.png"),
        ("consistency_segment", "average_pnl", "20_consistency_segment_average_pnl.png"),
        ("position_size_segment", "average_trade_size", "21_position_size_segment_average_size.png"),
        ("direction_bias_segment", "long_short_ratio", "22_direction_bias_long_short_ratio.png"),
        ("leverage_segment", "average_pnl", "23_leverage_segment_average_pnl.png"),
    ]
    for segment_type, metric, filename in chart_specs:
        _plot_segment_bar(summary, segment_type, metric, filename)

    _plot_segment_box(segmented, "frequency_segment", "total_pnl", "24_frequency_segment_pnl_box.png")
    _plot_segment_box(segmented, "consistency_segment", "total_pnl", "25_consistency_segment_pnl_box.png")
    _plot_segment_box(segmented, "position_size_segment", "avg_trade_size", "26_position_size_box.png")
    _plot_segment_pie(segmented, "frequency_segment", "27_frequency_segment_pie.png")
    _plot_segment_pie(segmented, "consistency_segment", "28_consistency_segment_pie.png")
    _plot_segment_pie(segmented, "direction_bias_segment", "29_direction_bias_pie.png")
    _plot_histogram(segmented, "trade_count", "30_trader_trade_count_histogram.png", "Trader Trade Count Distribution")
    _plot_histogram(segmented, "total_pnl", "31_trader_total_pnl_histogram.png", "Trader Total PnL Distribution")


def run_trader_segmentation(
    trader_agg: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run complete behavioral trader segmentation.

    Returns segmented trader features, segment summary table, and concise
    interpretations. All artifacts are saved to output/tables and output/charts.
    """
    segmented, rules = add_behavior_segments(trader_agg)
    segment_columns = [
        "leverage_segment",
        "frequency_segment",
        "consistency_segment",
        "aggression_segment",
        "position_size_segment",
        "direction_bias_segment",
    ]
    summary = summarize_segments(segmented, segment_columns)
    interpretations = build_segment_interpretations(summary, rules)
    generate_segment_visualizations(segmented, summary)

    TRADER_BEHAVIOR_SEGMENTS.parent.mkdir(parents=True, exist_ok=True)
    segmented.to_csv(TRADER_BEHAVIOR_SEGMENTS, index=False)
    summary.to_csv(TRADER_SEGMENT_SUMMARY, index=False)
    interpretations.to_csv(TRADER_SEGMENT_INTERPRETATIONS, index=False)

    logger.info("Saved behavioral trader segments to: %s", TRADER_BEHAVIOR_SEGMENTS)
    logger.info("Saved trader segment summary to: %s", TRADER_SEGMENT_SUMMARY)
    logger.info("Saved trader segment interpretations to: %s", TRADER_SEGMENT_INTERPRETATIONS)
    return segmented, summary, interpretations


def segment_traders(trader_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Apply K-Means clustering to trader aggregate features.

    Only uses features that are present and not entirely missing. Adds a segment
    integer column and a segment_label string column.
    """
    df = trader_agg.copy()

    available_features = [
        f
        for f in SEGMENT_FEATURES
        if f in df.columns and df[f].dropna().nunique() >= 2
    ]
    if len(available_features) < 2:
        logger.warning(
            "Not enough features for clustering (%d found). Skipping segmentation.",
            len(available_features),
        )
        df["segment"] = 0
        df["segment_label"] = "Unsegmented"
        return df

    X = df[available_features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = min(N_CLUSTERS, len(df))
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    df["segment"] = kmeans.fit_predict(X_scaled)

    segment_pnl = df.groupby("segment")["total_pnl"].median().sort_values()
    rank_map = {seg: rank for rank, seg in enumerate(segment_pnl.index)}
    df["segment_rank"] = df["segment"].map(rank_map)

    label_map = {
        0: "Low Performers",
        1: "Moderate Performers",
        2: "High Performers",
        3: "Elite Performers",
    }
    df["segment_label"] = df["segment_rank"].map(
        lambda r: label_map.get(r, f"Segment {r}")
    )

    logger.info(
        "K-Means segmentation complete. Cluster distribution:\n%s",
        df["segment_label"].value_counts().to_string(),
    )
    return df
