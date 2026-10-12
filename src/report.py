"""
report.py
---------
Generate a professional Markdown report from the project artifacts.
"""

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from src.config import FINAL_REPORT

logger = logging.getLogger(__name__)


def _fmt(value: Any, decimals: int = 4) -> str:
    """Format values for report text."""
    if value is None or pd.isna(value):
        return "not available"
    if isinstance(value, (int, float)):
        return f"{value:,.{decimals}f}"
    return str(value)


def _pct(value: Any) -> str:
    """Format a proportion as a percentage."""
    if value is None or pd.isna(value):
        return "not available"
    return f"{float(value) * 100:.2f}%"


def _table(df: pd.DataFrame, max_rows: int = 12) -> str:
    """Return a Markdown table or a no-data placeholder."""
    if df is None or df.empty:
        return "*No data available.*"
    return df.head(max_rows).to_markdown(index=False)


def _safe_col(df: pd.DataFrame, column: str, default: Any = None) -> Any:
    """Return a column if present, otherwise a default."""
    return df[column] if df is not None and column in df.columns else default


def _build_key_insights(
    sentiment_summary: pd.DataFrame,
    stats_summary: pd.DataFrame,
    ranked_correlations: pd.DataFrame,
    segment_summary: pd.DataFrame,
    behavior_segments: pd.DataFrame,
) -> list[dict[str, str]]:
    """Build evidence-based insights from actual output tables."""
    insights: list[dict[str, str]] = []

    if not sentiment_summary.empty:
        top_mean = sentiment_summary.sort_values("mean_pnl", ascending=False).iloc[0]
        top_win = sentiment_summary.sort_values("win_rate", ascending=False).iloc[0]
        insights.append(
            {
                "Observation": f"{top_mean['sentiment_regime']} has the highest observed mean PnL.",
                "Supporting evidence": (
                    f"Mean PnL is {_fmt(top_mean['mean_pnl'])} across "
                    f"{int(top_mean['trade_count']):,} trades."
                ),
                "Business interpretation": (
                    "Trader outcomes differ across sentiment regimes, so sentiment "
                    "should be considered when reviewing historical performance."
                ),
            }
        )
        insights.append(
            {
                "Observation": f"{top_win['sentiment_regime']} has the highest observed win rate.",
                "Supporting evidence": f"Win rate is {_pct(top_win['win_rate'])}.",
                "Business interpretation": (
                    "Win rate and average PnL should be evaluated together because "
                    "high win rates do not always imply the highest total profitability."
                ),
            }
        )

    if not stats_summary.empty:
        sig = stats_summary[stats_summary["significant"].astype(str).str.lower() == "yes"]
        if not sig.empty:
            first = sig.iloc[0]
            insights.append(
                {
                    "Observation": f"{first['variable']} differs significantly between Fear and Greed days.",
                    "Supporting evidence": (
                        f"{first['test_used']} statistic={_fmt(first['test_statistic'])}, "
                        f"p-value={_fmt(first['p_value'], 8)}."
                    ),
                    "Business interpretation": (
                        "The observed behavioral difference is unlikely to be explained "
                        "by random sampling noise alone, though it is not causal proof."
                    ),
                }
            )

    if not ranked_correlations.empty:
        top_corr = ranked_correlations.iloc[0]
        insights.append(
            {
                "Observation": (
                    f"The strongest numerical association is between "
                    f"{top_corr['variable_1']} and {top_corr['variable_2']}."
                ),
                "Supporting evidence": (
                    f"{top_corr['method']} correlation is "
                    f"{_fmt(top_corr['correlation'])}."
                ),
                "Business interpretation": (
                    "Highly correlated features may describe overlapping behavior and "
                    "should be handled carefully in future predictive models."
                ),
            }
        )

    if not segment_summary.empty:
        freq = segment_summary[segment_summary["segment_type"] == "frequency_segment"]
        if not freq.empty:
            best = freq.sort_values("average_pnl", ascending=False).iloc[0]
            insights.append(
                {
                    "Observation": f"{best['segment']} has the higher average PnL among frequency segments.",
                    "Supporting evidence": (
                        f"Average PnL is {_fmt(best['average_pnl'])}; average trade "
                        f"frequency is {_fmt(best['trade_frequency'])}."
                    ),
                    "Business interpretation": (
                        "Trading frequency is a useful behavioral lens, but it should "
                        "be reviewed alongside profitability and drawdown."
                    ),
                }
            )

    if behavior_segments is not None and not behavior_segments.empty:
        leverage_missing = behavior_segments["avg_leverage"].isna().all()
        if leverage_missing:
            insights.append(
                {
                    "Observation": "Leverage is unavailable in the source trade data.",
                    "Supporting evidence": "All avg_leverage values are missing after preprocessing.",
                    "Business interpretation": (
                        "Leverage-based conclusions should not be made until margin, "
                        "collateral, or leverage fields are supplied."
                    ),
                }
            )

    return insights[:5]


def _build_recommendations(
    insights: list[dict[str, str]], stats_summary: pd.DataFrame, segment_summary: pd.DataFrame
) -> list[dict[str, str]]:
    """Build actionable recommendations from analysis outputs."""
    recommendations: list[dict[str, str]] = []

    pnl_row = pd.DataFrame()
    if not stats_summary.empty:
        pnl_row = stats_summary[stats_summary["variable"].astype(str).str.contains("PnL", case=False)]
    if not pnl_row.empty:
        row = pnl_row.iloc[0]
        recommendations.append(
            {
                "Problem identified": "Trader performance varies across market sentiment regimes.",
                "Evidence supporting the recommendation": (
                    f"{row['test_used']} on {row['variable']} produced p-value "
                    f"{_fmt(row['p_value'], 8)}."
                ),
                "Suggested strategy": (
                    "Track Fear/Greed regime as a risk context variable before sizing "
                    "or evaluating trade performance."
                ),
                "Expected impact": (
                    "Better post-trade review and more disciplined risk allocation by sentiment state."
                ),
            }
        )

    if not segment_summary.empty:
        consistency = segment_summary[
            segment_summary["segment_type"] == "consistency_segment"
        ]
        if not consistency.empty:
            best = consistency.sort_values("win_rate", ascending=False).iloc[0]
            recommendations.append(
                {
                    "Problem identified": "Trader quality differs by consistency profile.",
                    "Evidence supporting the recommendation": (
                        f"{best['segment']} shows win rate {_pct(best['win_rate'])} "
                        f"and average PnL {_fmt(best['average_pnl'])}."
                    ),
                    "Suggested strategy": (
                        "Benchmark traders against the consistent-winner criteria and "
                        "study their risk controls before scaling exposure."
                    ),
                    "Expected impact": (
                        "More robust trader selection and less reliance on one-off large wins."
                    ),
                }
            )

    recommendations.append(
        {
            "Problem identified": "Leverage analysis cannot be completed from the current dataset.",
            "Evidence supporting the recommendation": (
                "The leverage segment is marked unavailable because avg_leverage has no valid values."
            ),
            "Suggested strategy": (
                "Collect leverage, margin, or collateral fields in future data exports."
            ),
            "Expected impact": (
                "Enables real high-leverage versus low-leverage risk analysis without proxies."
            ),
        }
    )

    if len(recommendations) < 3 and insights:
        recommendations.append(
            {
                "Problem identified": insights[0]["Observation"],
                "Evidence supporting the recommendation": insights[0]["Supporting evidence"],
                "Suggested strategy": "Use this finding as a monitoring signal in future analysis.",
                "Expected impact": "Improves evidence-based review of trader behavior.",
            }
        )

    return recommendations[:3]


def _data_quality_text(merged: pd.DataFrame, fg_raw: pd.DataFrame, trades_raw: pd.DataFrame) -> list[str]:
    """Create data quality summary bullets."""
    return [
        f"Fear & Greed raw records: {len(fg_raw):,}.",
        f"Trade raw records: {len(trades_raw):,}.",
        f"Merged records available for analysis: {len(merged):,}.",
        f"Merged date range: {merged['date'].min().date()} to {merged['date'].max().date()}.",
        f"Remaining missing cells in merged dataset: {int(merged.isna().sum().sum()):,}.",
        f"Duplicate merged rows: {int(merged.duplicated().sum()):,}.",
    ]


def generate_report(
    merged: pd.DataFrame,
    fg_raw: pd.DataFrame,
    trades_raw: pd.DataFrame,
    sentiment_summary: pd.DataFrame,
    stats_summary: pd.DataFrame,
    ranked_correlations: pd.DataFrame,
    chart_notes: pd.DataFrame,
    trader_segments: pd.DataFrame,
    behavior_segments: pd.DataFrame,
    segment_summary: pd.DataFrame,
    segment_interpretations: pd.DataFrame,
    eda_summary: pd.DataFrame,
) -> None:
    """
    Write the final professional project report to output/final_summary.md.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    insights = _build_key_insights(
        sentiment_summary,
        stats_summary,
        ranked_correlations,
        segment_summary,
        behavior_segments,
    )
    recommendations = _build_recommendations(insights, stats_summary, segment_summary)

    feature_cols = [
        "pnl",
        "trade_size",
        "is_win",
        "is_long",
        "is_short",
        "fg_value",
        "sentiment_regime",
        "daily_pnl",
        "total_pnl",
        "win_rate",
        "trade_count",
        "avg_trade_size",
        "long_short_ratio",
        "max_drawdown_proxy",
    ]
    used_features = [col for col in feature_cols if col in merged.columns or col in behavior_segments.columns]

    lines: list[str] = [
        "# Bitcoin Fear/Greed vs Hyperliquid Trader Performance",
        "",
        f"*Report generated: {ts}*",
        "",
        "## Introduction",
        "",
        "This project analyzes how Bitcoin market sentiment, represented by the Fear & Greed Index, relates to Hyperliquid trader behavior and performance.",
        "",
        "The objective is to align daily sentiment data with trade-level performance, engineer trader behavior features, test whether observed differences are statistically meaningful, and segment traders into interpretable behavior groups.",
        "",
        "Understanding trader behavior under different sentiment regimes is important because profitability, trade size, trade direction, and win rate can change when markets move from fear-dominant to greed-dominant conditions.",
        "",
        "## Dataset Summary",
        "",
        "- Fear & Greed dataset: daily Bitcoin sentiment index with value and classification fields.",
        "- Hyperliquid dataset: historical trade records with account, coin, side, timestamp, trade size, and closed PnL fields.",
    ]
    lines.extend(f"- {item}" for item in _data_quality_text(merged, fg_raw, trades_raw))
    lines.extend(
        [
            f"- Features used: {', '.join(used_features)}.",
            "",
            "## Methodology",
            "",
            "1. Data loading: both CSV files are loaded from the project data directory using flexible file and column detection.",
            "2. Data cleaning: timestamps are parsed, dates are normalized, numeric fields are coerced, missing values are detected, and duplicate rows are removed where appropriate.",
            "3. Date alignment: trades are merged with the Fear & Greed Index on normalized daily dates using an inner join.",
            "4. Feature engineering: trade PnL, win flags, trade size, long/short flags, daily trader metrics, cumulative PnL, drawdown proxy, and trader aggregates are created.",
            "5. Exploratory Data Analysis: publication-ready charts are generated with chart-level notes and cautious observations.",
            "6. Statistical analysis: continuous variables use t-tests or Mann-Whitney U tests based on normality diagnostics; categorical variables use chi-square tests; numerical associations use Pearson or Spearman correlations.",
            "7. Trader segmentation: traders are grouped using interpretable behavioral rules and an additional K-Means performance segmentation.",
            "",
            "## Data Cleaning",
            "",
        ]
    )
    for item in _data_quality_text(merged, fg_raw, trades_raw):
        lines.append(f"- {item}")
    lines.extend(
        [
            "- Timestamp conversion: trade timestamps are parsed with day-first handling and normalized to daily granularity.",
            "- Dataset merging: only dates present in both datasets are retained so every analyzed trade has a sentiment context.",
            "- Leverage note: no leverage, margin, or collateral field is present in the current source data, so leverage is reported as unavailable rather than estimated.",
            "",
            "## Exploratory Data Analysis",
            "",
            _table(chart_notes, max_rows=20),
            "",
            "### Sentiment and PnL Summary",
            "",
            _table(sentiment_summary),
            "",
            "## Statistical Analysis",
            "",
            _table(stats_summary, max_rows=20),
            "",
            "### Strongest Numerical Correlations",
            "",
            _table(ranked_correlations, max_rows=10),
            "",
            "## Trader Segmentation",
            "",
            "Behavioral segmentation uses median thresholds where valid numerical data exists. Leverage segmentation is retained as an explicit unavailable category because the dataset does not include leverage inputs.",
            "",
            "### Segment Summary Table",
            "",
            _table(segment_summary, max_rows=30),
            "",
            "### Segment Interpretations",
            "",
            _table(segment_interpretations, max_rows=40),
            "",
            "### K-Means Performance Segments",
            "",
        ]
    )

    if trader_segments is not None and not trader_segments.empty and "segment_label" in trader_segments.columns:
        kmeans_summary = (
            trader_segments.groupby("segment_label")
            .agg(
                trader_count=("account", "count"),
                median_pnl=("total_pnl", "median"),
                mean_win_rate=("win_rate", "mean"),
                mean_trade_count=("trade_count", "mean"),
            )
            .reset_index()
            .round(4)
        )
        lines.append(_table(kmeans_summary))
    else:
        lines.append("*K-Means segment data is unavailable.*")

    lines.extend(["", "## Key Insights", ""])
    for i, insight in enumerate(insights, 1):
        lines.extend(
            [
                f"### Insight {i}: {insight['Observation']}",
                "",
                f"- Observation: {insight['Observation']}",
                f"- Supporting evidence: {insight['Supporting evidence']}",
                f"- Business interpretation: {insight['Business interpretation']}",
                "",
            ]
        )

    lines.extend(["## Trading Strategy Recommendations", ""])
    for i, rec in enumerate(recommendations, 1):
        lines.extend(
            [
                f"### Recommendation {i}",
                "",
                f"- Problem identified: {rec['Problem identified']}",
                f"- Evidence supporting the recommendation: {rec['Evidence supporting the recommendation']}",
                f"- Suggested strategy: {rec['Suggested strategy']}",
                f"- Expected impact: {rec['Expected impact']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Limitations",
            "",
            "- The analysis is observational and does not establish causality.",
            "- Hyperliquid data may not represent all market participants.",
            "- Leverage-specific conclusions cannot be made because leverage inputs are absent.",
            "- Outliers are flagged but retained; extreme trades can influence aggregate PnL.",
            "- The Fear & Greed Index is daily, so intraday sentiment variation is not captured.",
            "",
            "## Future Improvements",
            "",
            "- Add predictive modeling to forecast trader outcomes by sentiment state.",
            "- Expand clustering with more trader risk features and validate cluster stability.",
            "- Build a Streamlit dashboard for interactive segment and sentiment exploration.",
            "- Add market indicators such as volatility, funding rates, volume, open interest, and BTC returns.",
            "- Collect leverage, margin, collateral, and liquidation data for more complete risk segmentation.",
            "",
            "## Conclusion",
            "",
        ]
    )
    if insights:
        lines.append(
            "The analysis shows that trader behavior and outcomes vary across market sentiment regimes and trader behavior segments. "
            "The strongest conclusions are based on observed PnL, win-rate, trade-size, frequency, and direction metrics, while leverage remains unavailable in the current dataset."
        )
    else:
        lines.append(
            "The pipeline completed, but no strong reportable insights were generated from the available summary tables."
        )
    lines.append("")
    lines.append("*End of Report*")

    FINAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    FINAL_REPORT.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Final report written to: %s", FINAL_REPORT)
