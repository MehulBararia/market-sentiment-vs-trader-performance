"""
main.py
-------
Orchestration entry point for the full analysis pipeline.

Usage:
    python main.py
"""

import logging
import sys

import pandas as pd


def _setup_logging() -> None:
    """Configure root logger to write to stdout and a log file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline.log", encoding="utf-8"),
        ],
    )


def main() -> None:
    """Run the complete analysis pipeline end-to-end."""
    _setup_logging()
    logger = logging.getLogger(__name__)

    from src.config import (
        CHARTS_DIR,
        DAILY_TRADER_STATS,
        MERGED_DATASET,
        TABLES_DIR,
        TRADER_FEATURES,
        TRADER_SEGMENTS,
    )
    from src.cleaner import clean_fear_greed, clean_trades
    from src.data_loader import load_fear_greed, load_trades, validate_dataframe
    from src.eda import save_eda_tables, sentiment_pnl_summary, summarise_merged
    from src.features import (
        add_pnl_features,
        add_rolling_features,
        add_sentiment_regime,
        build_daily_trader_stats,
        build_trader_aggregates,
    )
    from src.merger import merge_datasets
    from src.report import generate_report
    from src.segmentation import run_trader_segmentation, segment_traders
    from src.stats import run_statistical_analysis
    from src.visualizations import (
        generate_eda_visualizations,
        plot_correlation_heatmap,
        plot_daily_pnl_vs_fg,
        plot_fear_greed_over_time,
        plot_pnl_by_sentiment,
        plot_trader_segments,
        plot_win_rate_by_sentiment,
    )

    logger.info("=" * 60)
    logger.info("Pipeline started.")
    logger.info("=" * 60)

    logger.info("--- Stage 1: Loading data ---")
    fg_raw = load_fear_greed()
    trades_raw = load_trades()
    validate_dataframe(fg_raw, "Fear & Greed (raw)")
    validate_dataframe(trades_raw, "Trades (raw)")

    logger.info("--- Stage 2: Cleaning data ---")
    fg_clean = clean_fear_greed(fg_raw)
    trades_clean = clean_trades(trades_raw)

    logger.info("--- Stage 3: Merging datasets ---")
    merged = merge_datasets(fg_clean, trades_clean)

    logger.info("--- Stage 4: Feature engineering ---")
    merged = add_sentiment_regime(merged)
    merged = add_rolling_features(merged)
    merged = add_pnl_features(merged)
    daily_trader_stats = build_daily_trader_stats(merged)
    trader_agg = build_trader_aggregates(merged, daily_trader_stats)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MERGED_DATASET, index=False)
    logger.info("Merged dataset saved: %s", MERGED_DATASET)
    if not daily_trader_stats.empty:
        daily_trader_stats.to_csv(DAILY_TRADER_STATS, index=False)
        logger.info("Daily trader stats saved: %s", DAILY_TRADER_STATS)
    if not trader_agg.empty:
        trader_agg.to_csv(TRADER_FEATURES, index=False)
        logger.info("Trader feature table saved: %s", TRADER_FEATURES)

    logger.info("--- Stage 5: EDA ---")
    eda_summary = summarise_merged(merged)
    sentiment_summary = sentiment_pnl_summary(merged)
    save_eda_tables(eda_summary, sentiment_summary)
    chart_notes = generate_eda_visualizations(merged, daily_trader_stats, trader_agg)

    logger.info("--- Stage 6: Statistical analysis ---")
    stats_outputs = run_statistical_analysis(merged, daily_trader_stats, trader_agg)

    logger.info("--- Stage 7: Trader segmentation ---")
    behavior_segments = pd.DataFrame()
    segment_summary = pd.DataFrame()
    segment_interpretations = pd.DataFrame()
    if not trader_agg.empty:
        behavior_segments, segment_summary, segment_interpretations = (
            run_trader_segmentation(trader_agg)
        )
        trader_agg = segment_traders(trader_agg)
        trader_agg.to_csv(TRADER_SEGMENTS, index=False)
        logger.info("Trader segments saved: %s", TRADER_SEGMENTS)

    logger.info("--- Stage 8: Generating visualizations ---")
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_fear_greed_over_time(merged)
    plot_pnl_by_sentiment(merged)
    plot_win_rate_by_sentiment(merged)
    plot_correlation_heatmap(merged)
    plot_daily_pnl_vs_fg(merged)
    if not trader_agg.empty:
        plot_trader_segments(trader_agg)

    logger.info("--- Stage 9: Generating final report ---")
    generate_report(
        merged=merged,
        fg_raw=fg_raw,
        trades_raw=trades_raw,
        sentiment_summary=sentiment_summary,
        stats_summary=stats_outputs.get("summary", pd.DataFrame()),
        ranked_correlations=stats_outputs.get("ranked_correlations", pd.DataFrame()),
        chart_notes=chart_notes,
        trader_segments=trader_agg,
        behavior_segments=behavior_segments,
        segment_summary=segment_summary,
        segment_interpretations=segment_interpretations,
        eda_summary=eda_summary,
    )

    logger.info("=" * 60)
    logger.info("Pipeline complete. Check output/ for all results.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
