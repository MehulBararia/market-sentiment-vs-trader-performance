# Bitcoin Fear/Greed vs Hyperliquid Trader Performance

*Report generated: 2026-07-08 09:26:16*

## Introduction

This project analyzes how Bitcoin market sentiment, represented by the Fear & Greed Index, relates to Hyperliquid trader behavior and performance.

The objective is to align daily sentiment data with trade-level performance, engineer trader behavior features, test whether observed differences are statistically meaningful, and segment traders into interpretable behavior groups.

Understanding trader behavior under different sentiment regimes is important because profitability, trade size, trade direction, and win rate can change when markets move from fear-dominant to greed-dominant conditions.

## Dataset Summary

- Fear & Greed dataset: daily Bitcoin sentiment index with value and classification fields.
- Hyperliquid dataset: historical trade records with account, coin, side, timestamp, trade size, and closed PnL fields.
- Fear & Greed raw records: 2,644.
- Trade raw records: 211,224.
- Merged records available for analysis: 211,218.
- Merged date range: 2023-05-01 to 2025-05-01.
- Remaining missing cells in merged dataset: 211,221.
- Duplicate merged rows: 0.
- Features used: pnl, trade_size, is_win, is_long, is_short, fg_value, sentiment_regime, total_pnl, win_rate, trade_count, avg_trade_size, long_short_ratio, max_drawdown_proxy.

## Methodology

1. Data loading: both CSV files are loaded from the project data directory using flexible file and column detection.
2. Data cleaning: timestamps are parsed, dates are normalized, numeric fields are coerced, missing values are detected, and duplicate rows are removed where appropriate.
3. Date alignment: trades are merged with the Fear & Greed Index on normalized daily dates using an inner join.
4. Feature engineering: trade PnL, win flags, trade size, long/short flags, daily trader metrics, cumulative PnL, drawdown proxy, and trader aggregates are created.
5. Exploratory Data Analysis: publication-ready charts are generated with chart-level notes and cautious observations.
6. Statistical analysis: continuous variables use t-tests or Mann-Whitney U tests based on normality diagnostics; categorical variables use chi-square tests; numerical associations use Pearson or Spearman correlations.
7. Trader segmentation: traders are grouped using interpretable behavioral rules and an additional K-Means performance segmentation.

## Data Cleaning

- Fear & Greed raw records: 2,644.
- Trade raw records: 211,224.
- Merged records available for analysis: 211,218.
- Merged date range: 2023-05-01 to 2025-05-01.
- Remaining missing cells in merged dataset: 211,221.
- Duplicate merged rows: 0.
- Timestamp conversion: trade timestamps are parsed with day-first handling and normalized to daily granularity.
- Dataset merging: only dates present in both datasets are retained so every analyzed trade has a sentiment context.
- Leverage note: no leverage, margin, or collateral field is present in the current source data, so leverage is reported as unavailable rather than estimated.

## Exploratory Data Analysis

| filename                                 | represents                                                                                                            | observation                                                                                       |
|:-----------------------------------------|:----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------|
| 07_fear_vs_greed_pnl_boxplot.png         | Distribution of trade-level Closed PnL on Fear versus Greed days, with flagged PnL outliers excluded for readability. | Fear and Greed values are similar at about 0.00 median PnL.                                       |
| 08_fear_vs_greed_mean_pnl_bar.png        | Mean trade-level Closed PnL grouped into binary Fear and Greed market days.                                           | Greed has the higher observed value (Fear=46.02 mean PnL, Greed=51.36 mean PnL).                  |
| 09_win_rate_fear_vs_greed.png            | Win Rate on Fear vs Greed Days                                                                                        | Greed has the higher observed value (Fear=40.52%, Greed=41.79%).                                  |
| 10_trade_frequency_fear_vs_greed.png     | Trade Frequency on Fear vs Greed Days                                                                                 | Fear has the higher observed value (Fear=111116.00, Greed=100102.00).                             |
| 11_average_leverage_fear_vs_greed.png    | Average Leverage on Fear vs Greed Days                                                                                | Average Leverage on Fear vs Greed Days could not be plotted because leverage has no valid values. |
| 12_long_short_ratio_fear_vs_greed.png    | Ratio of long-direction trades to short-direction trades on Fear and Greed days.                                      | Fear has the higher observed value (Fear=1.79 long/short ratio, Greed=0.87 long/short ratio).     |
| 17_leverage_distribution.png             | Leverage Distribution                                                                                                 | Leverage Distribution could not be plotted because leverage has no numeric values.                |
| 04_correlation_heatmap.png               | Correlation heatmap across relevant numeric features in the merged dataset.                                           | Use the ranked correlation table for exact strongest positive and negative relationships.         |
| 13_trader_profitability_distribution.png | Trader Profitability Distribution                                                                                     | The observed range is -167621.12 to 2143382.60; median is 117655.08.                              |
| 14_average_trade_size_distribution.png   | Trade Size Distribution                                                                                               | The observed range is 0.00 to 3921430.72; median is 597.02.                                       |
| 16_daily_trade_volume_over_time.png      | Daily aggregate trade size over the merged analysis period.                                                           | Daily observed volume ranges from 0.11 to 55,914,617.49.                                          |
| 15_daily_trader_pnl_distribution.png     | Daily Trader PnL Distribution                                                                                         | The observed range is -358963.14 to 533974.66; median is 206.35.                                  |

### Sentiment and PnL Summary

| sentiment_regime   |   trade_count |   mean_pnl |   median_pnl |   std_pnl |   win_rate |        total_pnl |
|:-------------------|--------------:|-----------:|-------------:|----------:|-----------:|-----------------:|
| Extreme Fear       |         21400 |    34.5379 |            0 |  1136.06  |     0.3706 | 739110           |
| Extreme Greed      |         39992 |    67.8929 |            0 |   766.828 |     0.4649 |      2.71517e+06 |
| Fear               |         61837 |    54.2904 |            0 |   935.355 |     0.4208 |      3.35716e+06 |
| Greed              |         50303 |    42.7436 |            0 |  1116.03  |     0.3848 |      2.15013e+06 |
| Neutral            |         37686 |    34.3077 |            0 |   517.122 |     0.397  |      1.29292e+06 |

## Statistical Analysis

| variable                      | test_used                       |   test_statistic |      p_value | significant   | interpretation                                                                                                                                                                                                                                      |
|:------------------------------|:--------------------------------|-----------------:|-------------:|:--------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Closed PnL                    | Mann-Whitney U test             |      5.48167e+09 |   0          | Yes           | Closed PnL: the Fear vs Greed difference is statistically significant at alpha=0.05; the observed group summaries are similar (Fear=0.0000, Greed=0.0000). Test choice: normality diagnostics rejected normality or were not applicable.            |
| Trade Size                    | Mann-Whitney U test             |      5.93398e+09 |   0          | Yes           | Trade Size: the Fear vs Greed difference is statistically significant at alpha=0.05; Fear has the higher observed median (Fear=663.7800, Greed=554.3200). Test choice: normality diagnostics rejected normality or were not applicable.             |
| Leverage                      | Not run                         |    nan           | nan          | No            | Leverage: insufficient variation or valid observations for a two-group test.                                                                                                                                                                        |
| Average Daily PnL             | Mann-Whitney U test             | 655408           |   0.220847   | No            | Average Daily PnL: the Fear vs Greed difference is not statistically significant at alpha=0.05; Greed has the higher observed median (Fear=144.9931, Greed=240.7492). Test choice: normality diagnostics rejected normality or were not applicable. |
| Number of Trades              | Mann-Whitney U test             | 718238           |   0.00780047 | Yes           | Number of Trades: the Fear vs Greed difference is statistically significant at alpha=0.05; Fear has the higher observed median (Fear=32.0000, Greed=27.0000). Test choice: normality diagnostics rejected normality or were not applicable.         |
| Long vs Short Trades          | Chi-Square Test of Independence |  13120.5         |   0          | Yes           | Long vs Short Trades: trade_direction is associated with Fear vs Greed classification at alpha=0.05.                                                                                                                                                |
| Winning vs Losing Trades      | Chi-Square Test of Independence |     35.1797      |   0          | Yes           | Winning vs Losing Trades: is_win is associated with Fear vs Greed classification at alpha=0.05.                                                                                                                                                     |
| Fear vs Greed Classifications | Chi-Square Test of Independence | 182119           |   0          | Yes           | Fear vs Greed Classifications: sentiment_regime is associated with Fear vs Greed classification at alpha=0.05.                                                                                                                                      |

### Strongest Numerical Correlations

| variable_1          | variable_2          | method   |   correlation |   p_value |   abs_correlation | interpretation                                                                                                                |
|:--------------------|:--------------------|:---------|--------------:|----------:|------------------:|:------------------------------------------------------------------------------------------------------------------------------|
| trade_size          | log_size            | Spearman |      1        |         0 |          1        | trade_size and log_size have a spearman correlation of 1.0000; this describes association, not causation.                     |
| size                | trade_size          | Spearman |      1        |         0 |          1        | size and trade_size have a spearman correlation of 1.0000; this describes association, not causation.                         |
| size                | log_size            | Spearman |      1        |         0 |          1        | size and log_size have a spearman correlation of 1.0000; this describes association, not causation.                           |
| fg_rolling_mean_7d  | fg_rolling_mean_14d | Spearman |      0.999805 |         0 |          0.999805 | fg_rolling_mean_7d and fg_rolling_mean_14d have a spearman correlation of 0.9998; this describes association, not causation.  |
| fg_value            | fg_rolling_mean_7d  | Spearman |      0.999731 |         0 |          0.999731 | fg_value and fg_rolling_mean_7d have a spearman correlation of 0.9997; this describes association, not causation.             |
| fg_rolling_mean_14d | fg_rolling_mean_30d | Spearman |      0.999561 |         0 |          0.999561 | fg_rolling_mean_14d and fg_rolling_mean_30d have a spearman correlation of 0.9996; this describes association, not causation. |
| fg_value            | fg_rolling_mean_14d | Spearman |      0.999399 |         0 |          0.999399 | fg_value and fg_rolling_mean_14d have a spearman correlation of 0.9994; this describes association, not causation.            |
| fg_rolling_mean_7d  | fg_rolling_mean_30d | Spearman |      0.999154 |         0 |          0.999154 | fg_rolling_mean_7d and fg_rolling_mean_30d have a spearman correlation of 0.9992; this describes association, not causation.  |
| fg_value            | fg_rolling_mean_30d | Spearman |      0.998685 |         0 |          0.998685 | fg_value and fg_rolling_mean_30d have a spearman correlation of 0.9987; this describes association, not causation.            |
| Execution Price     | Size Tokens         | Spearman |     -0.772751 |         0 |          0.772751 | Execution Price and Size Tokens have a spearman correlation of -0.7728; this describes association, not causation.            |

## Trader Segmentation

Behavioral segmentation uses median thresholds where valid numerical data exists. Leverage segmentation is retained as an explicit unavailable category because the dataset does not include leverage inputs.

### Segment Summary Table

| segment_type           | segment                |   trader_count |   average_pnl |   median_pnl |   win_rate |   trade_frequency |   average_leverage |   average_trade_size |   long_short_ratio |
|:-----------------------|:-----------------------|---------------:|--------------:|-------------:|-----------:|------------------:|-------------------:|---------------------:|-------------------:|
| leverage_segment       | Leverage Unavailable   |             32 |        320453 |     117655   |     0.403  |           6600.56 |                nan |              6006.26 |            62.0347 |
| frequency_segment      | Frequent Traders       |             16 |        496528 |     220720   |     0.4136 |          11684.8  |                nan |              7912.91 |             3.268  |
| frequency_segment      | Infrequent Traders     |             16 |        144378 |      90788.9 |     0.3924 |           1516.31 |                nan |              4099.61 |           135.493  |
| consistency_segment    | Consistent Winners     |              9 |        280103 |     108731   |     0.4777 |           3763.33 |                nan |              3424.34 |           326.129  |
| consistency_segment    | Inconsistent Traders   |             23 |        336242 |     126579   |     0.3737 |           7710.78 |                nan |              7016.58 |             2.0133 |
| aggression_segment     | Aggressive Traders     |             16 |        416806 |     163042   |     0.3617 |           4185.31 |                nan |             10151.8  |             1.6239 |
| aggression_segment     | Conservative Traders   |             16 |        224099 |     107517   |     0.4443 |           9015.81 |                nan |              1860.75 |           127.092  |
| position_size_segment  | Large Position Traders |             16 |        416806 |     163042   |     0.3617 |           4185.31 |                nan |             10151.8  |             1.6239 |
| position_size_segment  | Small Position Traders |             16 |        224099 |     107517   |     0.4443 |           9015.81 |                nan |              1860.75 |           127.092  |
| direction_bias_segment | Direction Unavailable  |              5 |        231222 |      67845.6 |     0.4495 |           4607.6  |                nan |              4006.45 |           nan      |
| direction_bias_segment | Long-Biased Traders    |             12 |        312472 |     150561   |     0.4135 |           9569.42 |                nan |              8648.31 |           139.183  |
| direction_bias_segment | Short-Biased Traders   |             15 |        356581 |     108731   |     0.3791 |           4889.8  |                nan |              4559.23 |             0.3158 |

### Segment Interpretations

| segment_type           | segment                | interpretation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|:-----------------------|:-----------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| leverage_segment       | Leverage Unavailable   | Leverage Unavailable contains 32 traders. Average PnL is 320452.72, median PnL is 117655.08, win rate is 40.30%, average trade frequency is 6600.56, average leverage is unavailable in the source data, average trade size is 6006.26, and long/short ratio is 62.0347.                                                                                                                                                                                                                                                                                                                                                                  |
| frequency_segment      | Frequent Traders       | Frequent Traders contains 16 traders. Average PnL is 496527.66, median PnL is 220720.34, win rate is 41.36%, average trade frequency is 11684.81, average leverage is unavailable in the source data, average trade size is 7912.91, and long/short ratio is 3.2680.                                                                                                                                                                                                                                                                                                                                                                      |
| frequency_segment      | Infrequent Traders     | Infrequent Traders contains 16 traders. Average PnL is 144377.77, median PnL is 90788.85, win rate is 39.24%, average trade frequency is 1516.31, average leverage is unavailable in the source data, average trade size is 4099.61, and long/short ratio is 135.4931.                                                                                                                                                                                                                                                                                                                                                                    |
| consistency_segment    | Consistent Winners     | Consistent Winners contains 9 traders. Average PnL is 280103.42, median PnL is 108731.22, win rate is 47.77%, average trade frequency is 3763.33, average leverage is unavailable in the source data, average trade size is 3424.34, and long/short ratio is 326.1287.                                                                                                                                                                                                                                                                                                                                                                    |
| consistency_segment    | Inconsistent Traders   | Inconsistent Traders contains 23 traders. Average PnL is 336241.57, median PnL is 126578.94, win rate is 37.37%, average trade frequency is 7710.78, average leverage is unavailable in the source data, average trade size is 7016.58, and long/short ratio is 2.0133.                                                                                                                                                                                                                                                                                                                                                                   |
| aggression_segment     | Aggressive Traders     | Aggressive Traders contains 16 traders. Average PnL is 416805.97, median PnL is 163042.27, win rate is 36.17%, average trade frequency is 4185.31, average leverage is unavailable in the source data, average trade size is 10151.78, and long/short ratio is 1.6239.                                                                                                                                                                                                                                                                                                                                                                    |
| aggression_segment     | Conservative Traders   | Conservative Traders contains 16 traders. Average PnL is 224099.47, median PnL is 107517.05, win rate is 44.43%, average trade frequency is 9015.81, average leverage is unavailable in the source data, average trade size is 1860.75, and long/short ratio is 127.0924.                                                                                                                                                                                                                                                                                                                                                                 |
| position_size_segment  | Large Position Traders | Large Position Traders contains 16 traders. Average PnL is 416805.97, median PnL is 163042.27, win rate is 36.17%, average trade frequency is 4185.31, average leverage is unavailable in the source data, average trade size is 10151.78, and long/short ratio is 1.6239.                                                                                                                                                                                                                                                                                                                                                                |
| position_size_segment  | Small Position Traders | Small Position Traders contains 16 traders. Average PnL is 224099.47, median PnL is 107517.05, win rate is 44.43%, average trade frequency is 9015.81, average leverage is unavailable in the source data, average trade size is 1860.75, and long/short ratio is 127.0924.                                                                                                                                                                                                                                                                                                                                                               |
| direction_bias_segment | Direction Unavailable  | Direction Unavailable contains 5 traders. Average PnL is 231222.20, median PnL is 67845.62, win rate is 44.95%, average trade frequency is 4607.60, average leverage is unavailable in the source data, average trade size is 4006.45, and long/short ratio is nan.                                                                                                                                                                                                                                                                                                                                                                       |
| direction_bias_segment | Long-Biased Traders    | Long-Biased Traders contains 12 traders. Average PnL is 312471.67, median PnL is 150561.41, win rate is 41.35%, average trade frequency is 9569.42, average leverage is unavailable in the source data, average trade size is 8648.31, and long/short ratio is 139.1834.                                                                                                                                                                                                                                                                                                                                                                  |
| direction_bias_segment | Short-Biased Traders   | Short-Biased Traders contains 15 traders. Average PnL is 356581.06, median PnL is 108731.22, win rate is 37.91%, average trade frequency is 4889.80, average leverage is unavailable in the source data, average trade size is 4559.23, and long/short ratio is 0.3158.                                                                                                                                                                                                                                                                                                                                                                   |
| criteria               | Segmentation rules     | leverage_segment: unavailable because avg_leverage has no valid variation. frequency_segment: Frequent Traders if trade_count >= median (3699.0000); otherwise Infrequent Traders. position_size_segment: Large Position Traders if avg_trade_size >= median (3327.9720); otherwise Small Position Traders. aggression_segment: derived from average trade size median split. consistency_segment: Consistent Winners require positive mean_pnl, win_rate >= median (0.3920), and max_drawdown_proxy >= median (-16870.4502). direction_bias_segment: long_short_ratio > 1.05 is Long-Biased; < 0.95 is Short-Biased; otherwise Balanced. |

### K-Means Performance Segments

| segment_label       |   trader_count |   median_pnl |   mean_win_rate |   mean_trade_count |
|:--------------------|---------------:|-------------:|----------------:|-------------------:|
| Elite Performers    |              3 |     940164   |          0.411  |           25369.7  |
| High Performers     |              8 |     334758   |          0.3464 |            4314.25 |
| Low Performers      |             13 |      67845.6 |          0.3699 |            4600.77 |
| Moderate Performers |              8 |      92980.5 |          0.5102 |            5098.12 |

## Key Insights

### Insight 1: Extreme Greed has the highest observed mean PnL.

- Observation: Extreme Greed has the highest observed mean PnL.
- Supporting evidence: Mean PnL is 67.8929 across 39,992 trades.
- Business interpretation: Trader outcomes differ across sentiment regimes, so sentiment should be considered when reviewing historical performance.

### Insight 2: Extreme Greed has the highest observed win rate.

- Observation: Extreme Greed has the highest observed win rate.
- Supporting evidence: Win rate is 46.49%.
- Business interpretation: Win rate and average PnL should be evaluated together because high win rates do not always imply the highest total profitability.

### Insight 3: Closed PnL differs significantly between Fear and Greed days.

- Observation: Closed PnL differs significantly between Fear and Greed days.
- Supporting evidence: Mann-Whitney U test statistic=5,481,670,950.0000, p-value=0.00000000.
- Business interpretation: The observed behavioral difference is unlikely to be explained by random sampling noise alone, though it is not causal proof.

### Insight 4: The strongest numerical association is between trade_size and log_size.

- Observation: The strongest numerical association is between trade_size and log_size.
- Supporting evidence: Spearman correlation is 1.0000.
- Business interpretation: Highly correlated features may describe overlapping behavior and should be handled carefully in future predictive models.

### Insight 5: Frequent Traders has the higher average PnL among frequency segments.

- Observation: Frequent Traders has the higher average PnL among frequency segments.
- Supporting evidence: Average PnL is 496,527.6612; average trade frequency is 11,684.8125.
- Business interpretation: Trading frequency is a useful behavioral lens, but it should be reviewed alongside profitability and drawdown.

## Trading Strategy Recommendations

### Recommendation 1

- Problem identified: Trader performance varies across market sentiment regimes.
- Evidence supporting the recommendation: Mann-Whitney U test on Closed PnL produced p-value 0.00000000.
- Suggested strategy: Track Fear/Greed regime as a risk context variable before sizing or evaluating trade performance.
- Expected impact: Better post-trade review and more disciplined risk allocation by sentiment state.

### Recommendation 2

- Problem identified: Trader quality differs by consistency profile.
- Evidence supporting the recommendation: Consistent Winners shows win rate 47.77% and average PnL 280,103.4161.
- Suggested strategy: Benchmark traders against the consistent-winner criteria and study their risk controls before scaling exposure.
- Expected impact: More robust trader selection and less reliance on one-off large wins.

### Recommendation 3

- Problem identified: Leverage analysis cannot be completed from the current dataset.
- Evidence supporting the recommendation: The leverage segment is marked unavailable because avg_leverage has no valid values.
- Suggested strategy: Collect leverage, margin, or collateral fields in future data exports.
- Expected impact: Enables real high-leverage versus low-leverage risk analysis without proxies.

## Limitations

- The analysis is observational and does not establish causality.
- Hyperliquid data may not represent all market participants.
- Leverage-specific conclusions cannot be made because leverage inputs are absent.
- Outliers are flagged but retained; extreme trades can influence aggregate PnL.
- The Fear & Greed Index is daily, so intraday sentiment variation is not captured.

## Future Improvements

- Add predictive modeling to forecast trader outcomes by sentiment state.
- Expand clustering with more trader risk features and validate cluster stability.
- Build a Streamlit dashboard for interactive segment and sentiment exploration.
- Add market indicators such as volatility, funding rates, volume, open interest, and BTC returns.
- Collect leverage, margin, collateral, and liquidation data for more complete risk segmentation.

## Conclusion

The analysis shows that trader behavior and outcomes vary across market sentiment regimes and trader behavior segments. The strongest conclusions are based on observed PnL, win-rate, trade-size, frequency, and direction metrics, while leverage remains unavailable in the current dataset.

*End of Report*