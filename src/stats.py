"""
stats.py
--------
Statistical analysis: correlation tests, group comparisons, and significance testing.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    CHI_SQUARE_RESULTS,
    CORRELATION_MATRIX,
    RANKED_CORRELATIONS,
    STATISTICAL_SUMMARY,
)

logger = logging.getLogger(__name__)

ALPHA = 0.05


def pearson_correlation(df: pd.DataFrame, x: str, y: str) -> dict:
    """
    Compute Pearson correlation between two numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
    x : str
        Column name for the first variable.
    y : str
        Column name for the second variable.

    Returns
    -------
    dict
        Keys: r, p_value, n, significant.
    """
    clean = df[[x, y]].dropna()
    r, p = stats.pearsonr(clean[x], clean[y])
    result = {
        "r": round(r, 4),
        "p_value": round(p, 6),
        "n": len(clean),
        "significant": p < 0.05,
    }
    logger.info("Pearson(%s, %s): r=%.4f, p=%.6f, n=%d", x, y, r, p, len(clean))
    return result


def spearman_correlation(df: pd.DataFrame, x: str, y: str) -> dict:
    """
    Compute Spearman rank correlation between two numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
    x : str
    y : str

    Returns
    -------
    dict
        Keys: rho, p_value, n, significant.
    """
    clean = df[[x, y]].dropna()
    rho, p = stats.spearmanr(clean[x], clean[y])
    result = {
        "rho": round(rho, 4),
        "p_value": round(p, 6),
        "n": len(clean),
        "significant": p < 0.05,
    }
    logger.info(
        "Spearman(%s, %s): rho=%.4f, p=%.6f, n=%d", x, y, rho, p, len(clean)
    )
    return result


def kruskal_wallis_test(
    df: pd.DataFrame, group_col: str, value_col: str
) -> dict:
    """
    Kruskal-Wallis H-test: non-parametric one-way ANOVA across sentiment groups.

    Parameters
    ----------
    df : pd.DataFrame
    group_col : str
        Categorical grouping column (e.g., sentiment_regime).
    value_col : str
        Numeric outcome column (e.g., pnl).

    Returns
    -------
    dict
        Keys: H, p_value, significant, groups.
    """
    groups = [
        grp[value_col].dropna().values
        for _, grp in df.groupby(group_col, observed=True)
        if len(grp[value_col].dropna()) > 0
    ]
    H, p = stats.kruskal(*groups)
    result = {
        "H": round(H, 4),
        "p_value": round(p, 6),
        "significant": p < 0.05,
        "groups": df[group_col].nunique(),
    }
    logger.info(
        "Kruskal-Wallis(%s ~ %s): H=%.4f, p=%.6f", value_col, group_col, H, p
    )
    return result


def mann_whitney_fear_vs_greed(
    df: pd.DataFrame, value_col: str = "pnl"
) -> dict:
    """
    Mann-Whitney U test comparing a metric during Fear vs Greed periods.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain is_fear boolean column.
    value_col : str
        Numeric column to compare.

    Returns
    -------
    dict
        Keys: U, p_value, significant, fear_median, greed_median.
    """
    fear_vals = df[df["is_fear"]][value_col].dropna()
    greed_vals = df[~df["is_fear"]][value_col].dropna()
    U, p = stats.mannwhitneyu(fear_vals, greed_vals, alternative="two-sided")
    result = {
        "U": round(U, 2),
        "p_value": round(p, 6),
        "significant": p < 0.05,
        "fear_median": round(fear_vals.median(), 4),
        "greed_median": round(greed_vals.median(), 4),
    }
    logger.info(
        "Mann-Whitney U(%s: Fear vs Greed): U=%.2f, p=%.6f | "
        "Fear median=%.4f, Greed median=%.4f",
        value_col,
        U,
        p,
        fear_vals.median(),
        greed_vals.median(),
    )
    return result


def _normality_p_value(values: pd.Series) -> float | None:
    """
    Return a normality-test p-value for a numeric series.

    Uses D'Agostino-Pearson for samples with at least 20 observations and
    Shapiro-Wilk for smaller samples. Very large samples are capped for the
    normality test to keep the diagnostic stable and fast.
    """
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 3 or clean.nunique() < 2:
        return None
    sample = clean.sample(n=5000, random_state=42) if len(clean) > 5000 else clean
    if len(sample) >= 20:
        return float(stats.normaltest(sample).pvalue)
    return float(stats.shapiro(sample).pvalue)


def _is_normal(values: pd.Series) -> bool:
    """Return True when the normality diagnostic does not reject normality."""
    p_value = _normality_p_value(values)
    return bool(p_value is not None and p_value >= ALPHA)


def _interpret_significance(
    variable: str,
    p_value: float | None,
    fear_value: float | None,
    greed_value: float | None,
    metric: str,
) -> str:
    """Create a cautious, data-grounded interpretation sentence."""
    if p_value is None or pd.isna(p_value):
        return f"{variable}: not enough valid data to run the requested comparison."
    significance = "statistically significant" if p_value < ALPHA else "not statistically significant"
    if fear_value is None or greed_value is None or pd.isna(fear_value) or pd.isna(greed_value):
        return f"{variable}: the Fear vs Greed difference is {significance} at alpha={ALPHA}."
    if np.isclose(fear_value, greed_value):
        direction = "the observed group summaries are similar"
    else:
        higher = "Fear" if fear_value > greed_value else "Greed"
        direction = f"{higher} has the higher observed {metric}"
    return (
        f"{variable}: the Fear vs Greed difference is {significance} at alpha={ALPHA}; "
        f"{direction} (Fear={fear_value:.4f}, Greed={greed_value:.4f})."
    )


def compare_continuous_fear_greed(
    df: pd.DataFrame, variable: str, label: str | None = None
) -> dict:
    """
    Compare a continuous variable between Fear and Greed days.

    Automatically selects Welch's independent t-test when both groups satisfy
    the normality diagnostic. Otherwise uses the Mann-Whitney U test.
    """
    label = label or variable
    required = {"is_fear", variable}
    if not required.issubset(df.columns):
        return {
            "variable": label,
            "test_used": "Not run",
            "test_statistic": np.nan,
            "p_value": np.nan,
            "significant": "No",
            "interpretation": f"{label}: missing required columns {sorted(required - set(df.columns))}.",
        }

    fear = pd.to_numeric(df.loc[df["is_fear"], variable], errors="coerce").dropna()
    greed = pd.to_numeric(df.loc[~df["is_fear"], variable], errors="coerce").dropna()
    if len(fear) < 2 or len(greed) < 2 or fear.nunique() < 2 or greed.nunique() < 2:
        return {
            "variable": label,
            "test_used": "Not run",
            "test_statistic": np.nan,
            "p_value": np.nan,
            "significant": "No",
            "interpretation": f"{label}: insufficient variation or valid observations for a two-group test.",
        }

    fear_normal = _is_normal(fear)
    greed_normal = _is_normal(greed)
    if fear_normal and greed_normal:
        stat, p_value = stats.ttest_ind(fear, greed, equal_var=False, nan_policy="omit")
        test_name = "Welch independent t-test"
        metric = "mean"
        fear_summary = float(fear.mean())
        greed_summary = float(greed.mean())
        reason = "normality diagnostics did not reject normality in either group"
    else:
        stat, p_value = stats.mannwhitneyu(fear, greed, alternative="two-sided")
        test_name = "Mann-Whitney U test"
        metric = "median"
        fear_summary = float(fear.median())
        greed_summary = float(greed.median())
        reason = "normality diagnostics rejected normality or were not applicable"

    interpretation = _interpret_significance(
        label, float(p_value), fear_summary, greed_summary, metric
    )
    return {
        "variable": label,
        "test_used": test_name,
        "test_statistic": round(float(stat), 6),
        "p_value": round(float(p_value), 8),
        "significant": "Yes" if p_value < ALPHA else "No",
        "interpretation": f"{interpretation} Test choice: {reason}.",
    }


def chi_square_test(
    df: pd.DataFrame, category_col: str, label: str | None = None
) -> dict:
    """
    Run a Chi-Square Test of Independence between Fear/Greed and a category.
    """
    label = label or category_col
    required = {"is_fear", category_col}
    if not required.issubset(df.columns):
        return {
            "variable": label,
            "test_used": "Chi-Square Test of Independence",
            "chi_square": np.nan,
            "degrees_of_freedom": np.nan,
            "p_value": np.nan,
            "significant": "No",
            "interpretation": f"{label}: missing required columns {sorted(required - set(df.columns))}.",
        }

    work = df[["is_fear", category_col]].dropna().copy()
    if work.empty:
        return {
            "variable": label,
            "test_used": "Chi-Square Test of Independence",
            "chi_square": np.nan,
            "degrees_of_freedom": np.nan,
            "p_value": np.nan,
            "significant": "No",
            "interpretation": f"{label}: no valid rows available for chi-square testing.",
        }
    work["market_sentiment"] = np.where(work["is_fear"], "Fear", "Greed")
    table = pd.crosstab(work["market_sentiment"], work[category_col])
    if table.shape[0] < 2 or table.shape[1] < 2:
        p_value = np.nan
        chi2 = np.nan
        dof = np.nan
        interpretation = f"{label}: chi-square test requires at least two categories in both dimensions."
    else:
        chi2, p_value, dof, _ = stats.chi2_contingency(table)
        relation = "is associated with" if p_value < ALPHA else "does not show a statistically significant association with"
        interpretation = (
            f"{label}: {category_col} {relation} Fear vs Greed classification "
            f"at alpha={ALPHA}."
        )

    return {
        "variable": label,
        "test_used": "Chi-Square Test of Independence",
        "chi_square": round(float(chi2), 6) if not pd.isna(chi2) else np.nan,
        "degrees_of_freedom": int(dof) if not pd.isna(dof) else np.nan,
        "p_value": round(float(p_value), 8) if not pd.isna(p_value) else np.nan,
        "significant": "Yes" if not pd.isna(p_value) and p_value < ALPHA else "No",
        "interpretation": interpretation,
    }


def _relevant_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Select numerical analysis columns while excluding IDs and timestamps."""
    blocked = {
        "timestamp",
        "Timestamp",
        "Trade ID",
        "Order ID",
    }
    cols = []
    for col in df.select_dtypes(include="number").columns:
        if col in blocked:
            continue
        if df[col].dropna().nunique() < 2:
            continue
        cols.append(col)
    return cols


def correlation_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a correlation matrix and ranked strongest positive/negative pairs.

    Pearson is used only when both variables pass normality diagnostics;
    otherwise Spearman is used.
    """
    cols = _relevant_numeric_columns(df)
    if len(cols) < 2:
        return pd.DataFrame(), pd.DataFrame()

    corr_matrix = df[cols].corr(method="spearman")
    ranked_rows: list[dict] = []
    for i, left in enumerate(cols):
        for right in cols[i + 1:]:
            clean = df[[left, right]].dropna()
            if len(clean) < 3:
                continue
            use_pearson = _is_normal(clean[left]) and _is_normal(clean[right])
            method = "Pearson" if use_pearson else "Spearman"
            if use_pearson:
                corr, p_value = stats.pearsonr(clean[left], clean[right])
            else:
                corr, p_value = stats.spearmanr(clean[left], clean[right])
            ranked_rows.append(
                {
                    "variable_1": left,
                    "variable_2": right,
                    "method": method,
                    "correlation": round(float(corr), 6),
                    "p_value": round(float(p_value), 8),
                    "abs_correlation": abs(float(corr)),
                    "interpretation": (
                        f"{left} and {right} have a {method.lower()} correlation "
                        f"of {corr:.4f}; this describes association, not causation."
                    ),
                }
            )

    ranked = pd.DataFrame(ranked_rows)
    if not ranked.empty:
        ranked = ranked.sort_values("abs_correlation", ascending=False).reset_index(drop=True)
    return corr_matrix, ranked


def run_statistical_analysis(
    merged: pd.DataFrame,
    daily_trader_stats: pd.DataFrame | None = None,
    trader_features: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Run complete Stage 5 statistical analysis and save output tables.
    """
    continuous_results = [
        compare_continuous_fear_greed(merged, "pnl", "Closed PnL"),
        compare_continuous_fear_greed(merged, "trade_size", "Trade Size"),
        compare_continuous_fear_greed(merged, "leverage", "Leverage"),
    ]
    if (
        daily_trader_stats is not None
        and not daily_trader_stats.empty
        and "is_fear" in daily_trader_stats.columns
    ):
        continuous_results.extend(
            [
                compare_continuous_fear_greed(daily_trader_stats, "daily_pnl", "Average Daily PnL"),
                compare_continuous_fear_greed(daily_trader_stats, "daily_trade_count", "Number of Trades"),
            ]
        )

    summary = pd.DataFrame(continuous_results)

    categorical_source = merged.copy()
    if {"is_long", "is_short"}.issubset(categorical_source.columns):
        categorical_source["trade_direction"] = np.select(
            [categorical_source["is_long"], categorical_source["is_short"]],
            ["Long", "Short"],
            default="Other",
        )
    categorical_results = [
        chi_square_test(categorical_source, "trade_direction", "Long vs Short Trades"),
        chi_square_test(categorical_source, "is_win", "Winning vs Losing Trades"),
        chi_square_test(categorical_source, "sentiment_regime", "Fear vs Greed Classifications"),
    ]
    chi_square = pd.DataFrame(categorical_results)

    for _, row in chi_square.iterrows():
        summary = pd.concat(
            [
                summary,
                pd.DataFrame(
                    [
                        {
                            "variable": row["variable"],
                            "test_used": row["test_used"],
                            "test_statistic": row["chi_square"],
                            "p_value": row["p_value"],
                            "significant": row["significant"],
                            "interpretation": row["interpretation"],
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    corr_matrix, ranked = correlation_analysis(merged)

    STATISTICAL_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(STATISTICAL_SUMMARY, index=False)
    chi_square.to_csv(CHI_SQUARE_RESULTS, index=False)
    if not corr_matrix.empty:
        corr_matrix.to_csv(CORRELATION_MATRIX)
    if not ranked.empty:
        ranked.to_csv(RANKED_CORRELATIONS, index=False)

    logger.info("Saved statistical summary to: %s", STATISTICAL_SUMMARY)
    logger.info("Saved chi-square results to: %s", CHI_SQUARE_RESULTS)
    logger.info("Saved correlation matrix to: %s", CORRELATION_MATRIX)
    logger.info("Saved ranked correlations to: %s", RANKED_CORRELATIONS)
    return {
        "summary": summary,
        "chi_square": chi_square,
        "correlation_matrix": corr_matrix,
        "ranked_correlations": ranked,
    }
