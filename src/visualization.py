"""Chart generation for the Hermès market intelligence pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from config.settings import DISPLAY_LABELS, TARGET_LABEL


def _save(fig, path: Path) -> Path:
    """Save a matplotlib figure and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _apply_common_formatting(
    ax,
    title: str,
    ylabel: str | None = None,
    xlabel: str | None = None,
) -> None:
    """Apply common chart formatting."""
    ax.set_title(title, fontsize=13, fontweight="bold")

    if ylabel:
        ax.set_ylabel(ylabel)

    if xlabel:
        ax.set_xlabel(xlabel)

    ax.grid(True, alpha=0.3)


def _rename_columns_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Rename DataFrame columns using DISPLAY_LABELS from settings.py."""
    return df.rename(
        columns={col: DISPLAY_LABELS.get(col, col) for col in df.columns}
    )


def _clean_factor_coefficients(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Remove diagnostic rows from factor coefficients before plotting."""
    diagnostic_rows = [
        "Alpha",
        "Model R-Squared",
        "Adjusted R-Squared",
        "Residual Mean",
        "Residual Std Dev",
        "Observations",
    ]

    return coefficients.drop(index=diagnostic_rows, errors="ignore")


def _safe_loc(df: pd.DataFrame, labels: Iterable[str]) -> pd.DataFrame:
    """Select rows that exist, instead of failing if one label is missing."""
    existing = [label for label in labels if label in df.index]
    return df.loc[existing]


def _maybe_rotate_xticks(ax, rotation: int = 35) -> None:
    """Rotate x-axis labels for readability."""
    ax.tick_params(axis="x", labelrotation=rotation)


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    """Convert a series to numeric values and drop missing observations."""
    return pd.to_numeric(series, errors="coerce").dropna()


def _plot_horizontal_bar(
    series: pd.Series,
    ax,
    title: str,
    xlabel: str,
    ylabel: str | None = None,
    sort_values: bool = False,
    xlim: tuple[float, float] | None = None,
) -> None:
    """Plot a clean horizontal bar chart for categorical data."""
    plot_series = _coerce_numeric_series(series)

    if sort_values:
        plot_series = plot_series.sort_values()

    plot_series.plot(kind="barh", ax=ax)

    if xlim is not None:
        ax.set_xlim(*xlim)

    _apply_common_formatting(
        ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
    )


def _add_zero_line(ax) -> None:
    """Add a vertical zero line for charts with positive and negative values."""
    ax.axvline(0, linewidth=0.8, alpha=0.6)


def _format_percent_axis(ax) -> None:
    """Format x-axis values as percentages when data is already in percent units."""
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")


def save_price_chart(prices: pd.DataFrame, chart_dir: Path) -> Path:
    """Save Hermès price-history chart."""
    fig, ax = plt.subplots(figsize=(10, 5))

    if TARGET_LABEL in prices.columns:
        series = prices[TARGET_LABEL].dropna()
    elif "Hermès International" in prices.columns:
        series = prices["Hermès International"].dropna()
    else:
        series = prices.iloc[:, 0].dropna()

    series.plot(ax=ax)

    _apply_common_formatting(
        ax,
        title="Hermès International Price History",
        ylabel="Price",
    )

    return _save(fig, chart_dir / "hermes_price_history.png")


def save_peer_performance_chart(returns: pd.DataFrame, chart_dir: Path) -> Path:
    """Save normalized luxury peer-performance chart."""
    normalized = (1 + returns.dropna(how="all")).cumprod()

    fig, ax = plt.subplots(figsize=(10, 5))
    normalized.plot(ax=ax)

    _apply_common_formatting(
        ax,
        title="Luxury Peer Normalized Performance",
        ylabel="Growth of $1",
    )

    ax.legend(fontsize=8)

    return _save(fig, chart_dir / "luxury_peer_normalized_performance.png")


def save_strategy_chart(strategy_df: pd.DataFrame, chart_dir: Path) -> Path:
    """Save risk-control overlay vs buy-and-hold chart."""
    fig, ax = plt.subplots(figsize=(10, 5))

    columns = ["strategy_wealth", "buy_hold_wealth"]
    available_columns = [col for col in columns if col in strategy_df.columns]

    plot_df = _rename_columns_for_display(strategy_df[available_columns])
    plot_df.plot(ax=ax)

    _apply_common_formatting(
        ax,
        title="Hermès Risk-Control Overlay vs Buy & Hold",
        ylabel="Growth of $1",
    )

    ax.legend(fontsize=8)

    return _save(fig, chart_dir / "hermes_strategy_vs_buy_hold.png")


def save_strategy_comparison_chart(strategy_df: pd.DataFrame, chart_dir: Path) -> Path:
    """Save comparison chart for all available strategy overlays."""
    strategy_columns = [
        "buy_hold_wealth",
        "risk_control_wealth",
        "trend_following_wealth",
        "momentum_wealth",
        "defensive_cash_wealth",
        "volatility_target_wealth",
    ]

    available_columns = [col for col in strategy_columns if col in strategy_df.columns]

    if not available_columns:
        return save_strategy_chart(strategy_df, chart_dir)

    label_map = {
        "buy_hold_wealth": "Buy & Hold",
        "risk_control_wealth": "Risk-Control Overlay",
        "trend_following_wealth": "Trend Following Overlay",
        "momentum_wealth": "Momentum Overlay",
        "defensive_cash_wealth": "Defensive Cash Overlay",
        "volatility_target_wealth": "Volatility Target Overlay",
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    strategy_df[available_columns].rename(columns=label_map).plot(ax=ax)

    _apply_common_formatting(
        ax,
        title="Hermès Strategy Overlay Comparison",
        ylabel="Growth of $1",
    )

    ax.legend(fontsize=8)

    return _save(fig, chart_dir / "hermes_strategy_overlay_comparison.png")


def save_drawdown_chart(strategy_df: pd.DataFrame, chart_dir: Path) -> Path:
    """Save drawdown comparison chart."""
    fig, ax = plt.subplots(figsize=(10, 5))

    columns = ["strategy_drawdown", "buy_hold_drawdown"]
    available_columns = [col for col in columns if col in strategy_df.columns]

    plot_df = _rename_columns_for_display(strategy_df[available_columns])
    plot_df.plot(ax=ax)

    _apply_common_formatting(
        ax,
        title="Hermès Drawdown Analysis",
        ylabel="Drawdown",
    )

    ax.legend(fontsize=8)

    return _save(fig, chart_dir / "hermes_drawdown_analysis.png")


def save_strategy_drawdown_comparison_chart(
    strategy_df: pd.DataFrame,
    chart_dir: Path,
) -> Path:
    """Save drawdown comparison across strategy overlays."""
    drawdown_columns = [
        "buy_hold_drawdown",
        "risk_control_drawdown",
        "trend_following_drawdown",
        "momentum_drawdown",
        "defensive_cash_drawdown",
        "volatility_target_drawdown",
    ]

    available_columns = [col for col in drawdown_columns if col in strategy_df.columns]

    if not available_columns:
        return save_drawdown_chart(strategy_df, chart_dir)

    label_map = {
        "buy_hold_drawdown": "Buy & Hold",
        "risk_control_drawdown": "Risk-Control Overlay",
        "trend_following_drawdown": "Trend Following Overlay",
        "momentum_drawdown": "Momentum Overlay",
        "defensive_cash_drawdown": "Defensive Cash Overlay",
        "volatility_target_drawdown": "Volatility Target Overlay",
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    strategy_df[available_columns].rename(columns=label_map).plot(ax=ax)

    _apply_common_formatting(
        ax,
        title="Hermès Strategy Drawdown Comparison",
        ylabel="Drawdown",
    )

    ax.legend(fontsize=8)

    return _save(fig, chart_dir / "hermes_strategy_drawdown_comparison.png")


def save_forecast_chart(forecast: pd.DataFrame, chart_dir: Path) -> Path:
    """Save Monte Carlo forecast chart."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(forecast["Day"], forecast["P50"], label="Median")
    ax.plot(forecast["Day"], forecast["P05"], label="5th Percentile")
    ax.plot(forecast["Day"], forecast["P95"], label="95th Percentile")

    if {"P25", "P75"}.issubset(forecast.columns):
        ax.plot(
            forecast["Day"],
            forecast["P25"],
            label="25th Percentile",
            alpha=0.8,
        )
        ax.plot(
            forecast["Day"],
            forecast["P75"],
            label="75th Percentile",
            alpha=0.8,
        )

    _apply_common_formatting(
        ax,
        title="Hermès Monte Carlo Forecast Range",
        xlabel="Forecast Day",
        ylabel="Simulated Price",
    )

    ax.legend(fontsize=8)

    return _save(fig, chart_dir / "hermes_monte_carlo_forecast.png")


def save_var_chart(var_backtest: pd.DataFrame, chart_dir: Path) -> Path:
    """Save VaR backtest chart."""
    fig, ax = plt.subplots(figsize=(10, 5))

    preferred_columns = ["return", "garch_style_var_95", "historical_var_95"]
    fallback_columns = ["return", "garch_style_var", "historical_var"]

    if all(col in var_backtest.columns for col in preferred_columns):
        plot_columns = preferred_columns
    else:
        plot_columns = [col for col in fallback_columns if col in var_backtest.columns]

    label_map = {
        "return": "Daily Return",
        "garch_style_var_95": "GARCH-Style VaR 95%",
        "historical_var_95": "Historical VaR 95%",
        "garch_style_var": "GARCH-Style VaR",
        "historical_var": "Historical VaR",
    }

    var_backtest[plot_columns].rename(columns=label_map).plot(ax=ax)

    _apply_common_formatting(
        ax,
        title="Hermès VaR Backtest: Returns vs Risk Thresholds",
        ylabel="Daily Return",
    )

    ax.legend(fontsize=8)

    return _save(fig, chart_dir / "hermes_var_backtest.png")


def save_volatility_chart(var_backtest: pd.DataFrame, chart_dir: Path) -> Path:
    """Save annualized EWMA volatility chart."""
    fig, ax = plt.subplots(figsize=(10, 5))

    if "annualized_ewma_volatility" in var_backtest.columns:
        vol = var_backtest["annualized_ewma_volatility"]
    else:
        vol = var_backtest["ewma_volatility"] * (252 ** 0.5)

    vol.plot(ax=ax)

    _apply_common_formatting(
        ax,
        title="Hermès GARCH-Style Annualized Volatility",
        ylabel="Annualized Volatility",
    )

    return _save(fig, chart_dir / "hermes_garch_style_volatility.png")


def save_factor_exposure_chart(coefficients: pd.DataFrame, chart_dir: Path) -> Path:
    """Save factor exposure chart."""
    coefs = _clean_factor_coefficients(coefficients)

    fig, ax = plt.subplots(figsize=(9, 5))
    coefs["Coefficient"].plot(kind="bar", ax=ax)

    _apply_common_formatting(
        ax,
        title="Hermès Factor Exposure Model",
        ylabel="Daily Return Beta",
    )

    _maybe_rotate_xticks(ax)

    return _save(fig, chart_dir / "hermes_factor_exposure.png")


def save_scenario_chart(scenario_table: pd.DataFrame, chart_dir: Path) -> Path:
    """Save scenario return chart."""
    fig, ax = plt.subplots(figsize=(9, 5))

    scenario_table["Estimated Hermès Return"].plot(kind="bar", ax=ax)

    _apply_common_formatting(
        ax,
        title="Hermès Scenario Return Estimates",
        ylabel="Estimated Annual Return",
    )

    _maybe_rotate_xticks(ax)

    return _save(fig, chart_dir / "hermes_scenario_returns.png")


def save_intelligence_score_chart(score_table: pd.DataFrame, chart_dir: Path) -> Path:
    """Save intelligence score component chart."""
    components = score_table.drop(
        index="Hermès Luxury Intelligence Score",
        errors="ignore",
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    components["Score"].plot(kind="bar", ax=ax)

    ax.set_ylim(0, 100)

    _apply_common_formatting(
        ax,
        title="Hermès Luxury Intelligence Score Components",
        ylabel="Score",
    )

    _maybe_rotate_xticks(ax)

    return _save(fig, chart_dir / "hermes_intelligence_score.png")


def save_fundamentals_chart(
    fundamental_metrics: pd.DataFrame,
    chart_dir: Path,
) -> Path:
    """Save official fundamentals chart."""
    selected = _safe_loc(
        fundamental_metrics,
        [
            "Revenue",
            "Recurring operating income",
            "Adjusted free cash flows",
            "Restated net cash position",
        ],
    )

    plot_df = selected[["2024", "2025"]].astype(float)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.plot(kind="bar", ax=ax)

    _apply_common_formatting(
        ax,
        title="Hermès Official Fundamentals: 2024 vs 2025",
        ylabel="EUR millions",
    )

    ax.legend(fontsize=8)
    _maybe_rotate_xticks(ax)

    return _save(fig, chart_dir / "hermes_official_fundamentals.png")


def save_margin_profile_chart(
    fundamental_metrics: pd.DataFrame,
    chart_dir: Path,
) -> Path:
    """Save margin and growth profile chart."""
    selected = _safe_loc(
        fundamental_metrics,
        [
            "Recurring operating margin",
            "Growth at current exchange rates",
            "Growth at constant exchange rates",
        ],
    )

    plot_df = selected[["2024", "2025"]].astype(float)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.plot(kind="bar", ax=ax)

    _apply_common_formatting(
        ax,
        title="Hermès Growth and Margin Profile",
        ylabel="Percent",
    )

    ax.legend(fontsize=8)
    _maybe_rotate_xticks(ax)

    return _save(fig, chart_dir / "hermes_margin_growth_profile.png")


def save_dividend_chart(dividend_metrics: pd.DataFrame, chart_dir: Path) -> Path:
    """Save EPS vs dividend chart."""
    fig, ax = plt.subplots(figsize=(10, 5))

    columns = ["eps_eur", "dividend_per_share_eur"]
    available_columns = [col for col in columns if col in dividend_metrics.columns]

    plot_df = _rename_columns_for_display(dividend_metrics[available_columns])
    plot_df.plot(kind="bar", ax=ax)

    _apply_common_formatting(
        ax,
        title="Hermès EPS vs Dividend per Share",
        ylabel="EUR per share",
    )

    ax.legend(fontsize=8)

    return _save(fig, chart_dir / "hermes_eps_vs_dividend.png")


def save_dividend_yield_chart(
    dividend_metrics: pd.DataFrame,
    chart_dir: Path,
) -> Path:
    """Save dividend yield chart if the metric exists."""
    column = "dividend_yield_on_year_end_price_pct"

    fig, ax = plt.subplots(figsize=(10, 5))

    if column in dividend_metrics.columns:
        dividend_metrics[column].plot(kind="bar", ax=ax)
    else:
        fallback = (
            dividend_metrics["dividend_per_share_eur"]
            / dividend_metrics["year_end_share_price_eur"]
            * 100
        )
        fallback.plot(kind="bar", ax=ax)

    _apply_common_formatting(
        ax,
        title="Hermès Dividend Yield on Year-End Share Price",
        ylabel="Dividend Yield (%)",
    )

    return _save(fig, chart_dir / "hermes_dividend_yield.png")


def save_market_cap_chart(dividend_metrics: pd.DataFrame, chart_dir: Path) -> Path:
    """Save market capitalization chart."""
    fig, ax = plt.subplots(figsize=(10, 5))

    dividend_metrics["market_cap_eur_bn"].plot(kind="bar", ax=ax)

    _apply_common_formatting(
        ax,
        title="Hermès Market Capitalization History",
        ylabel="EUR billions",
    )

    return _save(fig, chart_dir / "hermes_market_cap_history.png")


def save_q1_growth_by_region_chart(
    q1_revenue: pd.DataFrame,
    chart_dir: Path,
) -> Path:
    """Save Q1 2026 constant-FX growth by region chart."""
    geo = q1_revenue[
        (q1_revenue["type"] == "Geography")
        & (q1_revenue["segment"] != "Total")
    ].set_index("segment")

    fig, ax = plt.subplots(figsize=(10, 5))

    _plot_horizontal_bar(
        series=geo["constant_fx_growth_pct"],
        ax=ax,
        title="Hermès Q1 2026 Constant-FX Growth by Region",
        xlabel="Constant-FX Growth (%)",
        ylabel="Region",
        sort_values=True,
    )

    _add_zero_line(ax)
    _format_percent_axis(ax)

    return _save(fig, chart_dir / "hermes_q1_growth_by_region.png")


def save_q1_growth_by_sector_chart(
    q1_revenue: pd.DataFrame,
    chart_dir: Path,
) -> Path:
    """Save Q1 2026 constant-FX growth by sector chart."""
    sector = q1_revenue[
        (q1_revenue["type"] == "Sector")
        & (q1_revenue["segment"] != "Total")
    ].set_index("segment")

    fig, ax = plt.subplots(figsize=(10, 5))

    _plot_horizontal_bar(
        series=sector["constant_fx_growth_pct"],
        ax=ax,
        title="Hermès Q1 2026 Constant-FX Growth by Sector",
        xlabel="Constant-FX Growth (%)",
        ylabel="Sector",
        sort_values=True,
    )

    _add_zero_line(ax)
    _format_percent_axis(ax)

    return _save(fig, chart_dir / "hermes_q1_growth_by_sector.png")


def save_q1_fx_gap_chart(q1_revenue: pd.DataFrame, chart_dir: Path) -> Path:
    """Save chart showing constant-FX growth minus reported growth."""
    df = q1_revenue.copy()

    if "fx_gap_pct_points" not in df.columns:
        df["fx_gap_pct_points"] = (
            df["constant_fx_growth_pct"] - df["published_growth_pct"]
        )

    geo = df[
        (df["type"] == "Geography")
        & (df["segment"] != "Total")
    ].set_index("segment")

    fig, ax = plt.subplots(figsize=(10, 5))

    _plot_horizontal_bar(
        series=geo["fx_gap_pct_points"],
        ax=ax,
        title="Hermès Q1 2026 FX Gap by Region",
        xlabel="FX Gap, pct. points",
        ylabel="Region",
        sort_values=True,
    )

    _add_zero_line(ax)

    return _save(fig, chart_dir / "hermes_q1_fx_gap_by_region.png")


def save_strategy_qualitative_score_chart(
    strategy_scores: pd.DataFrame,
    chart_dir: Path,
) -> Path:
    """Save strategic moat and qualitative score chart."""
    plot_df = strategy_scores.set_index("Factor")

    fig, ax = plt.subplots(figsize=(10, 5))

    _plot_horizontal_bar(
        series=plot_df["Score_0_to_100"],
        ax=ax,
        title="Hermès Strategic Moat and Qualitative Signal Scores",
        xlabel="Score",
        ylabel="Factor",
        sort_values=True,
        xlim=(0, 100),
    )

    return _save(fig, chart_dir / "hermes_strategy_qualitative_score.png")


# =============================================================================
# Valuation charts
# =============================================================================


def save_fair_value_range_chart(
    fair_value_summary: pd.DataFrame,
    chart_dir: Path,
) -> Path:
    """Save fair value range chart from valuation outputs."""
    if fair_value_summary.empty:
        raise ValueError("Fair value summary is empty.")

    if "Fair Value Per Share EUR" not in fair_value_summary.columns:
        raise KeyError("Fair value summary must include 'Fair Value Per Share EUR'.")

    fig, ax = plt.subplots(figsize=(10, 5))

    values = _coerce_numeric_series(fair_value_summary["Fair Value Per Share EUR"])
    values.plot(kind="bar", ax=ax)

    if "Current Price EUR" in fair_value_summary.columns:
        current_prices = _coerce_numeric_series(fair_value_summary["Current Price EUR"])

        if not current_prices.empty:
            current_price = float(current_prices.iloc[0])
            ax.axhline(
                current_price,
                linestyle="--",
                linewidth=1.5,
                label="Current Price",
            )
            ax.legend(fontsize=8)

    _apply_common_formatting(
        ax,
        title="Hermès Fair Value Range",
        ylabel="EUR per share",
    )

    _maybe_rotate_xticks(ax)

    return _save(fig, chart_dir / "hermes_fair_value_range.png")


def save_dcf_sensitivity_chart(
    dcf_sensitivity: pd.DataFrame,
    chart_dir: Path,
) -> Path:
    """Save DCF sensitivity matrix chart."""
    if dcf_sensitivity.empty:
        raise ValueError("DCF sensitivity table is empty.")

    matrix = dcf_sensitivity.copy()
    matrix = matrix.apply(pd.to_numeric, errors="coerce")

    fig, ax = plt.subplots(figsize=(10, 5))

    image = ax.imshow(matrix.values, aspect="auto")
    fig.colorbar(image, ax=ax, label="Fair Value per Share EUR")

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")

    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels([f"{float(idx):.1%}" for idx in matrix.index])

    _apply_common_formatting(
        ax,
        title="Hermès DCF Sensitivity: WACC vs Terminal Growth",
        xlabel="WACC",
        ylabel="Terminal Growth",
    )

    return _save(fig, chart_dir / "hermes_dcf_sensitivity.png")


# =============================================================================
# News and market-intelligence charts
# =============================================================================


def save_news_theme_chart(
    news_theme_summary: pd.DataFrame,
    chart_dir: Path,
) -> Path:
    """Save news theme summary chart."""
    if news_theme_summary.empty:
        raise ValueError("News theme summary is empty.")

    if not {"Theme", "Count"}.issubset(news_theme_summary.columns):
        raise KeyError("News theme summary must include Theme and Count columns.")

    plot_df = news_theme_summary.copy().head(12)
    plot_df = plot_df.set_index("Theme")

    fig, ax = plt.subplots(figsize=(10, 5))

    _plot_horizontal_bar(
        series=plot_df["Count"],
        ax=ax,
        title="Hermès News Intelligence Theme Summary",
        xlabel="Headline Count",
        ylabel="Theme",
        sort_values=True,
    )

    return _save(fig, chart_dir / "hermes_news_theme_summary.png")


def save_news_classification_chart(
    news_items: pd.DataFrame,
    chart_dir: Path,
) -> Path:
    """Save catalyst/risk/neutral news classification chart."""
    if news_items.empty:
        raise ValueError("News items table is empty.")

    if "Classification" not in news_items.columns:
        raise KeyError("News items table must include a Classification column.")

    counts = news_items["Classification"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))

    _plot_horizontal_bar(
        series=counts,
        ax=ax,
        title="Hermès News Classification Summary",
        xlabel="Headline Count",
        ylabel="Classification",
        sort_values=True,
    )

    return _save(fig, chart_dir / "hermes_news_classification_summary.png")


def save_market_intelligence_dashboard_chart(
    market_dashboard: pd.DataFrame,
    chart_dir: Path,
) -> Path:
    """Save market intelligence dashboard chart for numeric indicators."""
    if market_dashboard.empty:
        raise ValueError("Market intelligence dashboard is empty.")

    if not {"Indicator", "Value"}.issubset(market_dashboard.columns):
        raise KeyError("Market dashboard must include Indicator and Value columns.")

    plot_df = market_dashboard.copy()
    plot_df["Numeric Value"] = pd.to_numeric(plot_df["Value"], errors="coerce")
    plot_df = plot_df.dropna(subset=["Numeric Value"])

    if plot_df.empty:
        raise ValueError("Market dashboard has no numeric values to plot.")

    plot_df = plot_df.set_index("Indicator")

    fig, ax = plt.subplots(figsize=(10, 5))

    _plot_horizontal_bar(
        series=plot_df["Numeric Value"],
        ax=ax,
        title="Hermès Market Intelligence Dashboard: Numeric Signals",
        xlabel="Value",
        ylabel="Indicator",
        sort_values=True,
    )

    _add_zero_line(ax)

    return _save(fig, chart_dir / "hermes_market_intelligence_dashboard.png")


# =============================================================================
# Geopolitical risk charts
# =============================================================================


def save_geopolitical_risk_chart(
    geopolitical_risks: pd.DataFrame,
    chart_dir: Path,
) -> Path:
    """Save geopolitical risk factor score chart."""
    if geopolitical_risks.empty:
        raise ValueError("Geopolitical risk table is empty.")

    if not {"Risk Factor", "Risk Score 0-100"}.issubset(geopolitical_risks.columns):
        raise KeyError(
            "Geopolitical risk table must include Risk Factor and Risk Score 0-100."
        )

    plot_df = geopolitical_risks.copy().set_index("Risk Factor")

    fig, ax = plt.subplots(figsize=(10, 5))

    _plot_horizontal_bar(
        series=plot_df["Risk Score 0-100"],
        ax=ax,
        title="Hermès Geopolitical Risk Factors",
        xlabel="Risk Score",
        ylabel="Risk Factor",
        sort_values=True,
        xlim=(0, 100),
    )

    return _save(fig, chart_dir / "hermes_geopolitical_risk_factors.png")


def save_geopolitical_category_chart(
    geopolitical_category_summary: pd.DataFrame,
    chart_dir: Path,
) -> Path:
    """Save geopolitical risk category summary chart."""
    if geopolitical_category_summary.empty:
        raise ValueError("Geopolitical category summary is empty.")

    if not {"Category", "Average Risk Score"}.issubset(
        geopolitical_category_summary.columns
    ):
        raise KeyError(
            "Geopolitical category summary must include Category and Average Risk Score."
        )

    plot_df = geopolitical_category_summary.copy().set_index("Category")

    fig, ax = plt.subplots(figsize=(10, 5))

    _plot_horizontal_bar(
        series=plot_df["Average Risk Score"],
        ax=ax,
        title="Hermès Geopolitical Risk by Category",
        xlabel="Average Risk Score",
        ylabel="Category",
        sort_values=True,
        xlim=(0, 100),
    )

    return _save(fig, chart_dir / "hermes_geopolitical_category_summary.png")


def save_geopolitical_scenario_chart(
    geopolitical_scenarios: pd.DataFrame,
    chart_dir: Path,
) -> Path:
    """Save geopolitical scenario stress-score chart."""
    if geopolitical_scenarios.empty:
        raise ValueError("Geopolitical scenario table is empty.")

    scenario_df = geopolitical_scenarios.copy()

    if "Scenario" in scenario_df.columns:
        scenario_df = scenario_df.set_index("Scenario")

    if "Stress Score" not in scenario_df.columns:
        raise KeyError("Geopolitical scenario table must include Stress Score.")

    fig, ax = plt.subplots(figsize=(10, 5))

    _plot_horizontal_bar(
        series=scenario_df["Stress Score"],
        ax=ax,
        title="Hermès Geopolitical Scenario Stress Overlay",
        xlabel="Stress Score",
        ylabel="Scenario",
        sort_values=True,
        xlim=(0, 100),
    )

    return _save(fig, chart_dir / "hermes_geopolitical_scenario_overlay.png")


# =============================================================================
# Recommendation charts
# =============================================================================


def save_recommendation_framework_chart(
    recommendation_framework: pd.DataFrame,
    chart_dir: Path,
) -> Path:
    """Save recommendation framework point contribution chart."""
    if recommendation_framework.empty:
        raise ValueError("Recommendation framework is empty.")

    if not {"Component", "Points"}.issubset(recommendation_framework.columns):
        raise KeyError("Recommendation framework must include Component and Points.")

    exclude_components = {
        "Total Recommendation Points",
        "Final Recommendation",
        "Risk Rating",
        "Conviction",
    }

    plot_df = recommendation_framework[
        ~recommendation_framework["Component"].isin(exclude_components)
    ].copy()

    plot_df["Points"] = pd.to_numeric(plot_df["Points"], errors="coerce")
    plot_df = plot_df.dropna(subset=["Points"])

    if plot_df.empty:
        raise ValueError("Recommendation framework has no numeric points to plot.")

    plot_df = plot_df.set_index("Component")

    fig, ax = plt.subplots(figsize=(10, 5))

    _plot_horizontal_bar(
        series=plot_df["Points"],
        ax=ax,
        title="Hermès Recommendation Framework: Point Contributions",
        xlabel="Points",
        ylabel="Component",
        sort_values=True,
    )

    _add_zero_line(ax)

    return _save(fig, chart_dir / "hermes_recommendation_framework.png")


def save_recommendation_summary_chart(
    recommendation_summary: pd.DataFrame,
    chart_dir: Path,
) -> Path:
    """Save recommendation summary chart for numeric fields.

    This function is retained for optional standalone use, but the main PDF report
    should generally prefer the recommendation framework chart because it shows
    the point-contribution bridge.
    """
    if recommendation_summary.empty:
        raise ValueError("Recommendation summary is empty.")

    if not {"Metric", "Value"}.issubset(recommendation_summary.columns):
        raise KeyError("Recommendation summary must include Metric and Value columns.")

    plot_df = recommendation_summary.copy()
    plot_df["Numeric Value"] = pd.to_numeric(plot_df["Value"], errors="coerce")
    plot_df = plot_df.dropna(subset=["Numeric Value"])

    if plot_df.empty:
        raise ValueError("Recommendation summary has no numeric values to plot.")

    plot_df = plot_df.set_index("Metric")

    fig, ax = plt.subplots(figsize=(8, 5))

    _plot_horizontal_bar(
        series=plot_df["Numeric Value"],
        ax=ax,
        title="Hermès Recommendation Summary",
        xlabel="Value",
        ylabel="Metric",
        sort_values=True,
    )

    return _save(fig, chart_dir / "hermes_recommendation_summary.png")

# =============================================================================
# Dashboard-ready module upgrade charts
# =============================================================================


def _first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """Return the first column from candidates that exists in the DataFrame."""
    for column in candidates:
        if column in df.columns:
            return column

    return None


def save_risk_leaderboard_chart(
    risk_metric_leaderboard: pd.DataFrame,
    chart_dir: Path,
) -> Path:
    """Save peer risk leaderboard chart."""
    if risk_metric_leaderboard.empty:
        raise ValueError("Risk metric leaderboard is empty.")

    if "Asset" in risk_metric_leaderboard.columns:
        plot_df = risk_metric_leaderboard.set_index("Asset")
    else:
        plot_df = risk_metric_leaderboard.copy()

    value_column = _first_existing_column(
        plot_df,
        [
            "Sharpe Ratio",
            "Annualized Return",
            "Overall Risk Rank",
        ],
    )

    if value_column is None:
        raise KeyError(
            "Risk leaderboard must include Sharpe Ratio, Annualized Return, "
            "or Overall Risk Rank."
        )

    fig, ax = plt.subplots(figsize=(10, 5))

    _plot_horizontal_bar(
        series=plot_df[value_column],
        ax=ax,
        title=f"Hermès Peer Risk Leaderboard: {value_column}",
        xlabel=value_column,
        ylabel="Asset",
        sort_values=True,
    )

    if value_column in {"Sharpe Ratio", "Annualized Return"}:
        _add_zero_line(ax)

    return _save(fig, chart_dir / "hermes_risk_metric_leaderboard.png")


def save_target_risk_profile_chart(
    target_risk_profile: pd.DataFrame,
    chart_dir: Path,
) -> Path:
    """Save Hermès target risk profile chart."""
    if target_risk_profile.empty:
        raise ValueError("Target risk profile is empty.")

    if not {"Profile Area", "Value"}.issubset(target_risk_profile.columns):
        raise KeyError("Target risk profile must include Profile Area and Value.")

    plot_df = target_risk_profile.copy()
    plot_df["Numeric Value"] = pd.to_numeric(plot_df["Value"], errors="coerce")
    plot_df = plot_df.dropna(subset=["Numeric Value"])

    if plot_df.empty:
        raise ValueError("Target risk profile has no numeric values to plot.")

    plot_df = plot_df.set_index("Profile Area")

    fig, ax = plt.subplots(figsize=(10, 5))

    _plot_horizontal_bar(
        series=plot_df["Numeric Value"],
        ax=ax,
        title="Hermès Target Risk Profile",
        xlabel="Metric Value",
        ylabel="Profile Area",
        sort_values=True,
    )

    _add_zero_line(ax)

    return _save(fig, chart_dir / "hermes_target_risk_profile.png")


def save_revenue_leaderboard_chart(
    revenue_segment_leaderboard: pd.DataFrame,
    chart_dir: Path,
) -> Path:
    """Save Q1 revenue segment leaderboard chart."""
    if revenue_segment_leaderboard.empty:
        raise ValueError("Revenue segment leaderboard is empty.")

    required_columns = {
        "Category",
        "Segment",
        "Constant-FX Growth pct",
    }

    missing_columns = required_columns.difference(revenue_segment_leaderboard.columns)

    if missing_columns:
        raise KeyError(
            "Revenue leaderboard is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    plot_df = revenue_segment_leaderboard.copy()
    plot_df["Label"] = plot_df["Category"] + ": " + plot_df["Segment"]
    plot_df = plot_df.set_index("Label")

    fig, ax = plt.subplots(figsize=(10, 6))

    _plot_horizontal_bar(
        series=plot_df["Constant-FX Growth pct"],
        ax=ax,
        title="Hermès Q1 2026 Revenue Leaderboard",
        xlabel="Constant-FX Growth (%)",
        ylabel="Segment",
        sort_values=True,
    )

    _add_zero_line(ax)
    _format_percent_axis(ax)

    return _save(fig, chart_dir / "hermes_revenue_segment_leaderboard.png")


def save_revenue_fx_dashboard_chart(
    revenue_fx_dashboard: pd.DataFrame,
    chart_dir: Path,
) -> Path:
    """Save revenue FX dashboard chart."""
    if revenue_fx_dashboard.empty:
        raise ValueError("Revenue FX dashboard is empty.")

    required_columns = {
        "Category",
        "Segment",
        "FX Gap pct points",
    }

    missing_columns = required_columns.difference(revenue_fx_dashboard.columns)

    if missing_columns:
        raise KeyError(
            "Revenue FX dashboard is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    plot_df = revenue_fx_dashboard.copy()
    plot_df["Label"] = plot_df["Category"] + ": " + plot_df["Segment"]
    plot_df = plot_df.set_index("Label")

    fig, ax = plt.subplots(figsize=(10, 5))

    _plot_horizontal_bar(
        series=plot_df["FX Gap pct points"],
        ax=ax,
        title="Hermès Q1 2026 FX Impact Dashboard",
        xlabel="FX Gap, pct. points",
        ylabel="Segment",
        sort_values=True,
    )

    _add_zero_line(ax)

    return _save(fig, chart_dir / "hermes_revenue_fx_dashboard.png")


def save_shareholder_policy_chart(
    dividend_policy_dashboard: pd.DataFrame,
    chart_dir: Path,
) -> Path:
    """Save dividend payout policy chart."""
    if dividend_policy_dashboard.empty:
        raise ValueError("Dividend policy dashboard is empty.")

    required_columns = {
        "Year",
        "Dividend Payout Ratio",
    }

    missing_columns = required_columns.difference(dividend_policy_dashboard.columns)

    if missing_columns:
        raise KeyError(
            "Dividend policy dashboard is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    plot_df = dividend_policy_dashboard.copy().set_index("Year")

    fig, ax = plt.subplots(figsize=(10, 5))

    plot_df["Dividend Payout Ratio"].plot(kind="bar", ax=ax)

    _apply_common_formatting(
        ax,
        title="Hermès Dividend Payout Ratio",
        ylabel="Payout Ratio",
        xlabel="Year",
    )

    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")

    return _save(fig, chart_dir / "hermes_dividend_policy_dashboard.png")


def save_shareholder_return_leaderboard_chart(
    shareholder_return_leaderboard: pd.DataFrame,
    chart_dir: Path,
) -> Path:
    """Save shareholder return growth leaderboard chart."""
    if shareholder_return_leaderboard.empty:
        raise ValueError("Shareholder return leaderboard is empty.")

    required_columns = {
        "Year",
        "EPS Growth pct",
        "Dividend Growth pct",
    }

    missing_columns = required_columns.difference(shareholder_return_leaderboard.columns)

    if missing_columns:
        raise KeyError(
            "Shareholder return leaderboard is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    plot_df = shareholder_return_leaderboard.copy().set_index("Year")

    fig, ax = plt.subplots(figsize=(10, 5))

    plot_df[["EPS Growth pct", "Dividend Growth pct"]].plot(kind="bar", ax=ax)

    _apply_common_formatting(
        ax,
        title="Hermès EPS Growth vs Dividend Growth",
        ylabel="Growth (%)",
        xlabel="Year",
    )

    _add_zero_line(ax)
    ax.legend(fontsize=8)

    return _save(fig, chart_dir / "hermes_shareholder_return_leaderboard.png")


def save_strategy_leaderboard_chart(
    strategy_comparison_leaderboard: pd.DataFrame,
    chart_dir: Path,
) -> Path:
    """Save strategy comparison leaderboard chart."""
    if strategy_comparison_leaderboard.empty:
        raise ValueError("Strategy comparison leaderboard is empty.")

    required_columns = {
        "Strategy",
        "Overall Strategy Score",
    }

    missing_columns = required_columns.difference(strategy_comparison_leaderboard.columns)

    if missing_columns:
        raise KeyError(
            "Strategy leaderboard is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    plot_df = strategy_comparison_leaderboard.copy().set_index("Strategy")

    fig, ax = plt.subplots(figsize=(10, 5))

    _plot_horizontal_bar(
        series=plot_df["Overall Strategy Score"],
        ax=ax,
        title="Hermès Strategy Comparison Leaderboard",
        xlabel="Overall Strategy Score",
        ylabel="Strategy",
        sort_values=True,
    )

    return _save(fig, chart_dir / "hermes_strategy_comparison_leaderboard.png")


def save_strategy_signal_summary_chart(
    strategy_signal_summary: pd.DataFrame,
    chart_dir: Path,
) -> Path:
    """Save strategy signal and exposure summary chart."""
    if strategy_signal_summary.empty:
        raise ValueError("Strategy signal summary is empty.")

    required_columns = {
        "Strategy",
        "Average Exposure",
    }

    missing_columns = required_columns.difference(strategy_signal_summary.columns)

    if missing_columns:
        raise KeyError(
            "Strategy signal summary is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    plot_df = strategy_signal_summary.copy().set_index("Strategy")

    fig, ax = plt.subplots(figsize=(10, 5))

    _plot_horizontal_bar(
        series=plot_df["Average Exposure"],
        ax=ax,
        title="Hermès Strategy Average Exposure",
        xlabel="Average Exposure",
        ylabel="Strategy",
        sort_values=True,
        xlim=(0, 1),
    )

    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")

    return _save(fig, chart_dir / "hermes_strategy_signal_summary.png")


def save_valuation_bridge_chart(
    valuation_bridge: pd.DataFrame,
    chart_dir: Path,
) -> Path:
    """Save valuation bridge chart for fair value and market comparison items."""
    if valuation_bridge.empty:
        raise ValueError("Valuation bridge is empty.")

    if not {"Metric", "Numeric Value"}.issubset(valuation_bridge.columns):
        raise KeyError("Valuation bridge must include Metric and Numeric Value.")

    include_metrics = [
        "DCF Fair Value",
        "Peer Bear Fair Value",
        "Peer Base Fair Value",
        "Peer Bull Fair Value",
        "Base Fair Value",
        "Current Price",
    ]

    plot_df = valuation_bridge[
        valuation_bridge["Metric"].isin(include_metrics)
    ].copy()

    plot_df["Numeric Value"] = pd.to_numeric(
        plot_df["Numeric Value"],
        errors="coerce",
    )

    plot_df = plot_df.dropna(subset=["Numeric Value"])

    if plot_df.empty:
        raise ValueError("Valuation bridge has no fair-value metrics to plot.")

    plot_df = plot_df.set_index("Metric")

    fig, ax = plt.subplots(figsize=(10, 5))

    _plot_horizontal_bar(
        series=plot_df["Numeric Value"],
        ax=ax,
        title="Hermès Valuation Bridge",
        xlabel="EUR per share",
        ylabel="Metric",
        sort_values=True,
    )

    return _save(fig, chart_dir / "hermes_valuation_bridge.png")


def save_valuation_sensitivity_dashboard_chart(
    valuation_sensitivity_dashboard: pd.DataFrame,
    chart_dir: Path,
) -> Path:
    """Save valuation sensitivity dashboard heatmap."""
    if valuation_sensitivity_dashboard.empty:
        raise ValueError("Valuation sensitivity dashboard is empty.")

    required_columns = {
        "Terminal Growth",
        "WACC",
        "Fair Value Per Share EUR",
    }

    missing_columns = required_columns.difference(
        valuation_sensitivity_dashboard.columns
    )

    if missing_columns:
        raise KeyError(
            "Valuation sensitivity dashboard is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    matrix = valuation_sensitivity_dashboard.pivot_table(
        index="Terminal Growth",
        columns="WACC",
        values="Fair Value Per Share EUR",
        aggfunc="mean",
    )

    matrix = matrix.sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))

    image = ax.imshow(matrix.values, aspect="auto")
    fig.colorbar(image, ax=ax, label="Fair Value per Share EUR")

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([f"{float(col):.1%}" for col in matrix.columns], rotation=35)

    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels([f"{float(idx):.1%}" for idx in matrix.index])

    _apply_common_formatting(
        ax,
        title="Hermès Valuation Sensitivity Dashboard",
        xlabel="WACC",
        ylabel="Terminal Growth",
    )

    return _save(fig, chart_dir / "hermes_valuation_sensitivity_dashboard.png")


def save_news_source_summary_chart(
    news_source_summary: pd.DataFrame,
    chart_dir: Path,
) -> Path:
    """Save news source summary chart."""
    if news_source_summary.empty:
        raise ValueError("News source summary is empty.")

    category_column = _first_existing_column(
        news_source_summary,
        ["Source", "Publisher", "Domain", "Source Name"],
    )

    value_column = _first_existing_column(
        news_source_summary,
        ["Count", "Headline Count", "Articles", "Article Count"],
    )

    if category_column is None or value_column is None:
        raise KeyError(
            "News source summary must include a source/category column and "
            "a count column."
        )

    plot_df = news_source_summary.copy().head(12).set_index(category_column)

    fig, ax = plt.subplots(figsize=(10, 5))

    _plot_horizontal_bar(
        series=plot_df[value_column],
        ax=ax,
        title="Hermès News Source Summary",
        xlabel="Headline Count",
        ylabel="Source",
        sort_values=True,
    )

    return _save(fig, chart_dir / "hermes_news_source_summary.png")


def save_news_watchlist_classification_chart(
    news_watchlist: pd.DataFrame,
    chart_dir: Path,
) -> Path:
    """Save news watchlist classification or priority chart."""
    if news_watchlist.empty:
        raise ValueError("News watchlist is empty.")

    category_column = _first_existing_column(
        news_watchlist,
        ["Classification", "Priority", "Watchlist Priority", "Category"],
    )

    if category_column is None:
        raise KeyError(
            "News watchlist must include Classification, Priority, "
            "Watchlist Priority, or Category."
        )

    counts = news_watchlist[category_column].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))

    _plot_horizontal_bar(
        series=counts,
        ax=ax,
        title=f"Hermès News Watchlist by {category_column}",
        xlabel="Headline Count",
        ylabel=category_column,
        sort_values=True,
    )

    return _save(fig, chart_dir / "hermes_news_watchlist_classification.png")