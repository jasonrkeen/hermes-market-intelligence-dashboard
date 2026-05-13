"""Hermès official share and dividend history analytics.

This module loads Hermès official share and dividend history, calculates
shareholder-return, valuation-support, payout, yield, and share-price metrics,
and saves clean tables for the Hermès market intelligence and equity research
pipeline.

The module also creates dashboard-ready shareholder return cards, year-by-year
leaderboards, and dividend-policy interpretation tables for Streamlit and other
interactive reporting outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "year",
    "shares_at_dec_31",
    "market_cap_eur_bn",
    "eps_eur",
    "dividend_per_share_eur",
    "avg_daily_volume",
    "high_12m_eur",
    "low_12m_eur",
    "avg_share_price_eur",
    "year_end_share_price_eur",
}


def _default_share_dividend_history() -> pd.DataFrame:
    """Create the default Hermès share and dividend history table.

    Values are based on official Hermès share and dividend data.
    """
    rows = [
        {
            "year": 2021,
            "shares_at_dec_31": 105_569_412,
            "market_cap_eur_bn": 162.15,
            "eps_eur": 23.37,
            "dividend_per_share_eur": 8.00,
            "avg_daily_volume": 55_257,
            "high_12m_eur": 1678.00,
            "low_12m_eur": 839.40,
            "avg_share_price_eur": 1190.97,
            "year_end_share_price_eur": 1536.00,
        },
        {
            "year": 2022,
            "shares_at_dec_31": 105_569_412,
            "market_cap_eur_bn": 152.55,
            "eps_eur": 32.20,
            "dividend_per_share_eur": 13.00,
            "avg_daily_volume": 77_527,
            "high_12m_eur": 1585.50,
            "low_12m_eur": 957.60,
            "avg_share_price_eur": 1264.22,
            "year_end_share_price_eur": 1445.00,
        },
        {
            "year": 2023,
            "shares_at_dec_31": 105_569_412,
            "market_cap_eur_bn": 202.57,
            "eps_eur": 41.19,
            "dividend_per_share_eur": 25.00,
            "avg_daily_volume": 57_338,
            "high_12m_eur": 2063.50,
            "low_12m_eur": 1450.00,
            "avg_share_price_eur": 1846.59,
            "year_end_share_price_eur": 1918.80,
        },
        {
            "year": 2024,
            "shares_at_dec_31": 105_569_412,
            "market_cap_eur_bn": 245.13,
            "eps_eur": 43.93,
            "dividend_per_share_eur": 26.00,
            "avg_daily_volume": 55_445,
            "high_12m_eur": 2436.00,
            "low_12m_eur": 1788.80,
            "avg_share_price_eur": 2139.65,
            "year_end_share_price_eur": 2322.00,
        },
        {
            "year": 2025,
            "shares_at_dec_31": 105_569_412,
            "market_cap_eur_bn": 224.02,
            "eps_eur": 43.15,
            "dividend_per_share_eur": 18.00,
            "avg_daily_volume": 59_406,
            "high_12m_eur": 2957.00,
            "low_12m_eur": 1998.00,
            "avg_share_price_eur": 2314.26,
            "year_end_share_price_eur": 2122.00,
        },
    ]

    return pd.DataFrame(rows)


def _validate_share_dividend_history(history: pd.DataFrame) -> None:
    """Validate the Hermès share and dividend history table."""
    missing_columns = REQUIRED_COLUMNS.difference(history.columns)

    if missing_columns:
        raise ValueError(
            "Hermès share/dividend history is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if history.empty:
        raise ValueError("Hermès share/dividend history table is empty.")


def _safe_float(value: Any, default: float = np.nan) -> float:
    """Safely convert a value to float."""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _classify_dividend_policy(payout_ratio: float) -> str:
    """Classify dividend payout policy."""
    payout = _safe_float(payout_ratio)

    if pd.isna(payout):
        return "Not Available"

    if payout <= 0.45:
        return "Conservative"

    if payout <= 0.65:
        return "Balanced"

    return "Elevated"


def _classify_dividend_yield(dividend_yield_pct: float) -> str:
    """Classify dividend yield level."""
    yield_value = _safe_float(dividend_yield_pct)

    if pd.isna(yield_value):
        return "Not Available"

    if yield_value >= 2.5:
        return "High Yield"

    if yield_value >= 1.0:
        return "Moderate Yield"

    return "Low Yield"


def _classify_valuation_multiple(implied_pe: float) -> str:
    """Classify valuation multiple level based on implied P/E."""
    pe = _safe_float(implied_pe)

    if pd.isna(pe):
        return "Not Available"

    if pe >= 60:
        return "Premium Multiple"

    if pe >= 40:
        return "Elevated Multiple"

    if pe >= 25:
        return "Moderate Premium"

    return "Lower Multiple"


def _classify_share_price_range(range_pct: float) -> str:
    """Classify 12-month share price range width."""
    range_value = _safe_float(range_pct)

    if pd.isna(range_value):
        return "Not Available"

    if range_value >= 50:
        return "Wide Range"

    if range_value >= 30:
        return "Moderate Range"

    return "Narrow Range"


def _growth_signal(growth_pct: float) -> str:
    """Classify year-over-year growth percentage."""
    growth = _safe_float(growth_pct)

    if pd.isna(growth):
        return "Not Available"

    if growth >= 15:
        return "Strong Growth"

    if growth >= 5:
        return "Positive Growth"

    if growth >= 0:
        return "Stable / Low Growth"

    return "Decline"


def _policy_interpretation(
    payout_policy: str,
    yield_classification: str,
    valuation_classification: str,
) -> str:
    """Create a plain-English dividend policy interpretation."""
    if payout_policy == "Conservative":
        return (
            "Dividend payout appears conservative, suggesting earnings retention "
            "and balance-sheet flexibility remain important."
        )

    if payout_policy == "Balanced":
        return (
            "Dividend payout appears balanced, supporting shareholder returns "
            "while preserving reinvestment capacity."
        )

    if payout_policy == "Elevated":
        return (
            "Dividend payout appears elevated and should be monitored against EPS "
            "growth and free-cash-flow durability."
        )

    return (
        f"Dividend yield is classified as {yield_classification}; valuation is "
        f"classified as {valuation_classification}."
    )


def _extract_summary_value(
    summary: pd.DataFrame,
    metric: str,
    default: Any = "Not Available",
) -> Any:
    """Extract a metric value from the shareholder summary table."""
    if summary.empty or "Metric" not in summary.columns:
        return default

    rows = summary[summary["Metric"].eq(metric)]

    if rows.empty:
        return default

    return rows.iloc[0].get("Value", default)


def load_share_dividend_history(raw_data_dir: Path) -> pd.DataFrame:
    """Load Hermès share and dividend history.

    If the CSV does not exist, this function creates the default project dataset
    and saves it to the raw data directory.
    """
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    history_path = raw_data_dir / "hermes_share_dividend_history.csv"

    if history_path.exists():
        history = pd.read_csv(history_path)
    else:
        history = _default_share_dividend_history()
        history.to_csv(history_path, index=False)

    _validate_share_dividend_history(history)

    numeric_columns = [
        "year",
        "shares_at_dec_31",
        "market_cap_eur_bn",
        "eps_eur",
        "dividend_per_share_eur",
        "avg_daily_volume",
        "high_12m_eur",
        "low_12m_eur",
        "avg_share_price_eur",
        "year_end_share_price_eur",
    ]

    for column in numeric_columns:
        history[column] = pd.to_numeric(history[column], errors="coerce")

    return history.sort_values("year")


def build_dividend_metrics(history: pd.DataFrame) -> pd.DataFrame:
    """Create dividend, payout, EPS, market-cap, and share-price metrics."""
    _validate_share_dividend_history(history)

    df = history.copy().sort_values("year")

    df["dividend_payout_ratio"] = np.where(
        df["eps_eur"].replace(0, np.nan).notna(),
        df["dividend_per_share_eur"] / df["eps_eur"],
        np.nan,
    )

    df["eps_growth_pct"] = df["eps_eur"].pct_change() * 100
    df["dividend_growth_pct"] = df["dividend_per_share_eur"].pct_change() * 100
    df["market_cap_growth_pct"] = df["market_cap_eur_bn"].pct_change() * 100
    df["year_end_share_price_growth_pct"] = (
        df["year_end_share_price_eur"].pct_change() * 100
    )

    df["share_price_range_eur"] = df["high_12m_eur"] - df["low_12m_eur"]

    df["share_price_range_pct"] = np.where(
        df["avg_share_price_eur"].replace(0, np.nan).notna(),
        df["share_price_range_eur"] / df["avg_share_price_eur"] * 100,
        np.nan,
    )

    df["dividend_yield_on_avg_price_pct"] = np.where(
        df["avg_share_price_eur"].replace(0, np.nan).notna(),
        df["dividend_per_share_eur"] / df["avg_share_price_eur"] * 100,
        np.nan,
    )

    df["dividend_yield_on_year_end_price_pct"] = np.where(
        df["year_end_share_price_eur"].replace(0, np.nan).notna(),
        df["dividend_per_share_eur"] / df["year_end_share_price_eur"] * 100,
        np.nan,
    )

    df["earnings_yield_on_year_end_price_pct"] = np.where(
        df["year_end_share_price_eur"].replace(0, np.nan).notna(),
        df["eps_eur"] / df["year_end_share_price_eur"] * 100,
        np.nan,
    )

    df["implied_pe_year_end"] = np.where(
        df["eps_eur"].replace(0, np.nan).notna(),
        df["year_end_share_price_eur"] / df["eps_eur"],
        np.nan,
    )

    df["market_cap_per_share_eur"] = np.where(
        df["shares_at_dec_31"].replace(0, np.nan).notna(),
        (df["market_cap_eur_bn"] * 1_000_000_000) / df["shares_at_dec_31"],
        np.nan,
    )

    df["Dividend Policy Signal"] = df["dividend_payout_ratio"].apply(
        _classify_dividend_policy
    )

    df["Dividend Yield Signal"] = df["dividend_yield_on_year_end_price_pct"].apply(
        _classify_dividend_yield
    )

    df["Valuation Multiple Signal"] = df["implied_pe_year_end"].apply(
        _classify_valuation_multiple
    )

    df["Share Price Range Signal"] = df["share_price_range_pct"].apply(
        _classify_share_price_range
    )

    return df.set_index("year")


def build_shareholder_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Build a compact shareholder-return summary table."""
    if metrics.empty:
        raise ValueError("Dividend metrics table is empty.")

    latest_year = metrics.index.max()
    latest = metrics.loc[latest_year]

    rows = [
        {
            "Metric": "Latest Year",
            "Value": latest_year,
            "Unit": "Year",
        },
        {
            "Metric": "Market Capitalization",
            "Value": latest.get("market_cap_eur_bn"),
            "Unit": "EUR billions",
        },
        {
            "Metric": "EPS",
            "Value": latest.get("eps_eur"),
            "Unit": "EUR per share",
        },
        {
            "Metric": "Dividend per Share",
            "Value": latest.get("dividend_per_share_eur"),
            "Unit": "EUR per share",
        },
        {
            "Metric": "Dividend Payout Ratio",
            "Value": latest.get("dividend_payout_ratio"),
            "Unit": "Ratio",
        },
        {
            "Metric": "Dividend Yield on Year-End Price",
            "Value": latest.get("dividend_yield_on_year_end_price_pct"),
            "Unit": "%",
        },
        {
            "Metric": "Implied Year-End P/E",
            "Value": latest.get("implied_pe_year_end"),
            "Unit": "x",
        },
        {
            "Metric": "12-Month Share Price Range",
            "Value": latest.get("share_price_range_pct"),
            "Unit": "%",
        },
        {
            "Metric": "Dividend Policy Signal",
            "Value": latest.get("Dividend Policy Signal"),
            "Unit": "Category",
        },
    ]

    return pd.DataFrame(rows)


def build_shareholder_dashboard_cards(
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Build dashboard cards for the latest shareholder-return data.

    Parameters
    ----------
    summary:
        Output from build_shareholder_summary().
    metrics:
        Output from build_dividend_metrics().

    Returns
    -------
    pd.DataFrame
        Dashboard-ready shareholder return cards.
    """
    if summary.empty or metrics.empty:
        return pd.DataFrame(
            [
                {
                    "Card": "Shareholder Returns",
                    "Value": "Not Available",
                    "Display Value": "Not Available",
                    "Numeric Value": np.nan,
                    "Interpretation": "Shareholder return data was unavailable.",
                }
            ]
        )

    latest_year = metrics.index.max()
    latest = metrics.loc[latest_year]

    market_cap = _safe_float(latest.get("market_cap_eur_bn"))
    eps = _safe_float(latest.get("eps_eur"))
    dividend = _safe_float(latest.get("dividend_per_share_eur"))
    payout = _safe_float(latest.get("dividend_payout_ratio"))
    dividend_yield = _safe_float(latest.get("dividend_yield_on_year_end_price_pct"))
    implied_pe = _safe_float(latest.get("implied_pe_year_end"))
    share_price_range = _safe_float(latest.get("share_price_range_pct"))
    policy_signal = str(latest.get("Dividend Policy Signal", "Not Available"))
    yield_signal = str(latest.get("Dividend Yield Signal", "Not Available"))
    valuation_signal = str(latest.get("Valuation Multiple Signal", "Not Available"))

    rows = [
        {
            "Card": "Latest Year",
            "Value": int(latest_year),
            "Display Value": str(int(latest_year)),
            "Numeric Value": int(latest_year),
            "Interpretation": "Most recent year included in the shareholder-return dataset.",
        },
        {
            "Card": "Market Capitalization",
            "Value": market_cap,
            "Display Value": f"€{market_cap:,.2f}bn"
            if not pd.isna(market_cap)
            else "Not Available",
            "Numeric Value": market_cap,
            "Interpretation": "Latest official market capitalization in EUR billions.",
        },
        {
            "Card": "EPS",
            "Value": eps,
            "Display Value": f"€{eps:.2f}"
            if not pd.isna(eps)
            else "Not Available",
            "Numeric Value": eps,
            "Interpretation": "Latest earnings per share.",
        },
        {
            "Card": "Dividend per Share",
            "Value": dividend,
            "Display Value": f"€{dividend:.2f}"
            if not pd.isna(dividend)
            else "Not Available",
            "Numeric Value": dividend,
            "Interpretation": "Latest ordinary dividend per share.",
        },
        {
            "Card": "Dividend Payout Ratio",
            "Value": payout,
            "Display Value": f"{payout:.2%}"
            if not pd.isna(payout)
            else "Not Available",
            "Numeric Value": payout,
            "Interpretation": f"Payout policy is classified as {policy_signal}.",
        },
        {
            "Card": "Dividend Yield",
            "Value": dividend_yield,
            "Display Value": f"{dividend_yield:.2f}%"
            if not pd.isna(dividend_yield)
            else "Not Available",
            "Numeric Value": dividend_yield,
            "Interpretation": f"Dividend yield is classified as {yield_signal}.",
        },
        {
            "Card": "Implied Year-End P/E",
            "Value": implied_pe,
            "Display Value": f"{implied_pe:.2f}x"
            if not pd.isna(implied_pe)
            else "Not Available",
            "Numeric Value": implied_pe,
            "Interpretation": f"Year-end valuation multiple is classified as {valuation_signal}.",
        },
        {
            "Card": "12-Month Share Price Range",
            "Value": share_price_range,
            "Display Value": f"{share_price_range:.2f}%"
            if not pd.isna(share_price_range)
            else "Not Available",
            "Numeric Value": share_price_range,
            "Interpretation": "Measures the 12-month high-low range relative to average share price.",
        },
        {
            "Card": "Dividend Policy Signal",
            "Value": policy_signal,
            "Display Value": policy_signal,
            "Numeric Value": payout,
            "Interpretation": _policy_interpretation(
                payout_policy=policy_signal,
                yield_classification=yield_signal,
                valuation_classification=valuation_signal,
            ),
        },
    ]

    return pd.DataFrame(rows)


def build_shareholder_return_leaderboard(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Build a year-by-year shareholder-return leaderboard.

    Parameters
    ----------
    metrics:
        Output from build_dividend_metrics().

    Returns
    -------
    pd.DataFrame
        Clean year-by-year shareholder-return table.
    """
    if metrics.empty:
        return pd.DataFrame(
            columns=[
                "Year",
                "Market Cap EUR bn",
                "EPS EUR",
                "Dividend per Share EUR",
                "EPS Growth pct",
                "Dividend Growth pct",
                "Market Cap Growth pct",
                "Year-End Share Price Growth pct",
                "Dividend Yield pct",
                "Earnings Yield pct",
                "Implied P/E",
                "Dividend Policy Signal",
                "Dividend Yield Signal",
                "Valuation Multiple Signal",
            ]
        )

    leaderboard = metrics.copy().reset_index()

    leaderboard["EPS Growth Signal"] = leaderboard["eps_growth_pct"].apply(
        _growth_signal
    )

    leaderboard["Dividend Growth Signal"] = leaderboard[
        "dividend_growth_pct"
    ].apply(_growth_signal)

    leaderboard["Market Cap Growth Signal"] = leaderboard[
        "market_cap_growth_pct"
    ].apply(_growth_signal)

    leaderboard = leaderboard.rename(
        columns={
            "year": "Year",
            "market_cap_eur_bn": "Market Cap EUR bn",
            "eps_eur": "EPS EUR",
            "dividend_per_share_eur": "Dividend per Share EUR",
            "eps_growth_pct": "EPS Growth pct",
            "dividend_growth_pct": "Dividend Growth pct",
            "market_cap_growth_pct": "Market Cap Growth pct",
            "year_end_share_price_growth_pct": "Year-End Share Price Growth pct",
            "dividend_yield_on_year_end_price_pct": "Dividend Yield pct",
            "earnings_yield_on_year_end_price_pct": "Earnings Yield pct",
            "implied_pe_year_end": "Implied P/E",
        }
    )

    return (
        leaderboard[
            [
                "Year",
                "Market Cap EUR bn",
                "EPS EUR",
                "Dividend per Share EUR",
                "EPS Growth pct",
                "Dividend Growth pct",
                "Market Cap Growth pct",
                "Year-End Share Price Growth pct",
                "Dividend Yield pct",
                "Earnings Yield pct",
                "Implied P/E",
                "Dividend Policy Signal",
                "Dividend Yield Signal",
                "Valuation Multiple Signal",
                "EPS Growth Signal",
                "Dividend Growth Signal",
                "Market Cap Growth Signal",
            ]
        ]
        .sort_values("Year")
        .reset_index(drop=True)
    )


def build_dividend_policy_dashboard(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Build a dividend-policy dashboard table.

    Parameters
    ----------
    metrics:
        Output from build_dividend_metrics().

    Returns
    -------
    pd.DataFrame
        Dashboard-friendly dividend policy and valuation-support table.
    """
    if metrics.empty:
        return pd.DataFrame(
            columns=[
                "Year",
                "Dividend Payout Ratio",
                "Dividend Yield pct",
                "Earnings Yield pct",
                "Implied P/E",
                "Dividend Policy Signal",
                "Dividend Yield Signal",
                "Valuation Multiple Signal",
                "Interpretation",
            ]
        )

    dashboard = metrics.copy().reset_index()

    if "Dividend Yield Signal" not in dashboard.columns:
        dashboard["Dividend Yield Signal"] = dashboard[
            "dividend_yield_on_year_end_price_pct"
        ].apply(_classify_dividend_yield)

    if "Valuation Multiple Signal" not in dashboard.columns:
        dashboard["Valuation Multiple Signal"] = dashboard[
            "implied_pe_year_end"
        ].apply(_classify_valuation_multiple)

    dashboard["Interpretation"] = dashboard.apply(
        lambda row: _policy_interpretation(
            payout_policy=str(row["Dividend Policy Signal"]),
            yield_classification=str(row["Dividend Yield Signal"]),
            valuation_classification=str(row["Valuation Multiple Signal"]),
        ),
        axis=1,
    )

    dashboard = dashboard.rename(
        columns={
            "year": "Year",
            "dividend_payout_ratio": "Dividend Payout Ratio",
            "dividend_yield_on_year_end_price_pct": "Dividend Yield pct",
            "earnings_yield_on_year_end_price_pct": "Earnings Yield pct",
            "implied_pe_year_end": "Implied P/E",
        }
    )

    return (
        dashboard[
            [
                "Year",
                "Dividend Payout Ratio",
                "Dividend Yield pct",
                "Earnings Yield pct",
                "Implied P/E",
                "Dividend Policy Signal",
                "Dividend Yield Signal",
                "Valuation Multiple Signal",
                "Interpretation",
            ]
        ]
        .sort_values("Year")
        .reset_index(drop=True)
    )


def save_shareholder_outputs(
    raw_data_dir: Path,
    table_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, calculate, and save Hermès shareholder-return output tables."""
    history = load_share_dividend_history(raw_data_dir)
    metrics = build_dividend_metrics(history)
    summary = build_shareholder_summary(metrics)

    shareholder_dashboard_cards = build_shareholder_dashboard_cards(
        summary=summary,
        metrics=metrics,
    )

    shareholder_return_leaderboard = build_shareholder_return_leaderboard(metrics)

    dividend_policy_dashboard = build_dividend_policy_dashboard(metrics)

    table_dir.mkdir(parents=True, exist_ok=True)

    history.to_csv(
        table_dir / "hermes_share_dividend_history.csv",
        index=False,
    )

    metrics.to_csv(
        table_dir / "hermes_dividend_metrics.csv",
    )

    summary.to_csv(
        table_dir / "hermes_shareholder_summary.csv",
        index=False,
    )

    shareholder_dashboard_cards.to_csv(
        table_dir / "hermes_shareholder_dashboard_cards.csv",
        index=False,
    )

    shareholder_return_leaderboard.to_csv(
        table_dir / "hermes_shareholder_return_leaderboard.csv",
        index=False,
    )

    dividend_policy_dashboard.to_csv(
        table_dir / "hermes_dividend_policy_dashboard.csv",
        index=False,
    )

    return history, metrics