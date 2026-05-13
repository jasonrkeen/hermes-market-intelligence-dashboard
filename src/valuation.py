"""Valuation model for the Hermès market intelligence report.

This module builds a lightweight buy-side / sell-side style valuation layer for
Hermès. It includes a forward financial forecast, DCF valuation, peer multiple
valuation, fair value range, and valuation sensitivity tables.

The model is designed for educational market-intelligence use. It is not
investment advice and does not represent a formal valuation opinion.

The module also creates dashboard-ready valuation cards, valuation bridge tables,
and long-form sensitivity tables for Streamlit and other interactive reporting
outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import (
    VALUATION_DIVIDEND_PAYOUT_RATES,
    VALUATION_FCF_CONVERSION_RATES,
    VALUATION_FORECAST_YEARS,
    VALUATION_NET_INCOME_MARGIN_RATES,
    VALUATION_OPERATING_MARGIN_RATES,
    VALUATION_PE_MULTIPLES,
    VALUATION_REVENUE_GROWTH_RATES,
    VALUATION_TERMINAL_GROWTH,
    VALUATION_TERMINAL_GROWTH_SENSITIVITY,
    VALUATION_WACC,
    VALUATION_WACC_SENSITIVITY,
)


def _safe_float(value: Any, default: float = np.nan) -> float:
    """Safely convert a value to float."""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_metric_value(
    fundamental_metrics: pd.DataFrame,
    metric_name: str,
    year: str = "2025",
    default: float = np.nan,
) -> float:
    """Retrieve a metric value from the fundamental metrics table."""
    if fundamental_metrics is None or fundamental_metrics.empty:
        return default

    if metric_name not in fundamental_metrics.index:
        return default

    if year not in fundamental_metrics.columns:
        return default

    return _safe_float(fundamental_metrics.loc[metric_name, year], default=default)


def _get_latest_dividend_metric(
    dividend_metrics: pd.DataFrame,
    column: str,
    default: float = np.nan,
) -> float:
    """Retrieve latest value from dividend/shareholder metrics."""
    if dividend_metrics is None or dividend_metrics.empty:
        return default

    if column not in dividend_metrics.columns:
        return default

    latest_year = dividend_metrics.index.max()
    return _safe_float(dividend_metrics.loc[latest_year, column], default=default)


def _get_dcf_line_item(
    dcf_valuation: pd.DataFrame,
    line_item: str,
    default: float = np.nan,
) -> float:
    """Retrieve a DCF line-item value."""
    if dcf_valuation is None or dcf_valuation.empty:
        return default

    if "Line Item" not in dcf_valuation.columns or "Value EURm" not in dcf_valuation.columns:
        return default

    rows = dcf_valuation[dcf_valuation["Line Item"].eq(line_item)]

    if rows.empty:
        return default

    return _safe_float(rows.iloc[0]["Value EURm"], default=default)


def _get_valuation_signal_value(
    valuation_signal: pd.DataFrame,
    metric: str,
    default: Any = "Not Available",
) -> Any:
    """Retrieve a value from the valuation signal table."""
    if valuation_signal is None or valuation_signal.empty:
        return default

    if "Metric" not in valuation_signal.columns or "Value" not in valuation_signal.columns:
        return default

    rows = valuation_signal[valuation_signal["Metric"].eq(metric)]

    if rows.empty:
        return default

    return rows.iloc[0]["Value"]


def _classify_implied_upside(implied_upside: float) -> str:
    """Classify implied upside/downside."""
    value = _safe_float(implied_upside)

    if pd.isna(value):
        return "Not Available"

    if value >= 0.20:
        return "Material Upside"

    if value >= 0.10:
        return "Moderate Upside"

    if value > -0.10:
        return "Balanced / Near Fair Value"

    if value > -0.20:
        return "Moderate Downside"

    return "Material Downside"


def _classify_fair_value_position(
    fair_value: float,
    current_price_eur: float | None,
) -> str:
    """Classify fair value relative to current price."""
    fair_value_numeric = _safe_float(fair_value)
    current_price_numeric = _safe_float(current_price_eur)

    if pd.isna(fair_value_numeric) or pd.isna(current_price_numeric) or current_price_numeric <= 0:
        return "Not Available"

    implied_upside = fair_value_numeric / current_price_numeric - 1

    return _classify_implied_upside(implied_upside)


def _format_currency(value: float, suffix: str = "") -> str:
    """Format a euro value."""
    numeric_value = _safe_float(value)

    if pd.isna(numeric_value):
        return "Not Available"

    return f"€{numeric_value:,.2f}{suffix}"


def _format_percent(value: float) -> str:
    """Format a decimal percentage value."""
    numeric_value = _safe_float(value)

    if pd.isna(numeric_value):
        return "Not Available"

    return f"{numeric_value:.2%}"


def build_forecast_model(
    fundamental_metrics: pd.DataFrame,
    dividend_metrics: pd.DataFrame,
    forecast_years: list[int] | None = None,
    revenue_growth_rates: dict[int, float] | None = None,
    operating_margin_rates: dict[int, float] | None = None,
    net_income_margin_rates: dict[int, float] | None = None,
    fcf_conversion_rates: dict[int, float] | None = None,
    dividend_payout_rates: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Build a forward Hermès financial forecast.

    Parameters
    ----------
    fundamental_metrics:
        Official Hermès fundamental metrics table.
    dividend_metrics:
        Shareholder return and dividend metrics table.
    forecast_years:
        Years to forecast.
    revenue_growth_rates:
        Revenue growth assumptions by year.
    operating_margin_rates:
        Recurring operating margin assumptions by year.
    net_income_margin_rates:
        Net income margin assumptions by year.
    fcf_conversion_rates:
        Adjusted free cash flow as a percentage of revenue.
    dividend_payout_rates:
        Dividend payout assumptions as a percentage of EPS.

    Returns
    -------
    pd.DataFrame
        Forecast table indexed by year.
    """
    if forecast_years is None:
        forecast_years = VALUATION_FORECAST_YEARS

    revenue_growth_rates = revenue_growth_rates or VALUATION_REVENUE_GROWTH_RATES
    operating_margin_rates = operating_margin_rates or VALUATION_OPERATING_MARGIN_RATES
    net_income_margin_rates = (
        net_income_margin_rates or VALUATION_NET_INCOME_MARGIN_RATES
    )
    fcf_conversion_rates = fcf_conversion_rates or VALUATION_FCF_CONVERSION_RATES
    dividend_payout_rates = (
        dividend_payout_rates or VALUATION_DIVIDEND_PAYOUT_RATES
    )

    base_revenue = _get_metric_value(
        fundamental_metrics,
        metric_name="Revenue",
        year="2025",
    )

    base_net_income = _get_metric_value(
        fundamental_metrics,
        metric_name="Net profit Group share",
        year="2025",
    )

    shares_outstanding = _get_latest_dividend_metric(
        dividend_metrics,
        column="shares_at_dec_31",
    )

    if pd.isna(base_revenue) or base_revenue <= 0:
        raise ValueError("Revenue base year is missing or invalid.")

    if pd.isna(shares_outstanding) or shares_outstanding <= 0:
        raise ValueError("Shares outstanding are missing or invalid.")

    rows: list[dict[str, Any]] = []
    prior_revenue = base_revenue

    for year in forecast_years:
        revenue_growth = float(revenue_growth_rates.get(year, 0.05))
        operating_margin = float(operating_margin_rates.get(year, 0.40))
        net_income_margin = float(net_income_margin_rates.get(year, 0.28))
        fcf_conversion = float(fcf_conversion_rates.get(year, 0.24))
        dividend_payout = float(dividend_payout_rates.get(year, 0.43))

        revenue = prior_revenue * (1 + revenue_growth)
        recurring_operating_income = revenue * operating_margin
        net_income = revenue * net_income_margin
        adjusted_free_cash_flow = revenue * fcf_conversion

        eps = (net_income * 1_000_000) / shares_outstanding
        dividend_per_share = eps * dividend_payout

        rows.append(
            {
                "Year": year,
                "Revenue EURm": revenue,
                "Revenue Growth": revenue_growth,
                "Recurring Operating Margin": operating_margin,
                "Recurring Operating Income EURm": recurring_operating_income,
                "Net Income Margin": net_income_margin,
                "Net Income EURm": net_income,
                "Adjusted FCF Conversion": fcf_conversion,
                "Adjusted FCF EURm": adjusted_free_cash_flow,
                "Shares Outstanding": shares_outstanding,
                "EPS EUR": eps,
                "Dividend Payout Ratio": dividend_payout,
                "Dividend Per Share EUR": dividend_per_share,
            }
        )

        prior_revenue = revenue

    forecast = pd.DataFrame(rows).set_index("Year")

    if not pd.isna(base_net_income) and base_net_income > 0:
        forecast["Net Income Growth vs 2025"] = (
            forecast["Net Income EURm"] / base_net_income - 1
        )

    return forecast


def build_dcf_valuation(
    forecast: pd.DataFrame,
    dividend_metrics: pd.DataFrame,
    net_cash_eur_m: float | None = None,
    wacc: float = VALUATION_WACC,
    terminal_growth: float = VALUATION_TERMINAL_GROWTH,
) -> pd.DataFrame:
    """Build a DCF valuation from the forecast model."""
    if forecast.empty:
        raise ValueError("Forecast table is empty.")

    if "Adjusted FCF EURm" not in forecast.columns:
        raise KeyError("Forecast table must include 'Adjusted FCF EURm'.")

    if wacc <= terminal_growth:
        raise ValueError("WACC must be greater than terminal growth.")

    shares_outstanding = _get_latest_dividend_metric(
        dividend_metrics,
        column="shares_at_dec_31",
    )

    if pd.isna(shares_outstanding) or shares_outstanding <= 0:
        raise ValueError("Shares outstanding are missing or invalid.")

    net_cash_eur_m = 0.0 if net_cash_eur_m is None else float(net_cash_eur_m)

    forecast_years = list(forecast.index)
    base_year = min(forecast_years)

    rows: list[dict[str, Any]] = []
    present_value_fcf = 0.0

    for year in forecast_years:
        year_number = int(year) - int(base_year) + 1
        fcf = float(forecast.loc[year, "Adjusted FCF EURm"])
        discount_factor = 1 / ((1 + wacc) ** year_number)
        pv_fcf = fcf * discount_factor
        present_value_fcf += pv_fcf

        rows.append(
            {
                "Line Item": f"PV FCF {year}",
                "Value EURm": pv_fcf,
            }
        )

    final_year = forecast_years[-1]
    final_fcf = float(forecast.loc[final_year, "Adjusted FCF EURm"])

    terminal_value = final_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
    terminal_discount_factor = 1 / ((1 + wacc) ** len(forecast_years))
    present_value_terminal_value = terminal_value * terminal_discount_factor

    enterprise_value = present_value_fcf + present_value_terminal_value
    equity_value = enterprise_value + net_cash_eur_m
    fair_value_per_share = (equity_value * 1_000_000) / shares_outstanding

    rows.extend(
        [
            {
                "Line Item": "Present Value of Explicit FCF",
                "Value EURm": present_value_fcf,
            },
            {
                "Line Item": "Terminal Value",
                "Value EURm": terminal_value,
            },
            {
                "Line Item": "Present Value of Terminal Value",
                "Value EURm": present_value_terminal_value,
            },
            {
                "Line Item": "Enterprise Value",
                "Value EURm": enterprise_value,
            },
            {
                "Line Item": "Net Cash",
                "Value EURm": net_cash_eur_m,
            },
            {
                "Line Item": "Equity Value",
                "Value EURm": equity_value,
            },
            {
                "Line Item": "Shares Outstanding",
                "Value EURm": shares_outstanding,
            },
            {
                "Line Item": "DCF Fair Value Per Share EUR",
                "Value EURm": fair_value_per_share,
            },
            {
                "Line Item": "WACC",
                "Value EURm": wacc,
            },
            {
                "Line Item": "Terminal Growth",
                "Value EURm": terminal_growth,
            },
        ]
    )

    return pd.DataFrame(rows)


def build_peer_multiple_valuation(
    forecast: pd.DataFrame,
    dividend_metrics: pd.DataFrame,
    current_price_eur: float | None = None,
    multiples: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build a peer-style multiple valuation."""
    if forecast.empty:
        raise ValueError("Forecast table is empty.")

    multiples = multiples or VALUATION_PE_MULTIPLES

    shares_outstanding = _get_latest_dividend_metric(
        dividend_metrics,
        column="shares_at_dec_31",
    )

    if pd.isna(shares_outstanding) or shares_outstanding <= 0:
        raise ValueError("Shares outstanding are missing or invalid.")

    first_forecast_year = forecast.index.min()
    eps_next_year = float(forecast.loc[first_forecast_year, "EPS EUR"])

    rows: list[dict[str, Any]] = []

    for case, multiple in multiples.items():
        fair_value = eps_next_year * float(multiple)

        if current_price_eur is not None and current_price_eur > 0:
            implied_upside = fair_value / current_price_eur - 1
        else:
            implied_upside = np.nan

        rows.append(
            {
                "Case": case,
                "Forward EPS EUR": eps_next_year,
                "Applied P/E Multiple": float(multiple),
                "Fair Value Per Share EUR": fair_value,
                "Current Price EUR": current_price_eur,
                "Implied Upside / Downside": implied_upside,
            }
        )

    return pd.DataFrame(rows).set_index("Case")


def build_fair_value_summary(
    dcf_valuation: pd.DataFrame,
    peer_valuation: pd.DataFrame,
    current_price_eur: float | None = None,
) -> pd.DataFrame:
    """Combine DCF and peer multiple outputs into a fair value summary."""
    dcf_row = dcf_valuation[
        dcf_valuation["Line Item"].eq("DCF Fair Value Per Share EUR")
    ]

    dcf_fair_value = (
        float(dcf_row.iloc[0]["Value EURm"])
        if not dcf_row.empty
        else np.nan
    )

    peer_values = []

    if peer_valuation is not None and not peer_valuation.empty:
        if "Fair Value Per Share EUR" in peer_valuation.columns:
            peer_values = peer_valuation["Fair Value Per Share EUR"].dropna().tolist()

    all_values = [
        value for value in [dcf_fair_value, *peer_values] if not pd.isna(value)
    ]

    if all_values:
        bear_value = float(np.min(all_values))
        base_value = float(np.median(all_values))
        bull_value = float(np.max(all_values))
    else:
        bear_value = np.nan
        base_value = np.nan
        bull_value = np.nan

    rows = [
        {
            "Case": "Bear / Low Fair Value",
            "Fair Value Per Share EUR": bear_value,
        },
        {
            "Case": "Base / Median Fair Value",
            "Fair Value Per Share EUR": base_value,
        },
        {
            "Case": "Bull / High Fair Value",
            "Fair Value Per Share EUR": bull_value,
        },
        {
            "Case": "DCF Fair Value",
            "Fair Value Per Share EUR": dcf_fair_value,
        },
    ]

    summary = pd.DataFrame(rows)

    if current_price_eur is not None and current_price_eur > 0:
        summary["Current Price EUR"] = current_price_eur
        summary["Implied Upside / Downside"] = (
            summary["Fair Value Per Share EUR"] / current_price_eur - 1
        )
    else:
        summary["Current Price EUR"] = np.nan
        summary["Implied Upside / Downside"] = np.nan

    return summary.set_index("Case")


def build_dcf_sensitivity_table(
    forecast: pd.DataFrame,
    dividend_metrics: pd.DataFrame,
    net_cash_eur_m: float | None = None,
    wacc_values: list[float] | None = None,
    terminal_growth_values: list[float] | None = None,
) -> pd.DataFrame:
    """Build a DCF sensitivity table for WACC and terminal growth."""
    wacc_values = wacc_values or VALUATION_WACC_SENSITIVITY
    terminal_growth_values = (
        terminal_growth_values or VALUATION_TERMINAL_GROWTH_SENSITIVITY
    )

    rows: list[dict[str, Any]] = []

    for terminal_growth in terminal_growth_values:
        row: dict[str, Any] = {
            "Terminal Growth": terminal_growth,
        }

        for wacc in wacc_values:
            try:
                valuation = build_dcf_valuation(
                    forecast=forecast,
                    dividend_metrics=dividend_metrics,
                    net_cash_eur_m=net_cash_eur_m,
                    wacc=wacc,
                    terminal_growth=terminal_growth,
                )

                fair_value_row = valuation[
                    valuation["Line Item"].eq("DCF Fair Value Per Share EUR")
                ]

                fair_value = (
                    float(fair_value_row.iloc[0]["Value EURm"])
                    if not fair_value_row.empty
                    else np.nan
                )

            except ValueError:
                fair_value = np.nan

            row[f"WACC {wacc:.1%}"] = fair_value

        rows.append(row)

    return pd.DataFrame(rows).set_index("Terminal Growth")


def build_valuation_signal(
    fair_value_summary: pd.DataFrame,
    current_price_eur: float | None = None,
) -> pd.DataFrame:
    """Build a valuation signal from fair value and current price."""
    if fair_value_summary.empty:
        return pd.DataFrame(
            [
                {
                    "Metric": "Valuation Signal",
                    "Value": "Not Available",
                    "Interpretation": "Fair value summary was unavailable.",
                }
            ]
        )

    if "Base / Median Fair Value" in fair_value_summary.index:
        base_fair_value = _safe_float(
            fair_value_summary.loc[
                "Base / Median Fair Value",
                "Fair Value Per Share EUR",
            ]
        )
    else:
        base_fair_value = _safe_float(
            fair_value_summary["Fair Value Per Share EUR"].median()
        )

    if current_price_eur is None or current_price_eur <= 0:
        implied_upside = np.nan
        signal = "No Current Price"
    else:
        implied_upside = base_fair_value / current_price_eur - 1

        if implied_upside >= 0.15:
            signal = "Valuation Supportive"
        elif implied_upside <= -0.10:
            signal = "Valuation Stretched"
        else:
            signal = "Fairly Valued / Balanced"

    rows = [
        {
            "Metric": "Base Fair Value Per Share EUR",
            "Value": base_fair_value,
            "Interpretation": (
                "Median of available DCF and peer multiple fair value estimates."
            ),
        },
        {
            "Metric": "Current Price EUR",
            "Value": current_price_eur if current_price_eur is not None else np.nan,
            "Interpretation": (
                "Latest available market price used for valuation comparison."
            ),
        },
        {
            "Metric": "Implied Upside / Downside",
            "Value": implied_upside,
            "Interpretation": signal,
        },
        {
            "Metric": "Valuation Signal",
            "Value": signal,
            "Interpretation": (
                "Rules-based valuation view for the market intelligence report."
            ),
        },
    ]

    return pd.DataFrame(rows)


def build_valuation_dashboard_cards(
    fair_value_summary: pd.DataFrame,
    dcf_valuation: pd.DataFrame,
    valuation_signal: pd.DataFrame,
    current_price_eur: float | None = None,
) -> pd.DataFrame:
    """Build dashboard card rows for valuation outputs.

    Parameters
    ----------
    fair_value_summary:
        Output from build_fair_value_summary().
    dcf_valuation:
        Output from build_dcf_valuation().
    valuation_signal:
        Output from build_valuation_signal().
    current_price_eur:
        Latest available market price.

    Returns
    -------
    pd.DataFrame
        Dashboard-ready valuation cards.
    """
    if fair_value_summary.empty:
        return pd.DataFrame(
            [
                {
                    "Card": "Valuation",
                    "Value": "Not Available",
                    "Display Value": "Not Available",
                    "Numeric Value": np.nan,
                    "Interpretation": "Fair value summary was unavailable.",
                }
            ]
        )

    base_fair_value = _safe_float(
        _get_valuation_signal_value(
            valuation_signal,
            "Base Fair Value Per Share EUR",
            default=np.nan,
        )
    )

    implied_upside = _safe_float(
        _get_valuation_signal_value(
            valuation_signal,
            "Implied Upside / Downside",
            default=np.nan,
        )
    )

    valuation_signal_label = str(
        _get_valuation_signal_value(
            valuation_signal,
            "Valuation Signal",
            default="Not Available",
        )
    )

    current_price_value = (
        _safe_float(current_price_eur)
        if current_price_eur is not None
        else _safe_float(
            _get_valuation_signal_value(
                valuation_signal,
                "Current Price EUR",
                default=np.nan,
            )
        )
    )

    dcf_fair_value = _get_dcf_line_item(
        dcf_valuation,
        "DCF Fair Value Per Share EUR",
    )

    bear_fair_value = (
        _safe_float(fair_value_summary.loc["Bear / Low Fair Value", "Fair Value Per Share EUR"])
        if "Bear / Low Fair Value" in fair_value_summary.index
        else np.nan
    )

    bull_fair_value = (
        _safe_float(fair_value_summary.loc["Bull / High Fair Value", "Fair Value Per Share EUR"])
        if "Bull / High Fair Value" in fair_value_summary.index
        else np.nan
    )

    rows = [
        {
            "Card": "Current Price",
            "Value": current_price_value,
            "Display Value": _format_currency(current_price_value),
            "Numeric Value": current_price_value,
            "Interpretation": "Latest available market price used for valuation comparison.",
        },
        {
            "Card": "Base Fair Value",
            "Value": base_fair_value,
            "Display Value": _format_currency(base_fair_value),
            "Numeric Value": base_fair_value,
            "Interpretation": "Median fair value estimate across DCF and peer multiple outputs.",
        },
        {
            "Card": "Implied Upside / Downside",
            "Value": implied_upside,
            "Display Value": _format_percent(implied_upside),
            "Numeric Value": implied_upside,
            "Interpretation": _classify_implied_upside(implied_upside),
        },
        {
            "Card": "Valuation Signal",
            "Value": valuation_signal_label,
            "Display Value": valuation_signal_label,
            "Numeric Value": implied_upside,
            "Interpretation": "Rules-based valuation classification used in the recommendation framework.",
        },
        {
            "Card": "DCF Fair Value",
            "Value": dcf_fair_value,
            "Display Value": _format_currency(dcf_fair_value),
            "Numeric Value": dcf_fair_value,
            "Interpretation": _classify_fair_value_position(
                dcf_fair_value,
                current_price_value,
            ),
        },
        {
            "Card": "Bear Fair Value",
            "Value": bear_fair_value,
            "Display Value": _format_currency(bear_fair_value),
            "Numeric Value": bear_fair_value,
            "Interpretation": _classify_fair_value_position(
                bear_fair_value,
                current_price_value,
            ),
        },
        {
            "Card": "Bull Fair Value",
            "Value": bull_fair_value,
            "Display Value": _format_currency(bull_fair_value),
            "Numeric Value": bull_fair_value,
            "Interpretation": _classify_fair_value_position(
                bull_fair_value,
                current_price_value,
            ),
        },
    ]

    return pd.DataFrame(rows)


def build_valuation_bridge(
    forecast: pd.DataFrame,
    dcf_valuation: pd.DataFrame,
    peer_valuation: pd.DataFrame,
    fair_value_summary: pd.DataFrame,
    valuation_signal: pd.DataFrame,
    current_price_eur: float | None = None,
) -> pd.DataFrame:
    """Build a concise valuation bridge table.

    Parameters
    ----------
    forecast:
        Output from build_forecast_model().
    dcf_valuation:
        Output from build_dcf_valuation().
    peer_valuation:
        Output from build_peer_multiple_valuation().
    fair_value_summary:
        Output from build_fair_value_summary().
    valuation_signal:
        Output from build_valuation_signal().
    current_price_eur:
        Latest available market price.

    Returns
    -------
    pd.DataFrame
        Dashboard-ready valuation bridge table.
    """
    if forecast.empty:
        return pd.DataFrame(
            [
                {
                    "Step": "Valuation Bridge",
                    "Metric": "Forecast",
                    "Value": "Not Available",
                    "Numeric Value": np.nan,
                    "Interpretation": "Forecast table was unavailable.",
                }
            ]
        )

    first_forecast_year = forecast.index.min()
    final_forecast_year = forecast.index.max()

    first_revenue = _safe_float(forecast.loc[first_forecast_year, "Revenue EURm"])
    final_revenue = _safe_float(forecast.loc[final_forecast_year, "Revenue EURm"])
    first_eps = _safe_float(forecast.loc[first_forecast_year, "EPS EUR"])
    final_eps = _safe_float(forecast.loc[final_forecast_year, "EPS EUR"])
    final_fcf = _safe_float(forecast.loc[final_forecast_year, "Adjusted FCF EURm"])

    dcf_fair_value = _get_dcf_line_item(
        dcf_valuation,
        "DCF Fair Value Per Share EUR",
    )

    base_fair_value = _safe_float(
        _get_valuation_signal_value(
            valuation_signal,
            "Base Fair Value Per Share EUR",
            default=np.nan,
        )
    )

    implied_upside = _safe_float(
        _get_valuation_signal_value(
            valuation_signal,
            "Implied Upside / Downside",
            default=np.nan,
        )
    )

    valuation_signal_label = str(
        _get_valuation_signal_value(
            valuation_signal,
            "Valuation Signal",
            default="Not Available",
        )
    )

    current_price_value = (
        _safe_float(current_price_eur)
        if current_price_eur is not None
        else _safe_float(
            _get_valuation_signal_value(
                valuation_signal,
                "Current Price EUR",
                default=np.nan,
            )
        )
    )

    peer_bear_value = np.nan
    peer_base_value = np.nan
    peer_bull_value = np.nan

    if peer_valuation is not None and not peer_valuation.empty:
        if "Bear" in peer_valuation.index:
            peer_bear_value = _safe_float(
                peer_valuation.loc["Bear", "Fair Value Per Share EUR"]
            )

        if "Base" in peer_valuation.index:
            peer_base_value = _safe_float(
                peer_valuation.loc["Base", "Fair Value Per Share EUR"]
            )

        if "Bull" in peer_valuation.index:
            peer_bull_value = _safe_float(
                peer_valuation.loc["Bull", "Fair Value Per Share EUR"]
            )

    if pd.isna(peer_bear_value) and peer_valuation is not None and not peer_valuation.empty:
        peer_bear_value = _safe_float(peer_valuation["Fair Value Per Share EUR"].min())

    if pd.isna(peer_base_value) and peer_valuation is not None and not peer_valuation.empty:
        peer_base_value = _safe_float(peer_valuation["Fair Value Per Share EUR"].median())

    if pd.isna(peer_bull_value) and peer_valuation is not None and not peer_valuation.empty:
        peer_bull_value = _safe_float(peer_valuation["Fair Value Per Share EUR"].max())

    rows = [
        {
            "Step": "Forecast",
            "Metric": f"{first_forecast_year} Revenue",
            "Value": _format_currency(first_revenue, suffix="m"),
            "Numeric Value": first_revenue,
            "Interpretation": "First forecast-year revenue assumption.",
        },
        {
            "Step": "Forecast",
            "Metric": f"{final_forecast_year} Revenue",
            "Value": _format_currency(final_revenue, suffix="m"),
            "Numeric Value": final_revenue,
            "Interpretation": "Final forecast-year revenue assumption.",
        },
        {
            "Step": "Forecast",
            "Metric": f"{first_forecast_year} EPS",
            "Value": _format_currency(first_eps),
            "Numeric Value": first_eps,
            "Interpretation": "First forecast-year earnings per share.",
        },
        {
            "Step": "Forecast",
            "Metric": f"{final_forecast_year} EPS",
            "Value": _format_currency(final_eps),
            "Numeric Value": final_eps,
            "Interpretation": "Final forecast-year earnings per share.",
        },
        {
            "Step": "Forecast",
            "Metric": f"{final_forecast_year} Adjusted FCF",
            "Value": _format_currency(final_fcf, suffix="m"),
            "Numeric Value": final_fcf,
            "Interpretation": "Final forecast-year adjusted free cash flow.",
        },
        {
            "Step": "DCF",
            "Metric": "DCF Fair Value",
            "Value": _format_currency(dcf_fair_value),
            "Numeric Value": dcf_fair_value,
            "Interpretation": "Fair value per share from discounted cash flow valuation.",
        },
        {
            "Step": "Peer Multiple",
            "Metric": "Peer Bear Fair Value",
            "Value": _format_currency(peer_bear_value),
            "Numeric Value": peer_bear_value,
            "Interpretation": "Lower-end peer multiple fair value estimate.",
        },
        {
            "Step": "Peer Multiple",
            "Metric": "Peer Base Fair Value",
            "Value": _format_currency(peer_base_value),
            "Numeric Value": peer_base_value,
            "Interpretation": "Median or base peer multiple fair value estimate.",
        },
        {
            "Step": "Peer Multiple",
            "Metric": "Peer Bull Fair Value",
            "Value": _format_currency(peer_bull_value),
            "Numeric Value": peer_bull_value,
            "Interpretation": "Upper-end peer multiple fair value estimate.",
        },
        {
            "Step": "Fair Value Range",
            "Metric": "Base Fair Value",
            "Value": _format_currency(base_fair_value),
            "Numeric Value": base_fair_value,
            "Interpretation": "Median fair value across available DCF and peer outputs.",
        },
        {
            "Step": "Market Comparison",
            "Metric": "Current Price",
            "Value": _format_currency(current_price_value),
            "Numeric Value": current_price_value,
            "Interpretation": "Market price used for comparison.",
        },
        {
            "Step": "Market Comparison",
            "Metric": "Base Implied Upside / Downside",
            "Value": _format_percent(implied_upside),
            "Numeric Value": implied_upside,
            "Interpretation": _classify_implied_upside(implied_upside),
        },
        {
            "Step": "Valuation Signal",
            "Metric": "Valuation Signal",
            "Value": valuation_signal_label,
            "Numeric Value": implied_upside,
            "Interpretation": "Final rules-based valuation view.",
        },
    ]

    return pd.DataFrame(rows)


def build_valuation_sensitivity_dashboard(
    sensitivity: pd.DataFrame,
    current_price_eur: float | None = None,
) -> pd.DataFrame:
    """Convert DCF sensitivity matrix into long-form dashboard table.

    Parameters
    ----------
    sensitivity:
        Output from build_dcf_sensitivity_table().
    current_price_eur:
        Latest available market price.

    Returns
    -------
    pd.DataFrame
        Long-form valuation sensitivity dashboard.
    """
    if sensitivity.empty:
        return pd.DataFrame(
            columns=[
                "Terminal Growth",
                "WACC",
                "Fair Value Per Share EUR",
                "Current Price EUR",
                "Distance from Current Price",
                "Scenario Classification",
            ]
        )

    sensitivity_reset = sensitivity.copy().reset_index()

    if "Terminal Growth" not in sensitivity_reset.columns:
        first_column = sensitivity_reset.columns[0]
        sensitivity_reset = sensitivity_reset.rename(
            columns={first_column: "Terminal Growth"}
        )

    rows: list[dict[str, Any]] = []
    current_price_value = _safe_float(current_price_eur)

    for _, row in sensitivity_reset.iterrows():
        terminal_growth = _safe_float(row["Terminal Growth"])

        for column in sensitivity_reset.columns:
            if not str(column).startswith("WACC "):
                continue

            wacc_text = str(column).replace("WACC ", "").replace("%", "")
            try:
                wacc = float(wacc_text) / 100
            except ValueError:
                wacc = np.nan

            fair_value = _safe_float(row[column])

            if not pd.isna(current_price_value) and current_price_value > 0:
                distance_from_price = fair_value / current_price_value - 1
            else:
                distance_from_price = np.nan

            rows.append(
                {
                    "Terminal Growth": terminal_growth,
                    "WACC": wacc,
                    "Fair Value Per Share EUR": fair_value,
                    "Current Price EUR": current_price_value,
                    "Distance from Current Price": distance_from_price,
                    "Scenario Classification": _classify_implied_upside(
                        distance_from_price
                    ),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(["Terminal Growth", "WACC"])
        .reset_index(drop=True)
    )


def save_valuation_outputs(
    table_dir: Path,
    fundamental_metrics: pd.DataFrame,
    dividend_metrics: pd.DataFrame,
    current_price_eur: float | None = None,
    net_cash_eur_m: float | None = None,
    wacc: float = VALUATION_WACC,
    terminal_growth: float = VALUATION_TERMINAL_GROWTH,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build and save valuation model outputs."""
    table_dir.mkdir(parents=True, exist_ok=True)

    if net_cash_eur_m is None:
        net_cash_eur_m = _get_metric_value(
            fundamental_metrics,
            metric_name="Restated net cash position",
            year="2025",
            default=0.0,
        )

    forecast = build_forecast_model(
        fundamental_metrics=fundamental_metrics,
        dividend_metrics=dividend_metrics,
    )

    dcf_valuation = build_dcf_valuation(
        forecast=forecast,
        dividend_metrics=dividend_metrics,
        net_cash_eur_m=net_cash_eur_m,
        wacc=wacc,
        terminal_growth=terminal_growth,
    )

    peer_valuation = build_peer_multiple_valuation(
        forecast=forecast,
        dividend_metrics=dividend_metrics,
        current_price_eur=current_price_eur,
    )

    fair_value_summary = build_fair_value_summary(
        dcf_valuation=dcf_valuation,
        peer_valuation=peer_valuation,
        current_price_eur=current_price_eur,
    )

    sensitivity = build_dcf_sensitivity_table(
        forecast=forecast,
        dividend_metrics=dividend_metrics,
        net_cash_eur_m=net_cash_eur_m,
    )

    valuation_signal = build_valuation_signal(
        fair_value_summary=fair_value_summary,
        current_price_eur=current_price_eur,
    )

    valuation_dashboard_cards = build_valuation_dashboard_cards(
        fair_value_summary=fair_value_summary,
        dcf_valuation=dcf_valuation,
        valuation_signal=valuation_signal,
        current_price_eur=current_price_eur,
    )

    valuation_bridge = build_valuation_bridge(
        forecast=forecast,
        dcf_valuation=dcf_valuation,
        peer_valuation=peer_valuation,
        fair_value_summary=fair_value_summary,
        valuation_signal=valuation_signal,
        current_price_eur=current_price_eur,
    )

    valuation_sensitivity_dashboard = build_valuation_sensitivity_dashboard(
        sensitivity=sensitivity,
        current_price_eur=current_price_eur,
    )

    forecast.to_csv(
        table_dir / "hermes_forecast_model.csv",
    )

    dcf_valuation.to_csv(
        table_dir / "hermes_dcf_valuation.csv",
        index=False,
    )

    peer_valuation.to_csv(
        table_dir / "hermes_peer_multiple_valuation.csv",
    )

    fair_value_summary.to_csv(
        table_dir / "hermes_fair_value_summary.csv",
    )

    sensitivity.to_csv(
        table_dir / "hermes_dcf_sensitivity.csv",
    )

    valuation_signal.to_csv(
        table_dir / "hermes_valuation_signal.csv",
        index=False,
    )

    valuation_dashboard_cards.to_csv(
        table_dir / "hermes_valuation_dashboard_cards.csv",
        index=False,
    )

    valuation_bridge.to_csv(
        table_dir / "hermes_valuation_bridge.csv",
        index=False,
    )

    valuation_sensitivity_dashboard.to_csv(
        table_dir / "hermes_valuation_sensitivity_dashboard.csv",
        index=False,
    )

    return (
        forecast,
        dcf_valuation,
        peer_valuation,
        fair_value_summary,
        sensitivity,
        valuation_signal,
    )