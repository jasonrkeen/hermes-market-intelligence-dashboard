"""Market intelligence dashboard for the Hermès market intelligence pipeline.

This module combines major project signals into a market-intelligence summary:
market performance, fundamentals, Q1 revenue momentum, risk profile,
factor/scenario support, valuation, geopolitical risk, news flow, and strategic moat.

The module also creates dashboard-ready card, signal-breakdown, and watchlist
tables for Streamlit and other interactive reporting outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _safe_float(value: Any, default: float = np.nan) -> float:
    """Safely convert a value to float."""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _score_from_direction(
    value: float,
    positive_threshold: float,
    negative_threshold: float,
) -> str:
    """Classify a numeric value as Positive, Neutral, or Negative."""
    if pd.isna(value):
        return "Not Available"

    if value >= positive_threshold:
        return "Positive"

    if value <= negative_threshold:
        return "Negative"

    return "Neutral"


def _extract_total_q1_growth(revenue_momentum_summary: pd.DataFrame) -> float:
    """Extract total constant-FX growth from the Q1 revenue summary."""
    if revenue_momentum_summary.empty:
        return np.nan

    rows = revenue_momentum_summary[
        revenue_momentum_summary["Insight"].eq("Constant-FX growth")
    ]

    if rows.empty:
        return np.nan

    return _safe_float(rows.iloc[0]["Value"])


def _extract_news_tone(news_summary: pd.DataFrame | None) -> str:
    """Extract overall news tone from the news intelligence summary."""
    if news_summary is None or news_summary.empty:
        return "Not Available"

    rows = news_summary[news_summary["Metric"].eq("Average News Score")]

    if rows.empty:
        return "Not Available"

    return str(rows.iloc[0]["Interpretation"])


def _extract_geopolitical_posture(
    geopolitical_summary: pd.DataFrame | None,
) -> str:
    """Extract geopolitical risk posture from the geopolitical summary."""
    if geopolitical_summary is None or geopolitical_summary.empty:
        return "Not Available"

    rows = geopolitical_summary[geopolitical_summary["Metric"].eq("Risk Posture")]

    if rows.empty:
        return "Not Available"

    return str(rows.iloc[0]["Value"])


def _extract_valuation_signal(
    valuation_signal: pd.DataFrame | None,
) -> tuple[str, float]:
    """Extract valuation signal and implied upside/downside."""
    if valuation_signal is None or valuation_signal.empty:
        return "Not Available", np.nan

    signal_rows = valuation_signal[
        valuation_signal["Metric"].eq("Valuation Signal")
    ]

    upside_rows = valuation_signal[
        valuation_signal["Metric"].eq("Implied Upside / Downside")
    ]

    signal = (
        str(signal_rows.iloc[0]["Value"])
        if not signal_rows.empty
        else "Not Available"
    )

    implied_upside = (
        _safe_float(upside_rows.iloc[0]["Value"])
        if not upside_rows.empty
        else np.nan
    )

    return signal, implied_upside


def _classify_market_item_priority(
    item_type: str,
    category: str,
    assessment: str,
) -> str:
    """Assign a dashboard watchlist priority to a catalyst/risk item."""
    item_type_lower = str(item_type).lower()
    category_lower = str(category).lower()
    assessment_lower = str(assessment).lower()

    high_priority_categories = {
        "valuation",
        "geopolitical risk",
        "risk profile",
        "news flow",
    }

    if item_type_lower == "risk" and category_lower in high_priority_categories:
        return "High"

    if "high" in assessment_lower or "elevated" in assessment_lower:
        return "High"

    if item_type_lower == "risk":
        return "Medium"

    if item_type_lower == "catalyst":
        return "Medium"

    return "Monitor"


def _market_item_action_note(
    item_type: str,
    category: str,
    indicator: str,
    assessment: str,
) -> str:
    """Create a short action-oriented monitoring note."""
    item_type_lower = str(item_type).lower()
    category_lower = str(category).lower()

    if item_type_lower == "risk" and category_lower == "valuation":
        return "Monitor fair value gap, multiple sensitivity, and market price movement."

    if item_type_lower == "risk" and category_lower == "geopolitical risk":
        return "Monitor tourism, China demand, FX, trade, and regional disruption indicators."

    if item_type_lower == "risk" and category_lower == "news flow":
        return "Monitor recent headline tone and whether negative themes persist."

    if item_type_lower == "risk" and category_lower == "risk profile":
        return "Monitor volatility, drawdown, and downside-risk indicators."

    if item_type_lower == "catalyst":
        return "Monitor whether this catalyst continues to support the market intelligence view."

    return (
        f"Track {indicator} because the current assessment is {assessment}."
    )


def build_market_intelligence_dashboard(
    metrics: pd.DataFrame,
    intelligence_score: pd.DataFrame,
    scenario_table: pd.DataFrame,
    revenue_momentum_summary: pd.DataFrame,
    strategy_scores: pd.DataFrame,
    valuation_signal: pd.DataFrame | None = None,
    news_summary: pd.DataFrame | None = None,
    geopolitical_summary: pd.DataFrame | None = None,
    target_label: str = "Hermès International",
) -> pd.DataFrame:
    """Build a compact market-intelligence dashboard.

    Parameters
    ----------
    metrics:
        Risk/return metrics table.
    intelligence_score:
        Hermès Luxury Intelligence Score table.
    scenario_table:
        Scenario return estimates.
    revenue_momentum_summary:
        Q1 2026 revenue momentum summary.
    strategy_scores:
        Strategic moat and qualitative score table.
    valuation_signal:
        Optional valuation signal table from valuation.py.
    news_summary:
        Optional news intelligence summary.
    geopolitical_summary:
        Optional geopolitical risk summary.
    target_label:
        Name of the Hermès row in the metrics table.

    Returns
    -------
    pd.DataFrame
        Market intelligence dashboard table.
    """
    rows: list[dict[str, Any]] = []

    hermes_metrics = (
        metrics.loc[target_label]
        if target_label in metrics.index
        else pd.Series(dtype=float)
    )

    annualized_return = _safe_float(hermes_metrics.get("Annualized Return"))
    annualized_volatility = _safe_float(hermes_metrics.get("Annualized Volatility"))
    max_drawdown = _safe_float(hermes_metrics.get("Max Drawdown"))
    sharpe_ratio = _safe_float(hermes_metrics.get("Sharpe Ratio"))

    score_label = "Hermès Luxury Intelligence Score"

    final_score = (
        _safe_float(intelligence_score.loc[score_label, "Score"])
        if score_label in intelligence_score.index
        else np.nan
    )

    base_case_return = (
        _safe_float(scenario_table.loc["Base Case", "Estimated Hermès Return"])
        if "Base Case" in scenario_table.index
        and "Estimated Hermès Return" in scenario_table.columns
        else np.nan
    )

    q1_constant_fx_growth = _extract_total_q1_growth(revenue_momentum_summary)

    qualitative_score = (
        _safe_float(strategy_scores["Score_0_to_100"].mean())
        if strategy_scores is not None and not strategy_scores.empty
        else np.nan
    )

    valuation_signal_label, implied_upside = _extract_valuation_signal(
        valuation_signal
    )

    news_tone = _extract_news_tone(news_summary)
    geopolitical_posture = _extract_geopolitical_posture(geopolitical_summary)

    rows.extend(
        [
            {
                "Category": "Model Signal",
                "Indicator": "Hermès Luxury Intelligence Score",
                "Value": final_score,
                "Assessment": _score_from_direction(final_score, 75, 45),
                "Interpretation": (
                    "Composite score combining return, risk, factor, scenario, "
                    "moat, and revenue-momentum signals."
                ),
            },
            {
                "Category": "Market Performance",
                "Indicator": "Annualized Return",
                "Value": annualized_return,
                "Assessment": _score_from_direction(
                    annualized_return,
                    0.12,
                    0.00,
                ),
                "Interpretation": (
                    "Measures Hermès' annualized historical return over the "
                    "analysis period."
                ),
            },
            {
                "Category": "Risk Profile",
                "Indicator": "Annualized Volatility",
                "Value": annualized_volatility,
                "Assessment": _score_from_direction(
                    -annualized_volatility,
                    -0.30,
                    -0.20,
                ),
                "Interpretation": (
                    "Lower volatility improves the risk profile; elevated "
                    "volatility increases uncertainty."
                ),
            },
            {
                "Category": "Risk Profile",
                "Indicator": "Maximum Drawdown",
                "Value": max_drawdown,
                "Assessment": _score_from_direction(
                    max_drawdown,
                    -0.25,
                    -0.45,
                ),
                "Interpretation": (
                    "Large drawdowns highlight valuation sensitivity and "
                    "downside risk."
                ),
            },
            {
                "Category": "Risk-Adjusted Return",
                "Indicator": "Sharpe Ratio",
                "Value": sharpe_ratio,
                "Assessment": _score_from_direction(
                    sharpe_ratio,
                    0.75,
                    0.25,
                ),
                "Interpretation": "Measures return quality relative to volatility.",
            },
            {
                "Category": "Scenario Support",
                "Indicator": "Base Case Scenario Return",
                "Value": base_case_return,
                "Assessment": _score_from_direction(
                    base_case_return,
                    0.10,
                    0.00,
                ),
                "Interpretation": (
                    "Shows whether factor-based base-case assumptions support "
                    "positive estimated return."
                ),
            },
            {
                "Category": "Revenue Momentum",
                "Indicator": "Q1 2026 Constant-FX Growth",
                "Value": q1_constant_fx_growth,
                "Assessment": _score_from_direction(
                    q1_constant_fx_growth,
                    5.0,
                    0.0,
                ),
                "Interpretation": (
                    "Captures current reported operating momentum adjusted for "
                    "currency effects."
                ),
            },
            {
                "Category": "Strategic Moat",
                "Indicator": "Qualitative Moat Score",
                "Value": qualitative_score,
                "Assessment": _score_from_direction(
                    qualitative_score,
                    80,
                    50,
                ),
                "Interpretation": (
                    "Measures craftsmanship, exclusivity, brand strength, "
                    "vertical integration, and shareholder communication quality."
                ),
            },
            {
                "Category": "Valuation",
                "Indicator": "Valuation Signal",
                "Value": valuation_signal_label,
                "Assessment": valuation_signal_label,
                "Interpretation": (
                    "Rules-based valuation view comparing fair value range "
                    "against current market price."
                ),
            },
            {
                "Category": "Valuation",
                "Indicator": "Implied Upside / Downside",
                "Value": implied_upside,
                "Assessment": _score_from_direction(
                    implied_upside,
                    0.15,
                    -0.10,
                ),
                "Interpretation": (
                    "Compares base fair value estimate with current market price."
                ),
            },
            {
                "Category": "News Flow",
                "Indicator": "News Intelligence Tone",
                "Value": news_tone,
                "Assessment": news_tone,
                "Interpretation": (
                    "Rules-based summary of recent news headlines and market themes."
                ),
            },
            {
                "Category": "Geopolitical Risk",
                "Indicator": "Geopolitical Risk Posture",
                "Value": geopolitical_posture,
                "Assessment": geopolitical_posture,
                "Interpretation": (
                    "Summarizes geopolitical risk exposure across tourism, "
                    "China demand, FX, trade, and regional disruption."
                ),
            },
        ]
    )

    return pd.DataFrame(rows)


def build_market_intelligence_summary(
    dashboard: pd.DataFrame,
) -> pd.DataFrame:
    """Build a short market-intelligence summary from the dashboard."""
    if dashboard.empty:
        return pd.DataFrame(
            [
                {
                    "Section": "Market Intelligence",
                    "Summary": "No market intelligence dashboard was available.",
                }
            ]
        )

    positive_count = int(dashboard["Assessment"].eq("Positive").sum())
    negative_count = int(dashboard["Assessment"].eq("Negative").sum())
    neutral_count = int(dashboard["Assessment"].eq("Neutral").sum())

    risk_terms = {
        "Elevated",
        "High",
        "Risk-Leaning",
        "Valuation Stretched",
        "Cautious / risk-aware",
    }

    catalyst_terms = {
        "Positive / Catalyst-Leaning",
        "Valuation Supportive",
        "Constructive",
        "Moderately Constructive",
    }

    risk_leaning_count = int(dashboard["Assessment"].isin(risk_terms).sum())
    catalyst_leaning_count = int(dashboard["Assessment"].isin(catalyst_terms).sum())

    adjusted_positive = positive_count + catalyst_leaning_count
    adjusted_negative = negative_count + risk_leaning_count

    if adjusted_positive > adjusted_negative:
        overall_view = "Constructive but monitored"
    elif adjusted_negative > adjusted_positive:
        overall_view = "Cautious / risk-aware"
    else:
        overall_view = "Balanced / mixed"

    rows = [
        {
            "Section": "Overall Market Intelligence View",
            "Summary": overall_view,
        },
        {
            "Section": "Positive Signals",
            "Summary": positive_count,
        },
        {
            "Section": "Catalyst-Leaning External Signals",
            "Summary": catalyst_leaning_count,
        },
        {
            "Section": "Negative Signals",
            "Summary": negative_count,
        },
        {
            "Section": "Risk-Leaning External Signals",
            "Summary": risk_leaning_count,
        },
        {
            "Section": "Neutral Signals",
            "Summary": neutral_count,
        },
    ]

    return pd.DataFrame(rows)


def build_catalyst_risk_register(
    dashboard: pd.DataFrame,
) -> pd.DataFrame:
    """Create a catalyst/risk register from dashboard indicators."""
    rows: list[dict[str, Any]] = []

    for _, row in dashboard.iterrows():
        assessment = str(row.get("Assessment", ""))

        if assessment in {
            "Positive",
            "Positive / Catalyst-Leaning",
            "Valuation Supportive",
            "Constructive",
            "Moderately Constructive",
        }:
            item_type = "Catalyst"
        elif assessment in {
            "Negative",
            "Risk-Leaning",
            "Valuation Stretched",
            "Elevated",
            "High",
            "Cautious / risk-aware",
        }:
            item_type = "Risk"
        else:
            item_type = "Watch Item"

        rows.append(
            {
                "Type": item_type,
                "Category": row.get("Category", ""),
                "Indicator": row.get("Indicator", ""),
                "Assessment": assessment,
                "Monitoring Note": row.get("Interpretation", ""),
            }
        )

    return pd.DataFrame(rows)


def build_market_intelligence_cards(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build card-style market intelligence summary rows.

    Parameters
    ----------
    summary:
        Output from build_market_intelligence_summary().

    Returns
    -------
    pd.DataFrame
        Card-style table for Streamlit display.
    """
    if summary.empty:
        return pd.DataFrame(
            [
                {
                    "Card": "Overall Market Intelligence View",
                    "Value": "Not Available",
                    "Display Value": "Not Available",
                    "Interpretation": "No market intelligence summary was available.",
                }
            ]
        )

    summary_lookup = {
        str(row["Section"]): row["Summary"]
        for _, row in summary.iterrows()
        if "Section" in summary.columns and "Summary" in summary.columns
    }

    card_definitions = [
        {
            "Card": "Overall Market Intelligence View",
            "Section": "Overall Market Intelligence View",
            "Interpretation": "Overall project-level view after combining model, market, valuation, news, and geopolitical signals.",
        },
        {
            "Card": "Positive Signals",
            "Section": "Positive Signals",
            "Interpretation": "Number of dashboard indicators classified as positive.",
        },
        {
            "Card": "Catalyst-Leaning External Signals",
            "Section": "Catalyst-Leaning External Signals",
            "Interpretation": "Number of valuation, news, or external signals that lean toward catalysts.",
        },
        {
            "Card": "Negative Signals",
            "Section": "Negative Signals",
            "Interpretation": "Number of dashboard indicators classified as negative.",
        },
        {
            "Card": "Risk-Leaning External Signals",
            "Section": "Risk-Leaning External Signals",
            "Interpretation": "Number of valuation, news, or external signals that lean toward risks.",
        },
        {
            "Card": "Neutral Signals",
            "Section": "Neutral Signals",
            "Interpretation": "Number of dashboard indicators classified as neutral.",
        },
    ]

    rows: list[dict[str, Any]] = []

    for card in card_definitions:
        value = summary_lookup.get(card["Section"], "Not Available")

        rows.append(
            {
                "Card": card["Card"],
                "Value": value,
                "Display Value": str(value),
                "Interpretation": card["Interpretation"],
            }
        )

    return pd.DataFrame(rows)


def build_market_signal_breakdown(
    dashboard: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize market-intelligence signals by category and assessment.

    Parameters
    ----------
    dashboard:
        Output from build_market_intelligence_dashboard().

    Returns
    -------
    pd.DataFrame
        Category-level signal breakdown table for dashboard display.
    """
    if dashboard.empty:
        return pd.DataFrame(
            columns=[
                "Category",
                "Total Indicators",
                "Positive",
                "Neutral",
                "Negative",
                "Catalyst-Leaning",
                "Risk-Leaning",
                "Watch Items",
                "Primary Assessment",
            ]
        )

    required_columns = {"Category", "Assessment"}

    missing_columns = required_columns.difference(dashboard.columns)

    if missing_columns:
        raise KeyError(
            "Market intelligence dashboard is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    catalyst_terms = {
        "Positive / Catalyst-Leaning",
        "Valuation Supportive",
        "Constructive",
        "Moderately Constructive",
    }

    risk_terms = {
        "Risk-Leaning",
        "Valuation Stretched",
        "Elevated",
        "High",
        "Cautious / risk-aware",
    }

    rows: list[dict[str, Any]] = []

    for category, group in dashboard.groupby("Category"):
        assessments = group["Assessment"].astype(str)

        positive = int(assessments.eq("Positive").sum())
        neutral = int(assessments.eq("Neutral").sum())
        negative = int(assessments.eq("Negative").sum())
        catalyst_leaning = int(assessments.isin(catalyst_terms).sum())
        risk_leaning = int(assessments.isin(risk_terms).sum())

        watch_items = max(
            0,
            int(
                len(group)
                - positive
                - neutral
                - negative
                - catalyst_leaning
                - risk_leaning
            ),
        )

        if positive + catalyst_leaning > negative + risk_leaning:
            primary_assessment = "Constructive"
        elif negative + risk_leaning > positive + catalyst_leaning:
            primary_assessment = "Risk-Leaning"
        elif neutral > 0:
            primary_assessment = "Neutral"
        else:
            primary_assessment = "Watch Item"

        rows.append(
            {
                "Category": category,
                "Total Indicators": int(len(group)),
                "Positive": positive,
                "Neutral": neutral,
                "Negative": negative,
                "Catalyst-Leaning": catalyst_leaning,
                "Risk-Leaning": risk_leaning,
                "Watch Items": watch_items,
                "Primary Assessment": primary_assessment,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["Risk-Leaning", "Negative", "Catalyst-Leaning", "Positive"],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def build_market_watchlist(
    catalyst_risk_register: pd.DataFrame,
) -> pd.DataFrame:
    """Build a prioritized market watchlist from the catalyst/risk register.

    Parameters
    ----------
    catalyst_risk_register:
        Output from build_catalyst_risk_register().

    Returns
    -------
    pd.DataFrame
        Prioritized watchlist for dashboard display.
    """
    if catalyst_risk_register.empty:
        return pd.DataFrame(
            columns=[
                "Priority",
                "Type",
                "Category",
                "Indicator",
                "Assessment",
                "Monitoring Note",
                "Action Note",
            ]
        )

    required_columns = {
        "Type",
        "Category",
        "Indicator",
        "Assessment",
        "Monitoring Note",
    }

    missing_columns = required_columns.difference(catalyst_risk_register.columns)

    if missing_columns:
        raise KeyError(
            "Catalyst/risk register is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    watchlist = catalyst_risk_register.copy()

    watchlist["Priority"] = watchlist.apply(
        lambda row: _classify_market_item_priority(
            item_type=str(row["Type"]),
            category=str(row["Category"]),
            assessment=str(row["Assessment"]),
        ),
        axis=1,
    )

    watchlist["Action Note"] = watchlist.apply(
        lambda row: _market_item_action_note(
            item_type=str(row["Type"]),
            category=str(row["Category"]),
            indicator=str(row["Indicator"]),
            assessment=str(row["Assessment"]),
        ),
        axis=1,
    )

    priority_order = {
        "High": 0,
        "Medium": 1,
        "Monitor": 2,
    }

    watchlist["Priority Sort"] = watchlist["Priority"].map(priority_order).fillna(99)

    return (
        watchlist[
            [
                "Priority",
                "Type",
                "Category",
                "Indicator",
                "Assessment",
                "Monitoring Note",
                "Action Note",
                "Priority Sort",
            ]
        ]
        .sort_values(
            ["Priority Sort", "Type", "Category", "Indicator"],
            ascending=True,
        )
        .drop(columns=["Priority Sort"])
        .reset_index(drop=True)
    )


def save_market_intelligence_outputs(
    table_dir: Path,
    metrics: pd.DataFrame,
    intelligence_score: pd.DataFrame,
    scenario_table: pd.DataFrame,
    revenue_momentum_summary: pd.DataFrame,
    strategy_scores: pd.DataFrame,
    valuation_signal: pd.DataFrame | None = None,
    news_summary: pd.DataFrame | None = None,
    geopolitical_summary: pd.DataFrame | None = None,
    target_label: str = "Hermès International",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build and save market intelligence dashboard outputs."""
    table_dir.mkdir(parents=True, exist_ok=True)

    dashboard = build_market_intelligence_dashboard(
        metrics=metrics,
        intelligence_score=intelligence_score,
        scenario_table=scenario_table,
        revenue_momentum_summary=revenue_momentum_summary,
        strategy_scores=strategy_scores,
        valuation_signal=valuation_signal,
        news_summary=news_summary,
        geopolitical_summary=geopolitical_summary,
        target_label=target_label,
    )

    summary = build_market_intelligence_summary(dashboard)
    catalyst_risk_register = build_catalyst_risk_register(dashboard)

    market_cards = build_market_intelligence_cards(summary)
    signal_breakdown = build_market_signal_breakdown(dashboard)
    market_watchlist = build_market_watchlist(catalyst_risk_register)

    dashboard.to_csv(
        table_dir / "hermes_market_intelligence_dashboard.csv",
        index=False,
    )

    summary.to_csv(
        table_dir / "hermes_market_intelligence_summary.csv",
        index=False,
    )

    catalyst_risk_register.to_csv(
        table_dir / "hermes_catalyst_risk_register.csv",
        index=False,
    )

    market_cards.to_csv(
        table_dir / "hermes_market_intelligence_cards.csv",
        index=False,
    )

    signal_breakdown.to_csv(
        table_dir / "hermes_market_signal_breakdown.csv",
        index=False,
    )

    market_watchlist.to_csv(
        table_dir / "hermes_market_watchlist.csv",
        index=False,
    )

    return dashboard, summary, catalyst_risk_register