"""Official Hermès fundamentals for the market intelligence report.

This module loads Hermès official key figures, calculates year-over-year changes,
and saves clean tables for the Hermès market intelligence and equity research
pipeline.

The module also creates dashboard-friendly fundamental summary tables for
Streamlit and other interactive outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "metric",
    "2025",
    "2024",
    "unit",
}


def _default_key_figures() -> pd.DataFrame:
    """Create the official Hermès key-figures table used by the project.

    Values are sourced from Hermès Finance key figures and are expressed in the
    units shown in the unit column.
    """
    rows = [
        {
            "metric": "Revenue",
            "2025": 16002,
            "2024": 15170,
            "unit": "EUR millions",
        },
        {
            "metric": "Growth at current exchange rates",
            "2025": 5.5,
            "2024": 13.0,
            "unit": "Percent",
        },
        {
            "metric": "Growth at constant exchange rates",
            "2025": 8.9,
            "2024": 14.7,
            "unit": "Percent",
        },
        {
            "metric": "Recurring operating income",
            "2025": 6569,
            "2024": 6150,
            "unit": "EUR millions",
        },
        {
            "metric": "Recurring operating margin",
            "2025": 41.0,
            "2024": 40.5,
            "unit": "Percent",
        },
        {
            "metric": "Net profit Group share",
            "2025": 4524,
            "2024": 4603,
            "unit": "EUR millions",
        },
        {
            "metric": "Operating cash flows",
            "2025": 5607,
            "2024": 5378,
            "unit": "EUR millions",
        },
        {
            "metric": "Operating investments",
            "2025": 1161,
            "2024": 1067,
            "unit": "EUR millions",
        },
        {
            "metric": "Adjusted free cash flows",
            "2025": 3880,
            "2024": 3767,
            "unit": "EUR millions",
        },
        {
            "metric": "Equity Group share",
            "2025": 18840,
            "2024": 17327,
            "unit": "EUR millions",
        },
        {
            "metric": "Net cash position",
            "2025": 12239,
            "2024": 11642,
            "unit": "EUR millions",
        },
        {
            "metric": "Restated net cash position",
            "2025": 12773,
            "2024": 12039,
            "unit": "EUR millions",
        },
        {
            "metric": "Workforce",
            "2025": 26494,
            "2024": 25185,
            "unit": "Employees",
        },
    ]

    return pd.DataFrame(rows)


def _validate_key_figures(key_figures: pd.DataFrame) -> None:
    """Validate the Hermès key-figures table."""
    missing_columns = REQUIRED_COLUMNS.difference(key_figures.columns)

    if missing_columns:
        raise ValueError(
            "Hermès key-figures table is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if key_figures.empty:
        raise ValueError("Hermès key-figures table is empty.")


def _safe_metric_value(
    metrics: pd.DataFrame,
    metric: str,
    column: str,
    default: Any = np.nan,
) -> Any:
    """Safely retrieve a value from the indexed fundamental metrics table."""
    if metric not in metrics.index:
        return default

    if column not in metrics.columns:
        return default

    return metrics.loc[metric, column]


def _classify_direction(value: float | int | None) -> str:
    """Classify a numeric movement as positive, negative, or unchanged."""
    if value is None or pd.isna(value):
        return "Not Available"

    value_float = float(value)

    if value_float > 0:
        return "Improved"

    if value_float < 0:
        return "Declined"

    return "Unchanged"


def load_key_figures(raw_data_dir: Path) -> pd.DataFrame:
    """Load Hermès key figures from CSV.

    If the CSV does not exist, the function creates a default project dataset
    and saves it to the raw data directory.
    """
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    key_figures_path = raw_data_dir / "hermes_key_figures.csv"

    if key_figures_path.exists():
        key_figures = pd.read_csv(key_figures_path)
    else:
        key_figures = _default_key_figures()
        key_figures.to_csv(key_figures_path, index=False)

    _validate_key_figures(key_figures)

    return key_figures


def build_fundamental_metrics(key_figures: pd.DataFrame) -> pd.DataFrame:
    """Create derived YoY metrics from the official Hermès key figures table.

    For monetary and count metrics, Percent Change is calculated as the normal
    year-over-year percentage change.

    For metrics already expressed as percentages, Percent Change is replaced
    with the percentage-point change because that is more interpretable.
    """
    _validate_key_figures(key_figures)

    numeric = key_figures.copy()

    numeric["2025"] = pd.to_numeric(numeric["2025"], errors="coerce")
    numeric["2024"] = pd.to_numeric(numeric["2024"], errors="coerce")

    numeric["Absolute Change"] = numeric["2025"] - numeric["2024"]

    numeric["Percent Change"] = np.where(
        numeric["2024"].replace(0, np.nan).notna(),
        ((numeric["2025"] / numeric["2024"]) - 1.0) * 100.0,
        np.nan,
    )

    percent_mask = numeric["unit"].str.contains(
        "Percent",
        case=False,
        na=False,
    )

    numeric.loc[percent_mask, "Percent Change"] = numeric.loc[
        percent_mask,
        "Absolute Change",
    ]

    numeric.loc[percent_mask, "unit"] = (
        numeric.loc[percent_mask, "unit"] + " / percentage-point change"
    )

    numeric["Direction"] = np.where(
        numeric["Absolute Change"] > 0,
        "Improved",
        np.where(numeric["Absolute Change"] < 0, "Declined", "Unchanged"),
    )

    numeric["2025 Share of Revenue"] = np.nan

    revenue_2025 = numeric.loc[
        numeric["metric"].eq("Revenue"),
        "2025",
    ]

    if not revenue_2025.empty and float(revenue_2025.iloc[0]) != 0:
        revenue_value = float(revenue_2025.iloc[0])

        eur_metric_mask = numeric["unit"].str.contains(
            "EUR millions",
            case=False,
            na=False,
        )

        numeric.loc[eur_metric_mask, "2025 Share of Revenue"] = (
            numeric.loc[eur_metric_mask, "2025"] / revenue_value
        ) * 100.0

    return numeric.set_index("metric")


def build_fundamental_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Build a compact summary table for report narrative."""
    summary_rows = []

    selected_metrics = [
        "Revenue",
        "Recurring operating income",
        "Recurring operating margin",
        "Adjusted free cash flows",
        "Restated net cash position",
        "Workforce",
    ]

    for metric in selected_metrics:
        if metric not in metrics.index:
            continue

        row = metrics.loc[metric]

        summary_rows.append(
            {
                "Metric": metric,
                "2025": row.get("2025"),
                "2024": row.get("2024"),
                "Unit": row.get("unit"),
                "Change": row.get("Absolute Change"),
                "Percent / Point Change": row.get("Percent Change"),
                "Direction": row.get("Direction"),
            }
        )

    return pd.DataFrame(summary_rows)


def build_fundamental_dashboard_cards(metrics: pd.DataFrame) -> pd.DataFrame:
    """Build dashboard card metrics from Hermès fundamental metrics.

    Returns
    -------
    pd.DataFrame
        Compact card-style table for Streamlit dashboard display.
    """
    card_definitions = [
        {
            "Card": "Revenue",
            "Metric": "Revenue",
            "Primary Column": "2025",
            "Change Column": "Percent Change",
            "Description": "Official 2025 revenue in EUR millions.",
        },
        {
            "Card": "Constant-FX Growth",
            "Metric": "Growth at constant exchange rates",
            "Primary Column": "2025",
            "Change Column": "Percent Change",
            "Description": "Annual growth at constant exchange rates.",
        },
        {
            "Card": "Recurring Operating Margin",
            "Metric": "Recurring operating margin",
            "Primary Column": "2025",
            "Change Column": "Percent Change",
            "Description": "Recurring operating margin and year-over-year point change.",
        },
        {
            "Card": "Adjusted Free Cash Flow",
            "Metric": "Adjusted free cash flows",
            "Primary Column": "2025",
            "Change Column": "Percent Change",
            "Description": "Adjusted free cash flow in EUR millions.",
        },
        {
            "Card": "Restated Net Cash Position",
            "Metric": "Restated net cash position",
            "Primary Column": "2025",
            "Change Column": "Percent Change",
            "Description": "Restated net cash position in EUR millions.",
        },
        {
            "Card": "Workforce",
            "Metric": "Workforce",
            "Primary Column": "2025",
            "Change Column": "Percent Change",
            "Description": "Total workforce at year-end.",
        },
    ]

    rows: list[dict[str, Any]] = []

    for card in card_definitions:
        metric = card["Metric"]
        primary_column = card["Primary Column"]
        change_column = card["Change Column"]

        value = _safe_metric_value(metrics, metric, primary_column)
        change = _safe_metric_value(metrics, metric, change_column)
        absolute_change = _safe_metric_value(metrics, metric, "Absolute Change")
        unit = _safe_metric_value(metrics, metric, "unit", "")
        direction = _safe_metric_value(metrics, metric, "Direction", "Not Available")

        rows.append(
            {
                "Card": card["Card"],
                "Metric": metric,
                "Value": value,
                "Change": change,
                "Absolute Change": absolute_change,
                "Unit": unit,
                "Direction": direction,
                "Description": card["Description"],
            }
        )

    return pd.DataFrame(rows)


def build_fundamental_quality_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Build a compact qualitative summary of Hermès fundamental quality.

    The score is intentionally simple and transparent. It is not a valuation
    model; it summarizes whether Hermès' latest official fundamentals indicate
    strength, stability, or pressure.
    """
    revenue_growth = _safe_metric_value(
        metrics,
        "Growth at constant exchange rates",
        "2025",
    )
    operating_margin = _safe_metric_value(
        metrics,
        "Recurring operating margin",
        "2025",
    )
    free_cash_flow_change = _safe_metric_value(
        metrics,
        "Adjusted free cash flows",
        "Percent Change",
    )
    net_cash_change = _safe_metric_value(
        metrics,
        "Restated net cash position",
        "Percent Change",
    )
    workforce_change = _safe_metric_value(
        metrics,
        "Workforce",
        "Percent Change",
    )

    score_components: list[dict[str, Any]] = []

    revenue_growth_score = 0
    if not pd.isna(revenue_growth):
        if float(revenue_growth) >= 8:
            revenue_growth_score = 25
        elif float(revenue_growth) >= 5:
            revenue_growth_score = 20
        elif float(revenue_growth) > 0:
            revenue_growth_score = 12

    score_components.append(
        {
            "Component": "Revenue Growth",
            "Raw Value": revenue_growth,
            "Score": revenue_growth_score,
            "Interpretation": "Constant-FX revenue growth contribution.",
        }
    )

    margin_score = 0
    if not pd.isna(operating_margin):
        if float(operating_margin) >= 40:
            margin_score = 25
        elif float(operating_margin) >= 35:
            margin_score = 20
        elif float(operating_margin) >= 30:
            margin_score = 15

    score_components.append(
        {
            "Component": "Operating Margin",
            "Raw Value": operating_margin,
            "Score": margin_score,
            "Interpretation": "Recurring operating margin contribution.",
        }
    )

    fcf_score = 0
    if not pd.isna(free_cash_flow_change):
        if float(free_cash_flow_change) > 5:
            fcf_score = 20
        elif float(free_cash_flow_change) > 0:
            fcf_score = 15
        elif float(free_cash_flow_change) > -5:
            fcf_score = 8

    score_components.append(
        {
            "Component": "Free Cash Flow Momentum",
            "Raw Value": free_cash_flow_change,
            "Score": fcf_score,
            "Interpretation": "Adjusted free cash flow year-over-year contribution.",
        }
    )

    net_cash_score = 0
    if not pd.isna(net_cash_change):
        if float(net_cash_change) > 5:
            net_cash_score = 20
        elif float(net_cash_change) > 0:
            net_cash_score = 15
        elif float(net_cash_change) > -5:
            net_cash_score = 8

    score_components.append(
        {
            "Component": "Net Cash Strength",
            "Raw Value": net_cash_change,
            "Score": net_cash_score,
            "Interpretation": "Restated net cash position year-over-year contribution.",
        }
    )

    workforce_score = 0
    if not pd.isna(workforce_change):
        if float(workforce_change) > 0:
            workforce_score = 10
        elif float(workforce_change) == 0:
            workforce_score = 5

    score_components.append(
        {
            "Component": "Workforce Expansion",
            "Raw Value": workforce_change,
            "Score": workforce_score,
            "Interpretation": "Workforce growth contribution.",
        }
    )

    component_table = pd.DataFrame(score_components)

    total_score = float(component_table["Score"].sum())

    if total_score >= 80:
        quality_label = "Strong"
        quality_view = (
            "Hermès fundamentals remain strong, supported by high margins, "
            "positive constant-FX growth, free cash flow generation, and net cash strength."
        )
    elif total_score >= 60:
        quality_label = "Stable"
        quality_view = (
            "Hermès fundamentals remain stable, though some growth or cash-flow metrics "
            "show more moderate momentum."
        )
    elif total_score >= 40:
        quality_label = "Mixed"
        quality_view = (
            "Hermès fundamentals are mixed, with strength in some areas offset by pressure "
            "in growth, cash-flow, or balance-sheet indicators."
        )
    else:
        quality_label = "Pressured"
        quality_view = (
            "Hermès fundamentals appear pressured based on the selected summary indicators."
        )

    summary_rows = [
        {
            "Component": "Overall Fundamental Quality",
            "Raw Value": quality_label,
            "Score": total_score,
            "Interpretation": quality_view,
        }
    ]

    summary = pd.DataFrame(summary_rows)

    return pd.concat([summary, component_table], ignore_index=True)


def save_fundamental_outputs(
    raw_data_dir: Path,
    table_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, calculate, and save Hermès fundamental output tables."""
    key_figures = load_key_figures(raw_data_dir)
    metrics = build_fundamental_metrics(key_figures)
    summary = build_fundamental_summary(metrics)
    dashboard_cards = build_fundamental_dashboard_cards(metrics)
    quality_summary = build_fundamental_quality_summary(metrics)

    table_dir.mkdir(parents=True, exist_ok=True)

    key_figures.to_csv(
        table_dir / "hermes_key_figures.csv",
        index=False,
    )

    metrics.to_csv(
        table_dir / "hermes_fundamental_metrics.csv",
    )

    summary.to_csv(
        table_dir / "hermes_fundamental_summary.csv",
        index=False,
    )

    dashboard_cards.to_csv(
        table_dir / "hermes_fundamental_dashboard_cards.csv",
        index=False,
    )

    quality_summary.to_csv(
        table_dir / "hermes_fundamental_quality_summary.csv",
        index=False,
    )

    return key_figures, metrics