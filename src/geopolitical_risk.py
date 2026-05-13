"""Geopolitical risk analytics for the Hermès market intelligence report.

This module converts geopolitical risk assumptions into structured tables for
the README, PDF report, and market intelligence layer.

The module also creates dashboard-friendly geopolitical risk outputs including
summary cards, heatmap-style risk tables, and scenario severity interpretation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _classify_risk_category(risk_factor: str) -> str:
    """Classify geopolitical risk factors into broad report categories."""
    lowered = risk_factor.lower()

    if "middle east" in lowered or "tourism" in lowered:
        return "Regional / Tourism"

    if "china" in lowered:
        return "China / Demand"

    if "fx" in lowered or "euro" in lowered or "currency" in lowered:
        return "Currency"

    if "re-rating" in lowered or "valuation" in lowered:
        return "Market / Valuation"

    if "trade" in lowered or "tariff" in lowered:
        return "Trade Policy"

    if "supply chain" in lowered or "logistics" in lowered:
        return "Operations"

    if "energy" in lowered or "inflation" in lowered:
        return "Macro"

    if "cyber" in lowered:
        return "Technology / Operational"

    return "Other"


def _classify_score_severity(score: float) -> str:
    """Classify a 0-100 risk score into a severity label."""
    score_float = float(score)

    if score_float >= 80:
        return "Very High"

    if score_float >= 60:
        return "High"

    if score_float >= 40:
        return "Elevated"

    if score_float >= 20:
        return "Moderate"

    return "Low"


def _interpret_score_severity(severity: str) -> str:
    """Return a plain-English interpretation for a severity label."""
    if severity == "Very High":
        return "Risk level is very high and should be monitored as a major market-intelligence factor."

    if severity == "High":
        return "Risk level is high and could materially influence sentiment, valuation, or demand assumptions."

    if severity == "Elevated":
        return "Risk level is elevated and should be monitored, though it is not the dominant risk factor."

    if severity == "Moderate":
        return "Risk level is moderate and relevant mainly as a supporting risk consideration."

    if severity == "Low":
        return "Risk level is low based on the configured assumptions."

    return "Risk severity could not be classified."


def build_geopolitical_risk_table(
    risk_factors: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Build a table of geopolitical risk factors."""
    rows: list[dict[str, Any]] = []

    for factor, details in risk_factors.items():
        risk_level = float(details.get("risk_level", 0))
        risk_score = max(0, min(100, risk_level * 20))

        severity = _classify_score_severity(risk_score)

        rows.append(
            {
                "Category": _classify_risk_category(factor),
                "Risk Factor": factor,
                "Exposure": details.get("exposure", ""),
                "Risk Level 1-5": risk_level,
                "Risk Score 0-100": risk_score,
                "Severity": severity,
                "Direction": details.get("direction", "Negative"),
                "Description": details.get("description", ""),
            }
        )

    return pd.DataFrame(rows)


def build_geopolitical_risk_summary(
    risk_table: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize overall geopolitical risk posture."""
    if risk_table.empty:
        return pd.DataFrame(
            [
                {
                    "Metric": "Average Geopolitical Risk Score",
                    "Value": 0,
                    "Interpretation": "No geopolitical risks configured.",
                }
            ]
        )

    avg_score = float(risk_table["Risk Score 0-100"].mean())
    max_row = risk_table.loc[risk_table["Risk Score 0-100"].idxmax()]

    if avg_score >= 75:
        posture = "High"
    elif avg_score >= 50:
        posture = "Elevated"
    elif avg_score >= 30:
        posture = "Moderate"
    else:
        posture = "Low"

    rows = [
        {
            "Metric": "Average Geopolitical Risk Score",
            "Value": avg_score,
            "Interpretation": posture,
        },
        {
            "Metric": "Highest Risk Category",
            "Value": max_row["Category"],
            "Interpretation": "Category associated with the highest configured geopolitical risk score.",
        },
        {
            "Metric": "Highest Risk Factor",
            "Value": max_row["Risk Factor"],
            "Interpretation": max_row["Description"],
        },
        {
            "Metric": "Risk Posture",
            "Value": posture,
            "Interpretation": (
                "Geopolitical factors are incorporated as market-intelligence risks, "
                "not as standalone investment recommendations."
            ),
        },
    ]

    return pd.DataFrame(rows)


def build_geopolitical_category_summary(
    risk_table: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize geopolitical risk by category."""
    if risk_table.empty or "Category" not in risk_table.columns:
        return pd.DataFrame(
            columns=[
                "Category",
                "Risk Count",
                "Average Risk Score",
                "Maximum Risk Score",
                "Highest Risk Factor",
            ]
        )

    rows: list[dict[str, Any]] = []

    for category, group in risk_table.groupby("Category"):
        max_row = group.loc[group["Risk Score 0-100"].idxmax()]

        rows.append(
            {
                "Category": category,
                "Risk Count": len(group),
                "Average Risk Score": float(group["Risk Score 0-100"].mean()),
                "Maximum Risk Score": float(group["Risk Score 0-100"].max()),
                "Highest Risk Factor": max_row["Risk Factor"],
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["Maximum Risk Score", "Average Risk Score"],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def build_geopolitical_scenario_overlay(
    scenario_adjustments: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Build geopolitical scenario adjustment table."""
    rows: list[dict[str, Any]] = []

    for scenario, assumptions in scenario_adjustments.items():
        revenue_adjustment = float(
            assumptions.get("revenue_growth_adjustment_pct", 0.0)
        )
        margin_adjustment = float(
            assumptions.get("margin_adjustment_pct", 0.0)
        )
        multiple_adjustment = float(
            assumptions.get("valuation_multiple_adjustment_pct", 0.0)
        )

        stress_score = (
            abs(revenue_adjustment) * 8
            + abs(margin_adjustment) * 10
            + abs(multiple_adjustment) * 2
        )

        rows.append(
            {
                "Scenario": scenario,
                "Revenue Growth Adjustment pct": revenue_adjustment,
                "Margin Adjustment pct": margin_adjustment,
                "Valuation Multiple Adjustment pct": multiple_adjustment,
                "Stress Score": min(100, stress_score),
            }
        )

    return pd.DataFrame(rows)


def build_geopolitical_dashboard_summary(
    risk_table: pd.DataFrame,
    risk_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build dashboard summary cards for geopolitical risk.

    Parameters
    ----------
    risk_table:
        Geopolitical risk factor table returned by build_geopolitical_risk_table().
    risk_summary:
        Summary table returned by build_geopolitical_risk_summary().

    Returns
    -------
    pd.DataFrame
        Card-style dashboard summary table.
    """
    if risk_table.empty:
        return pd.DataFrame(
            [
                {
                    "Card": "Risk Posture",
                    "Value": "Not Available",
                    "Numeric Value": 0,
                    "Interpretation": "No geopolitical risk factors are configured.",
                }
            ]
        )

    avg_score = float(risk_table["Risk Score 0-100"].mean())
    max_row = risk_table.loc[risk_table["Risk Score 0-100"].idxmax()]
    total_risks = int(len(risk_table))
    category_count = int(risk_table["Category"].nunique())

    posture_row = risk_summary[risk_summary["Metric"].eq("Risk Posture")]
    posture = (
        str(posture_row.iloc[0]["Value"])
        if not posture_row.empty
        else _classify_score_severity(avg_score)
    )

    rows = [
        {
            "Card": "Risk Posture",
            "Value": posture,
            "Numeric Value": avg_score,
            "Interpretation": "Overall geopolitical risk posture based on configured risk factors.",
        },
        {
            "Card": "Average Risk Score",
            "Value": f"{avg_score:.2f}",
            "Numeric Value": avg_score,
            "Interpretation": "Average geopolitical risk score on a 0-100 scale.",
        },
        {
            "Card": "Highest Risk Category",
            "Value": max_row["Category"],
            "Numeric Value": float(max_row["Risk Score 0-100"]),
            "Interpretation": "Category attached to the highest individual geopolitical risk score.",
        },
        {
            "Card": "Highest Risk Factor",
            "Value": max_row["Risk Factor"],
            "Numeric Value": float(max_row["Risk Score 0-100"]),
            "Interpretation": str(max_row["Description"]),
        },
        {
            "Card": "Configured Risk Factors",
            "Value": total_risks,
            "Numeric Value": total_risks,
            "Interpretation": "Number of geopolitical risk factors included in the pipeline.",
        },
        {
            "Card": "Risk Categories Covered",
            "Value": category_count,
            "Numeric Value": category_count,
            "Interpretation": "Number of broad geopolitical risk categories represented.",
        },
    ]

    return pd.DataFrame(rows)


def build_geopolitical_risk_heatmap(
    risk_table: pd.DataFrame,
) -> pd.DataFrame:
    """Build a dashboard-friendly geopolitical risk heatmap table.

    Parameters
    ----------
    risk_table:
        Geopolitical risk factor table returned by build_geopolitical_risk_table().

    Returns
    -------
    pd.DataFrame
        Heatmap-style table with severity and interpretation fields.
    """
    if risk_table.empty:
        return pd.DataFrame(
            columns=[
                "Category",
                "Risk Factor",
                "Exposure",
                "Risk Level 1-5",
                "Risk Score 0-100",
                "Severity",
                "Direction",
                "Interpretation",
                "Description",
            ]
        )

    required_columns = {
        "Category",
        "Risk Factor",
        "Exposure",
        "Risk Level 1-5",
        "Risk Score 0-100",
        "Direction",
        "Description",
    }

    missing_columns = required_columns.difference(risk_table.columns)

    if missing_columns:
        raise KeyError(
            "Geopolitical risk table is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    heatmap = risk_table.copy()

    heatmap["Severity"] = heatmap["Risk Score 0-100"].apply(_classify_score_severity)
    heatmap["Interpretation"] = heatmap["Severity"].apply(_interpret_score_severity)

    return (
        heatmap[
            [
                "Category",
                "Risk Factor",
                "Exposure",
                "Risk Level 1-5",
                "Risk Score 0-100",
                "Severity",
                "Direction",
                "Interpretation",
                "Description",
            ]
        ]
        .sort_values(
            ["Risk Score 0-100", "Risk Level 1-5"],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def build_geopolitical_scenario_dashboard(
    scenario_overlay: pd.DataFrame,
) -> pd.DataFrame:
    """Build dashboard-friendly geopolitical scenario table.

    Parameters
    ----------
    scenario_overlay:
        Scenario overlay table returned by build_geopolitical_scenario_overlay().

    Returns
    -------
    pd.DataFrame
        Scenario table with severity labels and plain-English interpretations.
    """
    if scenario_overlay.empty:
        return pd.DataFrame(
            columns=[
                "Scenario",
                "Revenue Growth Adjustment pct",
                "Margin Adjustment pct",
                "Valuation Multiple Adjustment pct",
                "Stress Score",
                "Severity",
                "Interpretation",
            ]
        )

    required_columns = {
        "Scenario",
        "Revenue Growth Adjustment pct",
        "Margin Adjustment pct",
        "Valuation Multiple Adjustment pct",
        "Stress Score",
    }

    missing_columns = required_columns.difference(scenario_overlay.columns)

    if missing_columns:
        raise KeyError(
            "Geopolitical scenario overlay table is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    dashboard = scenario_overlay.copy()

    dashboard["Severity"] = dashboard["Stress Score"].apply(_classify_score_severity)

    def _scenario_interpretation(row: pd.Series) -> str:
        severity = row["Severity"]
        revenue = float(row["Revenue Growth Adjustment pct"])
        margin = float(row["Margin Adjustment pct"])
        multiple = float(row["Valuation Multiple Adjustment pct"])

        if severity in {"Very High", "High"}:
            return (
                "Scenario implies a meaningful geopolitical stress case through "
                f"{revenue:.2f} percentage-point revenue-growth adjustment, "
                f"{margin:.2f} percentage-point margin adjustment, "
                f"and {multiple:.2f}% valuation-multiple adjustment."
            )

        if severity == "Elevated":
            return (
                "Scenario implies a moderate stress case that should be monitored "
                "alongside valuation and risk-model outputs."
            )

        if severity == "Moderate":
            return (
                "Scenario implies a limited but relevant geopolitical adjustment."
            )

        return "Scenario implies a low geopolitical adjustment under current assumptions."

    dashboard["Interpretation"] = dashboard.apply(_scenario_interpretation, axis=1)

    return (
        dashboard[
            [
                "Scenario",
                "Revenue Growth Adjustment pct",
                "Margin Adjustment pct",
                "Valuation Multiple Adjustment pct",
                "Stress Score",
                "Severity",
                "Interpretation",
            ]
        ]
        .sort_values("Stress Score", ascending=False)
        .reset_index(drop=True)
    )


def save_geopolitical_risk_outputs(
    table_dir: Path,
    risk_factors: dict[str, dict[str, Any]],
    scenario_adjustments: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create and save geopolitical risk output tables."""
    table_dir.mkdir(parents=True, exist_ok=True)

    risk_table = build_geopolitical_risk_table(risk_factors)
    risk_summary = build_geopolitical_risk_summary(risk_table)
    category_summary = build_geopolitical_category_summary(risk_table)
    scenario_overlay = build_geopolitical_scenario_overlay(scenario_adjustments)

    dashboard_summary = build_geopolitical_dashboard_summary(
        risk_table=risk_table,
        risk_summary=risk_summary,
    )

    risk_heatmap = build_geopolitical_risk_heatmap(risk_table)

    scenario_dashboard = build_geopolitical_scenario_dashboard(scenario_overlay)

    risk_table.to_csv(
        table_dir / "hermes_geopolitical_risk_factors.csv",
        index=False,
    )

    risk_summary.to_csv(
        table_dir / "hermes_geopolitical_risk_summary.csv",
        index=False,
    )

    category_summary.to_csv(
        table_dir / "hermes_geopolitical_category_summary.csv",
        index=False,
    )

    scenario_overlay.to_csv(
        table_dir / "hermes_geopolitical_scenario_overlay.csv",
        index=False,
    )

    dashboard_summary.to_csv(
        table_dir / "hermes_geopolitical_dashboard_summary.csv",
        index=False,
    )

    risk_heatmap.to_csv(
        table_dir / "hermes_geopolitical_risk_heatmap.csv",
        index=False,
    )

    scenario_dashboard.to_csv(
        table_dir / "hermes_geopolitical_scenario_dashboard.csv",
        index=False,
    )

    return risk_table, risk_summary, scenario_overlay