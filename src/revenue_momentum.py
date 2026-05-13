"""Hermès Q1 2026 revenue momentum analytics for the market intelligence report.

This module loads Hermès Q1 2026 revenue data, builds revenue-momentum,
FX-impact, and momentum-signal tables, and saves clean outputs for the PDF report,
README, and market intelligence dashboard.

The module also creates dashboard-ready revenue cards, segment leaderboards,
and FX-impact interpretation tables for Streamlit and other interactive outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "segment",
    "type",
    "revenue_2026_eur_m",
    "revenue_2025_eur_m",
    "published_growth_pct",
    "constant_fx_growth_pct",
}


def _default_q1_2026_revenue() -> pd.DataFrame:
    """Create the default Hermès Q1 2026 revenue table.

    Values are expressed in EUR millions and percentage growth rates.
    """
    rows = [
        {
            "segment": "France",
            "type": "Geography",
            "revenue_2026_eur_m": 347,
            "revenue_2025_eur_m": 357,
            "published_growth_pct": -2.8,
            "constant_fx_growth_pct": -2.8,
        },
        {
            "segment": "Europe excluding France",
            "type": "Geography",
            "revenue_2026_eur_m": 538,
            "revenue_2025_eur_m": 501,
            "published_growth_pct": 7.6,
            "constant_fx_growth_pct": 9.7,
        },
        {
            "segment": "Total Europe",
            "type": "Geography",
            "revenue_2026_eur_m": 885,
            "revenue_2025_eur_m": 857,
            "published_growth_pct": 3.2,
            "constant_fx_growth_pct": 4.5,
        },
        {
            "segment": "Japan",
            "type": "Geography",
            "revenue_2026_eur_m": 404,
            "revenue_2025_eur_m": 421,
            "published_growth_pct": -3.9,
            "constant_fx_growth_pct": 9.6,
        },
        {
            "segment": "Asia-Pacific excluding Japan",
            "type": "Geography",
            "revenue_2026_eur_m": 1881,
            "revenue_2025_eur_m": 1971,
            "published_growth_pct": -4.6,
            "constant_fx_growth_pct": 2.2,
        },
        {
            "segment": "Total Asia",
            "type": "Geography",
            "revenue_2026_eur_m": 2285,
            "revenue_2025_eur_m": 2392,
            "published_growth_pct": -4.5,
            "constant_fx_growth_pct": 3.5,
        },
        {
            "segment": "Americas",
            "type": "Geography",
            "revenue_2026_eur_m": 739,
            "revenue_2025_eur_m": 695,
            "published_growth_pct": 6.4,
            "constant_fx_growth_pct": 17.2,
        },
        {
            "segment": "Other Middle East",
            "type": "Geography",
            "revenue_2026_eur_m": 160,
            "revenue_2025_eur_m": 185,
            "published_growth_pct": -13.4,
            "constant_fx_growth_pct": -5.9,
        },
        {
            "segment": "Total",
            "type": "Geography",
            "revenue_2026_eur_m": 4070,
            "revenue_2025_eur_m": 4129,
            "published_growth_pct": -1.4,
            "constant_fx_growth_pct": 5.6,
        },
        {
            "segment": "Leather Goods and Saddlery",
            "type": "Sector",
            "revenue_2026_eur_m": 1849,
            "revenue_2025_eur_m": 1813,
            "published_growth_pct": 2.0,
            "constant_fx_growth_pct": 9.4,
        },
        {
            "segment": "Ready-to-wear and Accessories",
            "type": "Sector",
            "revenue_2026_eur_m": 1076,
            "revenue_2025_eur_m": 1149,
            "published_growth_pct": -6.4,
            "constant_fx_growth_pct": 0.4,
        },
        {
            "segment": "Silk and Textiles",
            "type": "Sector",
            "revenue_2026_eur_m": 257,
            "revenue_2025_eur_m": 256,
            "published_growth_pct": 0.4,
            "constant_fx_growth_pct": 7.8,
        },
        {
            "segment": "Other Hermès sectors",
            "type": "Sector",
            "revenue_2026_eur_m": 540,
            "revenue_2025_eur_m": 544,
            "published_growth_pct": -0.6,
            "constant_fx_growth_pct": 6.8,
        },
        {
            "segment": "Perfume and Beauty",
            "type": "Sector",
            "revenue_2026_eur_m": 127,
            "revenue_2025_eur_m": 129,
            "published_growth_pct": -2.2,
            "constant_fx_growth_pct": 0.2,
        },
        {
            "segment": "Watches",
            "type": "Sector",
            "revenue_2026_eur_m": 135,
            "revenue_2025_eur_m": 151,
            "published_growth_pct": -10.3,
            "constant_fx_growth_pct": -3.7,
        },
        {
            "segment": "Other products",
            "type": "Sector",
            "revenue_2026_eur_m": 86,
            "revenue_2025_eur_m": 87,
            "published_growth_pct": -0.7,
            "constant_fx_growth_pct": 3.6,
        },
        {
            "segment": "Total",
            "type": "Sector",
            "revenue_2026_eur_m": 4070,
            "revenue_2025_eur_m": 4129,
            "published_growth_pct": -1.4,
            "constant_fx_growth_pct": 5.6,
        },
    ]

    return pd.DataFrame(rows)


def _validate_q1_revenue(q1: pd.DataFrame) -> None:
    """Validate the Q1 2026 revenue table."""
    missing_columns = REQUIRED_COLUMNS.difference(q1.columns)

    if missing_columns:
        raise ValueError(
            "Hermès Q1 revenue table is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if q1.empty:
        raise ValueError("Hermès Q1 revenue table is empty.")


def _safe_float(value: Any, default: float = np.nan) -> float:
    """Safely convert a value to float."""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _momentum_signal_from_growth(constant_fx_growth_pct: float) -> str:
    """Classify revenue momentum from constant-FX growth."""
    growth = _safe_float(constant_fx_growth_pct)

    if pd.isna(growth):
        return "Not Available"

    if growth >= 8:
        return "Strong"

    if growth >= 3:
        return "Stable"

    if growth >= 0:
        return "Soft"

    return "Weak"


def _severity_from_fx_gap(fx_gap_pct_points: float) -> str:
    """Classify FX gap severity."""
    gap = _safe_float(fx_gap_pct_points)

    if pd.isna(gap):
        return "Not Available"

    if gap >= 8:
        return "Severe FX Drag"

    if gap >= 4:
        return "Material FX Drag"

    if gap >= 1:
        return "Moderate FX Drag"

    if gap > -1:
        return "Limited FX Impact"

    return "FX Tailwind / Offset"


def _interpret_momentum_signal(signal: str) -> str:
    """Return a plain-English interpretation for a momentum signal."""
    if signal == "Strong":
        return "Strong constant-FX growth indicates meaningful underlying demand momentum."

    if signal == "Stable":
        return "Stable constant-FX growth indicates positive but moderate operating momentum."

    if signal == "Soft":
        return "Soft constant-FX growth indicates limited positive momentum."

    if signal == "Weak":
        return "Negative constant-FX growth indicates a weak operating momentum signal."

    return "Momentum signal could not be classified."


def _extract_summary_row(
    summary: pd.DataFrame,
    category: str,
    insight: str,
) -> pd.Series | None:
    """Extract a row from the revenue momentum summary."""
    if summary.empty:
        return None

    rows = summary[
        summary["Category"].eq(category)
        & summary["Insight"].eq(insight)
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


def load_q1_2026_revenue(raw_data_dir: Path) -> pd.DataFrame:
    """Load Hermès Q1 2026 revenue data.

    If the CSV does not exist, this function creates the default project dataset
    and saves it to the raw data directory.
    """
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    q1_path = raw_data_dir / "hermes_q1_2026_revenue.csv"

    if q1_path.exists():
        q1 = pd.read_csv(q1_path)
    else:
        q1 = _default_q1_2026_revenue()
        q1.to_csv(q1_path, index=False)

    _validate_q1_revenue(q1)

    numeric_columns = [
        "revenue_2026_eur_m",
        "revenue_2025_eur_m",
        "published_growth_pct",
        "constant_fx_growth_pct",
    ]

    for column in numeric_columns:
        q1[column] = pd.to_numeric(q1[column], errors="coerce")

    return q1


def build_revenue_momentum_summary(q1: pd.DataFrame) -> pd.DataFrame:
    """Build a concise Q1 2026 revenue momentum summary."""
    _validate_q1_revenue(q1)

    rows: list[dict[str, object]] = []

    for revenue_type in ["Geography", "Sector"]:
        sub = q1[
            (q1["type"] == revenue_type)
            & (~q1["segment"].isin(["Total"]))
        ].copy()

        if sub.empty:
            continue

        strongest = sub.loc[sub["constant_fx_growth_pct"].idxmax()]
        weakest = sub.loc[sub["constant_fx_growth_pct"].idxmin()]
        largest = sub.loc[sub["revenue_2026_eur_m"].idxmax()]

        rows.extend(
            [
                {
                    "Category": revenue_type,
                    "Insight": "Strongest constant-FX growth",
                    "Segment": strongest["segment"],
                    "Value": strongest["constant_fx_growth_pct"],
                    "Unit": "%",
                },
                {
                    "Category": revenue_type,
                    "Insight": "Weakest constant-FX growth",
                    "Segment": weakest["segment"],
                    "Value": weakest["constant_fx_growth_pct"],
                    "Unit": "%",
                },
                {
                    "Category": revenue_type,
                    "Insight": "Largest Q1 2026 revenue contributor",
                    "Segment": largest["segment"],
                    "Value": largest["revenue_2026_eur_m"],
                    "Unit": "EUR millions",
                },
            ]
        )

    total = q1[
        (q1["segment"] == "Total")
        & (q1["type"] == "Geography")
    ]

    if not total.empty:
        t = total.iloc[0]

        rows.extend(
            [
                {
                    "Category": "Total",
                    "Insight": "Q1 2026 revenue",
                    "Segment": "Hermès Group",
                    "Value": t["revenue_2026_eur_m"],
                    "Unit": "EUR millions",
                },
                {
                    "Category": "Total",
                    "Insight": "Reported growth",
                    "Segment": "Hermès Group",
                    "Value": t["published_growth_pct"],
                    "Unit": "%",
                },
                {
                    "Category": "Total",
                    "Insight": "Constant-FX growth",
                    "Segment": "Hermès Group",
                    "Value": t["constant_fx_growth_pct"],
                    "Unit": "%",
                },
            ]
        )

    return pd.DataFrame(rows)


def build_revenue_momentum_metrics(q1: pd.DataFrame) -> pd.DataFrame:
    """Create additional derived Q1 2026 revenue metrics."""
    _validate_q1_revenue(q1)

    metrics = q1.copy()

    metrics["absolute_revenue_change_eur_m"] = (
        metrics["revenue_2026_eur_m"] - metrics["revenue_2025_eur_m"]
    )

    metrics["reported_growth_calculated_pct"] = np.where(
        metrics["revenue_2025_eur_m"].replace(0, np.nan).notna(),
        ((metrics["revenue_2026_eur_m"] / metrics["revenue_2025_eur_m"]) - 1) * 100,
        np.nan,
    )

    total_revenue_by_type = metrics.groupby("type")["revenue_2026_eur_m"].transform(
        lambda x: x[metrics.loc[x.index, "segment"].eq("Total")].iloc[0]
        if metrics.loc[x.index, "segment"].eq("Total").any()
        else x.sum()
    )

    metrics["share_of_2026_revenue_pct"] = np.where(
        total_revenue_by_type.replace(0, np.nan).notna(),
        metrics["revenue_2026_eur_m"] / total_revenue_by_type * 100,
        np.nan,
    )

    metrics["fx_gap_pct_points"] = (
        metrics["constant_fx_growth_pct"] - metrics["published_growth_pct"]
    )

    metrics["Momentum Signal"] = metrics["constant_fx_growth_pct"].apply(
        _momentum_signal_from_growth
    )

    return metrics


def build_fx_impact_summary(q1: pd.DataFrame) -> pd.DataFrame:
    """Summarize the gap between reported and constant-FX growth."""
    _validate_q1_revenue(q1)

    metrics = build_revenue_momentum_metrics(q1)

    rows = []

    for revenue_type in ["Geography", "Sector"]:
        sub = metrics[
            (metrics["type"] == revenue_type)
            & (~metrics["segment"].isin(["Total"]))
        ]

        if sub.empty:
            continue

        largest_fx_gap = sub.loc[sub["fx_gap_pct_points"].idxmax()]
        smallest_fx_gap = sub.loc[sub["fx_gap_pct_points"].idxmin()]

        rows.extend(
            [
                {
                    "Category": revenue_type,
                    "Insight": "Largest FX drag / constant-FX gap",
                    "Segment": largest_fx_gap["segment"],
                    "Value": largest_fx_gap["fx_gap_pct_points"],
                    "Unit": "percentage points",
                },
                {
                    "Category": revenue_type,
                    "Insight": "Smallest FX drag / constant-FX gap",
                    "Segment": smallest_fx_gap["segment"],
                    "Value": smallest_fx_gap["fx_gap_pct_points"],
                    "Unit": "percentage points",
                },
            ]
        )

    total = metrics[
        (metrics["segment"] == "Total")
        & (metrics["type"] == "Geography")
    ]

    if not total.empty:
        t = total.iloc[0]

        rows.append(
            {
                "Category": "Total",
                "Insight": "Group FX gap",
                "Segment": "Hermès Group",
                "Value": t["fx_gap_pct_points"],
                "Unit": "percentage points",
            }
        )

    return pd.DataFrame(rows)


def build_revenue_dashboard_cards(
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Build card-style revenue dashboard rows.

    Parameters
    ----------
    summary:
        Output from build_revenue_momentum_summary().
    metrics:
        Output from build_revenue_momentum_metrics().

    Returns
    -------
    pd.DataFrame
        Dashboard-ready revenue summary cards.
    """
    if summary.empty or metrics.empty:
        return pd.DataFrame(
            [
                {
                    "Card": "Revenue Momentum",
                    "Value": "Not Available",
                    "Display Value": "Not Available",
                    "Numeric Value": np.nan,
                    "Interpretation": "Revenue momentum data was unavailable.",
                }
            ]
        )

    total_row = metrics[
        metrics["segment"].eq("Total")
        & metrics["type"].eq("Geography")
    ]

    if total_row.empty:
        group_revenue = np.nan
        reported_growth = np.nan
        constant_fx_growth = np.nan
        fx_gap = np.nan
        group_signal = "Not Available"
    else:
        t = total_row.iloc[0]
        group_revenue = _safe_float(t["revenue_2026_eur_m"])
        reported_growth = _safe_float(t["published_growth_pct"])
        constant_fx_growth = _safe_float(t["constant_fx_growth_pct"])
        fx_gap = _safe_float(t["fx_gap_pct_points"])
        group_signal = str(t["Momentum Signal"])

    strongest_region = _extract_summary_row(
        summary,
        category="Geography",
        insight="Strongest constant-FX growth",
    )

    strongest_sector = _extract_summary_row(
        summary,
        category="Sector",
        insight="Strongest constant-FX growth",
    )

    largest_geography = _extract_summary_row(
        summary,
        category="Geography",
        insight="Largest Q1 2026 revenue contributor",
    )

    largest_sector = _extract_summary_row(
        summary,
        category="Sector",
        insight="Largest Q1 2026 revenue contributor",
    )

    rows = [
        {
            "Card": "Q1 2026 Revenue",
            "Value": group_revenue,
            "Display Value": f"€{group_revenue:,.0f}m"
            if not pd.isna(group_revenue)
            else "Not Available",
            "Numeric Value": group_revenue,
            "Interpretation": "Total Hermès Group Q1 2026 revenue in EUR millions.",
        },
        {
            "Card": "Reported Growth",
            "Value": reported_growth,
            "Display Value": f"{reported_growth:.1f}%"
            if not pd.isna(reported_growth)
            else "Not Available",
            "Numeric Value": reported_growth,
            "Interpretation": "Reported year-over-year Q1 growth including currency effects.",
        },
        {
            "Card": "Constant-FX Growth",
            "Value": constant_fx_growth,
            "Display Value": f"{constant_fx_growth:.1f}%"
            if not pd.isna(constant_fx_growth)
            else "Not Available",
            "Numeric Value": constant_fx_growth,
            "Interpretation": "Underlying Q1 growth excluding currency effects.",
        },
        {
            "Card": "Group FX Gap",
            "Value": fx_gap,
            "Display Value": f"{fx_gap:.1f} pp"
            if not pd.isna(fx_gap)
            else "Not Available",
            "Numeric Value": fx_gap,
            "Interpretation": "Difference between constant-FX growth and reported growth.",
        },
        {
            "Card": "Group Momentum Signal",
            "Value": group_signal,
            "Display Value": group_signal,
            "Numeric Value": constant_fx_growth,
            "Interpretation": _interpret_momentum_signal(group_signal),
        },
        {
            "Card": "Strongest Region",
            "Value": strongest_region["Segment"] if strongest_region is not None else "Not Available",
            "Display Value": (
                f"{strongest_region['Segment']} "
                f"({float(strongest_region['Value']):.1f}%)"
                if strongest_region is not None
                else "Not Available"
            ),
            "Numeric Value": float(strongest_region["Value"])
            if strongest_region is not None
            else np.nan,
            "Interpretation": "Geographic segment with the strongest constant-FX growth.",
        },
        {
            "Card": "Strongest Sector",
            "Value": strongest_sector["Segment"] if strongest_sector is not None else "Not Available",
            "Display Value": (
                f"{strongest_sector['Segment']} "
                f"({float(strongest_sector['Value']):.1f}%)"
                if strongest_sector is not None
                else "Not Available"
            ),
            "Numeric Value": float(strongest_sector["Value"])
            if strongest_sector is not None
            else np.nan,
            "Interpretation": "Sector with the strongest constant-FX growth.",
        },
        {
            "Card": "Largest Region Contributor",
            "Value": largest_geography["Segment"] if largest_geography is not None else "Not Available",
            "Display Value": (
                f"{largest_geography['Segment']} "
                f"(€{float(largest_geography['Value']):,.0f}m)"
                if largest_geography is not None
                else "Not Available"
            ),
            "Numeric Value": float(largest_geography["Value"])
            if largest_geography is not None
            else np.nan,
            "Interpretation": "Largest Q1 2026 geographic revenue contributor.",
        },
        {
            "Card": "Largest Sector Contributor",
            "Value": largest_sector["Segment"] if largest_sector is not None else "Not Available",
            "Display Value": (
                f"{largest_sector['Segment']} "
                f"(€{float(largest_sector['Value']):,.0f}m)"
                if largest_sector is not None
                else "Not Available"
            ),
            "Numeric Value": float(largest_sector["Value"])
            if largest_sector is not None
            else np.nan,
            "Interpretation": "Largest Q1 2026 sector revenue contributor.",
        },
    ]

    return pd.DataFrame(rows)


def build_revenue_segment_leaderboard(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Build a revenue segment leaderboard for dashboard display.

    Parameters
    ----------
    metrics:
        Output from build_revenue_momentum_metrics().

    Returns
    -------
    pd.DataFrame
        Ranked segment table by type, revenue, growth, share, and momentum.
    """
    if metrics.empty:
        return pd.DataFrame(
            columns=[
                "Rank",
                "Category",
                "Segment",
                "Revenue 2026 EUR m",
                "Revenue 2025 EUR m",
                "Absolute Revenue Change EUR m",
                "Published Growth pct",
                "Constant-FX Growth pct",
                "FX Gap pct points",
                "Share of 2026 Revenue pct",
                "Momentum Signal",
                "Interpretation",
            ]
        )

    required_columns = {
        "segment",
        "type",
        "revenue_2026_eur_m",
        "revenue_2025_eur_m",
        "absolute_revenue_change_eur_m",
        "published_growth_pct",
        "constant_fx_growth_pct",
        "fx_gap_pct_points",
        "share_of_2026_revenue_pct",
        "Momentum Signal",
    }

    missing_columns = required_columns.difference(metrics.columns)

    if missing_columns:
        raise KeyError(
            "Revenue momentum metrics table is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    leaderboard = metrics[~metrics["segment"].eq("Total")].copy()

    leaderboard = leaderboard.sort_values(
        ["type", "revenue_2026_eur_m"],
        ascending=[True, False],
    )

    leaderboard["Rank"] = (
        leaderboard.groupby("type")["revenue_2026_eur_m"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    leaderboard["Interpretation"] = leaderboard["Momentum Signal"].apply(
        _interpret_momentum_signal
    )

    leaderboard = leaderboard.rename(
        columns={
            "type": "Category",
            "segment": "Segment",
            "revenue_2026_eur_m": "Revenue 2026 EUR m",
            "revenue_2025_eur_m": "Revenue 2025 EUR m",
            "absolute_revenue_change_eur_m": "Absolute Revenue Change EUR m",
            "published_growth_pct": "Published Growth pct",
            "constant_fx_growth_pct": "Constant-FX Growth pct",
            "fx_gap_pct_points": "FX Gap pct points",
            "share_of_2026_revenue_pct": "Share of 2026 Revenue pct",
        }
    )

    return (
        leaderboard[
            [
                "Rank",
                "Category",
                "Segment",
                "Revenue 2026 EUR m",
                "Revenue 2025 EUR m",
                "Absolute Revenue Change EUR m",
                "Published Growth pct",
                "Constant-FX Growth pct",
                "FX Gap pct points",
                "Share of 2026 Revenue pct",
                "Momentum Signal",
                "Interpretation",
            ]
        ]
        .sort_values(["Category", "Rank"])
        .reset_index(drop=True)
    )


def build_revenue_fx_dashboard(
    fx_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build a dashboard-friendly FX impact table.

    Parameters
    ----------
    fx_summary:
        Output from build_fx_impact_summary().

    Returns
    -------
    pd.DataFrame
        FX impact dashboard table with severity labels and interpretation.
    """
    if fx_summary.empty:
        return pd.DataFrame(
            columns=[
                "Category",
                "Insight",
                "Segment",
                "FX Gap pct points",
                "Severity",
                "Interpretation",
            ]
        )

    required_columns = {
        "Category",
        "Insight",
        "Segment",
        "Value",
    }

    missing_columns = required_columns.difference(fx_summary.columns)

    if missing_columns:
        raise KeyError(
            "FX impact summary is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    dashboard = fx_summary.copy()

    dashboard["FX Gap pct points"] = pd.to_numeric(
        dashboard["Value"],
        errors="coerce",
    )

    dashboard["Severity"] = dashboard["FX Gap pct points"].apply(
        _severity_from_fx_gap
    )

    dashboard["Interpretation"] = dashboard.apply(
        lambda row: (
            f"{row['Segment']} shows a {row['FX Gap pct points']:.1f} percentage-point "
            f"gap between constant-FX growth and reported growth, classified as "
            f"{row['Severity']}."
        )
        if not pd.isna(row["FX Gap pct points"])
        else "FX impact could not be classified.",
        axis=1,
    )

    return (
        dashboard[
            [
                "Category",
                "Insight",
                "Segment",
                "FX Gap pct points",
                "Severity",
                "Interpretation",
            ]
        ]
        .sort_values("FX Gap pct points", ascending=False)
        .reset_index(drop=True)
    )


def save_revenue_momentum_outputs(
    raw_data_dir: Path,
    table_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, calculate, and save Hermès Q1 2026 revenue output tables."""
    q1 = load_q1_2026_revenue(raw_data_dir)

    summary = build_revenue_momentum_summary(q1)
    metrics = build_revenue_momentum_metrics(q1)
    fx_summary = build_fx_impact_summary(q1)

    revenue_dashboard_cards = build_revenue_dashboard_cards(
        summary=summary,
        metrics=metrics,
    )

    revenue_segment_leaderboard = build_revenue_segment_leaderboard(metrics)

    revenue_fx_dashboard = build_revenue_fx_dashboard(fx_summary)

    table_dir.mkdir(parents=True, exist_ok=True)

    q1.to_csv(
        table_dir / "hermes_q1_2026_revenue.csv",
        index=False,
    )

    summary.to_csv(
        table_dir / "hermes_revenue_momentum_summary.csv",
        index=False,
    )

    metrics.to_csv(
        table_dir / "hermes_revenue_momentum_metrics.csv",
        index=False,
    )

    fx_summary.to_csv(
        table_dir / "hermes_q1_fx_impact_summary.csv",
        index=False,
    )

    revenue_dashboard_cards.to_csv(
        table_dir / "hermes_revenue_dashboard_cards.csv",
        index=False,
    )

    revenue_segment_leaderboard.to_csv(
        table_dir / "hermes_revenue_segment_leaderboard.csv",
        index=False,
    )

    revenue_fx_dashboard.to_csv(
        table_dir / "hermes_revenue_fx_dashboard.csv",
        index=False,
    )

    return q1, summary