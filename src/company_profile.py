"""Company profile tables and official source references for Hermès.

This module converts Hermès configuration dictionaries into clean tabular
outputs for the README, PDF report, Streamlit dashboard, and GitHub project
artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _format_field_name(value: str) -> str:
    """Convert snake_case configuration keys into report-friendly labels."""
    return value.replace("_", " ").title()


def dictionary_to_table(
    data: dict[str, Any],
    key_name: str = "Field",
    value_name: str = "Value",
    prettify_keys: bool = True,
) -> pd.DataFrame:
    """Convert a dictionary into a two-column DataFrame.

    Parameters
    ----------
    data:
        Dictionary to convert.
    key_name:
        Name of the output column containing dictionary keys.
    value_name:
        Name of the output column containing dictionary values.
    prettify_keys:
        If True, converts snake_case keys into title-case labels.

    Returns
    -------
    pd.DataFrame
        Two-column table.
    """
    rows: list[dict[str, Any]] = []

    for key, value in data.items():
        display_key = _format_field_name(str(key)) if prettify_keys else str(key)
        rows.append({key_name: display_key, value_name: value})

    return pd.DataFrame(rows)


def build_source_table(
    source_urls: dict[str, str],
    source_categories: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Build a source-reference table.

    If source categories are provided, each source receives a category label.
    """
    category_lookup: dict[str, str] = {}

    if source_categories:
        for category, source_names in source_categories.items():
            for source_name in source_names:
                category_lookup[source_name] = category

    rows: list[dict[str, str]] = []

    for source, url in source_urls.items():
        rows.append(
            {
                "Category": category_lookup.get(source, "Other"),
                "Source": source,
                "URL": url,
            }
        )

    return pd.DataFrame(rows)


def build_strategy_pillars_table(strategy_pillars: list[str]) -> pd.DataFrame:
    """Build a strategy-pillars table."""
    return pd.DataFrame(
        {
            "Pillar Number": range(1, len(strategy_pillars) + 1),
            "Strategy Pillar": strategy_pillars,
        }
    )


def build_strategy_score_table(
    strategy_scores: dict[str, int | float],
    qualitative_scores: dict[str, int | float],
) -> pd.DataFrame:
    """Build a combined strategic-moat and qualitative-signal score table."""
    rows: list[dict[str, Any]] = []

    for category, scores in [
        ("Strategic Moat", strategy_scores),
        ("Qualitative Signal", qualitative_scores),
    ]:
        for factor, score in scores.items():
            score_float = float(score)
            rows.append(
                {
                    "Category": category,
                    "Factor": factor,
                    "Score_1_to_5": score_float,
                    "Score_0_to_100": score_float * 20,
                }
            )

    return pd.DataFrame(rows)


def build_company_profile_summary(
    share_info: dict[str, Any],
    business_model_factors: dict[str, Any],
    strategy_pillars: list[str],
) -> pd.DataFrame:
    """Build a compact high-level company profile table."""
    rows = [
        {
            "Section": "Listing",
            "Field": "Primary Ticker",
            "Value": share_info.get("yahoo_finance", "RMS.PA"),
        },
        {
            "Section": "Listing",
            "Field": "Exchange",
            "Value": share_info.get("exchange", "Euronext Paris"),
        },
        {
            "Section": "Listing",
            "Field": "ISIN",
            "Value": share_info.get("isin", "FR0000052292"),
        },
        {
            "Section": "Listing",
            "Field": "U.S. OTC Reference",
            "Value": share_info.get("us_otc", "HESAY"),
        },
        {
            "Section": "Business Model",
            "Field": "Ownership",
            "Value": business_model_factors.get("ownership", ""),
        },
        {
            "Section": "Business Model",
            "Field": "Model",
            "Value": business_model_factors.get("model", ""),
        },
        {
            "Section": "Business Model",
            "Field": "Distribution",
            "Value": business_model_factors.get("distribution", ""),
        },
        {
            "Section": "Strategy",
            "Field": "Strategy Pillars",
            "Value": "; ".join(strategy_pillars),
        },
    ]

    return pd.DataFrame(rows)


def build_dashboard_company_cards(
    share_info: dict[str, Any],
    business_model_factors: dict[str, Any],
    strategy_pillars: list[str],
) -> pd.DataFrame:
    """Build compact company profile cards for the Streamlit dashboard.

    This table is designed for dashboard display rather than the PDF report.
    It highlights the most important listing, business model, and strategy facts.
    """
    rows = [
        {
            "Card": "Primary Ticker",
            "Value": share_info.get("yahoo_finance", "RMS.PA"),
            "Description": "Primary Yahoo Finance ticker used by the pipeline.",
        },
        {
            "Card": "Exchange",
            "Value": share_info.get("exchange", "Euronext Paris"),
            "Description": "Primary listing venue.",
        },
        {
            "Card": "ISIN",
            "Value": share_info.get("isin", "FR0000052292"),
            "Description": "International securities identifier.",
        },
        {
            "Card": "U.S. OTC Reference",
            "Value": share_info.get("us_otc", "HESAY"),
            "Description": "U.S. over-the-counter reference ticker.",
        },
        {
            "Card": "Ownership",
            "Value": business_model_factors.get("ownership", ""),
            "Description": "High-level ownership profile.",
        },
        {
            "Card": "Business Model",
            "Value": business_model_factors.get("model", ""),
            "Description": "Core operating model.",
        },
        {
            "Card": "Distribution",
            "Value": business_model_factors.get("distribution", ""),
            "Description": "Distribution model and channel strategy.",
        },
        {
            "Card": "Strategy Pillars",
            "Value": "; ".join(strategy_pillars),
            "Description": "Official strategy themes summarized for the dashboard.",
        },
    ]

    return pd.DataFrame(rows)


def save_company_profile_outputs(
    table_dir: Path,
    source_urls: dict[str, str],
    share_info: dict[str, Any],
    shareholder_guide: dict[str, Any],
    business_model_factors: dict[str, Any],
    strategy_pillars: list[str],
    strategic_moat_score: dict[str, int | float],
    qualitative_signal_weights: dict[str, int | float],
    source_categories: dict[str, list[str]] | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create and save Hermès company-profile output tables.

    Parameters
    ----------
    table_dir:
        Output directory for CSV tables.
    source_urls:
        Dictionary of source names and URLs.
    share_info:
        Share-identification and listing information.
    shareholder_guide:
        Shareholder administration information.
    business_model_factors:
        Official Hermès business-model factors.
    strategy_pillars:
        List of Hermès strategy pillars.
    strategic_moat_score:
        Qualitative moat scores on a 1-to-5 scale.
    qualitative_signal_weights:
        Qualitative signal scores on a 1-to-5 scale.
    source_categories:
        Optional mapping of source categories to source names.

    Returns
    -------
    tuple[pd.DataFrame, ...]
        Sources, share reference, shareholder reference, business model,
        strategy pillars, strategy score, and company profile summary tables.

    Notes
    -----
    The dashboard company cards table is saved as a CSV but is not returned in
    the tuple. This keeps existing main.py unpacking logic backward-compatible.
    """
    table_dir.mkdir(parents=True, exist_ok=True)

    sources = build_source_table(
        source_urls=source_urls,
        source_categories=source_categories,
    )

    share_reference = dictionary_to_table(
        share_info,
        key_name="Field",
        value_name="Value",
    )

    shareholder_reference = dictionary_to_table(
        shareholder_guide,
        key_name="Field",
        value_name="Value",
    )

    business_model = dictionary_to_table(
        business_model_factors,
        key_name="Business Model Factor",
        value_name="Description",
    )

    pillars = build_strategy_pillars_table(strategy_pillars)

    strategy_scores = build_strategy_score_table(
        strategy_scores=strategic_moat_score,
        qualitative_scores=qualitative_signal_weights,
    )

    company_profile_summary = build_company_profile_summary(
        share_info=share_info,
        business_model_factors=business_model_factors,
        strategy_pillars=strategy_pillars,
    )

    dashboard_company_cards = build_dashboard_company_cards(
        share_info=share_info,
        business_model_factors=business_model_factors,
        strategy_pillars=strategy_pillars,
    )

    sources.to_csv(table_dir / "hermes_official_sources.csv", index=False)
    share_reference.to_csv(table_dir / "hermes_share_reference.csv", index=False)

    shareholder_reference.to_csv(
        table_dir / "hermes_shareholder_reference.csv",
        index=False,
    )

    business_model.to_csv(
        table_dir / "hermes_business_model_factors.csv",
        index=False,
    )

    pillars.to_csv(table_dir / "hermes_strategy_pillars.csv", index=False)

    strategy_scores.to_csv(
        table_dir / "hermes_strategy_qualitative_score.csv",
        index=False,
    )

    company_profile_summary.to_csv(
        table_dir / "hermes_company_profile_summary.csv",
        index=False,
    )

    dashboard_company_cards.to_csv(
        table_dir / "hermes_dashboard_company_cards.csv",
        index=False,
    )

    return (
        sources,
        share_reference,
        shareholder_reference,
        business_model,
        pillars,
        strategy_scores,
        company_profile_summary,
    )