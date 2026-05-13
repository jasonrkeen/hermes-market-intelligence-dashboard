"""Luxury intelligence score for the Hermès market intelligence pipeline.

This module combines quantitative market signals, risk analytics, factor-model
support, scenario analysis, official strategic moat variables, and recent revenue
momentum into a composite Hermès Luxury Intelligence Score.

The score measures company quality, market behavior, and risk-adjusted support.
Broader market intelligence inputs such as valuation, news flow, and geopolitical
risk are handled separately in the market intelligence and recommendation modules.

The module also provides dashboard-friendly summary and contribution tables for
interactive reporting.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


TARGET_LABEL = "Hermès International"
FINAL_SCORE_LABEL = "Hermès Luxury Intelligence Score"


def _clip_score(value: float) -> float:
    """Clip a score to the 0-100 range."""
    return float(np.clip(value, 0, 100))


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float safely."""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_metric(
    metrics: pd.DataFrame,
    asset_label: str,
    metric_name: str,
    default: float = 0.0,
) -> float:
    """Safely retrieve a metric value from the performance table."""
    if asset_label not in metrics.index:
        raise KeyError(
            f"Could not find {asset_label!r} in metrics table. "
            f"Available rows: {list(metrics.index)}"
        )

    if metric_name not in metrics.columns:
        raise KeyError(
            f"Could not find {metric_name!r} in metrics table. "
            f"Available columns: {list(metrics.columns)}"
        )

    return _safe_float(metrics.loc[asset_label, metric_name], default=default)


def _get_base_case_return(
    scenario_table: pd.DataFrame,
    fallback_return: float,
) -> float:
    """Retrieve base-case scenario return, with fallback to annualized return."""
    if (
        "Base Case" in scenario_table.index
        and "Estimated Hermès Return" in scenario_table.columns
    ):
        return _safe_float(
            scenario_table.loc["Base Case", "Estimated Hermès Return"],
            default=fallback_return,
        )

    return fallback_return


def _get_average_breach_rate(var_summary: pd.DataFrame) -> float:
    """Calculate the average VaR breach rate across available models."""
    if var_summary.empty or "Breach Rate" not in var_summary.columns:
        return 0.05

    breach_rate = pd.to_numeric(var_summary["Breach Rate"], errors="coerce").mean()

    if pd.isna(breach_rate):
        return 0.05

    return float(breach_rate)


def _normalize_weights(
    components: dict[str, float],
    base_weights: dict[str, float],
) -> dict[str, float]:
    """Keep only available component weights and normalize them to sum to 1."""
    weights = {key: value for key, value in base_weights.items() if key in components}
    weight_sum = sum(weights.values())

    if weight_sum == 0:
        equal_weight = 1 / len(components)
        return {key: equal_weight for key in components}

    return {key: value / weight_sum for key, value in weights.items()}


def _classify_component_score(score: float) -> str:
    """Classify an individual score component."""
    score_float = _safe_float(score, default=0.0)

    if score_float >= 75:
        return "Strong"
    if score_float >= 60:
        return "Supportive"
    if score_float >= 45:
        return "Neutral"
    return "Weak"


def _component_interpretation(component: str, score: float) -> str:
    """Create a plain-English interpretation for a score component."""
    classification = _classify_component_score(score)

    if classification == "Strong":
        return f"{component} is a strong contributor to the composite score."

    if classification == "Supportive":
        return f"{component} is supportive but not the strongest driver."

    if classification == "Neutral":
        return f"{component} is broadly neutral in the composite score."

    return f"{component} is a weaker component and may weigh on the overall score."


def build_intelligence_score(
    metrics: pd.DataFrame,
    factor_r2: float,
    var_summary: pd.DataFrame,
    scenario_table: pd.DataFrame,
    qualitative_score: float | None = None,
    revenue_momentum_score: float | None = None,
    target_label: str = TARGET_LABEL,
) -> pd.DataFrame:
    """Create the composite Hermès Luxury Intelligence Score.

    Parameters
    ----------
    metrics:
        Risk/return metrics table.
    factor_r2:
        Factor-model R-squared.
    var_summary:
        VaR backtest summary table.
    scenario_table:
        Scenario return estimates.
    qualitative_score:
        Optional qualitative moat score on a 0-100 scale.
    revenue_momentum_score:
        Optional recent revenue momentum score on a 0-100 scale.
    target_label:
        Name of the Hermès row in the metrics table.

    Returns
    -------
    pd.DataFrame
        Intelligence score table with component scores, weights, and weighted
        scores.
    """
    ann_return = _get_metric(metrics, target_label, "Annualized Return")
    ann_vol = _get_metric(metrics, target_label, "Annualized Volatility")
    max_drawdown = abs(_get_metric(metrics, target_label, "Max Drawdown"))
    sharpe = _get_metric(metrics, target_label, "Sharpe Ratio")

    factor_r2_clean = _safe_float(factor_r2, default=0.0)
    avg_breach_rate = _get_average_breach_rate(var_summary)
    base_return = _get_base_case_return(scenario_table, fallback_return=ann_return)

    components = {
        "Return Quality": _clip_score(50 + ann_return * 120),
        "Volatility Control": _clip_score(100 - ann_vol * 180),
        "Drawdown Resilience": _clip_score(100 - max_drawdown * 160),
        "Risk-Adjusted Return": _clip_score(50 + sharpe * 18),
        "Factor Explainability": _clip_score(40 + factor_r2_clean * 60),
        "VaR Calibration": _clip_score(100 - abs(avg_breach_rate - 0.05) * 800),
        "Scenario Support": _clip_score(50 + base_return * 120),
    }

    if qualitative_score is not None:
        components["Strategic Moat / Qualitative Signal"] = _clip_score(
            _safe_float(qualitative_score, default=60.0)
        )

    if revenue_momentum_score is not None:
        components["Recent Revenue Momentum"] = _clip_score(
            _safe_float(revenue_momentum_score, default=60.0)
        )

    base_weights = {
        "Return Quality": 0.12,
        "Volatility Control": 0.12,
        "Drawdown Resilience": 0.14,
        "Risk-Adjusted Return": 0.14,
        "Factor Explainability": 0.08,
        "VaR Calibration": 0.08,
        "Scenario Support": 0.12,
        "Strategic Moat / Qualitative Signal": 0.12,
        "Recent Revenue Momentum": 0.08,
    }

    weights = _normalize_weights(components, base_weights)

    total_score = sum(components[key] * weights[key] for key in components)

    rows = []

    for component, score in components.items():
        weight = weights[component]

        rows.append(
            {
                "Component": component,
                "Score": score,
                "Weight": weight,
                "Weight Display": f"{weight:.2%}",
                "Weighted Score": score * weight,
                "Classification": _classify_component_score(score),
                "Interpretation": _component_interpretation(component, score),
            }
        )

    rows.append(
        {
            "Component": FINAL_SCORE_LABEL,
            "Score": total_score,
            "Weight": 1.0,
            "Weight Display": "100.00%",
            "Weighted Score": total_score,
            "Classification": signal_from_score(total_score),
            "Interpretation": (
                "Composite score combining market behavior, risk-adjusted support, "
                "factor explainability, scenario support, strategic quality, and "
                "recent revenue momentum."
            ),
        }
    )

    return pd.DataFrame(rows).set_index("Component")


def signal_from_score(score: float) -> str:
    """Convert the composite score into an investment-style signal label."""
    clean_score = _safe_float(score, default=0.0)

    if clean_score >= 75:
        return "Constructive"
    if clean_score >= 60:
        return "Moderately Constructive"
    if clean_score >= 45:
        return "Neutral"
    return "Cautious"


def summarize_intelligence_score(score_table: pd.DataFrame) -> dict[str, Any]:
    """Create a compact summary of the score table for reports or logs."""
    if FINAL_SCORE_LABEL not in score_table.index:
        raise KeyError(f"Score table must include {FINAL_SCORE_LABEL!r}.")

    final_score = _safe_float(score_table.loc[FINAL_SCORE_LABEL, "Score"])
    signal = signal_from_score(final_score)

    component_table = score_table.drop(index=FINAL_SCORE_LABEL, errors="ignore")

    strongest_component = None
    weakest_component = None

    if not component_table.empty:
        strongest_component = component_table["Score"].idxmax()
        weakest_component = component_table["Score"].idxmin()

    return {
        "final_score": final_score,
        "signal": signal,
        "strongest_component": strongest_component,
        "weakest_component": weakest_component,
    }


def build_intelligence_score_contribution_table(
    score_table: pd.DataFrame,
) -> pd.DataFrame:
    """Build a dashboard-friendly component contribution table.

    Parameters
    ----------
    score_table:
        Score table returned by build_intelligence_score().

    Returns
    -------
    pd.DataFrame
        Component contribution table excluding the final composite score row.
    """
    if score_table.empty:
        return pd.DataFrame(
            columns=[
                "Component",
                "Score",
                "Weight",
                "Weight Display",
                "Weighted Score",
                "Contribution Share",
                "Contribution Share Display",
                "Classification",
                "Interpretation",
            ]
        )

    required_columns = {"Score", "Weight", "Weighted Score"}
    missing_columns = required_columns.difference(score_table.columns)

    if missing_columns:
        raise KeyError(
            "Score table is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    component_table = score_table.drop(index=FINAL_SCORE_LABEL, errors="ignore").copy()

    if component_table.empty:
        return pd.DataFrame(
            columns=[
                "Component",
                "Score",
                "Weight",
                "Weight Display",
                "Weighted Score",
                "Contribution Share",
                "Contribution Share Display",
                "Classification",
                "Interpretation",
            ]
        )

    total_weighted_score = float(component_table["Weighted Score"].sum())

    if total_weighted_score == 0:
        component_table["Contribution Share"] = np.nan
    else:
        component_table["Contribution Share"] = (
            component_table["Weighted Score"] / total_weighted_score
        )

    if "Weight Display" not in component_table.columns:
        component_table["Weight Display"] = component_table["Weight"].apply(
            lambda value: f"{float(value):.2%}"
        )

    if "Classification" not in component_table.columns:
        component_table["Classification"] = component_table["Score"].apply(
            _classify_component_score
        )

    if "Interpretation" not in component_table.columns:
        component_table["Interpretation"] = [
            _component_interpretation(component, row["Score"])
            for component, row in component_table.iterrows()
        ]

    component_table["Contribution Share Display"] = component_table[
        "Contribution Share"
    ].apply(lambda value: f"{float(value):.2%}" if not pd.isna(value) else "")

    component_table = component_table.reset_index()

    return component_table[
        [
            "Component",
            "Score",
            "Weight",
            "Weight Display",
            "Weighted Score",
            "Contribution Share",
            "Contribution Share Display",
            "Classification",
            "Interpretation",
        ]
    ]


def build_intelligence_score_dashboard(
    score_table: pd.DataFrame,
) -> pd.DataFrame:
    """Build dashboard card rows for the Hermès Luxury Intelligence Score.

    Parameters
    ----------
    score_table:
        Score table returned by build_intelligence_score().

    Returns
    -------
    pd.DataFrame
        Card-style summary table for Streamlit.
    """
    summary = summarize_intelligence_score(score_table)

    final_score = _safe_float(summary["final_score"])
    signal = str(summary["signal"])
    strongest_component = summary["strongest_component"]
    weakest_component = summary["weakest_component"]

    component_table = score_table.drop(index=FINAL_SCORE_LABEL, errors="ignore")

    strongest_score = (
        _safe_float(component_table.loc[strongest_component, "Score"])
        if strongest_component in component_table.index
        else np.nan
    )

    weakest_score = (
        _safe_float(component_table.loc[weakest_component, "Score"])
        if weakest_component in component_table.index
        else np.nan
    )

    rows = [
        {
            "Card": "Hermès Luxury Intelligence Score",
            "Value": final_score,
            "Display Value": f"{final_score:.2f} / 100",
            "Interpretation": "Composite score based on market, risk, scenario, moat, and revenue-momentum inputs.",
        },
        {
            "Card": "Signal",
            "Value": signal,
            "Display Value": signal,
            "Interpretation": "Investment-style signal derived from the composite score.",
        },
        {
            "Card": "Strongest Component",
            "Value": strongest_component,
            "Display Value": str(strongest_component),
            "Interpretation": (
                f"Highest-scoring component with a score of {strongest_score:.2f}."
                if not pd.isna(strongest_score)
                else "Highest-scoring component could not be determined."
            ),
        },
        {
            "Card": "Weakest Component",
            "Value": weakest_component,
            "Display Value": str(weakest_component),
            "Interpretation": (
                f"Lowest-scoring component with a score of {weakest_score:.2f}."
                if not pd.isna(weakest_score)
                else "Lowest-scoring component could not be determined."
            ),
        },
        {
            "Card": "Component Count",
            "Value": int(len(component_table)),
            "Display Value": str(int(len(component_table))),
            "Interpretation": "Number of component scores included in the composite score.",
        },
    ]

    return pd.DataFrame(rows)


def build_intelligence_score_summary_table(
    score_table: pd.DataFrame,
) -> pd.DataFrame:
    """Convert the intelligence score summary dictionary into a DataFrame.

    Parameters
    ----------
    score_table:
        Score table returned by build_intelligence_score().

    Returns
    -------
    pd.DataFrame
        Summary table with metric, value, and interpretation fields.
    """
    summary = summarize_intelligence_score(score_table)

    final_score = _safe_float(summary["final_score"])
    signal = str(summary["signal"])

    rows = [
        {
            "Metric": "Final Score",
            "Value": final_score,
            "Interpretation": "Composite Hermès Luxury Intelligence Score.",
        },
        {
            "Metric": "Signal",
            "Value": signal,
            "Interpretation": "Investment-style signal derived from the final score.",
        },
        {
            "Metric": "Strongest Component",
            "Value": summary["strongest_component"],
            "Interpretation": "Component with the highest standalone score.",
        },
        {
            "Metric": "Weakest Component",
            "Value": summary["weakest_component"],
            "Interpretation": "Component with the lowest standalone score.",
        },
    ]

    return pd.DataFrame(rows)