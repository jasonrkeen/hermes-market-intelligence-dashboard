"""Recommendation framework for the Hermès market intelligence report.

This module converts the Hermès Luxury Intelligence Score, valuation signal,
news flow, geopolitical risk posture, and market intelligence dashboard into a
formal recommendation-style output.

The recommendation is rules-based and intended for educational market
intelligence use. It is not investment advice.

The module also creates dashboard-ready recommendation cards, component
breakdowns, and action-summary tables for Streamlit and other interactive
reporting outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import RECOMMENDATION_LABELS, RECOMMENDATION_THRESHOLDS


def _safe_float(value: Any, default: float = np.nan) -> float:
    """Safely convert a value to float."""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _threshold(name: str, default: float | int) -> float | int:
    """Retrieve a recommendation threshold from settings.py."""
    return RECOMMENDATION_THRESHOLDS.get(name, default)


def _label(name: str, default: str) -> str:
    """Retrieve a recommendation label from settings.py."""
    return RECOMMENDATION_LABELS.get(name, default)


def _get_intelligence_score(
    intelligence_score: pd.DataFrame,
    score_label: str = "Hermès Luxury Intelligence Score",
) -> float:
    """Extract final intelligence score."""
    if intelligence_score is None or intelligence_score.empty:
        return np.nan

    if score_label not in intelligence_score.index:
        return np.nan

    if "Score" not in intelligence_score.columns:
        return np.nan

    return _safe_float(intelligence_score.loc[score_label, "Score"])


def _get_valuation_signal(valuation_signal: pd.DataFrame | None) -> tuple[str, float]:
    """Extract valuation signal and implied upside/downside."""
    if valuation_signal is None or valuation_signal.empty:
        return "Not Available", np.nan

    signal_rows = valuation_signal[valuation_signal["Metric"].eq("Valuation Signal")]
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


def _get_market_intelligence_view(
    market_summary: pd.DataFrame | None,
) -> str:
    """Extract overall market intelligence view."""
    if market_summary is None or market_summary.empty:
        return "Not Available"

    rows = market_summary[
        market_summary["Section"].eq("Overall Market Intelligence View")
    ]

    if rows.empty:
        return "Not Available"

    return str(rows.iloc[0]["Summary"])


def _get_news_tone(news_summary: pd.DataFrame | None) -> str:
    """Extract rules-based news tone."""
    if news_summary is None or news_summary.empty:
        return "Not Available"

    rows = news_summary[news_summary["Metric"].eq("Average News Score")]

    if rows.empty:
        return "Not Available"

    return str(rows.iloc[0]["Interpretation"])


def _get_geopolitical_risk_posture(
    geopolitical_summary: pd.DataFrame | None,
) -> str:
    """Extract geopolitical risk posture."""
    if geopolitical_summary is None or geopolitical_summary.empty:
        return "Not Available"

    rows = geopolitical_summary[geopolitical_summary["Metric"].eq("Risk Posture")]

    if rows.empty:
        return "Not Available"

    return str(rows.iloc[0]["Value"])


def _score_intelligence_component(score: float) -> int:
    """Convert intelligence score into recommendation points."""
    if pd.isna(score):
        return 0

    if score >= 75:
        return 2
    if score >= 60:
        return 1
    if score >= 45:
        return 0
    return -2


def _score_valuation_component(
    valuation_signal: str,
    implied_upside: float,
) -> int:
    """Convert valuation signal and upside/downside into recommendation points."""
    supportive_upside = float(_threshold("supportive_upside", 0.15))
    moderate_upside = float(_threshold("moderate_upside", 0.05))
    stretched_downside = float(_threshold("stretched_downside", -0.10))
    severe_downside = float(_threshold("severe_downside", -0.15))

    signal_lower = valuation_signal.lower()

    if not pd.isna(implied_upside):
        if implied_upside >= supportive_upside:
            return 2
        if implied_upside >= moderate_upside:
            return 1
        if implied_upside <= severe_downside:
            return -2
        if implied_upside <= stretched_downside:
            return -1

    if "supportive" in signal_lower:
        return 1
    if "stretched" in signal_lower:
        return -1

    return 0


def _score_news_component(news_tone: str) -> int:
    """Convert news tone into recommendation points."""
    tone = news_tone.lower()

    if "positive" in tone or "catalyst" in tone:
        return 1
    if "risk" in tone or "negative" in tone:
        return -1

    return 0


def _score_geopolitical_component(geopolitical_posture: str) -> int:
    """Convert geopolitical risk posture into recommendation points."""
    posture = geopolitical_posture.lower()

    if "high" in posture:
        return -2
    if "elevated" in posture:
        return -1
    if "moderate" in posture:
        return 0
    if "low" in posture:
        return 1

    return 0


def _score_market_intelligence_component(market_view: str) -> int:
    """Convert market intelligence view into recommendation points."""
    view = market_view.lower()

    if "constructive" in view:
        return 1
    if "cautious" in view or "risk-aware" in view:
        return -1

    return 0


def _recommendation_from_points(points: int) -> str:
    """Convert total points into a formal recommendation."""
    outperform_threshold = int(_threshold("outperform_points", 4))
    underperform_threshold = int(_threshold("underperform_points", -3))

    if points >= outperform_threshold:
        return _label("positive", "Outperform")
    if points <= underperform_threshold:
        return _label("negative", "Underperform")
    return _label("neutral", "Neutral")


def _risk_rating_from_inputs(
    intelligence_score_value: float,
    geopolitical_posture: str,
    implied_upside: float,
) -> str:
    """Create a simple risk rating."""
    severe_downside = float(_threshold("severe_downside", -0.15))

    posture = geopolitical_posture.lower()

    if "high" in posture:
        return "High"

    if "elevated" in posture:
        return "Medium-High"

    if not pd.isna(implied_upside) and implied_upside <= severe_downside:
        return "Medium-High"

    if not pd.isna(intelligence_score_value) and intelligence_score_value < 45:
        return "High"

    if not pd.isna(intelligence_score_value) and intelligence_score_value >= 75:
        return "Medium"

    return "Medium"


def _conviction_from_points(points: int) -> str:
    """Create a simple conviction label from recommendation score."""
    abs_points = abs(points)

    if abs_points >= 5:
        return "High"
    if abs_points >= 3:
        return "Medium"
    return "Low"


def _classify_points_direction(points: int | float) -> str:
    """Classify recommendation points as positive, neutral, or negative."""
    points_value = _safe_float(points, default=0.0)

    if points_value > 0:
        return "Positive Contributor"

    if points_value < 0:
        return "Negative Contributor"

    return "Neutral Contributor"


def _extract_framework_input(
    recommendation_framework: pd.DataFrame,
    component: str,
    default: Any = "Not Available",
) -> Any:
    """Extract a component input value from the recommendation framework."""
    if recommendation_framework.empty or "Component" not in recommendation_framework.columns:
        return default

    rows = recommendation_framework[
        recommendation_framework["Component"].eq(component)
    ]

    if rows.empty:
        return default

    return rows.iloc[0].get("Input", default)


def _extract_framework_points(
    recommendation_framework: pd.DataFrame,
    component: str,
    default: int = 0,
) -> int:
    """Extract a component points value from the recommendation framework."""
    if recommendation_framework.empty or "Component" not in recommendation_framework.columns:
        return default

    rows = recommendation_framework[
        recommendation_framework["Component"].eq(component)
    ]

    if rows.empty:
        return default

    return int(_safe_float(rows.iloc[0].get("Points", default), default=float(default)))


def build_recommendation_framework(
    intelligence_score: pd.DataFrame,
    valuation_signal: pd.DataFrame | None = None,
    market_summary: pd.DataFrame | None = None,
    news_summary: pd.DataFrame | None = None,
    geopolitical_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a formal recommendation framework.

    Parameters
    ----------
    intelligence_score:
        Hermès Luxury Intelligence Score table.
    valuation_signal:
        Valuation signal table from valuation.py.
    market_summary:
        Market intelligence summary table.
    news_summary:
        News intelligence summary table.
    geopolitical_summary:
        Geopolitical risk summary table.

    Returns
    -------
    pd.DataFrame
        Recommendation framework with component scores and final recommendation.
    """
    intelligence_score_value = _get_intelligence_score(intelligence_score)
    valuation_signal_label, implied_upside = _get_valuation_signal(valuation_signal)
    market_view = _get_market_intelligence_view(market_summary)
    news_tone = _get_news_tone(news_summary)
    geopolitical_posture = _get_geopolitical_risk_posture(geopolitical_summary)

    intelligence_points = _score_intelligence_component(intelligence_score_value)

    valuation_points = _score_valuation_component(
        valuation_signal=valuation_signal_label,
        implied_upside=implied_upside,
    )

    market_points = _score_market_intelligence_component(market_view)
    news_points = _score_news_component(news_tone)
    geopolitical_points = _score_geopolitical_component(geopolitical_posture)

    total_points = (
        intelligence_points
        + valuation_points
        + market_points
        + news_points
        + geopolitical_points
    )

    recommendation = _recommendation_from_points(total_points)

    risk_rating = _risk_rating_from_inputs(
        intelligence_score_value=intelligence_score_value,
        geopolitical_posture=geopolitical_posture,
        implied_upside=implied_upside,
    )

    conviction = _conviction_from_points(total_points)

    rows = [
        {
            "Component": "Hermès Luxury Intelligence Score",
            "Input": intelligence_score_value,
            "Points": intelligence_points,
            "Scoring Role": "Scoring Component",
            "Interpretation": (
                "Measures combined return, risk, scenario, moat, and "
                "revenue-momentum support."
            ),
        },
        {
            "Component": "Valuation Signal",
            "Input": valuation_signal_label,
            "Points": valuation_points,
            "Scoring Role": "Scoring Component",
            "Interpretation": "Compares fair value range against current market price.",
        },
        {
            "Component": "Implied Upside / Downside",
            "Input": implied_upside,
            "Points": 0,
            "Scoring Role": "Supporting Evidence",
            "Interpretation": (
                "Supporting evidence behind the valuation score. The valuation "
                "points are assigned once through the Valuation Signal component."
            ),
        },
        {
            "Component": "Market Intelligence View",
            "Input": market_view,
            "Points": market_points,
            "Scoring Role": "Scoring Component",
            "Interpretation": "Summarizes dashboard-level market intelligence conditions.",
        },
        {
            "Component": "News Flow Tone",
            "Input": news_tone,
            "Points": news_points,
            "Scoring Role": "Scoring Component",
            "Interpretation": "Rules-based classification of recent news flow.",
        },
        {
            "Component": "Geopolitical Risk Posture",
            "Input": geopolitical_posture,
            "Points": geopolitical_points,
            "Scoring Role": "Scoring Component",
            "Interpretation": (
                "Captures geopolitical pressure from tourism, FX, China, trade, "
                "and regional risk."
            ),
        },
        {
            "Component": "Total Recommendation Points",
            "Input": total_points,
            "Points": total_points,
            "Scoring Role": "Final Output",
            "Interpretation": "Aggregate score used to assign recommendation category.",
        },
        {
            "Component": "Final Recommendation",
            "Input": recommendation,
            "Points": total_points,
            "Scoring Role": "Final Output",
            "Interpretation": "Rules-based recommendation classification.",
        },
        {
            "Component": "Risk Rating",
            "Input": risk_rating,
            "Points": total_points,
            "Scoring Role": "Final Output",
            "Interpretation": (
                "Risk rating based on score quality, valuation downside, and "
                "geopolitical posture."
            ),
        },
        {
            "Component": "Conviction",
            "Input": conviction,
            "Points": total_points,
            "Scoring Role": "Final Output",
            "Interpretation": (
                "Conviction level based on the strength of the aggregate "
                "recommendation score."
            ),
        },
    ]

    return pd.DataFrame(rows)


def build_recommendation_summary(
    recommendation_framework: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact recommendation summary table."""
    if recommendation_framework.empty:
        return pd.DataFrame(
            [
                {
                    "Metric": "Final Recommendation",
                    "Value": "Not Available",
                    "Interpretation": "Recommendation framework was unavailable.",
                }
            ]
        )

    def get_input(component: str, default: Any = "Not Available") -> Any:
        rows = recommendation_framework[
            recommendation_framework["Component"].eq(component)
        ]

        if rows.empty:
            return default

        return rows.iloc[0]["Input"]

    recommendation = get_input("Final Recommendation")
    risk_rating = get_input("Risk Rating")
    conviction = get_input("Conviction")
    total_points = get_input("Total Recommendation Points", default=np.nan)

    if recommendation == _label("positive", "Outperform"):
        interpretation = (
            "The framework indicates favorable support from quality, valuation, "
            "market intelligence, or news/risk conditions."
        )
    elif recommendation == _label("negative", "Underperform"):
        interpretation = (
            "The framework indicates that risk, valuation, or external pressure "
            "outweighs current support."
        )
    else:
        interpretation = (
            "The framework indicates a balanced setup where business quality is "
            "recognized but valuation and risk factors limit conviction."
        )

    rows = [
        {
            "Metric": "Final Recommendation",
            "Value": recommendation,
            "Interpretation": interpretation,
        },
        {
            "Metric": "Risk Rating",
            "Value": risk_rating,
            "Interpretation": (
                "Risk rating based on valuation, geopolitical posture, and model score."
            ),
        },
        {
            "Metric": "Conviction",
            "Value": conviction,
            "Interpretation": (
                "Conviction level based on aggregate recommendation points."
            ),
        },
        {
            "Metric": "Total Recommendation Points",
            "Value": total_points,
            "Interpretation": "Aggregate recommendation score from all components.",
        },
    ]

    return pd.DataFrame(rows)


def build_recommendation_rules_table() -> pd.DataFrame:
    """Document the recommendation rules used by the model."""
    outperform_threshold = int(_threshold("outperform_points", 4))
    underperform_threshold = int(_threshold("underperform_points", -3))
    supportive_upside = float(_threshold("supportive_upside", 0.15))
    moderate_upside = float(_threshold("moderate_upside", 0.05))
    stretched_downside = float(_threshold("stretched_downside", -0.10))
    severe_downside = float(_threshold("severe_downside", -0.15))

    rows = [
        {
            "Rule Area": "Final Recommendation",
            "Rule": f"Total points >= {outperform_threshold}",
            "Output": _label("positive", "Outperform"),
        },
        {
            "Rule Area": "Final Recommendation",
            "Rule": (
                f"Total points between {underperform_threshold + 1} "
                f"and {outperform_threshold - 1}"
            ),
            "Output": _label("neutral", "Neutral"),
        },
        {
            "Rule Area": "Final Recommendation",
            "Rule": f"Total points <= {underperform_threshold}",
            "Output": _label("negative", "Underperform"),
        },
        {
            "Rule Area": "Valuation",
            "Rule": f"Implied upside >= {supportive_upside:.0%}",
            "Output": "+2 points",
        },
        {
            "Rule Area": "Valuation",
            "Rule": f"Implied upside >= {moderate_upside:.0%}",
            "Output": "+1 point",
        },
        {
            "Rule Area": "Valuation",
            "Rule": f"Implied downside <= {severe_downside:.0%}",
            "Output": "-2 points",
        },
        {
            "Rule Area": "Valuation",
            "Rule": f"Implied downside <= {stretched_downside:.0%}",
            "Output": "-1 point",
        },
        {
            "Rule Area": "Intelligence Score",
            "Rule": "Score >= 75",
            "Output": "+2 points",
        },
        {
            "Rule Area": "Intelligence Score",
            "Rule": "Score 60-74",
            "Output": "+1 point",
        },
        {
            "Rule Area": "Intelligence Score",
            "Rule": "Score < 45",
            "Output": "-2 points",
        },
        {
            "Rule Area": "Geopolitical Risk",
            "Rule": "High risk posture",
            "Output": "-2 points",
        },
        {
            "Rule Area": "Geopolitical Risk",
            "Rule": "Elevated risk posture",
            "Output": "-1 point",
        },
        {
            "Rule Area": "News Flow",
            "Rule": "Positive / catalyst-leaning tone",
            "Output": "+1 point",
        },
        {
            "Rule Area": "News Flow",
            "Rule": "Risk-leaning tone",
            "Output": "-1 point",
        },
    ]

    return pd.DataFrame(rows)


def build_recommendation_dashboard_cards(
    recommendation_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build card-style recommendation summary rows for the dashboard.

    Parameters
    ----------
    recommendation_summary:
        Output from build_recommendation_summary().

    Returns
    -------
    pd.DataFrame
        Dashboard-ready recommendation cards.
    """
    if recommendation_summary.empty:
        return pd.DataFrame(
            [
                {
                    "Card": "Final Recommendation",
                    "Value": "Not Available",
                    "Display Value": "Not Available",
                    "Interpretation": "Recommendation summary was unavailable.",
                }
            ]
        )

    summary_lookup = {
        str(row["Metric"]): row
        for _, row in recommendation_summary.iterrows()
        if "Metric" in recommendation_summary.columns
    }

    card_order = [
        "Final Recommendation",
        "Risk Rating",
        "Conviction",
        "Total Recommendation Points",
    ]

    rows: list[dict[str, Any]] = []

    for card in card_order:
        row = summary_lookup.get(card)

        if row is None:
            value = "Not Available"
            interpretation = "Metric was not available in the recommendation summary."
        else:
            value = row.get("Value", "Not Available")
            interpretation = row.get("Interpretation", "")

        rows.append(
            {
                "Card": card,
                "Value": value,
                "Display Value": str(value),
                "Interpretation": interpretation,
            }
        )

    return pd.DataFrame(rows)


def build_recommendation_component_breakdown(
    recommendation_framework: pd.DataFrame,
) -> pd.DataFrame:
    """Build a clean dashboard component breakdown.

    Parameters
    ----------
    recommendation_framework:
        Output from build_recommendation_framework().

    Returns
    -------
    pd.DataFrame
        Recommendation driver table with scoring role and direction.
    """
    if recommendation_framework.empty:
        return pd.DataFrame(
            columns=[
                "Component",
                "Input",
                "Points",
                "Direction",
                "Scoring Role",
                "Interpretation",
            ]
        )

    required_columns = {
        "Component",
        "Input",
        "Points",
        "Interpretation",
    }

    missing_columns = required_columns.difference(recommendation_framework.columns)

    if missing_columns:
        raise KeyError(
            "Recommendation framework is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    breakdown = recommendation_framework.copy()

    if "Scoring Role" not in breakdown.columns:
        breakdown["Scoring Role"] = "Scoring Component"

    final_output_components = {
        "Total Recommendation Points",
        "Final Recommendation",
        "Risk Rating",
        "Conviction",
    }

    breakdown["Direction"] = breakdown.apply(
        lambda row: "Final Output"
        if row["Component"] in final_output_components
        else _classify_points_direction(row["Points"]),
        axis=1,
    )

    scoring_role_order = {
        "Scoring Component": 0,
        "Supporting Evidence": 1,
        "Final Output": 2,
    }

    breakdown["Scoring Role Sort"] = (
        breakdown["Scoring Role"].map(scoring_role_order).fillna(99)
    )

    return (
        breakdown[
            [
                "Component",
                "Input",
                "Points",
                "Direction",
                "Scoring Role",
                "Interpretation",
                "Scoring Role Sort",
            ]
        ]
        .sort_values(
            ["Scoring Role Sort", "Direction", "Component"],
            ascending=True,
        )
        .drop(columns=["Scoring Role Sort"])
        .reset_index(drop=True)
    )


def build_recommendation_action_summary(
    recommendation_framework: pd.DataFrame,
    recommendation_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build a concise action-style recommendation summary.

    Parameters
    ----------
    recommendation_framework:
        Output from build_recommendation_framework().
    recommendation_summary:
        Output from build_recommendation_summary().

    Returns
    -------
    pd.DataFrame
        Action-summary table for dashboard and README use.
    """
    recommendation = _extract_framework_input(
        recommendation_framework,
        "Final Recommendation",
    )
    risk_rating = _extract_framework_input(
        recommendation_framework,
        "Risk Rating",
    )
    conviction = _extract_framework_input(
        recommendation_framework,
        "Conviction",
    )
    total_points = _extract_framework_input(
        recommendation_framework,
        "Total Recommendation Points",
        default=np.nan,
    )

    if "Scoring Role" in recommendation_framework.columns:
        scoring_components = recommendation_framework[
            recommendation_framework["Scoring Role"].eq("Scoring Component")
        ].copy()
    else:
        scoring_components = recommendation_framework.copy()
        
    positive_components = []
    negative_components = []

    if not scoring_components.empty:
        positive_components = (
            scoring_components[scoring_components["Points"] > 0]["Component"]
            .astype(str)
            .tolist()
        )

        negative_components = (
            scoring_components[scoring_components["Points"] < 0]["Component"]
            .astype(str)
            .tolist()
        )

    positive_text = (
        "; ".join(positive_components)
        if positive_components
        else "No positive scoring contributors."
    )

    negative_text = (
        "; ".join(negative_components)
        if negative_components
        else "No negative scoring contributors."
    )

    if str(recommendation).lower() == str(_label("positive", "Outperform")).lower():
        stance = (
            "Framework leans positive, supported by enough favorable signals to "
            "clear the outperform threshold."
        )
    elif str(recommendation).lower() == str(_label("negative", "Underperform")).lower():
        stance = (
            "Framework leans negative, with risk or valuation pressure outweighing "
            "current support."
        )
    else:
        stance = (
            "Framework remains neutral, recognizing Hermès' quality while valuation, "
            "risk, and external signals limit conviction."
        )

    monitoring_focus_items = []

    if negative_components:
        monitoring_focus_items.extend(negative_components)

    if "Medium-High" in str(risk_rating) or "High" in str(risk_rating):
        monitoring_focus_items.append("Risk Rating")

    if str(conviction).lower() == "low":
        monitoring_focus_items.append("Conviction")

    monitoring_focus = (
        "; ".join(dict.fromkeys(monitoring_focus_items))
        if monitoring_focus_items
        else "Monitor valuation gap, news tone, geopolitical risk, and market intelligence changes."
    )

    rows = [
        {
            "Section": "Recommendation Stance",
            "Value": recommendation,
            "Interpretation": stance,
        },
        {
            "Section": "Risk Posture",
            "Value": risk_rating,
            "Interpretation": (
                "Risk rating combines valuation downside, geopolitical posture, "
                "and intelligence-score quality."
            ),
        },
        {
            "Section": "Conviction Level",
            "Value": conviction,
            "Interpretation": (
                "Conviction is based on the absolute strength of total recommendation points."
            ),
        },
        {
            "Section": "Total Recommendation Points",
            "Value": total_points,
            "Interpretation": "Aggregate score across all active recommendation drivers.",
        },
        {
            "Section": "Key Positive Contributors",
            "Value": positive_text,
            "Interpretation": "Scoring components that improved the recommendation.",
        },
        {
            "Section": "Key Negative Contributors",
            "Value": negative_text,
            "Interpretation": "Scoring components that weakened the recommendation.",
        },
        {
            "Section": "Monitoring Focus",
            "Value": monitoring_focus,
            "Interpretation": (
                "Primary items to monitor before changing the recommendation stance."
            ),
        },
        {
            "Section": "Use Limitation",
            "Value": "Educational market intelligence; not investment advice.",
            "Interpretation": (
                "The framework is rules-based and should be interpreted as a project "
                "output, not a standalone investment recommendation."
            ),
        },
    ]

    return pd.DataFrame(rows)


def save_recommendation_outputs(
    table_dir: Path,
    intelligence_score: pd.DataFrame,
    valuation_signal: pd.DataFrame | None = None,
    market_summary: pd.DataFrame | None = None,
    news_summary: pd.DataFrame | None = None,
    geopolitical_summary: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build and save recommendation output tables."""
    table_dir.mkdir(parents=True, exist_ok=True)

    recommendation_framework = build_recommendation_framework(
        intelligence_score=intelligence_score,
        valuation_signal=valuation_signal,
        market_summary=market_summary,
        news_summary=news_summary,
        geopolitical_summary=geopolitical_summary,
    )

    recommendation_summary = build_recommendation_summary(
        recommendation_framework=recommendation_framework,
    )

    recommendation_rules = build_recommendation_rules_table()

    recommendation_dashboard_cards = build_recommendation_dashboard_cards(
        recommendation_summary=recommendation_summary,
    )

    recommendation_component_breakdown = build_recommendation_component_breakdown(
        recommendation_framework=recommendation_framework,
    )

    recommendation_action_summary = build_recommendation_action_summary(
        recommendation_framework=recommendation_framework,
        recommendation_summary=recommendation_summary,
    )

    recommendation_framework.to_csv(
        table_dir / "hermes_recommendation_framework.csv",
        index=False,
    )

    recommendation_summary.to_csv(
        table_dir / "hermes_recommendation_summary.csv",
        index=False,
    )

    recommendation_rules.to_csv(
        table_dir / "hermes_recommendation_rules.csv",
        index=False,
    )

    recommendation_dashboard_cards.to_csv(
        table_dir / "hermes_recommendation_dashboard_cards.csv",
        index=False,
    )

    recommendation_component_breakdown.to_csv(
        table_dir / "hermes_recommendation_component_breakdown.csv",
        index=False,
    )

    recommendation_action_summary.to_csv(
        table_dir / "hermes_recommendation_action_summary.csv",
        index=False,
    )

    return recommendation_framework, recommendation_summary, recommendation_rules