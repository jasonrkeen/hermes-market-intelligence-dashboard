"""Run-to-run change detection for the Hermès market intelligence pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def load_previous_snapshot(path: Path) -> dict[str, Any] | None:
    """Load the previous run snapshot if it exists.

    Parameters
    ----------
    path:
        Path to the saved JSON snapshot.

    Returns
    -------
    dict | None
        Previous snapshot dictionary, or None if unavailable.
    """
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _optional_string(value: str | None) -> str | None:
    """Return a clean optional string value."""
    if value is None:
        return None

    value_clean = str(value).strip()

    if not value_clean:
        return None

    return value_clean


def _optional_float(value: float | int | str | None) -> float | None:
    """Return a clean optional float value."""
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_snapshot(
    score: float,
    signal: str,
    ann_return: float,
    max_drawdown: float,
    var_breach_rate: float,
    recommendation: str | None = None,
    risk_rating: str | None = None,
    valuation_signal: str | None = None,
    implied_upside: float | None = None,
    geopolitical_posture: str | None = None,
    news_tone: str | None = None,
    market_view: str | None = None,
) -> dict[str, Any]:
    """Build the current run snapshot.

    Parameters
    ----------
    score:
        Hermès Luxury Intelligence Score.
    signal:
        Final model signal.
    ann_return:
        Annualized return for Hermès.
    max_drawdown:
        Maximum drawdown for Hermès.
    var_breach_rate:
        Observed VaR breach rate.
    recommendation:
        Optional final recommendation label.
    risk_rating:
        Optional recommendation risk rating.
    valuation_signal:
        Optional valuation signal label.
    implied_upside:
        Optional implied upside/downside from fair value.
    geopolitical_posture:
        Optional geopolitical risk posture.
    news_tone:
        Optional news intelligence tone.
    market_view:
        Optional overall market intelligence view.

    Returns
    -------
    dict
        Snapshot of the current model run.
    """
    snapshot: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "intelligence_score": round(float(score), 2),
        "signal": str(signal),
        "annualized_return": round(float(ann_return), 4),
        "max_drawdown": round(float(max_drawdown), 4),
        "var_breach_rate": round(float(var_breach_rate), 4),
    }

    optional_strings = {
        "recommendation": recommendation,
        "risk_rating": risk_rating,
        "valuation_signal": valuation_signal,
        "geopolitical_posture": geopolitical_posture,
        "news_tone": news_tone,
        "market_view": market_view,
    }

    for key, value in optional_strings.items():
        clean_value = _optional_string(value)

        if clean_value is not None:
            snapshot[key] = clean_value

    implied_upside_float = _optional_float(implied_upside)

    if implied_upside_float is not None:
        snapshot["implied_upside"] = round(implied_upside_float, 4)

    return snapshot


def _format_numeric_change(
    label: str,
    previous_value: float | int | str | None,
    current_value: float | int | str | None,
    higher_is_better: bool = True,
    precision: int = 4,
) -> str | None:
    """Format a numeric change with directional interpretation."""
    if previous_value is None or current_value is None:
        return None

    try:
        previous_float = float(previous_value)
        current_float = float(current_value)
    except (TypeError, ValueError):
        return None

    if previous_float == current_float:
        return None

    change = current_float - previous_float
    direction = "increased" if change > 0 else "decreased"

    is_improvement = change > 0 if higher_is_better else change < 0
    interpretation = "improved" if is_improvement else "weakened"

    return (
        f"{label} {direction} from {previous_float:.{precision}f} "
        f"to {current_float:.{precision}f} "
        f"({change:+.{precision}f}); this {interpretation} the signal."
    )


def _format_categorical_change(
    label: str,
    previous_value: Any,
    current_value: Any,
) -> str | None:
    """Format a categorical field change."""
    if previous_value is None or current_value is None:
        return None

    if previous_value == current_value:
        return None

    return f"{label} changed from {previous_value} to {current_value}."


def compare_snapshots(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
) -> list[str]:
    """Compare the previous snapshot with the current run.

    Parameters
    ----------
    previous:
        Previous snapshot dictionary.
    current:
        Current snapshot dictionary.

    Returns
    -------
    list[str]
        Human-readable change descriptions.
    """
    if previous is None:
        return ["No previous snapshot found. This run establishes the baseline."]

    changes: list[str] = []

    categorical_checks = [
        ("signal", "Signal"),
        ("recommendation", "Recommendation"),
        ("risk_rating", "Risk rating"),
        ("valuation_signal", "Valuation signal"),
        ("geopolitical_posture", "Geopolitical risk posture"),
        ("news_tone", "News tone"),
        ("market_view", "Market intelligence view"),
    ]

    for key, label in categorical_checks:
        change_text = _format_categorical_change(
            label=label,
            previous_value=previous.get(key),
            current_value=current.get(key),
        )

        if change_text:
            changes.append(change_text)

    numeric_checks = [
        {
            "key": "intelligence_score",
            "label": "Hermès Luxury Intelligence Score",
            "higher_is_better": True,
            "precision": 2,
        },
        {
            "key": "annualized_return",
            "label": "Annualized return",
            "higher_is_better": True,
            "precision": 4,
        },
        {
            "key": "max_drawdown",
            "label": "Maximum drawdown",
            "higher_is_better": True,
            "precision": 4,
        },
        {
            "key": "var_breach_rate",
            "label": "VaR breach rate",
            "higher_is_better": False,
            "precision": 4,
        },
        {
            "key": "implied_upside",
            "label": "Implied upside/downside",
            "higher_is_better": True,
            "precision": 4,
        },
    ]

    for check in numeric_checks:
        change_text = _format_numeric_change(
            label=check["label"],
            previous_value=previous.get(check["key"]),
            current_value=current.get(check["key"]),
            higher_is_better=check["higher_is_better"],
            precision=check["precision"],
        )

        if change_text:
            changes.append(change_text)

    return changes or ["No material changes versus the previous run."]


def build_change_detection_table(changes: list[str]) -> pd.DataFrame:
    """Convert change-detection messages into a dashboard/report table.

    Parameters
    ----------
    changes:
        Human-readable change descriptions from compare_snapshots().

    Returns
    -------
    pd.DataFrame
        Table with change number, classification, and description.
    """
    rows: list[dict[str, Any]] = []

    for index, change in enumerate(changes, start=1):
        change_lower = change.lower()

        if "no material changes" in change_lower:
            classification = "No Change"
        elif "baseline" in change_lower:
            classification = "Baseline"
        elif "weakened" in change_lower:
            classification = "Weakened"
        elif "improved" in change_lower:
            classification = "Improved"
        elif "changed" in change_lower:
            classification = "Changed"
        elif "decreased" in change_lower:
            classification = "Decreased"
        elif "increased" in change_lower:
            classification = "Increased"
        else:
            classification = "Update"

        rows.append(
            {
                "Change #": index,
                "Classification": classification,
                "Description": change,
            }
        )

    return pd.DataFrame(rows)


def save_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    """Save the current run snapshot as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")