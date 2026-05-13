"""GARCH-style volatility and VaR backtesting for the Hermès market intelligence pipeline.

This module uses an EWMA volatility model as a lightweight GARCH-style
approximation. It captures volatility clustering without requiring compiled
third-party econometrics packages and supports the report's downside-risk,
stress-testing, and market-intelligence sections.

The module also provides dashboard-friendly helper tables for VaR model
interpretation, latest-risk snapshots, and model comparison views.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.risk import kupiec_pof_test


def _normal_var_z_score(confidence: float) -> float:
    """Return the one-tailed normal z-score for supported VaR confidence levels.

    This avoids adding scipy as a dependency. The project currently uses 95% VaR,
    but common levels are included for flexibility.
    """
    z_scores = {
        0.90: 1.2815515655446004,
        0.95: 1.6448536269514722,
        0.975: 1.959963984540054,
        0.99: 2.3263478740408408,
    }

    rounded_confidence = round(float(confidence), 3)

    if rounded_confidence not in z_scores:
        raise ValueError(
            "Unsupported confidence level. Use one of: "
            f"{sorted(z_scores.keys())}. Received: {confidence}"
        )

    return z_scores[rounded_confidence]


def _classify_breach_behavior(
    breach_rate: float,
    expected_breach_rate: float,
) -> str:
    """Classify VaR breach behavior versus the expected breach rate."""
    if pd.isna(breach_rate) or pd.isna(expected_breach_rate):
        return "Not Available"

    lower_band = expected_breach_rate * 0.75
    upper_band = expected_breach_rate * 1.25

    if breach_rate < lower_band:
        return "Conservative"

    if breach_rate > upper_band:
        return "Underestimating Risk"

    return "Close to Expected"


def _interpret_breach_behavior(
    classification: str,
    breach_rate: float,
    expected_breach_rate: float,
) -> str:
    """Create a plain-English interpretation of VaR breach behavior."""
    if classification == "Conservative":
        return (
            "Observed breaches are below the expected rate, suggesting the VaR "
            "model may be conservative over the backtest period."
        )

    if classification == "Underestimating Risk":
        return (
            "Observed breaches are above the expected rate, suggesting the VaR "
            "model may be underestimating downside risk."
        )

    if classification == "Close to Expected":
        return (
            "Observed breaches are broadly in line with the expected breach rate."
        )

    return "Insufficient information to classify VaR breach behavior."


def build_garch_style_var_backtest(
    returns: pd.Series,
    confidence: float = 0.95,
    lambda_: float = 0.94,
    historical_window: int = 252,
    min_periods: int = 60,
) -> pd.DataFrame:
    """Build an EWMA volatility and historical VaR backtest.

    Parameters
    ----------
    returns:
        Daily return series.
    confidence:
        VaR confidence level. For 95% VaR, the left-tail threshold is 5%.
    lambda_:
        EWMA decay factor. A value of 0.94 is commonly used for daily returns.
    historical_window:
        Rolling window length for historical simulation VaR.
    min_periods:
        Minimum observations required before calculating historical VaR.

    Returns
    -------
    pd.DataFrame
        Backtest table containing returns, EWMA volatility, VaR estimates,
        expected shortfall estimates, and breach indicators.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")

    if not 0 < lambda_ < 1:
        raise ValueError("lambda_ must be between 0 and 1.")

    if historical_window <= 0:
        raise ValueError("historical_window must be greater than zero.")

    if min_periods <= 0:
        raise ValueError("min_periods must be greater than zero.")

    r = returns.dropna().replace([np.inf, -np.inf], np.nan).dropna().copy()

    if r.empty:
        raise ValueError("Return series is empty.")

    z_score = _normal_var_z_score(confidence)

    ewma_variance = r.pow(2).ewm(
        alpha=1 - lambda_,
        adjust=False,
    ).mean()

    ewma_volatility = np.sqrt(ewma_variance)

    garch_style_var = -z_score * ewma_volatility.shift(1)

    historical_var = (
        r.rolling(
            window=historical_window,
            min_periods=min_periods,
        )
        .quantile(1 - confidence)
        .shift(1)
    )

    rolling_expected_shortfall = (
        r.rolling(
            window=historical_window,
            min_periods=min_periods,
        )
        .apply(
            lambda x: float(np.mean(x[x <= np.quantile(x, 1 - confidence)]))
            if np.any(x <= np.quantile(x, 1 - confidence))
            else np.nan,
            raw=True,
        )
        .shift(1)
    )

    out = pd.DataFrame(
        {
            "return": r,
            "ewma_volatility": ewma_volatility,
            "annualized_ewma_volatility": ewma_volatility * np.sqrt(252),
            "garch_style_var": garch_style_var,
            "historical_var": historical_var,
            "historical_expected_shortfall": rolling_expected_shortfall,
        }
    ).dropna()

    out["garch_style_breach"] = out["return"] < out["garch_style_var"]
    out["historical_breach"] = out["return"] < out["historical_var"]

    # Keep legacy column names for compatibility with existing chart/report code.
    confidence_label = int(confidence * 100)
    out[f"garch_style_var_{confidence_label}"] = out["garch_style_var"]
    out[f"historical_var_{confidence_label}"] = out["historical_var"]

    return out


def summarize_var_backtest(
    backtest: pd.DataFrame,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Summarize VaR backtest performance.

    Parameters
    ----------
    backtest:
        VaR backtest output from build_garch_style_var_backtest.
    confidence:
        VaR confidence level. The expected breach rate is 1 - confidence.

    Returns
    -------
    pd.DataFrame
        Summary table with breach count, breach rate, Kupiec statistic,
        average VaR, and average expected shortfall where available.
    """
    if backtest.empty:
        raise ValueError("Backtest DataFrame is empty.")

    expected_breach_rate = 1 - confidence
    n_observations = len(backtest)

    model_specs: list[dict[str, Any]] = [
        {
            "Model": "GARCH-Style EWMA VaR",
            "Breach Column": "garch_style_breach",
            "VaR Column": "garch_style_var",
            "Expected Shortfall Column": None,
        },
        {
            "Model": "Historical Simulation VaR",
            "Breach Column": "historical_breach",
            "VaR Column": "historical_var",
            "Expected Shortfall Column": "historical_expected_shortfall",
        },
    ]

    rows: list[dict[str, Any]] = []

    for spec in model_specs:
        breach_col = spec["Breach Column"]
        var_col = spec["VaR Column"]
        es_col = spec["Expected Shortfall Column"]

        if breach_col not in backtest.columns:
            raise KeyError(f"Backtest DataFrame is missing column: {breach_col}")

        breaches = int(backtest[breach_col].sum())

        test = kupiec_pof_test(
            breaches=breaches,
            observations=n_observations,
            expected_breach_rate=expected_breach_rate,
        )

        avg_var = (
            float(backtest[var_col].mean())
            if var_col in backtest.columns
            else np.nan
        )

        avg_expected_shortfall = (
            float(backtest[es_col].mean())
            if es_col and es_col in backtest.columns
            else np.nan
        )

        rows.append(
            {
                "Model": spec["Model"],
                "Observations": n_observations,
                "VaR Breaches": breaches,
                "Expected Breach Rate": expected_breach_rate,
                "Breach Rate": test["breach_rate"],
                "Kupiec LR Statistic": test["lr_stat"],
                "Average VaR": avg_var,
                "Average Expected Shortfall": avg_expected_shortfall,
            }
        )

    return pd.DataFrame(rows).set_index("Model")


def build_var_dashboard_summary(
    var_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build a dashboard-friendly VaR summary table.

    Parameters
    ----------
    var_summary:
        Summary table returned by summarize_var_backtest().

    Returns
    -------
    pd.DataFrame
        Dashboard table with model-level VaR interpretation.
    """
    if var_summary.empty:
        return pd.DataFrame(
            columns=[
                "Model",
                "Observations",
                "VaR Breaches",
                "Expected Breach Rate",
                "Breach Rate",
                "Average VaR",
                "Average Expected Shortfall",
                "Kupiec LR Statistic",
                "Classification",
                "Interpretation",
            ]
        )

    required_columns = {
        "Observations",
        "VaR Breaches",
        "Expected Breach Rate",
        "Breach Rate",
        "Kupiec LR Statistic",
        "Average VaR",
        "Average Expected Shortfall",
    }

    missing_columns = required_columns.difference(var_summary.columns)

    if missing_columns:
        raise KeyError(
            "VaR summary table is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    rows: list[dict[str, Any]] = []

    for model_name, row in var_summary.iterrows():
        breach_rate = float(row["Breach Rate"])
        expected_breach_rate = float(row["Expected Breach Rate"])

        classification = _classify_breach_behavior(
            breach_rate=breach_rate,
            expected_breach_rate=expected_breach_rate,
        )

        rows.append(
            {
                "Model": model_name,
                "Observations": int(row["Observations"]),
                "VaR Breaches": int(row["VaR Breaches"]),
                "Expected Breach Rate": expected_breach_rate,
                "Breach Rate": breach_rate,
                "Average VaR": float(row["Average VaR"]),
                "Average Expected Shortfall": (
                    float(row["Average Expected Shortfall"])
                    if not pd.isna(row["Average Expected Shortfall"])
                    else np.nan
                ),
                "Kupiec LR Statistic": float(row["Kupiec LR Statistic"]),
                "Classification": classification,
                "Interpretation": _interpret_breach_behavior(
                    classification=classification,
                    breach_rate=breach_rate,
                    expected_breach_rate=expected_breach_rate,
                ),
            }
        )

    return pd.DataFrame(rows)


def build_var_model_comparison(
    var_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Compare EWMA VaR and Historical Simulation VaR performance.

    Parameters
    ----------
    var_summary:
        Summary table returned by summarize_var_backtest().

    Returns
    -------
    pd.DataFrame
        Model-comparison table for dashboard display.
    """
    dashboard_summary = build_var_dashboard_summary(var_summary)

    if dashboard_summary.empty:
        return pd.DataFrame(
            columns=[
                "Comparison Metric",
                "GARCH-Style EWMA VaR",
                "Historical Simulation VaR",
                "Interpretation",
            ]
        )

    comparison_metrics = [
        "Breach Rate",
        "Expected Breach Rate",
        "Average VaR",
        "Average Expected Shortfall",
        "Kupiec LR Statistic",
        "Classification",
    ]

    rows: list[dict[str, Any]] = []

    ewma_row = dashboard_summary[
        dashboard_summary["Model"].eq("GARCH-Style EWMA VaR")
    ]

    historical_row = dashboard_summary[
        dashboard_summary["Model"].eq("Historical Simulation VaR")
    ]

    ewma_values = ewma_row.iloc[0] if not ewma_row.empty else None
    historical_values = historical_row.iloc[0] if not historical_row.empty else None

    for metric in comparison_metrics:
        ewma_value = (
            ewma_values[metric]
            if ewma_values is not None and metric in ewma_values.index
            else np.nan
        )

        historical_value = (
            historical_values[metric]
            if historical_values is not None and metric in historical_values.index
            else np.nan
        )

        if metric == "Breach Rate":
            interpretation = (
                "Lower breach rates suggest a more conservative downside-risk threshold."
            )
        elif metric == "Average VaR":
            interpretation = (
                "More negative VaR values indicate a larger average downside-risk estimate."
            )
        elif metric == "Average Expected Shortfall":
            interpretation = (
                "Expected shortfall is available for the historical simulation model."
            )
        elif metric == "Kupiec LR Statistic":
            interpretation = (
                "Lower Kupiec LR values generally indicate closer alignment with the expected breach rate."
            )
        elif metric == "Classification":
            interpretation = (
                "Plain-English classification based on observed versus expected breaches."
            )
        else:
            interpretation = "Model comparison metric."

        rows.append(
            {
                "Comparison Metric": metric,
                "GARCH-Style EWMA VaR": ewma_value,
                "Historical Simulation VaR": historical_value,
                "Interpretation": interpretation,
            }
        )

    return pd.DataFrame(rows)


def build_latest_risk_snapshot(
    backtest: pd.DataFrame,
) -> pd.DataFrame:
    """Build a latest-risk snapshot from the final VaR backtest row.

    Parameters
    ----------
    backtest:
        VaR backtest output from build_garch_style_var_backtest().

    Returns
    -------
    pd.DataFrame
        Dashboard-friendly latest risk snapshot table.
    """
    if backtest.empty:
        return pd.DataFrame(
            columns=[
                "Metric",
                "Value",
                "Interpretation",
            ]
        )

    required_columns = {
        "return",
        "ewma_volatility",
        "annualized_ewma_volatility",
        "garch_style_var",
        "historical_var",
        "historical_expected_shortfall",
        "garch_style_breach",
        "historical_breach",
    }

    missing_columns = required_columns.difference(backtest.columns)

    if missing_columns:
        raise KeyError(
            "Backtest table is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    latest = backtest.iloc[-1]
    latest_date = backtest.index[-1]

    latest_date_display = (
        str(latest_date.date())
        if hasattr(latest_date, "date")
        else str(latest_date)
    )

    rows = [
        {
            "Metric": "Latest Date",
            "Value": latest_date_display,
            "Interpretation": "Most recent date included in the VaR backtest.",
        },
        {
            "Metric": "Latest Daily Return",
            "Value": float(latest["return"]),
            "Interpretation": "Most recent Hermès daily return in the backtest sample.",
        },
        {
            "Metric": "Latest EWMA Volatility",
            "Value": float(latest["ewma_volatility"]),
            "Interpretation": "Latest daily EWMA volatility estimate.",
        },
        {
            "Metric": "Latest Annualized EWMA Volatility",
            "Value": float(latest["annualized_ewma_volatility"]),
            "Interpretation": "Latest EWMA volatility annualized using 252 trading days.",
        },
        {
            "Metric": "Latest GARCH-Style EWMA VaR",
            "Value": float(latest["garch_style_var"]),
            "Interpretation": "Latest one-day downside threshold from the EWMA VaR model.",
        },
        {
            "Metric": "Latest Historical Simulation VaR",
            "Value": float(latest["historical_var"]),
            "Interpretation": "Latest one-day downside threshold from rolling historical simulation.",
        },
        {
            "Metric": "Latest Historical Expected Shortfall",
            "Value": float(latest["historical_expected_shortfall"]),
            "Interpretation": "Average tail loss below the latest historical VaR threshold.",
        },
        {
            "Metric": "Latest GARCH-Style Breach",
            "Value": bool(latest["garch_style_breach"]),
            "Interpretation": "Whether the latest daily return breached the EWMA VaR threshold.",
        },
        {
            "Metric": "Latest Historical VaR Breach",
            "Value": bool(latest["historical_breach"]),
            "Interpretation": "Whether the latest daily return breached the historical VaR threshold.",
        },
    ]

    return pd.DataFrame(rows)