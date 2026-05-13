"""Risk and performance analytics for the Hermès market intelligence pipeline.

This module calculates return, volatility, drawdown, downside risk, VaR,
expected shortfall, and VaR backtest diagnostics for the Hermès market
intelligence and equity research report.

The module also provides dashboard-ready helper tables for Streamlit and other
interactive reporting outputs.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from config.settings import TRADING_DAYS


def _clean_returns(return_series: pd.Series) -> pd.Series:
    """Clean a return series by removing missing and infinite values."""
    return (
        return_series.dropna()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )


def _safe_float(value: Any, default: float = np.nan) -> float:
    """Safely convert a value to float."""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _max_drawdown(return_series: pd.Series) -> float:
    """Calculate maximum drawdown from a return series."""
    r = _clean_returns(return_series)

    if r.empty:
        return np.nan

    wealth = (1 + r).cumprod()
    drawdown = wealth / wealth.cummax() - 1

    return float(drawdown.min())


def _annualized_return(return_series: pd.Series) -> float:
    """Calculate annualized return from a daily return series."""
    r = _clean_returns(return_series)

    if r.empty:
        return np.nan

    total_return = float((1 + r).prod() - 1)
    years = len(r) / TRADING_DAYS

    if years <= 0:
        return np.nan

    return float((1 + total_return) ** (1 / years) - 1)


def _annualized_volatility(return_series: pd.Series) -> float:
    """Calculate annualized volatility."""
    r = _clean_returns(return_series)

    if r.empty:
        return np.nan

    return float(r.std(ddof=1) * math.sqrt(TRADING_DAYS))


def _downside_volatility(
    return_series: pd.Series,
    risk_free_rate: float = 0.03,
) -> float:
    """Calculate annualized downside volatility versus daily risk-free rate."""
    r = _clean_returns(return_series)

    if r.empty:
        return np.nan

    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    downside = r[r < daily_rf] - daily_rf

    if downside.empty:
        return 0.0

    return float(downside.std(ddof=1) * math.sqrt(TRADING_DAYS))


def _sharpe_ratio(
    return_series: pd.Series,
    risk_free_rate: float = 0.03,
) -> float:
    """Calculate annualized Sharpe ratio."""
    r = _clean_returns(return_series)

    if r.empty:
        return np.nan

    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    daily_vol = r.std(ddof=1)

    if daily_vol == 0 or pd.isna(daily_vol):
        return np.nan

    return float(((r.mean() - daily_rf) / daily_vol) * math.sqrt(TRADING_DAYS))


def _sortino_ratio(
    return_series: pd.Series,
    risk_free_rate: float = 0.03,
) -> float:
    """Calculate annualized Sortino ratio."""
    r = _clean_returns(return_series)

    if r.empty:
        return np.nan

    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    downside = r[r < daily_rf] - daily_rf

    if downside.empty:
        return np.nan

    downside_std = downside.std(ddof=1)

    if downside_std == 0 or pd.isna(downside_std):
        return np.nan

    return float(((r.mean() - daily_rf) / downside_std) * math.sqrt(TRADING_DAYS))


def _calmar_ratio(return_series: pd.Series) -> float:
    """Calculate Calmar ratio as annualized return divided by absolute max drawdown."""
    ann_return = _annualized_return(return_series)
    max_drawdown = abs(_max_drawdown(return_series))

    if max_drawdown == 0 or pd.isna(max_drawdown):
        return np.nan

    return float(ann_return / max_drawdown)


def _historical_var(return_series: pd.Series, confidence: float = 0.95) -> float:
    """Calculate historical VaR at the selected confidence level."""
    r = _clean_returns(return_series)

    if r.empty:
        return np.nan

    return float(r.quantile(1 - confidence))


def _expected_shortfall(return_series: pd.Series, confidence: float = 0.95) -> float:
    """Calculate historical expected shortfall at the selected confidence level."""
    r = _clean_returns(return_series)

    if r.empty:
        return np.nan

    var_level = _historical_var(r, confidence=confidence)
    tail_returns = r[r <= var_level]

    if tail_returns.empty:
        return np.nan

    return float(tail_returns.mean())


def _classify_return_quality(annualized_return: float) -> str:
    """Classify annualized return quality."""
    value = _safe_float(annualized_return)

    if pd.isna(value):
        return "Not Available"

    if value >= 0.15:
        return "Strong"

    if value >= 0.08:
        return "Supportive"

    if value >= 0.00:
        return "Positive but Modest"

    return "Negative"


def _classify_volatility_level(annualized_volatility: float) -> str:
    """Classify annualized volatility level."""
    value = _safe_float(annualized_volatility)

    if pd.isna(value):
        return "Not Available"

    if value >= 0.35:
        return "High"

    if value >= 0.25:
        return "Elevated"

    if value >= 0.15:
        return "Moderate"

    return "Low"


def _classify_drawdown_severity(max_drawdown: float) -> str:
    """Classify maximum drawdown severity."""
    value = abs(_safe_float(max_drawdown))

    if pd.isna(value):
        return "Not Available"

    if value >= 0.50:
        return "Severe"

    if value >= 0.35:
        return "High"

    if value >= 0.20:
        return "Moderate"

    return "Low"


def _classify_sharpe_quality(sharpe_ratio: float) -> str:
    """Classify Sharpe ratio quality."""
    value = _safe_float(sharpe_ratio)

    if pd.isna(value):
        return "Not Available"

    if value >= 1.00:
        return "Strong"

    if value >= 0.50:
        return "Acceptable"

    if value >= 0.00:
        return "Weak Positive"

    return "Negative"


def _classify_tail_risk(var_value: float, expected_shortfall: float) -> str:
    """Classify tail-risk severity using VaR and expected shortfall."""
    var_abs = abs(_safe_float(var_value))
    es_abs = abs(_safe_float(expected_shortfall))

    if pd.isna(var_abs) and pd.isna(es_abs):
        return "Not Available"

    tail_value = np.nanmax([var_abs, es_abs])

    if tail_value >= 0.08:
        return "High"

    if tail_value >= 0.05:
        return "Elevated"

    if tail_value >= 0.03:
        return "Moderate"

    return "Low"


def _overall_risk_view(
    annualized_volatility: float,
    max_drawdown: float,
    var_value: float,
    expected_shortfall: float,
    sharpe_ratio: float,
) -> str:
    """Create an overall plain-English risk view."""
    volatility_level = _classify_volatility_level(annualized_volatility)
    drawdown_severity = _classify_drawdown_severity(max_drawdown)
    tail_risk = _classify_tail_risk(var_value, expected_shortfall)
    sharpe_quality = _classify_sharpe_quality(sharpe_ratio)

    high_risk_terms = {"High", "Severe"}
    elevated_risk_terms = {"Elevated", "Moderate"}

    if (
        volatility_level in high_risk_terms
        or drawdown_severity in high_risk_terms
        or tail_risk in high_risk_terms
    ):
        return "High Risk"

    if (
        volatility_level in elevated_risk_terms
        or drawdown_severity in elevated_risk_terms
        or tail_risk in elevated_risk_terms
    ):
        if sharpe_quality in {"Strong", "Acceptable"}:
            return "Moderate Risk / Compensated"

        return "Moderate Risk / Watch"

    if sharpe_quality in {"Strong", "Acceptable"}:
        return "Lower Risk / Supportive"

    return "Lower Risk / Low Return Compensation"


def _metric_interpretation(metric: str, value: Any) -> str:
    """Create a plain-English interpretation for risk dashboard metrics."""
    numeric_value = _safe_float(value)

    if metric == "Annualized Return":
        classification = _classify_return_quality(numeric_value)
        return f"Annualized return is classified as {classification}."

    if metric == "Annualized Volatility":
        classification = _classify_volatility_level(numeric_value)
        return f"Annualized volatility is classified as {classification}."

    if metric == "Sharpe Ratio":
        classification = _classify_sharpe_quality(numeric_value)
        return f"Sharpe ratio quality is classified as {classification}."

    if metric == "Sortino Ratio":
        return "Sortino ratio measures return quality relative to downside volatility."

    if metric == "Calmar Ratio":
        return "Calmar ratio measures annualized return relative to maximum drawdown."

    if metric == "Max Drawdown":
        classification = _classify_drawdown_severity(numeric_value)
        return f"Maximum drawdown severity is classified as {classification}."

    if metric == "VaR 95%":
        return "Historical one-day 95% VaR estimates the left-tail daily return threshold."

    if metric == "Expected Shortfall 95%":
        return "Expected shortfall estimates the average daily loss beyond the VaR threshold."

    if metric == "Best Daily Return":
        return "Best single-day return observed in the sample."

    if metric == "Worst Daily Return":
        return "Worst single-day return observed in the sample."

    if metric == "Skew":
        return "Skew measures return distribution asymmetry."

    if metric == "Kurtosis":
        return "Kurtosis measures the heaviness of return-distribution tails."

    return "Risk and performance metric used in the market intelligence dashboard."


def summarize_performance(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.03,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Create performance, drawdown, VaR, and expected shortfall summary.

    Parameters
    ----------
    returns:
        DataFrame of daily returns.
    risk_free_rate:
        Annual risk-free rate used for Sharpe and Sortino ratios.
    confidence:
        Confidence level for VaR and expected shortfall.

    Returns
    -------
    pd.DataFrame
        Risk and return summary indexed by asset name.
    """
    if returns.empty:
        raise ValueError("Returns DataFrame is empty. Cannot summarize performance.")

    rows: list[dict[str, Any]] = []

    for name in returns.columns:
        r = _clean_returns(returns[name])

        if r.empty:
            continue

        total_return = float((1 + r).prod() - 1)
        ann_return = _annualized_return(r)
        ann_vol = _annualized_volatility(r)
        downside_vol = _downside_volatility(r, risk_free_rate=risk_free_rate)
        sharpe = _sharpe_ratio(r, risk_free_rate=risk_free_rate)
        sortino = _sortino_ratio(r, risk_free_rate=risk_free_rate)
        max_drawdown = _max_drawdown(r)
        calmar = _calmar_ratio(r)
        var_95 = _historical_var(r, confidence=confidence)
        es_95 = _expected_shortfall(r, confidence=confidence)

        rows.append(
            {
                "Asset": name,
                "Observations": len(r),
                "Total Return": total_return,
                "Annualized Return": ann_return,
                "Annualized Volatility": ann_vol,
                "Downside Volatility": downside_vol,
                "Sharpe Ratio": sharpe,
                "Sortino Ratio": sortino,
                "Calmar Ratio": calmar,
                "Max Drawdown": max_drawdown,
                "VaR 95%": var_95,
                "Expected Shortfall 95%": es_95,
                "Best Daily Return": float(r.max()),
                "Worst Daily Return": float(r.min()),
                "Skew": float(r.skew()),
                "Kurtosis": float(r.kurtosis()),
            }
        )

    if not rows:
        raise ValueError("No valid return series were available for performance summary.")

    return pd.DataFrame(rows).set_index("Asset")


def build_risk_dashboard_cards(
    performance_summary: pd.DataFrame,
    target_label: str = "Hermès International",
) -> pd.DataFrame:
    """Build dashboard card rows for the target asset.

    Parameters
    ----------
    performance_summary:
        Output from summarize_performance().
    target_label:
        Asset row to use for dashboard cards.

    Returns
    -------
    pd.DataFrame
        Dashboard-ready risk and performance cards.
    """
    if performance_summary.empty or target_label not in performance_summary.index:
        return pd.DataFrame(
            [
                {
                    "Card": "Risk Profile",
                    "Value": "Not Available",
                    "Display Value": "Not Available",
                    "Numeric Value": np.nan,
                    "Interpretation": f"No performance summary available for {target_label}.",
                }
            ]
        )

    row = performance_summary.loc[target_label]

    annualized_return = _safe_float(row.get("Annualized Return"))
    annualized_volatility = _safe_float(row.get("Annualized Volatility"))
    sharpe_ratio = _safe_float(row.get("Sharpe Ratio"))
    sortino_ratio = _safe_float(row.get("Sortino Ratio"))
    max_drawdown = _safe_float(row.get("Max Drawdown"))
    var_95 = _safe_float(row.get("VaR 95%"))
    expected_shortfall_95 = _safe_float(row.get("Expected Shortfall 95%"))

    risk_profile = _overall_risk_view(
        annualized_volatility=annualized_volatility,
        max_drawdown=max_drawdown,
        var_value=var_95,
        expected_shortfall=expected_shortfall_95,
        sharpe_ratio=sharpe_ratio,
    )

    rows = [
        {
            "Card": "Annualized Return",
            "Value": annualized_return,
            "Display Value": f"{annualized_return:.2%}"
            if not pd.isna(annualized_return)
            else "Not Available",
            "Numeric Value": annualized_return,
            "Interpretation": _metric_interpretation(
                "Annualized Return",
                annualized_return,
            ),
        },
        {
            "Card": "Annualized Volatility",
            "Value": annualized_volatility,
            "Display Value": f"{annualized_volatility:.2%}"
            if not pd.isna(annualized_volatility)
            else "Not Available",
            "Numeric Value": annualized_volatility,
            "Interpretation": _metric_interpretation(
                "Annualized Volatility",
                annualized_volatility,
            ),
        },
        {
            "Card": "Sharpe Ratio",
            "Value": sharpe_ratio,
            "Display Value": f"{sharpe_ratio:.2f}"
            if not pd.isna(sharpe_ratio)
            else "Not Available",
            "Numeric Value": sharpe_ratio,
            "Interpretation": _metric_interpretation(
                "Sharpe Ratio",
                sharpe_ratio,
            ),
        },
        {
            "Card": "Sortino Ratio",
            "Value": sortino_ratio,
            "Display Value": f"{sortino_ratio:.2f}"
            if not pd.isna(sortino_ratio)
            else "Not Available",
            "Numeric Value": sortino_ratio,
            "Interpretation": _metric_interpretation(
                "Sortino Ratio",
                sortino_ratio,
            ),
        },
        {
            "Card": "Max Drawdown",
            "Value": max_drawdown,
            "Display Value": f"{max_drawdown:.2%}"
            if not pd.isna(max_drawdown)
            else "Not Available",
            "Numeric Value": max_drawdown,
            "Interpretation": _metric_interpretation(
                "Max Drawdown",
                max_drawdown,
            ),
        },
        {
            "Card": "VaR 95%",
            "Value": var_95,
            "Display Value": f"{var_95:.2%}"
            if not pd.isna(var_95)
            else "Not Available",
            "Numeric Value": var_95,
            "Interpretation": _metric_interpretation("VaR 95%", var_95),
        },
        {
            "Card": "Expected Shortfall 95%",
            "Value": expected_shortfall_95,
            "Display Value": f"{expected_shortfall_95:.2%}"
            if not pd.isna(expected_shortfall_95)
            else "Not Available",
            "Numeric Value": expected_shortfall_95,
            "Interpretation": _metric_interpretation(
                "Expected Shortfall 95%",
                expected_shortfall_95,
            ),
        },
        {
            "Card": "Risk Profile",
            "Value": risk_profile,
            "Display Value": risk_profile,
            "Numeric Value": np.nan,
            "Interpretation": (
                "Overall risk view based on volatility, drawdown, tail risk, "
                "and risk-adjusted return quality."
            ),
        },
    ]

    return pd.DataFrame(rows)


def build_risk_metric_leaderboard(
    performance_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build a peer risk and performance leaderboard.

    Parameters
    ----------
    performance_summary:
        Output from summarize_performance().

    Returns
    -------
    pd.DataFrame
        Dashboard-ready peer comparison table.
    """
    if performance_summary.empty:
        return pd.DataFrame(
            columns=[
                "Asset",
                "Annualized Return Rank",
                "Volatility Rank",
                "Sharpe Rank",
                "Drawdown Rank",
                "VaR Rank",
                "Expected Shortfall Rank",
                "Overall Risk Rank",
                "Return Quality",
                "Volatility Level",
                "Drawdown Severity",
                "Sharpe Quality",
                "Tail-Risk Severity",
            ]
        )

    required_columns = {
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Max Drawdown",
        "VaR 95%",
        "Expected Shortfall 95%",
    }

    missing_columns = required_columns.difference(performance_summary.columns)

    if missing_columns:
        raise KeyError(
            "Performance summary is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    leaderboard = performance_summary.copy().reset_index()

    leaderboard["Annualized Return Rank"] = leaderboard["Annualized Return"].rank(
        ascending=False,
        method="min",
    )

    leaderboard["Volatility Rank"] = leaderboard["Annualized Volatility"].rank(
        ascending=True,
        method="min",
    )

    leaderboard["Sharpe Rank"] = leaderboard["Sharpe Ratio"].rank(
        ascending=False,
        method="min",
    )

    leaderboard["Drawdown Rank"] = leaderboard["Max Drawdown"].abs().rank(
        ascending=True,
        method="min",
    )

    leaderboard["VaR Rank"] = leaderboard["VaR 95%"].abs().rank(
        ascending=True,
        method="min",
    )

    leaderboard["Expected Shortfall Rank"] = leaderboard[
        "Expected Shortfall 95%"
    ].abs().rank(
        ascending=True,
        method="min",
    )

    rank_columns = [
        "Annualized Return Rank",
        "Volatility Rank",
        "Sharpe Rank",
        "Drawdown Rank",
        "VaR Rank",
        "Expected Shortfall Rank",
    ]

    leaderboard["Overall Risk Rank"] = leaderboard[rank_columns].mean(axis=1).rank(
        ascending=True,
        method="min",
    )

    leaderboard["Return Quality"] = leaderboard["Annualized Return"].apply(
        _classify_return_quality
    )

    leaderboard["Volatility Level"] = leaderboard["Annualized Volatility"].apply(
        _classify_volatility_level
    )

    leaderboard["Drawdown Severity"] = leaderboard["Max Drawdown"].apply(
        _classify_drawdown_severity
    )

    leaderboard["Sharpe Quality"] = leaderboard["Sharpe Ratio"].apply(
        _classify_sharpe_quality
    )

    leaderboard["Tail-Risk Severity"] = leaderboard.apply(
        lambda row: _classify_tail_risk(
            row["VaR 95%"],
            row["Expected Shortfall 95%"],
        ),
        axis=1,
    )

    for column in rank_columns + ["Overall Risk Rank"]:
        leaderboard[column] = leaderboard[column].astype("Int64")

    return (
        leaderboard[
            [
                "Asset",
                "Annualized Return",
                "Annualized Volatility",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Calmar Ratio",
                "Max Drawdown",
                "VaR 95%",
                "Expected Shortfall 95%",
                "Annualized Return Rank",
                "Volatility Rank",
                "Sharpe Rank",
                "Drawdown Rank",
                "VaR Rank",
                "Expected Shortfall Rank",
                "Overall Risk Rank",
                "Return Quality",
                "Volatility Level",
                "Drawdown Severity",
                "Sharpe Quality",
                "Tail-Risk Severity",
            ]
        ]
        .sort_values(["Overall Risk Rank", "Annualized Return Rank"])
        .reset_index(drop=True)
    )


def build_target_risk_profile(
    performance_summary: pd.DataFrame,
    target_label: str = "Hermès International",
) -> pd.DataFrame:
    """Build a plain-English target risk profile.

    Parameters
    ----------
    performance_summary:
        Output from summarize_performance().
    target_label:
        Asset row to profile.

    Returns
    -------
    pd.DataFrame
        Risk-profile table for the target asset.
    """
    if performance_summary.empty or target_label not in performance_summary.index:
        return pd.DataFrame(
            [
                {
                    "Profile Area": "Overall Risk View",
                    "Value": "Not Available",
                    "Classification": "Not Available",
                    "Interpretation": f"No performance summary available for {target_label}.",
                }
            ]
        )

    row = performance_summary.loc[target_label]

    annualized_return = _safe_float(row.get("Annualized Return"))
    annualized_volatility = _safe_float(row.get("Annualized Volatility"))
    sharpe_ratio = _safe_float(row.get("Sharpe Ratio"))
    max_drawdown = _safe_float(row.get("Max Drawdown"))
    var_95 = _safe_float(row.get("VaR 95%"))
    expected_shortfall_95 = _safe_float(row.get("Expected Shortfall 95%"))

    return_quality = _classify_return_quality(annualized_return)
    volatility_level = _classify_volatility_level(annualized_volatility)
    drawdown_severity = _classify_drawdown_severity(max_drawdown)
    sharpe_quality = _classify_sharpe_quality(sharpe_ratio)
    tail_risk = _classify_tail_risk(var_95, expected_shortfall_95)

    overall_view = _overall_risk_view(
        annualized_volatility=annualized_volatility,
        max_drawdown=max_drawdown,
        var_value=var_95,
        expected_shortfall=expected_shortfall_95,
        sharpe_ratio=sharpe_ratio,
    )

    rows = [
        {
            "Profile Area": "Return Quality",
            "Value": annualized_return,
            "Classification": return_quality,
            "Interpretation": _metric_interpretation(
                "Annualized Return",
                annualized_return,
            ),
        },
        {
            "Profile Area": "Volatility Level",
            "Value": annualized_volatility,
            "Classification": volatility_level,
            "Interpretation": _metric_interpretation(
                "Annualized Volatility",
                annualized_volatility,
            ),
        },
        {
            "Profile Area": "Drawdown Severity",
            "Value": max_drawdown,
            "Classification": drawdown_severity,
            "Interpretation": _metric_interpretation(
                "Max Drawdown",
                max_drawdown,
            ),
        },
        {
            "Profile Area": "Sharpe Quality",
            "Value": sharpe_ratio,
            "Classification": sharpe_quality,
            "Interpretation": _metric_interpretation(
                "Sharpe Ratio",
                sharpe_ratio,
            ),
        },
        {
            "Profile Area": "Tail-Risk Severity",
            "Value": expected_shortfall_95,
            "Classification": tail_risk,
            "Interpretation": (
                "Tail-risk severity is based on historical VaR and expected shortfall."
            ),
        },
        {
            "Profile Area": "Overall Risk View",
            "Value": overall_view,
            "Classification": overall_view,
            "Interpretation": (
                "Overall view combines return quality, volatility, drawdown, "
                "Sharpe quality, and tail-risk severity."
            ),
        },
    ]

    return pd.DataFrame(rows)


def kupiec_pof_test(
    breaches: int | None = None,
    observations: int | None = None,
    expected_breach_rate: float = 0.05,
    num_breaches: int | None = None,
    num_observations: int | None = None,
    expected_prob: float | None = None,
) -> dict[str, float]:
    """Kupiec Proportion of Failures test for VaR breach frequency.

    This function supports both the newer argument names:

    ``breaches``, ``observations``, ``expected_breach_rate``

    and the older argument names used in earlier project files:

    ``num_breaches``, ``num_observations``, ``expected_prob``.
    """
    if breaches is None:
        breaches = num_breaches

    if observations is None:
        observations = num_observations

    if expected_prob is not None:
        expected_breach_rate = expected_prob

    if breaches is None or observations is None:
        raise ValueError("breaches and observations must be provided.")

    if observations <= 0:
        return {
            "breach_rate": np.nan,
            "lr_stat": np.nan,
            "expected_breach_rate": expected_breach_rate,
        }

    if not 0 < expected_breach_rate < 1:
        raise ValueError("expected_breach_rate must be between 0 and 1.")

    breaches = int(breaches)
    observations = int(observations)

    if breaches < 0:
        raise ValueError("breaches cannot be negative.")

    if breaches > observations:
        raise ValueError("breaches cannot exceed observations.")

    observed_breach_rate = breaches / observations

    eps = 1e-12

    phat_safe = min(max(observed_breach_rate, eps), 1 - eps)
    p_safe = min(max(expected_breach_rate, eps), 1 - eps)

    no_breach_count = observations - breaches

    log_likelihood_null = (
        no_breach_count * math.log(1 - p_safe)
        + breaches * math.log(p_safe)
    )

    log_likelihood_alt = (
        no_breach_count * math.log(1 - phat_safe)
        + breaches * math.log(phat_safe)
    )

    lr_stat = -2 * (log_likelihood_null - log_likelihood_alt)

    return {
        "breach_rate": float(observed_breach_rate),
        "lr_stat": float(lr_stat),
        "expected_breach_rate": float(expected_breach_rate),
    }