"""Hermès risk-control and strategy research module.

This module backtests several simple research overlays for the Hermès market
intelligence pipeline. These are not trading recommendations. They are strategy
experiments used to compare how trend, momentum, volatility targeting, and
drawdown filters affect risk exposure versus a simple buy-and-hold position.

The module also provides dashboard-ready strategy cards, comparison leaderboards,
and signal/exposure summaries for Streamlit and other interactive reporting
outputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _validate_strategy_inputs(
    features: pd.DataFrame,
    strategy_config: dict[str, Any],
) -> None:
    """Validate required feature and strategy configuration inputs."""
    required_feature_columns = {
        "return",
        "short_ema",
        "long_ema",
        "rsi",
        "rolling_volatility",
        "drawdown",
    }

    missing_feature_columns = required_feature_columns.difference(features.columns)

    if missing_feature_columns:
        raise KeyError(
            "Feature DataFrame is missing required columns: "
            f"{sorted(missing_feature_columns)}"
        )

    required_config_keys = {
        "rsi_max_buy",
        "max_daily_vol",
    }

    missing_config_keys = required_config_keys.difference(strategy_config)

    if missing_config_keys:
        raise KeyError(
            "Strategy configuration is missing required keys: "
            f"{sorted(missing_config_keys)}"
        )

    if features.empty:
        raise ValueError("Feature DataFrame is empty. Cannot build strategy returns.")


def _safe_float(value: Any, default: float = np.nan) -> float:
    """Safely convert a value to float."""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _calculate_drawdown(wealth: pd.Series) -> pd.Series:
    """Calculate drawdown from a wealth index."""
    return wealth / wealth.cummax() - 1


def _add_strategy_performance_columns(
    df: pd.DataFrame,
    strategy_return_col: str,
    prefix: str,
) -> pd.DataFrame:
    """Add wealth and drawdown columns for a strategy return series."""
    df[f"{prefix}_wealth"] = (1 + df[strategy_return_col].fillna(0)).cumprod()
    df[f"{prefix}_drawdown"] = _calculate_drawdown(df[f"{prefix}_wealth"])

    return df


def _build_risk_control_signal(
    df: pd.DataFrame,
    strategy_config: dict[str, Any],
) -> pd.Series:
    """Build the original Hermès risk-control overlay signal."""
    trend_signal = df["short_ema"] > df["long_ema"]
    momentum_filter = df["rsi"] < float(strategy_config["rsi_max_buy"])
    volatility_filter = df["rolling_volatility"] < float(
        strategy_config["max_daily_vol"]
    )

    return (trend_signal & momentum_filter & volatility_filter).astype(int)


def _build_trend_following_signal(df: pd.DataFrame) -> pd.Series:
    """Build a simple trend-following signal using short and long EMAs."""
    return (df["short_ema"] > df["long_ema"]).astype(int)


def _build_momentum_signal(df: pd.DataFrame) -> pd.Series:
    """Build a momentum signal using available momentum columns when present."""
    if "momentum_63d" in df.columns:
        medium_momentum = df["momentum_63d"] > 0
    else:
        medium_momentum = df["return"].rolling(63).sum() > 0

    if "momentum_126d" in df.columns:
        longer_momentum = df["momentum_126d"] > 0
    else:
        longer_momentum = df["return"].rolling(126).sum() > 0

    trend_confirmation = df["short_ema"] > df["long_ema"]

    return (medium_momentum & longer_momentum & trend_confirmation).astype(int)


def _build_defensive_cash_signal(df: pd.DataFrame) -> pd.Series:
    """Build a defensive signal that exits during downtrends or deep drawdowns."""
    price_above_long_ema = df["short_ema"] > df["long_ema"]
    drawdown_not_extreme = df["drawdown"] > -0.20

    return (price_above_long_ema & drawdown_not_extreme).astype(int)


def _build_volatility_target_exposure(
    df: pd.DataFrame,
    target_annual_vol: float = 0.18,
    max_exposure: float = 1.0,
    min_exposure: float = 0.0,
) -> pd.Series:
    """Build a volatility-target exposure series.

    Exposure is scaled down when realized volatility rises above the target.
    """
    annualized_vol = df["rolling_volatility"] * np.sqrt(252)

    exposure = target_annual_vol / annualized_vol.replace(0, np.nan)
    exposure = exposure.clip(lower=min_exposure, upper=max_exposure)
    exposure = exposure.replace([np.inf, -np.inf], np.nan).fillna(0)

    trend_filter = df["short_ema"] > df["long_ema"]

    return exposure.where(trend_filter, 0.0)


def _classify_strategy_quality(
    annualized_return: float,
    sharpe_ratio: float,
    max_drawdown: float,
) -> str:
    """Classify strategy quality using return, Sharpe, and drawdown."""
    ann_return = _safe_float(annualized_return)
    sharpe = _safe_float(sharpe_ratio)
    drawdown = abs(_safe_float(max_drawdown))

    if pd.isna(ann_return) or pd.isna(sharpe) or pd.isna(drawdown):
        return "Not Available"

    if ann_return > 0 and sharpe >= 0.75 and drawdown <= 0.30:
        return "Strong Risk-Adjusted Profile"

    if ann_return > 0 and sharpe >= 0.40:
        return "Constructive Profile"

    if ann_return > 0:
        return "Positive Return / Weak Efficiency"

    if drawdown <= 0.20:
        return "Defensive but Low Return"

    return "Weak Profile"


def _classify_exposure_level(average_exposure: float) -> str:
    """Classify average strategy exposure."""
    exposure = _safe_float(average_exposure)

    if pd.isna(exposure):
        return "Not Available"

    if exposure >= 0.85:
        return "Mostly Invested"

    if exposure >= 0.50:
        return "Partially Invested"

    if exposure >= 0.20:
        return "Defensive Exposure"

    return "Mostly Cash / Low Exposure"


def _metric_display_value(metric: str, value: Any) -> str:
    """Format strategy dashboard card values."""
    numeric_value = _safe_float(value)

    percent_metrics = {
        "Total Return",
        "Annualized Return",
        "Annualized Volatility",
        "Max Drawdown",
        "Average Exposure",
        "Drawdown Improvement",
    }

    if metric in percent_metrics and not pd.isna(numeric_value):
        return f"{numeric_value:.2%}"

    if metric in {"Sharpe Ratio", "Overall Strategy Score"} and not pd.isna(
        numeric_value
    ):
        return f"{numeric_value:.2f}"

    return str(value)


def _strategy_interpretation(
    strategy: str,
    annualized_return: float,
    sharpe_ratio: float,
    max_drawdown: float,
    average_exposure: float,
) -> str:
    """Create a plain-English strategy interpretation."""
    quality = _classify_strategy_quality(
        annualized_return=annualized_return,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
    )
    exposure = _classify_exposure_level(average_exposure)

    return (
        f"{strategy} is classified as {quality}, with exposure classified as "
        f"{exposure}."
    )


def build_strategy_research_table(
    features: pd.DataFrame,
    strategy_config: dict[str, Any],
) -> pd.DataFrame:
    """Build multiple Hermès strategy research overlays.

    Parameters
    ----------
    features:
        Hermès feature table from ``build_hermes_feature_set``.
    strategy_config:
        Strategy configuration dictionary from settings.py.

    Returns
    -------
    pd.DataFrame
        Feature table with multiple strategy signals, returns, wealth indexes,
        and drawdown series.
    """
    _validate_strategy_inputs(features, strategy_config)

    df = features.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    df["buy_hold_return"] = df["return"]

    # Original risk-control strategy.
    df["risk_control_signal"] = _build_risk_control_signal(df, strategy_config)
    df["risk_control_return"] = (
        df["risk_control_signal"].shift(1).fillna(0) * df["return"]
    )

    # Additional experimental overlays.
    df["trend_following_signal"] = _build_trend_following_signal(df)
    df["trend_following_return"] = (
        df["trend_following_signal"].shift(1).fillna(0) * df["return"]
    )

    df["momentum_signal"] = _build_momentum_signal(df)
    df["momentum_return"] = df["momentum_signal"].shift(1).fillna(0) * df["return"]

    df["defensive_cash_signal"] = _build_defensive_cash_signal(df)
    df["defensive_cash_return"] = (
        df["defensive_cash_signal"].shift(1).fillna(0) * df["return"]
    )

    target_vol = float(strategy_config.get("target_annual_vol", 0.18))
    max_exposure = float(strategy_config.get("max_exposure", 1.0))

    df["volatility_target_exposure"] = _build_volatility_target_exposure(
        df,
        target_annual_vol=target_vol,
        max_exposure=max_exposure,
    )

    df["volatility_target_return"] = (
        df["volatility_target_exposure"].shift(1).fillna(0) * df["return"]
    )

    # Compatibility columns for the existing report and chart functions.
    df["signal"] = df["risk_control_signal"]
    df["strategy_return"] = df["risk_control_return"]

    return_columns = [
        ("buy_hold_return", "buy_hold"),
        ("strategy_return", "strategy"),
        ("risk_control_return", "risk_control"),
        ("trend_following_return", "trend_following"),
        ("momentum_return", "momentum"),
        ("defensive_cash_return", "defensive_cash"),
        ("volatility_target_return", "volatility_target"),
    ]

    for return_col, prefix in return_columns:
        df = _add_strategy_performance_columns(df, return_col, prefix)

    # Extra compatibility columns expected by existing visualization code.
    df["strategy_wealth"] = df["risk_control_wealth"]
    df["strategy_drawdown"] = df["risk_control_drawdown"]

    return df.dropna()


def build_strategy_summary(strategy_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize strategy research overlays.

    This table is useful for comparing whether each overlay improves drawdown,
    volatility, or risk-adjusted returns versus buy-and-hold.
    """
    strategy_specs = {
        "Buy & Hold": "buy_hold_return",
        "Risk-Control Overlay": "risk_control_return",
        "Trend Following Overlay": "trend_following_return",
        "Momentum Overlay": "momentum_return",
        "Defensive Cash Overlay": "defensive_cash_return",
        "Volatility Target Overlay": "volatility_target_return",
    }

    rows: list[dict[str, float | str]] = []

    for name, return_col in strategy_specs.items():
        if return_col not in strategy_df.columns:
            continue

        returns = strategy_df[return_col].dropna()

        if returns.empty:
            continue

        wealth = (1 + returns.fillna(0)).cumprod()
        drawdown = wealth / wealth.cummax() - 1

        total_return = float(wealth.iloc[-1] - 1)
        annualized_return = float((1 + total_return) ** (252 / len(returns)) - 1)
        annualized_volatility = float(returns.std() * np.sqrt(252))
        max_drawdown = float(drawdown.min())

        sharpe_ratio = (
            float((returns.mean() / returns.std()) * np.sqrt(252))
            if returns.std() != 0
            else np.nan
        )

        exposure_col = None

        if name == "Risk-Control Overlay":
            exposure_col = "risk_control_signal"
        elif name == "Trend Following Overlay":
            exposure_col = "trend_following_signal"
        elif name == "Momentum Overlay":
            exposure_col = "momentum_signal"
        elif name == "Defensive Cash Overlay":
            exposure_col = "defensive_cash_signal"
        elif name == "Volatility Target Overlay":
            exposure_col = "volatility_target_exposure"

        average_exposure = (
            float(strategy_df[exposure_col].mean())
            if exposure_col and exposure_col in strategy_df.columns
            else 1.0
        )

        rows.append(
            {
                "Strategy": name,
                "Total Return": total_return,
                "Annualized Return": annualized_return,
                "Annualized Volatility": annualized_volatility,
                "Sharpe Ratio": sharpe_ratio,
                "Max Drawdown": max_drawdown,
                "Average Exposure": average_exposure,
            }
        )

    return pd.DataFrame(rows).set_index("Strategy")


def build_strategy_dashboard_cards(
    strategy_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build dashboard cards from the strategy summary.

    Parameters
    ----------
    strategy_summary:
        Output from build_strategy_summary().

    Returns
    -------
    pd.DataFrame
        Dashboard-ready strategy cards.
    """
    if strategy_summary.empty:
        return pd.DataFrame(
            [
                {
                    "Card": "Strategy Research",
                    "Value": "Not Available",
                    "Display Value": "Not Available",
                    "Numeric Value": np.nan,
                    "Interpretation": "Strategy summary was unavailable.",
                }
            ]
        )

    required_columns = {
        "Total Return",
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Max Drawdown",
        "Average Exposure",
    }

    missing_columns = required_columns.difference(strategy_summary.columns)

    if missing_columns:
        raise KeyError(
            "Strategy summary is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    summary = strategy_summary.copy()

    best_return_strategy = summary["Total Return"].idxmax()
    best_sharpe_strategy = summary["Sharpe Ratio"].idxmax()
    lowest_drawdown_strategy = summary["Max Drawdown"].abs().idxmin()
    lowest_vol_strategy = summary["Annualized Volatility"].idxmin()

    buy_hold_return = (
        summary.loc["Buy & Hold", "Total Return"]
        if "Buy & Hold" in summary.index
        else np.nan
    )

    risk_control_return = (
        summary.loc["Risk-Control Overlay", "Total Return"]
        if "Risk-Control Overlay" in summary.index
        else np.nan
    )

    buy_hold_drawdown = (
        summary.loc["Buy & Hold", "Max Drawdown"]
        if "Buy & Hold" in summary.index
        else np.nan
    )

    risk_control_drawdown = (
        summary.loc["Risk-Control Overlay", "Max Drawdown"]
        if "Risk-Control Overlay" in summary.index
        else np.nan
    )

    drawdown_improvement = (
        abs(_safe_float(buy_hold_drawdown)) - abs(_safe_float(risk_control_drawdown))
        if not pd.isna(_safe_float(buy_hold_drawdown))
        and not pd.isna(_safe_float(risk_control_drawdown))
        else np.nan
    )

    drawdown_result = (
        "Improved"
        if not pd.isna(drawdown_improvement) and drawdown_improvement > 0
        else "Not Improved"
        if not pd.isna(drawdown_improvement)
        else "Not Available"
    )

    rows = [
        {
            "Card": "Best Total Return Overlay",
            "Value": best_return_strategy,
            "Display Value": str(best_return_strategy),
            "Numeric Value": _safe_float(summary.loc[best_return_strategy, "Total Return"]),
            "Interpretation": (
                "Strategy with the highest cumulative total return in the backtest."
            ),
        },
        {
            "Card": "Best Sharpe Overlay",
            "Value": best_sharpe_strategy,
            "Display Value": str(best_sharpe_strategy),
            "Numeric Value": _safe_float(summary.loc[best_sharpe_strategy, "Sharpe Ratio"]),
            "Interpretation": (
                "Strategy with the strongest return per unit of realized volatility."
            ),
        },
        {
            "Card": "Lowest Drawdown Overlay",
            "Value": lowest_drawdown_strategy,
            "Display Value": str(lowest_drawdown_strategy),
            "Numeric Value": _safe_float(
                summary.loc[lowest_drawdown_strategy, "Max Drawdown"]
            ),
            "Interpretation": "Strategy with the smallest absolute maximum drawdown.",
        },
        {
            "Card": "Lowest Volatility Overlay",
            "Value": lowest_vol_strategy,
            "Display Value": str(lowest_vol_strategy),
            "Numeric Value": _safe_float(
                summary.loc[lowest_vol_strategy, "Annualized Volatility"]
            ),
            "Interpretation": "Strategy with the lowest annualized volatility.",
        },
        {
            "Card": "Buy & Hold Total Return",
            "Value": buy_hold_return,
            "Display Value": _metric_display_value("Total Return", buy_hold_return),
            "Numeric Value": _safe_float(buy_hold_return),
            "Interpretation": "Cumulative return from remaining fully invested.",
        },
        {
            "Card": "Risk-Control Total Return",
            "Value": risk_control_return,
            "Display Value": _metric_display_value("Total Return", risk_control_return),
            "Numeric Value": _safe_float(risk_control_return),
            "Interpretation": "Cumulative return from the primary risk-control overlay.",
        },
        {
            "Card": "Risk-Control Drawdown Improvement",
            "Value": drawdown_result,
            "Display Value": drawdown_result,
            "Numeric Value": drawdown_improvement,
            "Interpretation": (
                "Compares absolute maximum drawdown for risk-control versus buy-and-hold."
            ),
        },
        {
            "Card": "Drawdown Improvement Amount",
            "Value": drawdown_improvement,
            "Display Value": _metric_display_value(
                "Drawdown Improvement",
                drawdown_improvement,
            ),
            "Numeric Value": drawdown_improvement,
            "Interpretation": (
                "Positive values mean the risk-control overlay reduced drawdown."
            ),
        },
    ]

    return pd.DataFrame(rows)


def build_strategy_comparison_leaderboard(
    strategy_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build a ranked strategy comparison leaderboard.

    Parameters
    ----------
    strategy_summary:
        Output from build_strategy_summary().

    Returns
    -------
    pd.DataFrame
        Ranked strategy comparison table.
    """
    if strategy_summary.empty:
        return pd.DataFrame(
            columns=[
                "Strategy",
                "Total Return",
                "Annualized Return",
                "Annualized Volatility",
                "Sharpe Ratio",
                "Max Drawdown",
                "Average Exposure",
                "Return Rank",
                "Volatility Rank",
                "Sharpe Rank",
                "Drawdown Rank",
                "Exposure Rank",
                "Overall Strategy Score",
                "Strategy Quality",
                "Exposure Level",
                "Interpretation",
            ]
        )

    required_columns = {
        "Total Return",
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Max Drawdown",
        "Average Exposure",
    }

    missing_columns = required_columns.difference(strategy_summary.columns)

    if missing_columns:
        raise KeyError(
            "Strategy summary is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    leaderboard = strategy_summary.copy().reset_index()

    leaderboard["Return Rank"] = leaderboard["Total Return"].rank(
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

    leaderboard["Exposure Rank"] = leaderboard["Average Exposure"].rank(
        ascending=True,
        method="min",
    )

    rank_columns = [
        "Return Rank",
        "Volatility Rank",
        "Sharpe Rank",
        "Drawdown Rank",
        "Exposure Rank",
    ]

    leaderboard["Overall Strategy Score"] = leaderboard[rank_columns].mean(axis=1)

    leaderboard["Overall Strategy Rank"] = leaderboard[
        "Overall Strategy Score"
    ].rank(
        ascending=True,
        method="min",
    )

    leaderboard["Strategy Quality"] = leaderboard.apply(
        lambda row: _classify_strategy_quality(
            annualized_return=row["Annualized Return"],
            sharpe_ratio=row["Sharpe Ratio"],
            max_drawdown=row["Max Drawdown"],
        ),
        axis=1,
    )

    leaderboard["Exposure Level"] = leaderboard["Average Exposure"].apply(
        _classify_exposure_level
    )

    leaderboard["Interpretation"] = leaderboard.apply(
        lambda row: _strategy_interpretation(
            strategy=str(row["Strategy"]),
            annualized_return=row["Annualized Return"],
            sharpe_ratio=row["Sharpe Ratio"],
            max_drawdown=row["Max Drawdown"],
            average_exposure=row["Average Exposure"],
        ),
        axis=1,
    )

    for column in rank_columns + ["Overall Strategy Rank"]:
        leaderboard[column] = leaderboard[column].astype("Int64")

    return (
        leaderboard[
            [
                "Strategy",
                "Total Return",
                "Annualized Return",
                "Annualized Volatility",
                "Sharpe Ratio",
                "Max Drawdown",
                "Average Exposure",
                "Return Rank",
                "Volatility Rank",
                "Sharpe Rank",
                "Drawdown Rank",
                "Exposure Rank",
                "Overall Strategy Score",
                "Overall Strategy Rank",
                "Strategy Quality",
                "Exposure Level",
                "Interpretation",
            ]
        ]
        .sort_values(["Overall Strategy Rank", "Return Rank"])
        .reset_index(drop=True)
    )


def build_strategy_signal_summary(
    strategy_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize strategy signal and exposure behavior.

    Parameters
    ----------
    strategy_df:
        Output from build_strategy_research_table().

    Returns
    -------
    pd.DataFrame
        Dashboard-ready signal and exposure summary.
    """
    if strategy_df.empty:
        return pd.DataFrame(
            columns=[
                "Strategy",
                "Signal Column",
                "Average Exposure",
                "Invested Days",
                "Cash / Reduced Days",
                "Invested Day Share",
                "Exposure Level",
                "Interpretation",
            ]
        )

    signal_specs = {
        "Risk-Control Overlay": "risk_control_signal",
        "Trend Following Overlay": "trend_following_signal",
        "Momentum Overlay": "momentum_signal",
        "Defensive Cash Overlay": "defensive_cash_signal",
        "Volatility Target Overlay": "volatility_target_exposure",
    }

    rows: list[dict[str, Any]] = []

    total_days = len(strategy_df)

    for strategy_name, signal_col in signal_specs.items():
        if signal_col not in strategy_df.columns:
            continue

        exposure = (
            pd.to_numeric(strategy_df[signal_col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

        average_exposure = float(exposure.mean())
        invested_days = int((exposure > 0).sum())
        cash_or_reduced_days = int((exposure <= 0).sum())
        invested_day_share = invested_days / total_days if total_days else np.nan
        exposure_level = _classify_exposure_level(average_exposure)

        rows.append(
            {
                "Strategy": strategy_name,
                "Signal Column": signal_col,
                "Average Exposure": average_exposure,
                "Invested Days": invested_days,
                "Cash / Reduced Days": cash_or_reduced_days,
                "Invested Day Share": invested_day_share,
                "Exposure Level": exposure_level,
                "Interpretation": (
                    f"{strategy_name} was invested on "
                    f"{invested_day_share:.2%} of available days, with average "
                    f"exposure classified as {exposure_level}."
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["Average Exposure", "Invested Day Share"],
        ascending=False,
    ).reset_index(drop=True)


def build_hermes_strategy_returns(
    features: pd.DataFrame,
    strategy_config: dict[str, Any],
) -> pd.DataFrame:
    """Backtest the main Hermès risk-control overlay.

    This function keeps the original project API. It now also includes additional
    experimental strategy columns from ``build_strategy_research_table``.
    """
    return build_strategy_research_table(features, strategy_config)