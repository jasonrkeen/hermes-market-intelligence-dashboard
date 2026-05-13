"""Feature engineering for the Hermès market intelligence pipeline.

This module creates daily returns, momentum indicators, trend indicators,
volatility measures, cumulative return, drawdown, and valuation-reset features
used by the Hermès market intelligence and equity research pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily simple returns from price data.

    Parameters
    ----------
    prices:
        DataFrame of adjusted close prices.

    Returns
    -------
    pd.DataFrame
        Daily simple returns.
    """
    if prices.empty:
        raise ValueError("Price DataFrame is empty. Cannot compute returns.")

    returns = prices.pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(how="all")

    return returns


def compute_log_returns(price: pd.Series) -> pd.Series:
    """Compute daily log returns from a price series."""
    clean_price = price.dropna()

    if clean_price.empty:
        raise ValueError("Price series is empty. Cannot compute log returns.")

    log_returns = np.log(clean_price / clean_price.shift(1))
    return log_returns.replace([np.inf, -np.inf], np.nan)


def compute_rsi(price: pd.Series, window: int = 14) -> pd.Series:
    """Compute the Relative Strength Index.

    Parameters
    ----------
    price:
        Price series.
    window:
        RSI lookback window.

    Returns
    -------
    pd.Series
        RSI values.
    """
    if window <= 0:
        raise ValueError("RSI window must be greater than zero.")

    delta = price.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def compute_drawdown(returns: pd.Series) -> pd.Series:
    """Compute drawdown from a return series."""
    wealth = (1 + returns.fillna(0)).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1

    return drawdown


def compute_rolling_beta(
    target_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 252,
) -> pd.Series:
    """Compute rolling beta against a benchmark return series."""
    aligned = pd.concat(
        [target_returns.rename("target"), benchmark_returns.rename("benchmark")],
        axis=1,
    ).dropna()

    if aligned.empty:
        return pd.Series(dtype=float, name="rolling_beta")

    rolling_cov = aligned["target"].rolling(window).cov(aligned["benchmark"])
    rolling_var = aligned["benchmark"].rolling(window).var()

    beta = rolling_cov / rolling_var.replace(0, np.nan)
    beta.name = "rolling_beta"

    return beta


def build_hermes_feature_set(
    price: pd.Series,
    strategy_config: dict,
) -> pd.DataFrame:
    """Create trend, momentum, volatility, and drawdown features for Hermès.

    Parameters
    ----------
    price:
        Hermès adjusted close price series.
    strategy_config:
        Dictionary containing strategy parameters such as short EMA, long EMA,
        RSI window, and volatility window.

    Returns
    -------
    pd.DataFrame
        Feature table used by the strategy, risk, and reporting modules.
    """
    required_keys = {
        "short_ema",
        "long_ema",
        "rsi_window",
        "vol_window",
    }

    missing_keys = required_keys.difference(strategy_config)

    if missing_keys:
        raise KeyError(f"Strategy configuration is missing keys: {sorted(missing_keys)}")

    clean_price = price.dropna()

    if clean_price.empty:
        raise ValueError("Hermès price series is empty. Cannot build features.")

    df = pd.DataFrame({"price": clean_price})

    df["return"] = df["price"].pct_change()
    df["log_return"] = compute_log_returns(df["price"])

    df["short_ema"] = df["price"].ewm(
        span=int(strategy_config["short_ema"]),
        adjust=False,
    ).mean()

    df["long_ema"] = df["price"].ewm(
        span=int(strategy_config["long_ema"]),
        adjust=False,
    ).mean()

    df["ema_spread"] = df["short_ema"] / df["long_ema"] - 1

    df["rsi"] = compute_rsi(
        df["price"],
        window=int(strategy_config["rsi_window"]),
    )

    df["rolling_volatility"] = df["return"].rolling(
        window=int(strategy_config["vol_window"]),
        min_periods=int(strategy_config["vol_window"]),
    ).std()

    df["annualized_rolling_volatility"] = df["rolling_volatility"] * np.sqrt(252)

    df["cumulative_return"] = (1 + df["return"].fillna(0)).cumprod() - 1
    df["drawdown"] = compute_drawdown(df["return"])

    df["price_above_short_ema"] = df["price"] > df["short_ema"]
    df["price_above_long_ema"] = df["price"] > df["long_ema"]
    df["short_ema_above_long_ema"] = df["short_ema"] > df["long_ema"]

    df["momentum_21d"] = df["price"].pct_change(21)
    df["momentum_63d"] = df["price"].pct_change(63)
    df["momentum_126d"] = df["price"].pct_change(126)
    df["momentum_252d"] = df["price"].pct_change(252)

    df["rolling_high_252d"] = df["price"].rolling(window=252, min_periods=20).max()
    df["distance_from_252d_high"] = df["price"] / df["rolling_high_252d"] - 1

    df = df.replace([np.inf, -np.inf], np.nan)

    return df.dropna()


def build_feature_snapshot(features: pd.DataFrame) -> pd.DataFrame:
    """Build a latest-period feature snapshot for dashboard/report display."""
    if features.empty:
        return pd.DataFrame(
            columns=["Metric", "Value", "Interpretation"]
        )

    latest = features.iloc[-1]

    rows = [
        {
            "Metric": "Latest Price",
            "Value": latest.get("price", np.nan),
            "Interpretation": "Most recent Hermès adjusted close price in the feature set.",
        },
        {
            "Metric": "Latest Daily Return",
            "Value": latest.get("return", np.nan),
            "Interpretation": "Most recent one-day simple return.",
        },
        {
            "Metric": "Latest RSI",
            "Value": latest.get("rsi", np.nan),
            "Interpretation": "Momentum indicator; values above 70 may suggest overbought conditions.",
        },
        {
            "Metric": "Annualized Rolling Volatility",
            "Value": latest.get("annualized_rolling_volatility", np.nan),
            "Interpretation": "Latest annualized volatility estimate based on the configured rolling window.",
        },
        {
            "Metric": "Current Drawdown",
            "Value": latest.get("drawdown", np.nan),
            "Interpretation": "Current decline from the prior cumulative wealth peak.",
        },
        {
            "Metric": "Distance From 252-Day High",
            "Value": latest.get("distance_from_252d_high", np.nan),
            "Interpretation": "Current price discount relative to the trailing 252-day high.",
        },
        {
            "Metric": "21-Day Momentum",
            "Value": latest.get("momentum_21d", np.nan),
            "Interpretation": "Short-term price momentum over approximately one trading month.",
        },
        {
            "Metric": "63-Day Momentum",
            "Value": latest.get("momentum_63d", np.nan),
            "Interpretation": "Medium-term price momentum over approximately one trading quarter.",
        },
        {
            "Metric": "252-Day Momentum",
            "Value": latest.get("momentum_252d", np.nan),
            "Interpretation": "Longer-term price momentum over approximately one trading year.",
        },
    ]

    return pd.DataFrame(rows)


def build_trend_signal_snapshot(features: pd.DataFrame) -> pd.DataFrame:
    """Build a trend and momentum signal snapshot from the latest feature row."""
    if features.empty:
        return pd.DataFrame(
            columns=["Signal", "Value", "Assessment", "Interpretation"]
        )

    latest = features.iloc[-1]

    price_above_short = bool(latest.get("price_above_short_ema", False))
    price_above_long = bool(latest.get("price_above_long_ema", False))
    short_above_long = bool(latest.get("short_ema_above_long_ema", False))

    rows = [
        {
            "Signal": "Price Above Short EMA",
            "Value": price_above_short,
            "Assessment": "Positive" if price_above_short else "Negative",
            "Interpretation": "Short-term trend is supportive when price is above the short EMA.",
        },
        {
            "Signal": "Price Above Long EMA",
            "Value": price_above_long,
            "Assessment": "Positive" if price_above_long else "Negative",
            "Interpretation": "Long-term trend is supportive when price is above the long EMA.",
        },
        {
            "Signal": "Short EMA Above Long EMA",
            "Value": short_above_long,
            "Assessment": "Positive" if short_above_long else "Negative",
            "Interpretation": "Trend structure is constructive when the short EMA is above the long EMA.",
        },
    ]

    rsi = latest.get("rsi", np.nan)

    if pd.isna(rsi):
        rsi_assessment = "Not Available"
    elif rsi >= 70:
        rsi_assessment = "Overbought"
    elif rsi <= 30:
        rsi_assessment = "Oversold"
    elif rsi >= 55:
        rsi_assessment = "Constructive"
    elif rsi <= 45:
        rsi_assessment = "Weakening"
    else:
        rsi_assessment = "Neutral"

    rows.append(
        {
            "Signal": "RSI Signal",
            "Value": rsi,
            "Assessment": rsi_assessment,
            "Interpretation": "RSI summarizes recent momentum and potential overbought or oversold conditions.",
        }
    )

    return pd.DataFrame(rows)