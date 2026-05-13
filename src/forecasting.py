"""Monte Carlo forecasting utilities for the Hermès market intelligence pipeline.

This module uses historical daily Hermès returns to simulate potential future
price paths. The simulation is not a prediction; it is a scenario-distribution
tool based on historical return behavior and is used alongside valuation,
risk, and market-intelligence outputs.

The primary forecast output is a percentile-band table used for charts and
reports. Dashboard helper functions convert that forecast into compact summary
and probability tables.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _validate_monte_carlo_inputs(
    last_price: float,
    returns: pd.Series,
    config: dict[str, Any],
) -> None:
    """Validate Monte Carlo forecast inputs."""
    if last_price <= 0:
        raise ValueError("Last price must be greater than zero.")

    if returns.empty:
        raise ValueError("Returns series is empty.")

    required_keys = {"forecast_days", "num_simulations"}
    missing_keys = required_keys.difference(config)

    if missing_keys:
        raise KeyError(f"Monte Carlo config is missing keys: {sorted(missing_keys)}")

    if int(config["forecast_days"]) <= 0:
        raise ValueError("forecast_days must be greater than zero.")

    if int(config["num_simulations"]) <= 0:
        raise ValueError("num_simulations must be greater than zero.")


def _clean_daily_returns(returns: pd.Series) -> np.ndarray:
    """Clean a return series and convert it to a NumPy array."""
    daily_returns = (
        returns.dropna()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )

    if len(daily_returns) == 0:
        raise ValueError("Returns series has no valid numeric observations.")

    return daily_returns


def _simulate_price_paths(
    last_price: float,
    daily_returns: np.ndarray,
    forecast_days: int,
    num_simulations: int,
    random_seed: int,
) -> np.ndarray:
    """Simulate bootstrap Monte Carlo price paths."""
    rng = np.random.default_rng(random_seed)

    paths = np.zeros((forecast_days + 1, num_simulations))
    paths[0, :] = float(last_price)

    sampled_returns = rng.choice(
        daily_returns,
        size=(forecast_days, num_simulations),
        replace=True,
    )

    for day in range(1, forecast_days + 1):
        paths[day, :] = paths[day - 1, :] * (1 + sampled_returns[day - 1, :])

    return paths


def _build_forecast_table(paths: np.ndarray, last_price: float) -> pd.DataFrame:
    """Build the daily Monte Carlo percentile forecast table."""
    forecast_days = paths.shape[0] - 1

    forecast = pd.DataFrame(
        {
            "Day": range(forecast_days + 1),
            "P01": np.percentile(paths, 1, axis=1),
            "P05": np.percentile(paths, 5, axis=1),
            "P25": np.percentile(paths, 25, axis=1),
            "P50": np.percentile(paths, 50, axis=1),
            "P75": np.percentile(paths, 75, axis=1),
            "P95": np.percentile(paths, 95, axis=1),
            "P99": np.percentile(paths, 99, axis=1),
            "Mean": paths.mean(axis=1),
            "Std Dev": paths.std(axis=1),
            "Min": paths.min(axis=1),
            "Max": paths.max(axis=1),
        }
    )

    forecast["Expected Return"] = forecast["Mean"] / float(last_price) - 1
    forecast["Median Return"] = forecast["P50"] / float(last_price) - 1
    forecast["Downside P05 Return"] = forecast["P05"] / float(last_price) - 1
    forecast["Upside P95 Return"] = forecast["P95"] / float(last_price) - 1

    return forecast


def monte_carlo_forecast(
    last_price: float,
    returns: pd.Series,
    config: dict[str, Any],
    return_final_prices: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
    """Run a bootstrap Monte Carlo forecast using historical daily returns.

    Parameters
    ----------
    last_price:
        Most recent Hermès price.
    returns:
        Historical daily Hermès returns.
    config:
        Monte Carlo configuration dictionary. Expected keys include
        ``forecast_days``, ``num_simulations``, and optionally ``random_seed``.
    return_final_prices:
        If True, also return a Series of simulated final-day prices. This is
        useful for probability summaries in the dashboard.

    Returns
    -------
    pd.DataFrame | tuple[pd.DataFrame, pd.Series]
        Forecast distribution by day. If return_final_prices=True, returns
        ``(forecast, final_prices)``.
    """
    _validate_monte_carlo_inputs(
        last_price=last_price,
        returns=returns,
        config=config,
    )

    forecast_days = int(config["forecast_days"])
    num_simulations = int(config["num_simulations"])
    random_seed = int(config.get("random_seed", 42))

    daily_returns = _clean_daily_returns(returns)

    paths = _simulate_price_paths(
        last_price=last_price,
        daily_returns=daily_returns,
        forecast_days=forecast_days,
        num_simulations=num_simulations,
        random_seed=random_seed,
    )

    forecast = _build_forecast_table(paths=paths, last_price=last_price)

    if return_final_prices:
        final_prices = pd.Series(
            paths[-1, :],
            name="Simulated Final Price",
        )

        return forecast, final_prices

    return forecast


def build_monte_carlo_summary_table(
    forecast: pd.DataFrame,
    current_price: float,
) -> pd.DataFrame:
    """Build a dashboard-friendly Monte Carlo final-day summary table.

    Parameters
    ----------
    forecast:
        Forecast DataFrame returned by monte_carlo_forecast().
    current_price:
        Current Hermès share price used as the simulation starting point.

    Returns
    -------
    pd.DataFrame
        Final-day Monte Carlo summary table with price and return metrics.
    """
    if current_price <= 0:
        raise ValueError("Current price must be greater than zero.")

    if forecast.empty:
        return pd.DataFrame(
            columns=[
                "Metric",
                "Value",
                "Interpretation",
            ]
        )

    required_columns = {
        "Mean",
        "P50",
        "P05",
        "P95",
        "Expected Return",
        "Median Return",
        "Downside P05 Return",
        "Upside P95 Return",
    }

    missing_columns = required_columns.difference(forecast.columns)

    if missing_columns:
        raise KeyError(
            "Forecast table is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    final_row = forecast.iloc[-1]

    rows = [
        {
            "Metric": "Current Price",
            "Value": float(current_price),
            "Interpretation": "Starting share price used in the Monte Carlo simulation.",
        },
        {
            "Metric": "Expected Price",
            "Value": float(final_row["Mean"]),
            "Interpretation": "Average simulated price at the end of the forecast horizon.",
        },
        {
            "Metric": "Median Price",
            "Value": float(final_row["P50"]),
            "Interpretation": "Median simulated price at the end of the forecast horizon.",
        },
        {
            "Metric": "Downside P05 Price",
            "Value": float(final_row["P05"]),
            "Interpretation": "Fifth-percentile simulated price; downside stress estimate.",
        },
        {
            "Metric": "Upside P95 Price",
            "Value": float(final_row["P95"]),
            "Interpretation": "Ninety-fifth-percentile simulated price; upside scenario estimate.",
        },
        {
            "Metric": "Expected Return",
            "Value": float(final_row["Expected Return"]),
            "Interpretation": "Expected return implied by the mean simulated final price.",
        },
        {
            "Metric": "Median Return",
            "Value": float(final_row["Median Return"]),
            "Interpretation": "Return implied by the median simulated final price.",
        },
        {
            "Metric": "Downside P05 Return",
            "Value": float(final_row["Downside P05 Return"]),
            "Interpretation": "Return implied by the fifth-percentile simulated final price.",
        },
        {
            "Metric": "Upside P95 Return",
            "Value": float(final_row["Upside P95 Return"]),
            "Interpretation": "Return implied by the ninety-fifth-percentile simulated final price.",
        },
    ]

    return pd.DataFrame(rows)


def build_monte_carlo_probability_table(
    final_prices: pd.Series,
    current_price: float,
    downside_threshold: float = -0.10,
    upside_threshold: float = 0.10,
) -> pd.DataFrame:
    """Build a probability summary table from simulated final-day prices.

    Parameters
    ----------
    final_prices:
        Series of simulated final-day prices returned by monte_carlo_forecast()
        when return_final_prices=True.
    current_price:
        Starting Hermès share price.
    downside_threshold:
        Return threshold used to define material downside. Default is -10%.
    upside_threshold:
        Return threshold used to define material upside. Default is +10%.

    Returns
    -------
    pd.DataFrame
        Probability table for dashboard display.
    """
    if current_price <= 0:
        raise ValueError("Current price must be greater than zero.")

    if final_prices.empty:
        return pd.DataFrame(
            columns=[
                "Metric",
                "Value",
                "Interpretation",
            ]
        )

    final_prices_clean = (
        final_prices.dropna()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )

    if final_prices_clean.empty:
        raise ValueError("Final simulated prices have no valid numeric observations.")

    simulated_returns = final_prices_clean / float(current_price) - 1

    probability_positive = float((simulated_returns > 0).mean())
    probability_negative = float((simulated_returns < 0).mean())
    probability_downside = float((simulated_returns <= downside_threshold).mean())
    probability_upside = float((simulated_returns >= upside_threshold).mean())

    rows = [
        {
            "Metric": "Probability of Positive Return",
            "Value": probability_positive,
            "Interpretation": "Share of simulated final prices above the current price.",
        },
        {
            "Metric": "Probability of Negative Return",
            "Value": probability_negative,
            "Interpretation": "Share of simulated final prices below the current price.",
        },
        {
            "Metric": f"Probability of Return <= {downside_threshold:.0%}",
            "Value": probability_downside,
            "Interpretation": "Share of simulations at or below the downside threshold.",
        },
        {
            "Metric": f"Probability of Return >= {upside_threshold:.0%}",
            "Value": probability_upside,
            "Interpretation": "Share of simulations at or above the upside threshold.",
        },
        {
            "Metric": "Simulated Final Price Mean",
            "Value": float(final_prices_clean.mean()),
            "Interpretation": "Average simulated final price across all paths.",
        },
        {
            "Metric": "Simulated Final Price Median",
            "Value": float(final_prices_clean.median()),
            "Interpretation": "Median simulated final price across all paths.",
        },
        {
            "Metric": "Simulated Final Price Std Dev",
            "Value": float(final_prices_clean.std()),
            "Interpretation": "Dispersion of simulated final prices across all paths.",
        },
    ]

    return pd.DataFrame(rows)


def build_monte_carlo_dashboard_table(
    summary_table: pd.DataFrame,
    probability_table: pd.DataFrame,
) -> pd.DataFrame:
    """Combine Monte Carlo summary and probability outputs for dashboard display.

    Parameters
    ----------
    summary_table:
        Output from build_monte_carlo_summary_table().
    probability_table:
        Output from build_monte_carlo_probability_table().

    Returns
    -------
    pd.DataFrame
        Combined dashboard-ready Monte Carlo table.
    """
    summary = summary_table.copy()
    probabilities = probability_table.copy()

    if not summary.empty:
        summary.insert(0, "Section", "Forecast Summary")

    if not probabilities.empty:
        probabilities.insert(0, "Section", "Simulation Probabilities")

    combined = pd.concat(
        [summary, probabilities],
        ignore_index=True,
    )

    return combined