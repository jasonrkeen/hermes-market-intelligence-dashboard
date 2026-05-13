"""Factor regression model for the Hermès market intelligence pipeline.

This module estimates Hermès return sensitivity to selected market,
luxury-sector, FX, and China-demand proxy factors. It also provides
dashboard-friendly helper tables for diagnostics and scenario contribution
analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _validate_factor_inputs(
    returns: pd.DataFrame,
    target: str,
    factors: list[str],
) -> None:
    """Validate that the target and factor columns exist in the returns DataFrame."""
    required_columns = [target] + factors
    missing_columns = [col for col in required_columns if col not in returns.columns]

    if missing_columns:
        raise KeyError(
            "The returns DataFrame is missing required columns: "
            f"{missing_columns}. Available columns: {list(returns.columns)}"
        )


def fit_factor_model(
    returns: pd.DataFrame,
    target: str,
    factors: list[str],
) -> tuple[pd.DataFrame, pd.Series, float]:
    """Estimate target return exposure to selected factors using OLS.

    Parameters
    ----------
    returns:
        DataFrame of daily returns.
    target:
        Target return column, usually Hermès International.
    factors:
        List of explanatory factor columns.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, float]
        Coefficient table, fitted-return series, and model R-squared.
    """
    _validate_factor_inputs(returns, target, factors)

    cols = [target] + factors
    data = returns[cols].dropna()

    if data.empty:
        raise ValueError(
            "No overlapping return observations are available for the factor model."
        )

    min_required_observations = len(factors) + 2

    if len(data) < min_required_observations:
        raise ValueError(
            "Not enough observations to estimate the factor model. "
            f"Required at least {min_required_observations}, received {len(data)}."
        )

    y = data[target].to_numpy(dtype=float)
    x = data[factors].to_numpy(dtype=float)

    x_design = np.column_stack([np.ones(len(x)), x])

    beta, *_ = np.linalg.lstsq(x_design, y, rcond=None)

    fitted = x_design @ beta
    residuals = y - fitted

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    r_squared = 1 - ss_res / ss_tot if ss_tot else np.nan

    n_obs = len(y)
    n_factors = len(factors)

    if n_obs > n_factors + 1 and not np.isnan(r_squared):
        adjusted_r_squared = 1 - (
            (1 - r_squared) * (n_obs - 1) / (n_obs - n_factors - 1)
        )
    else:
        adjusted_r_squared = np.nan

    residual_std = float(np.std(residuals, ddof=n_factors + 1))
    residual_mean = float(np.mean(residuals))

    coef = pd.DataFrame(
        {
            "Coefficient": beta,
        },
        index=["Alpha"] + factors,
    )

    diagnostics = pd.DataFrame(
        {
            "Coefficient": [
                r_squared,
                adjusted_r_squared,
                residual_mean,
                residual_std,
                n_obs,
            ],
        },
        index=[
            "Model R-Squared",
            "Adjusted R-Squared",
            "Residual Mean",
            "Residual Std Dev",
            "Observations",
        ],
    )

    coef = pd.concat([coef, diagnostics])

    fitted_series = pd.Series(
        fitted,
        index=data.index,
        name="fitted_return",
    )

    return coef, fitted_series, float(r_squared)


def build_factor_diagnostics_table(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Build a dashboard-friendly factor model diagnostics table.

    Parameters
    ----------
    coefficients:
        Coefficient table returned by fit_factor_model().

    Returns
    -------
    pd.DataFrame
        Diagnostics table with metric, value, and interpretation columns.
    """
    if "Coefficient" not in coefficients.columns:
        raise KeyError("Coefficient table must include a 'Coefficient' column.")

    diagnostic_rows = [
        "Model R-Squared",
        "Adjusted R-Squared",
        "Residual Mean",
        "Residual Std Dev",
        "Observations",
    ]

    rows: list[dict[str, Any]] = []

    for metric in diagnostic_rows:
        if metric not in coefficients.index:
            continue

        value = float(coefficients.loc[metric, "Coefficient"])

        if metric == "Model R-Squared":
            interpretation = (
                "Share of Hermès daily return variation explained by the selected factors."
            )
        elif metric == "Adjusted R-Squared":
            interpretation = (
                "R-squared adjusted for the number of explanatory factors."
            )
        elif metric == "Residual Mean":
            interpretation = (
                "Average unexplained daily return after accounting for factors."
            )
        elif metric == "Residual Std Dev":
            interpretation = (
                "Volatility of unexplained daily returns after factor adjustment."
            )
        elif metric == "Observations":
            interpretation = "Number of overlapping daily observations used."
        else:
            interpretation = "Factor model diagnostic metric."

        rows.append(
            {
                "Metric": metric,
                "Value": value,
                "Interpretation": interpretation,
            }
        )

    return pd.DataFrame(rows)


def build_factor_exposure_table(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Build a clean factor exposure table excluding model diagnostics.

    Parameters
    ----------
    coefficients:
        Coefficient table returned by fit_factor_model().

    Returns
    -------
    pd.DataFrame
        Factor exposure table with Alpha and explanatory factor coefficients.
    """
    if "Coefficient" not in coefficients.columns:
        raise KeyError("Coefficient table must include a 'Coefficient' column.")

    diagnostic_rows = {
        "Model R-Squared",
        "Adjusted R-Squared",
        "Residual Mean",
        "Residual Std Dev",
        "Observations",
    }

    exposure_rows = []

    for factor_name, row in coefficients.iterrows():
        if factor_name in diagnostic_rows:
            continue

        coefficient = float(row["Coefficient"])

        if factor_name == "Alpha":
            interpretation = (
                "Average daily return unexplained by the selected factors."
            )
        elif coefficient > 0:
            interpretation = (
                "Positive exposure; Hermès tends to move in the same direction as this factor."
            )
        elif coefficient < 0:
            interpretation = (
                "Negative exposure; Hermès tends to move opposite this factor."
            )
        else:
            interpretation = "No estimated directional exposure."

        exposure_rows.append(
            {
                "Factor": factor_name,
                "Coefficient": coefficient,
                "Interpretation": interpretation,
            }
        )

    return pd.DataFrame(exposure_rows)


def estimate_scenarios(
    coefficients: pd.DataFrame,
    scenarios: dict[str, dict[str, float]],
    trading_days: int = 252,
) -> pd.DataFrame:
    """Estimate annualized Hermès return under each factor scenario.

    Parameters
    ----------
    coefficients:
        Factor coefficient table from fit_factor_model.
    scenarios:
        Dictionary of scenario assumptions, where each scenario maps factor names
        to assumed annual factor returns.
    trading_days:
        Number of trading days used to annualize the daily alpha.

    Returns
    -------
    pd.DataFrame
        Scenario table with estimated Hermès return and factor assumptions.
    """
    if "Alpha" not in coefficients.index:
        raise KeyError("Coefficient table must include an 'Alpha' row.")

    if "Coefficient" not in coefficients.columns:
        raise KeyError("Coefficient table must include a 'Coefficient' column.")

    alpha_daily = float(coefficients.loc["Alpha", "Coefficient"])

    rows: list[dict[str, Any]] = []

    for scenario_name, assumptions in scenarios.items():
        estimated_annual = alpha_daily * trading_days

        factor_contributions: dict[str, float] = {}

        for factor, annual_factor_return in assumptions.items():
            if factor in coefficients.index:
                beta = float(coefficients.loc[factor, "Coefficient"])
                contribution = beta * float(annual_factor_return)
                estimated_annual += contribution
                factor_contributions[f"{factor} Contribution"] = contribution
            else:
                factor_contributions[f"{factor} Contribution"] = np.nan

        rows.append(
            {
                "Scenario": scenario_name,
                "Estimated Hermès Return": estimated_annual,
                "Annualized Alpha Contribution": alpha_daily * trading_days,
                **assumptions,
                **factor_contributions,
            }
        )

    scenario_table = pd.DataFrame(rows).set_index("Scenario")

    return scenario_table


def build_scenario_contribution_table(
    scenario_table: pd.DataFrame,
) -> pd.DataFrame:
    """Convert wide scenario contribution columns into a long-form table.

    Parameters
    ----------
    scenario_table:
        Scenario table returned by estimate_scenarios().

    Returns
    -------
    pd.DataFrame
        Long-form scenario contribution table for dashboard display.
    """
    if scenario_table.empty:
        return pd.DataFrame(
            columns=[
                "Scenario",
                "Driver",
                "Contribution",
                "Contribution Type",
            ]
        )

    rows: list[dict[str, Any]] = []

    contribution_columns = [
        col for col in scenario_table.columns if str(col).endswith(" Contribution")
    ]

    for scenario_name, scenario_row in scenario_table.iterrows():
        if "Annualized Alpha Contribution" in scenario_table.columns:
            rows.append(
                {
                    "Scenario": scenario_name,
                    "Driver": "Annualized Alpha",
                    "Contribution": float(
                        scenario_row["Annualized Alpha Contribution"]
                    ),
                    "Contribution Type": "Alpha",
                }
            )

        for column in contribution_columns:
            contribution_value = scenario_row[column]

            if pd.isna(contribution_value):
                continue

            driver = str(column).replace(" Contribution", "")

            rows.append(
                {
                    "Scenario": scenario_name,
                    "Driver": driver,
                    "Contribution": float(contribution_value),
                    "Contribution Type": "Factor",
                }
            )

    return pd.DataFrame(rows)