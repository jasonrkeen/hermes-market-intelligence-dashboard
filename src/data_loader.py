"""Data loading utilities for the Hermès market intelligence pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import (
    ALLOW_OFFLINE_SAMPLE_DATA,
    OFFLINE_MODE_ENV_VAR,
    RAW_DATA_DIR,
    TARGET_LABEL,
)


def _is_offline_mode_enabled() -> bool:
    """Return True if offline mode is enabled through the environment variable."""
    value = os.getenv(OFFLINE_MODE_ENV_VAR, "").strip().lower()
    return value in {"1", "true", "yes", "y"}


def save_data_source_status(
    table_dir: Path,
    source: str,
    status: str,
    message: str,
) -> pd.DataFrame:
    """Save a small data-source status table for the dashboard and report.

    Parameters
    ----------
    table_dir:
        Output directory for CSV tables.
    source:
        Name of the data source used.
    status:
        Short status label.
    message:
        Human-readable explanation of the latest data-source behavior.

    Returns
    -------
    pd.DataFrame
        One-row data source status table.
    """
    status_table = pd.DataFrame(
        [
            {
                "Source": source,
                "Status": status,
                "Message": message,
                "Timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
            }
        ]
    )

    table_dir.mkdir(parents=True, exist_ok=True)
    status_table.to_csv(table_dir / "hermes_data_source_status.csv", index=False)

    return status_table


def _generate_sample_price_data(
    tickers: dict[str, str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """Generate deterministic sample price data for offline testing.

    This allows the full project pipeline to run even when Yahoo Finance is
    unavailable. The data is synthetic and should only be used for testing.
    """
    rng = np.random.default_rng(42)

    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end) if end else pd.Timestamp.today().normalize()

    dates = pd.bdate_range(start=start_date, end=end_date)

    if dates.empty:
        raise RuntimeError(
            "Offline sample data could not be generated because the date range is empty."
        )

    price_data: dict[str, pd.Series] = {}

    base_specs = {
        TARGET_LABEL: {"start_price": 450.0, "drift": 0.00065, "vol": 0.016},
        "LVMH": {"start_price": 250.0, "drift": 0.00035, "vol": 0.018},
        "Kering": {"start_price": 420.0, "drift": -0.00010, "vol": 0.021},
        "Richemont": {"start_price": 80.0, "drift": 0.00030, "vol": 0.019},
        "CAC 40": {"start_price": 5200.0, "drift": 0.00020, "vol": 0.011},
        "EUR/USD": {"start_price": 1.12, "drift": -0.00002, "vol": 0.004},
        "China Proxy ETF": {"start_price": 42.0, "drift": 0.00005, "vol": 0.020},
    }

    for label in tickers:
        spec = base_specs.get(
            label,
            {"start_price": 100.0, "drift": 0.00020, "vol": 0.015},
        )

        daily_returns = rng.normal(
            loc=spec["drift"],
            scale=spec["vol"],
            size=len(dates),
        )

        prices = spec["start_price"] * np.exp(np.cumsum(daily_returns))
        price_data[label] = pd.Series(prices, index=dates)

    sample_prices = pd.DataFrame(price_data)
    sample_prices.index.name = "Date"

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    sample_prices.to_csv(RAW_DATA_DIR / "offline_sample_price_data.csv")

    return sample_prices


def _extract_close_prices(raw: pd.DataFrame) -> pd.DataFrame:
    """Extract close prices from a yfinance download response."""
    if raw.empty:
        raise RuntimeError("No price data was downloaded.")

    if isinstance(raw.columns, pd.MultiIndex):
        top_level = raw.columns.get_level_values(0)

        if "Close" in top_level:
            close = raw["Close"].copy()
        elif "Adj Close" in top_level:
            close = raw["Adj Close"].copy()
        else:
            available_columns = ", ".join(sorted(set(map(str, top_level))))
            raise RuntimeError(
                "Could not find Close or Adj Close in downloaded data. "
                f"Available top-level columns: {available_columns}"
            )
    else:
        if "Close" in raw.columns:
            close = raw[["Close"]].copy()
        elif "Adj Close" in raw.columns:
            close = raw[["Adj Close"]].copy()
        else:
            close = raw.copy()

    return close


def _rename_downloaded_columns(
    close: pd.DataFrame,
    tickers: dict[str, str],
) -> pd.DataFrame:
    """Rename downloaded yfinance columns to project-friendly labels."""
    reverse = {ticker: label for label, ticker in tickers.items()}

    renamed = close.rename(columns=reverse)

    if len(tickers) == 1 and list(renamed.columns) == ["Close"]:
        only_label = next(iter(tickers.keys()))
        renamed = renamed.rename(columns={"Close": only_label})

    return renamed


def download_adjusted_close(
    tickers: dict[str, str],
    start: str,
    end: str | None = None,
    table_dir: Path | None = None,
) -> pd.DataFrame:
    """Download adjusted close prices and return friendly column names.

    Parameters
    ----------
    tickers:
        Dictionary mapping friendly labels to Yahoo Finance ticker symbols.
    start:
        Start date for price download.
    end:
        Optional end date for price download.
    table_dir:
        Optional output directory for saving dashboard/report data-source status.

    Returns
    -------
    pd.DataFrame
        Price DataFrame with friendly column names such as Hermès International,
        LVMH, Kering, Richemont, CAC 40, EUR/USD, and China Proxy ETF.
    """
    if _is_offline_mode_enabled():
        if not ALLOW_OFFLINE_SAMPLE_DATA:
            raise RuntimeError(
                f"{OFFLINE_MODE_ENV_VAR} is enabled, but offline sample data is disabled."
            )

        print("Offline mode enabled. Using synthetic sample price data.")

        if table_dir is not None:
            save_data_source_status(
                table_dir=table_dir,
                source="Offline sample data",
                status="Synthetic data used",
                message=(
                    f"{OFFLINE_MODE_ENV_VAR} is enabled. The pipeline used "
                    "deterministic synthetic sample data instead of live market data."
                ),
            )

        return _generate_sample_price_data(tickers=tickers, start=start, end=end)

    try:
        raw = yf.download(
            list(tickers.values()),
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )

        close = _extract_close_prices(raw)

    except Exception as exc:
        if not ALLOW_OFFLINE_SAMPLE_DATA:
            raise RuntimeError(
                "Yahoo Finance download failed and offline sample data is disabled."
            ) from exc

        print("Yahoo Finance download failed. Falling back to offline sample price data.")
        print(f"Reason: {exc}")

        if table_dir is not None:
            save_data_source_status(
                table_dir=table_dir,
                source="Offline sample data",
                status="Fallback used",
                message=(
                    "Yahoo Finance download failed, so the pipeline used "
                    f"deterministic synthetic sample data. Reason: {exc}"
                ),
            )

        return _generate_sample_price_data(tickers=tickers, start=start, end=end)

    close = _rename_downloaded_columns(close=close, tickers=tickers)

    expected_columns = list(tickers.keys())
    available_columns = [col for col in expected_columns if col in close.columns]

    if not available_columns:
        raise RuntimeError(
            "Downloaded data did not match the expected ticker labels. "
            f"Expected one of: {expected_columns}. Received: {list(close.columns)}"
        )

    close = close[available_columns]
    close = close.sort_index().dropna(how="all")
    close.index.name = "Date"

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    close.to_csv(RAW_DATA_DIR / "downloaded_adjusted_close_prices.csv")

    if table_dir is not None:
        save_data_source_status(
            table_dir=table_dir,
            source="Yahoo Finance",
            status="Live download successful",
            message=(
                "The pipeline successfully downloaded live adjusted close price data "
                "from Yahoo Finance using yfinance."
            ),
        )

    return close