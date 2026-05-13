"""Project configuration for the Hermès market intelligence and equity research pipeline."""

from __future__ import annotations

from pathlib import Path

# =============================================================================
# Project paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = OUTPUT_DIR / "reports"
SNAPSHOT_DIR = OUTPUT_DIR / "snapshots"

REPORT_FILENAME = "hermes_market_intelligence_report.pdf"
REPORT_PATH = REPORT_DIR / REPORT_FILENAME

SNAPSHOT_PATH = SNAPSHOT_DIR / "last_run_snapshot.json"

for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CHART_DIR,
    TABLE_DIR,
    REPORT_DIR,
    SNAPSHOT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Project identity
# =============================================================================

PROJECT_NAME = "Hermès International Market Intelligence & Equity Research Pipeline"
REPORT_TITLE = "Hermès International Market Intelligence Report"

TARGET_COMPANY = "Hermès International SCA"
TARGET_TICKER = "RMS.PA"
US_OTC_TICKER = "HESAY"

TARGET_LABEL = "Hermès International"
BENCHMARK_LABEL = "CAC 40"

CURRENCY = "EUR"
US_ADR_CURRENCY = "USD"

KEY_TAKEAWAY = (
    "Hermès has materially outperformed major luxury peers since 2018, but the stock has also "
    "experienced a meaningful valuation reset from its 2025 peak. Official fundamentals remain "
    "strong, supported by high operating margins, strong net cash, and resilient Q1 2026 constant-FX "
    "growth. The model assigns a moderately constructive signal, with strength in brand moat and "
    "fundamentals offset by drawdown risk, valuation sensitivity, FX effects, and geopolitical risk."
)

REPORT_SIGNAL_LABELS = {
    "strong": "Constructive",
    "moderate": "Moderately Constructive",
    "neutral": "Neutral",
    "cautious": "Cautious",
    "negative": "Negative",
}


# =============================================================================
# Dashboard settings
# =============================================================================

DASHBOARD_TITLE = "Hermès International Market Intelligence Dashboard"

DASHBOARD_SUBTITLE = (
    "Interactive dashboard for valuation, market intelligence, geopolitical risk, "
    "news monitoring, shareholder returns, strategy overlays, and downloadable "
    "equity research outputs."
)

DASHBOARD_PAGES = [
    "Executive Overview",
    "Market Performance",
    "Valuation",
    "Shareholder Returns",
    "Market Intelligence",
    "News Intelligence",
    "Geopolitical Risk",
    "Strategy Overlay",
    "Export Center",
]

DASHBOARD_KEY_METRICS = [
    "Hermès Luxury Intelligence Score",
    "Valuation Signal",
    "Final Recommendation",
    "Risk Rating",
    "Market Intelligence View",
    "Implied Upside / Downside",
]

DASHBOARD_EXPORT_BASENAME = "hermes_market_intelligence"

DASHBOARD_TABLE_OUTPUTS = [
    "hermes_risk_dashboard_cards.csv",
    "hermes_risk_metric_leaderboard.csv",
    "hermes_target_risk_profile.csv",
    "hermes_valuation_dashboard_cards.csv",
    "hermes_valuation_bridge.csv",
    "hermes_valuation_sensitivity_dashboard.csv",
    "hermes_shareholder_dashboard_cards.csv",
    "hermes_shareholder_return_leaderboard.csv",
    "hermes_dividend_policy_dashboard.csv",
    "hermes_strategy_dashboard_cards.csv",
    "hermes_strategy_comparison_leaderboard.csv",
    "hermes_strategy_signal_summary.csv",
    "hermes_news_dashboard_cards.csv",
    "hermes_news_risk_catalyst_watchlist.csv",
    "hermes_news_source_summary.csv",
    "hermes_revenue_dashboard_cards.csv",
    "hermes_revenue_segment_leaderboard.csv",
    "hermes_revenue_fx_dashboard.csv",
    "hermes_geopolitical_dashboard_summary.csv",
    "hermes_geopolitical_risk_heatmap.csv",
    "hermes_geopolitical_scenario_dashboard.csv",
    "hermes_intelligence_score_dashboard.csv",
    "hermes_intelligence_score_contribution.csv",
    "hermes_intelligence_score_summary.csv",
    "hermes_market_intelligence_cards.csv",
    "hermes_market_signal_breakdown.csv",
    "hermes_market_watchlist.csv",
    "hermes_recommendation_dashboard_cards.csv",
    "hermes_recommendation_component_breakdown.csv",
    "hermes_recommendation_action_summary.csv",
]

DASHBOARD_CHART_OUTPUTS = [
    "hermes_risk_metric_leaderboard.png",
    "hermes_target_risk_profile.png",
    "hermes_revenue_segment_leaderboard.png",
    "hermes_revenue_fx_dashboard.png",
    "hermes_dividend_policy_dashboard.png",
    "hermes_shareholder_return_leaderboard.png",
    "hermes_strategy_comparison_leaderboard.png",
    "hermes_strategy_signal_summary.png",
    "hermes_valuation_bridge.png",
    "hermes_valuation_sensitivity_dashboard.png",
    "hermes_news_source_summary.png",
    "hermes_news_watchlist_classification.png",
]


# =============================================================================
# Tickers and labels
# =============================================================================

TICKERS = {
    TARGET_LABEL: TARGET_TICKER,
    "LVMH": "MC.PA",
    "Kering": "KER.PA",
    "Richemont": "CFR.SW",
    BENCHMARK_LABEL: "^FCHI",
    "EUR/USD": "EURUSD=X",
    "China Proxy ETF": "FXI",
}

PEER_LABELS = [
    TARGET_LABEL,
    "LVMH",
    "Kering",
    "Richemont",
    BENCHMARK_LABEL,
]

FACTOR_LABELS = [
    BENCHMARK_LABEL,
    "LVMH",
    "EUR/USD",
    "China Proxy ETF",
]

DISPLAY_LABELS = {
    "strategy_wealth": "Risk-Control Overlay",
    "buy_hold_wealth": "Buy & Hold",
    "strategy_drawdown": "Risk-Control Overlay",
    "buy_hold_drawdown": "Buy & Hold",
    "eps_eur": "EPS",
    "dividend_per_share_eur": "Dividend per Share",
    "market_cap_eur_bn": "Market Capitalization",
    "revenue": "Revenue",
    "revenue_eur_m": "Revenue",
    "recurring_operating_income": "Recurring Operating Income",
    "recurring_operating_income_eur_m": "Recurring Operating Income",
    "adjusted_free_cash_flows": "Adjusted Free Cash Flow",
    "adjusted_free_cash_flows_eur_m": "Adjusted Free Cash Flow",
    "restated_net_cash_position": "Restated Net Cash Position",
    "restated_net_cash_position_eur_m": "Restated Net Cash Position",
    "published_growth_pct": "Published Growth",
    "constant_fx_growth_pct": "Constant-FX Growth",
    "operating_margin_pct": "Operating Margin",
    "recurring_operating_margin": "Recurring Operating Margin",
    "growth_at_current_exchange_rates": "Growth at Current Exchange Rates",
    "growth_at_constant_exchange_rates": "Growth at Constant Exchange Rates",
}


# =============================================================================
# Market data settings
# =============================================================================

START_DATE = "2018-01-01"
END_DATE = None

TRADING_DAYS = 252
RISK_FREE_RATE = 0.03

OFFLINE_MODE_ENV_VAR = "HERMES_OFFLINE_MODE"
ALLOW_OFFLINE_SAMPLE_DATA = True


# =============================================================================
# Source URLs
# =============================================================================

SOURCE_URLS = {
    "Hermès Investor Relations": "https://finance.hermes.com/en/",
    "Hermès 2025 Digital Activity Report": (
        "https://hermes.fr.digital-report.net/en/report/5309"
    ),
    "Hermès Strategy": "https://finance.hermes.com/en/strategy/",
    "Hermès Key Figures": "https://finance.hermes.com/en/key-figures/",
    "Hermès Shareholder Guide": "https://finance.hermes.com/en/shareholders-guide/",
    "Hermès Share and Dividend": "https://finance.hermes.com/en/share-and-dividend/",
    "Hermès March 2026 Letter to Shareholders": (
        "https://assets-finance.hermes.com/s3fs-public/node/pdf_file/2026-03/"
        "1774373084/letter-to-shareholders_25032026_en.pdf"
    ),
    "Hermès Q1 2026 Revenue Release": (
        "https://assets-finance.hermes.com/s3fs-public/node/pdf_file/2026-04/"
        "1776186951/hermes_20260415_revenue_q1_en.pdf"
    ),
    "Yahoo Finance RMS.PA": "https://finance.yahoo.com/quote/RMS.PA/",
    "MarketBeat HESAY": "https://www.marketbeat.com/stocks/OTCMKTS/HESAY/",
}

SOURCE_CATEGORIES = {
    "Market Data": [
        "Yahoo Finance RMS.PA",
    ],
    "Official Hermès Sources": [
        "Hermès Investor Relations",
        "Hermès 2025 Digital Activity Report",
        "Hermès Strategy",
        "Hermès Key Figures",
        "Hermès Shareholder Guide",
        "Hermès Share and Dividend",
        "Hermès March 2026 Letter to Shareholders",
        "Hermès Q1 2026 Revenue Release",
    ],
    "U.S. OTC Reference": [
        "MarketBeat HESAY",
    ],
}


# =============================================================================
# Share information
# =============================================================================

SHARE_INFO = {
    "shares_outstanding": 105_569_412,
    "nominal_value_eur": 0.51,
    "exchange": "Euronext Paris",
    "compartment": "A",
    "isin": "FR0000052292",
    "bloomberg": "RMS-FP",
    "reuters": "HRMS.PA",
    "yahoo_finance": "RMS.PA",
    "us_otc": "HESAY",
}

SHAREHOLDER_GUIDE = {
    "registrar": "Uptevia",
    "registered_share_custody_fees_eur": 0.00,
    "bank_commission_per_order_eur": 7.62,
    "brokerage_cost_percent": 0.45,
    "brokerage_minimum_eur": 6.10,
}


# =============================================================================
# Hermès strategy and qualitative business model variables
# =============================================================================

STRATEGY_PILLARS = [
    "Creation",
    "Craftsmanship",
    "Exclusive and balanced distribution network",
]

BUSINESS_MODEL_FACTORS = {
    "ownership": "Independent company backed by family shareholders",
    "model": "Integrated craftsmanship business model",
    "distribution": "Exclusive omnichannel and multi-local distribution network",
    "production": "Craftsmanship-centered production capacity",
    "financial_policy": "Strict financial management and self-financed long-term investment",
}

STRATEGIC_MOAT_SCORE = {
    "Creative Freedom": 5,
    "Craftsmanship Model": 5,
    "Exclusive Distribution": 5,
    "Family Independence": 5,
    "Supply Chain Integration": 4,
    "Responsible Growth": 4,
}

QUALITATIVE_SIGNAL_WEIGHTS = {
    "Artisanal Model Strength": 5,
    "Vertical Integration": 5,
    "Exclusive Distribution": 5,
    "Production Capacity Expansion": 4,
    "Brand Desirability": 5,
    "Sustainability Recognition": 4,
    "Shareholder Communication Quality": 4,
}


# =============================================================================
# Q1 2026 revenue momentum
# =============================================================================

Q1_2026_HIGHLIGHTS = {
    "revenue_eur_m": 4070,
    "published_growth_pct": -1.4,
    "constant_fx_growth_pct": 5.6,
    "currency_headwind_eur_m": -290,
    "store_sales_growth_pct": 7.0,
    "americas_constant_fx_growth_pct": 17.2,
    "japan_constant_fx_growth_pct": 9.6,
    "europe_ex_france_constant_fx_growth_pct": 9.7,
    "asia_ex_japan_constant_fx_growth_pct": 2.2,
    "leather_goods_constant_fx_growth_pct": 9.4,
}


# =============================================================================
# Risk-control overlay settings
# =============================================================================

STRATEGY_NAME = "Risk-Control Overlay"

STRATEGY_DESCRIPTION = (
    "A rules-based volatility and trend filter designed to reduce downside exposure, "
    "not necessarily to outperform a long-only buy-and-hold position."
)

STRATEGY = {
    "short_ema": 50,
    "long_ema": 200,
    "rsi_window": 14,
    "rsi_max_buy": 70,
    "vol_window": 20,
    "max_daily_vol": 0.03,
    "target_annual_vol": 0.18,
    "max_exposure": 1.0,
}


# =============================================================================
# Risk model settings
# =============================================================================

RISK_MODEL = {
    "var_confidence_level": 0.95,
    "historical_var_window": 252,
    "ewma_lambda": 0.94,
    "expected_shortfall_confidence_level": 0.95,
    "kupiec_expected_breach_rate": 0.05,
}


# =============================================================================
# Monte Carlo settings
# =============================================================================

MONTE_CARLO = {
    "num_simulations": 1000,
    "forecast_days": 252,
    "random_seed": 42,
}


# =============================================================================
# Scenario assumptions
# =============================================================================

SCENARIOS = {
    "Bear Case": {
        BENCHMARK_LABEL: -0.12,
        "LVMH": -0.15,
        "EUR/USD": 0.04,
        "China Proxy ETF": -0.18,
    },
    "Base Case": {
        BENCHMARK_LABEL: 0.04,
        "LVMH": 0.05,
        "EUR/USD": 0.00,
        "China Proxy ETF": 0.03,
    },
    "Bull Case": {
        BENCHMARK_LABEL: 0.10,
        "LVMH": 0.14,
        "EUR/USD": -0.03,
        "China Proxy ETF": 0.12,
    },
    "China Slowdown": {
        BENCHMARK_LABEL: -0.06,
        "LVMH": -0.10,
        "EUR/USD": 0.01,
        "China Proxy ETF": -0.25,
    },
    "Luxury Re-rating": {
        BENCHMARK_LABEL: 0.06,
        "LVMH": 0.18,
        "EUR/USD": -0.01,
        "China Proxy ETF": 0.08,
    },
}


# =============================================================================
# News and market intelligence settings
# =============================================================================

NEWS_QUERIES = [
    "Hermès RMS.PA stock news",
    "Hermès luxury sales China Americas FX",
    "Hermès Q1 2026 revenue luxury demand",
    "Hermès LVMH Kering luxury sector demand",
    "Hermès Middle East tourism luxury sales",
]

NEWS_MAX_ITEMS_PER_QUERY = 8


# =============================================================================
# Geopolitical risk settings
# =============================================================================

GEOPOLITICAL_RISK_FACTORS = {
    "Middle East Conflict / Tourism Disruption": {
        "exposure": "Tourist flows, Middle East stores, airport concessions, European tourism",
        "risk_level": 4,
        "direction": "Negative",
        "description": (
            "Regional conflict can reduce tourist flows, disrupt Middle East luxury demand, "
            "and pressure airport and concession sales."
        ),
    },
    "China Demand / Policy Risk": {
        "exposure": "Asia-Pacific excluding Japan, Greater China demand, luxury consumer confidence",
        "risk_level": 4,
        "direction": "Negative",
        "description": (
            "China demand softness, policy shifts, wealth effects, or consumer confidence pressure "
            "can affect luxury-sector growth expectations."
        ),
    },
    "FX / Strong Euro Risk": {
        "exposure": "Reported revenue translation, tourist purchasing behavior, international affordability",
        "risk_level": 4,
        "direction": "Negative",
        "description": (
            "Currency moves can create reported revenue headwinds and affect cross-border luxury purchasing."
        ),
    },
    "Luxury Sector Re-rating": {
        "exposure": "Equity valuation multiples, peer sentiment, institutional positioning",
        "risk_level": 4,
        "direction": "Negative",
        "description": (
            "Luxury-sector multiple compression can weigh on Hermès even when company fundamentals remain strong."
        ),
    },
    "Trade / Tariff Risk": {
        "exposure": "Cross-border luxury sales, pricing, margins, U.S. and China demand",
        "risk_level": 3,
        "direction": "Negative",
        "description": (
            "Tariffs or trade restrictions could pressure pricing, margins, or regional demand."
        ),
    },
    "Supply Chain / Logistics Risk": {
        "exposure": "Production flow, raw materials, distribution, store inventory",
        "risk_level": 2,
        "direction": "Negative",
        "description": (
            "Hermès' integrated craftsmanship model is resilient, but logistics disruption can still affect operations."
        ),
    },
    "Energy / Inflation Risk": {
        "exposure": "Consumer spending, travel costs, operating costs, discount rates",
        "risk_level": 3,
        "direction": "Negative",
        "description": (
            "Energy shocks and inflation can reduce discretionary demand and pressure valuation multiples."
        ),
    },
    "Cyber / Operational Risk": {
        "exposure": "Digital commerce, operations, customer data, brand trust",
        "risk_level": 2,
        "direction": "Negative",
        "description": (
            "Cyber incidents can disrupt operations and damage brand trust, even for high-quality luxury brands."
        ),
    },
}

GEOPOLITICAL_SCENARIO_ADJUSTMENTS = {
    "Base Geopolitical Risk": {
        "revenue_growth_adjustment_pct": 0.0,
        "margin_adjustment_pct": 0.0,
        "valuation_multiple_adjustment_pct": 0.0,
    },
    "Middle East / Tourism Shock": {
        "revenue_growth_adjustment_pct": -1.5,
        "margin_adjustment_pct": -0.3,
        "valuation_multiple_adjustment_pct": -5.0,
    },
    "China Demand Shock": {
        "revenue_growth_adjustment_pct": -3.0,
        "margin_adjustment_pct": -0.5,
        "valuation_multiple_adjustment_pct": -8.0,
    },
    "FX and Rate Pressure": {
        "revenue_growth_adjustment_pct": -2.0,
        "margin_adjustment_pct": -0.2,
        "valuation_multiple_adjustment_pct": -6.0,
    },
    "Combined Geopolitical Stress": {
        "revenue_growth_adjustment_pct": -5.0,
        "margin_adjustment_pct": -1.0,
        "valuation_multiple_adjustment_pct": -12.0,
    },
}


# =============================================================================
# Valuation model settings
# =============================================================================

VALUATION_FORECAST_YEARS = [2026, 2027, 2028, 2029, 2030]

VALUATION_REVENUE_GROWTH_RATES = {
    2026: 0.075,
    2027: 0.070,
    2028: 0.065,
    2029: 0.060,
    2030: 0.055,
}

VALUATION_OPERATING_MARGIN_RATES = {
    2026: 0.405,
    2027: 0.407,
    2028: 0.408,
    2029: 0.409,
    2030: 0.410,
}

VALUATION_NET_INCOME_MARGIN_RATES = {
    2026: 0.285,
    2027: 0.286,
    2028: 0.287,
    2029: 0.288,
    2030: 0.289,
}

VALUATION_FCF_CONVERSION_RATES = {
    2026: 0.245,
    2027: 0.247,
    2028: 0.248,
    2029: 0.249,
    2030: 0.250,
}

VALUATION_DIVIDEND_PAYOUT_RATES = {
    2026: 0.42,
    2027: 0.43,
    2028: 0.44,
    2029: 0.45,
    2030: 0.45,
}

VALUATION_PE_MULTIPLES = {
    "Bear P/E": 38.0,
    "Base P/E": 45.0,
    "Bull P/E": 52.0,
}

VALUATION_WACC = 0.085
VALUATION_TERMINAL_GROWTH = 0.025

VALUATION_WACC_SENSITIVITY = [0.075, 0.080, 0.085, 0.090, 0.095]
VALUATION_TERMINAL_GROWTH_SENSITIVITY = [0.015, 0.020, 0.025, 0.030, 0.035]


# =============================================================================
# Recommendation framework settings
# =============================================================================

RECOMMENDATION_LABELS = {
    "positive": "Outperform",
    "neutral": "Neutral",
    "negative": "Underperform",
}

RECOMMENDATION_THRESHOLDS = {
    "outperform_points": 4,
    "underperform_points": -3,
    "supportive_upside": 0.15,
    "moderate_upside": 0.05,
    "stretched_downside": -0.10,
    "severe_downside": -0.15,
}