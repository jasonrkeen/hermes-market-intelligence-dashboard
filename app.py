"""Streamlit dashboard for the Hermès market intelligence pipeline.

Run from the project root:

    streamlit run app.py

or:

    python -m streamlit run app.py

The dashboard reads generated outputs from the existing pipeline:
- outputs/tables
- outputs/charts
- outputs/reports/hermes_market_intelligence_report.pdf
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


# =============================================================================
# Paths
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
CHART_DIR = OUTPUT_DIR / "charts"
REPORT_DIR = OUTPUT_DIR / "reports"
REPORT_PATH = REPORT_DIR / "hermes_market_intelligence_report.pdf"


# =============================================================================
# Page config
# =============================================================================

st.set_page_config(
    page_title="Hermès Market Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
)


# =============================================================================
# Helpers
# =============================================================================

@st.cache_data
def load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV table from the outputs/tables directory."""
    path = TABLE_DIR / filename

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def show_dataframe(
    df: pd.DataFrame,
    title: str,
    height: int = 320,
) -> None:
    """Display a dataframe section if available."""
    st.subheader(title)

    if df.empty:
        st.info(f"{title} is not available yet. Run `python main.py` first.")
        return

    st.dataframe(df, width="stretch", height=height)


def show_chart(filename: str, caption: str | None = None) -> None:
    """Display a generated chart image if available."""
    path = CHART_DIR / filename

    if not path.exists():
        st.info(f"Chart not found: {filename}")
        return

    st.image(str(path), caption=caption, width="stretch")


def extract_metric_value(
    df: pd.DataFrame,
    metric_col: str,
    metric_name: str,
    value_col: str = "Value",
    default: str = "Not Available",
) -> str:
    """Extract a metric from a long-form metric table."""
    if df.empty:
        return default

    if metric_col not in df.columns or value_col not in df.columns:
        return default

    rows = df[df[metric_col].astype(str).eq(metric_name)]

    if rows.empty:
        return default

    return str(rows.iloc[0][value_col])


def format_number(value: object) -> str:
    """Format simple numeric values for cards."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if abs(numeric_value) >= 1_000:
        return f"{numeric_value:,.2f}"

    return f"{numeric_value:.2f}"


def format_card_value(value: Any, display_value: Any | None = None) -> str:
    """Prefer a display value when present, otherwise format the raw value."""
    if display_value is not None and not pd.isna(display_value):
        return str(display_value)

    if value is None or pd.isna(value):
        return "Not Available"

    return format_number(value)


def show_card_grid(
    cards: pd.DataFrame,
    title: str,
    max_cards: int | None = None,
    columns: int = 4,
) -> None:
    """Display dashboard card tables as Streamlit metric cards."""
    st.subheader(title)

    if cards.empty:
        st.info(f"{title} is not available yet. Run `python main.py` first.")
        return

    card_col = "Card" if "Card" in cards.columns else None
    value_col = "Display Value" if "Display Value" in cards.columns else "Value"
    raw_value_col = "Value" if "Value" in cards.columns else value_col

    interpretation_col = None

    for candidate in ["Interpretation", "Description"]:
        if candidate in cards.columns:
            interpretation_col = candidate
            break

    if card_col is None or value_col not in cards.columns:
        st.dataframe(cards, width="stretch")
        return

    display_cards = cards.copy()

    if max_cards is not None:
        display_cards = display_cards.head(max_cards)

    for start in range(0, len(display_cards), columns):
        row_cards = display_cards.iloc[start:start + columns]
        streamlit_columns = st.columns(len(row_cards))

        for column, (_, row) in zip(streamlit_columns, row_cards.iterrows()):
            label = str(row[card_col])
            value = format_card_value(
                row.get(raw_value_col),
                row.get(value_col),
            )

            with column:
                value_text = str(value)

                numeric_like_value = (
                    value_text.replace(".", "", 1)
                    .replace(",", "")
                    .replace("-", "", 1)
                    .isdigit()
                )

                is_long_text_value = (
                    len(value_text) > 18
                    and not numeric_like_value
                    and "%" not in value_text
                    and "€" not in value_text
                )

                if is_long_text_value:
                    st.markdown(f"**{label}**")
                    st.markdown(
                        (
                            "<div style='font-size: 1.65rem; line-height: 1.2; "
                            "font-weight: 500; margin-bottom: 0.35rem;'>"
                            f"{value_text}"
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    st.metric(label, value)

                if interpretation_col is not None:
                    interpretation = row.get(interpretation_col)

                    if interpretation is not None and not pd.isna(interpretation):
                        st.caption(str(interpretation))


def download_pdf_button(key: str = "download_pdf_report") -> None:
    """Render PDF download button."""
    if not REPORT_PATH.exists():
        st.warning("PDF report not found. Run `python main.py` first.")
        return

    with REPORT_PATH.open("rb") as file:
        st.download_button(
            label="Download PDF Report",
            data=file,
            file_name=REPORT_PATH.name,
            mime="application/pdf",
            width="stretch",
            key=key,
        )


# =============================================================================
# Load core outputs
# =============================================================================

metrics = load_csv("hermes_risk_return_metrics.csv")
var_summary = load_csv("hermes_var_backtest_summary.csv")
intelligence_score = load_csv("hermes_luxury_intelligence_score.csv")
market_summary = load_csv("hermes_market_intelligence_summary.csv")
market_dashboard = load_csv("hermes_market_intelligence_dashboard.csv")
recommendation_summary = load_csv("hermes_recommendation_summary.csv")
recommendation_framework = load_csv("hermes_recommendation_framework.csv")
valuation_signal = load_csv("hermes_valuation_signal.csv")
fair_value_summary = load_csv("hermes_fair_value_summary.csv")
dcf_valuation = load_csv("hermes_dcf_valuation.csv")
peer_valuation = load_csv("hermes_peer_multiple_valuation.csv")
dcf_sensitivity = load_csv("hermes_dcf_sensitivity.csv")
news_summary = load_csv("hermes_news_intelligence_summary.csv")
news_theme_summary = load_csv("hermes_news_theme_summary.csv")
geopolitical_summary = load_csv("hermes_geopolitical_risk_summary.csv")
geopolitical_risks = load_csv("hermes_geopolitical_risk_factors.csv")
geopolitical_scenarios = load_csv("hermes_geopolitical_scenario_overlay.csv")
revenue_momentum = load_csv("hermes_revenue_momentum_summary.csv")
strategy_summary = load_csv("hermes_strategy_summary.csv")
strategy_scores = load_csv("hermes_strategy_qualitative_score.csv")


# =============================================================================
# Load upgraded dashboard outputs
# =============================================================================

risk_dashboard_cards = load_csv("hermes_risk_dashboard_cards.csv")
risk_metric_leaderboard = load_csv("hermes_risk_metric_leaderboard.csv")
target_risk_profile = load_csv("hermes_target_risk_profile.csv")

valuation_dashboard_cards = load_csv("hermes_valuation_dashboard_cards.csv")
valuation_bridge = load_csv("hermes_valuation_bridge.csv")
valuation_sensitivity_dashboard = load_csv(
    "hermes_valuation_sensitivity_dashboard.csv"
)

shareholder_dashboard_cards = load_csv("hermes_shareholder_dashboard_cards.csv")
shareholder_return_leaderboard = load_csv(
    "hermes_shareholder_return_leaderboard.csv"
)
dividend_policy_dashboard = load_csv("hermes_dividend_policy_dashboard.csv")
shareholder_summary = load_csv("hermes_shareholder_summary.csv")
dividend_metrics = load_csv("hermes_dividend_metrics.csv")

strategy_dashboard_cards = load_csv("hermes_strategy_dashboard_cards.csv")
strategy_comparison_leaderboard = load_csv(
    "hermes_strategy_comparison_leaderboard.csv"
)
strategy_signal_summary = load_csv("hermes_strategy_signal_summary.csv")

news_dashboard_cards = load_csv("hermes_news_dashboard_cards.csv")
news_risk_catalyst_watchlist = load_csv(
    "hermes_news_risk_catalyst_watchlist.csv"
)
news_source_summary = load_csv("hermes_news_source_summary.csv")

revenue_dashboard_cards = load_csv("hermes_revenue_dashboard_cards.csv")
revenue_segment_leaderboard = load_csv("hermes_revenue_segment_leaderboard.csv")
revenue_fx_dashboard = load_csv("hermes_revenue_fx_dashboard.csv")

geopolitical_dashboard_summary = load_csv(
    "hermes_geopolitical_dashboard_summary.csv"
)
geopolitical_risk_heatmap = load_csv("hermes_geopolitical_risk_heatmap.csv")
geopolitical_scenario_dashboard = load_csv(
    "hermes_geopolitical_scenario_dashboard.csv"
)

intelligence_score_dashboard = load_csv("hermes_intelligence_score_dashboard.csv")
intelligence_score_contribution = load_csv(
    "hermes_intelligence_score_contribution.csv"
)
intelligence_score_summary = load_csv("hermes_intelligence_score_summary.csv")

market_intelligence_cards = load_csv("hermes_market_intelligence_cards.csv")
market_signal_breakdown = load_csv("hermes_market_signal_breakdown.csv")
market_watchlist = load_csv("hermes_market_watchlist.csv")

recommendation_dashboard_cards = load_csv(
    "hermes_recommendation_dashboard_cards.csv"
)
recommendation_component_breakdown = load_csv(
    "hermes_recommendation_component_breakdown.csv"
)
recommendation_action_summary = load_csv(
    "hermes_recommendation_action_summary.csv"
)

data_source_status = load_csv("hermes_data_source_status.csv")
change_detection = load_csv("hermes_change_detection.csv")


# =============================================================================
# Sidebar
# =============================================================================

st.sidebar.title("Hermès Dashboard")
st.sidebar.caption("Market Intelligence & Equity Research Pipeline")

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Market Performance",
        "Valuation",
        "Shareholder Returns",
        "Market Intelligence",
        "News Intelligence",
        "Geopolitical Risk",
        "Strategy Overlay",
        "Export Center",
    ],
)

st.sidebar.divider()
download_pdf_button(key="sidebar_download_pdf")

st.sidebar.divider()
st.sidebar.caption("Run `python main.py` to refresh the data, charts, and PDF.")


# =============================================================================
# Header
# =============================================================================

st.title("Hermès International Market Intelligence Dashboard")
st.caption(
    "Interactive dashboard layer for the Hermès market intelligence and equity research pipeline."
)


# =============================================================================
# Executive Overview
# =============================================================================

if page == "Executive Overview":
    st.header("Executive Overview")

    final_recommendation = extract_metric_value(
        recommendation_summary,
        metric_col="Metric",
        metric_name="Final Recommendation",
    )

    risk_rating = extract_metric_value(
        recommendation_summary,
        metric_col="Metric",
        metric_name="Risk Rating",
    )

    valuation_view = extract_metric_value(
        valuation_signal,
        metric_col="Metric",
        metric_name="Valuation Signal",
    )

    implied_upside = extract_metric_value(
        valuation_signal,
        metric_col="Metric",
        metric_name="Implied Upside / Downside",
    )

    market_view = extract_metric_value(
        market_summary,
        metric_col="Section",
        metric_name="Overall Market Intelligence View",
        value_col="Summary",
    )

    score_value = "Not Available"

    if not intelligence_score.empty:
        if "Component" in intelligence_score.columns and "Score" in intelligence_score.columns:
            score_rows = intelligence_score[
                intelligence_score["Component"].astype(str).eq(
                    "Hermès Luxury Intelligence Score"
                )
            ]

            if not score_rows.empty:
                score_value = format_number(score_rows.iloc[0]["Score"])

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Luxury Intelligence Score", score_value)
    col2.metric("Valuation Signal", valuation_view)
    col3.metric("Recommendation", final_recommendation)
    col4.metric("Risk Rating", risk_rating)

    col5, col6 = st.columns(2)
    col5.metric("Market Intelligence View", market_view)
    col6.metric("Implied Upside / Downside", implied_upside)

    st.divider()

    show_card_grid(
        recommendation_dashboard_cards,
        "Recommendation Dashboard Cards",
        max_cards=6,
    )

    show_card_grid(
        valuation_dashboard_cards,
        "Valuation Dashboard Cards",
        max_cards=4,
    )

    show_card_grid(
        market_intelligence_cards,
        "Market Intelligence Cards",
        max_cards=4,
    )

    st.divider()

    left, right = st.columns([1.1, 1])

    with left:
        show_dataframe(market_dashboard, "Market Intelligence Dashboard", height=420)

    with right:
        show_chart(
            "hermes_recommendation_framework.png",
            "Recommendation point contribution bridge",
        )

    st.divider()

    show_dataframe(recommendation_summary, "Recommendation Summary", height=240)
    show_dataframe(valuation_signal, "Valuation Signal", height=240)
    show_dataframe(change_detection, "Run-to-Run Change Detection", height=240)


# =============================================================================
# Market Performance
# =============================================================================

elif page == "Market Performance":
    st.header("Market Performance")

    show_card_grid(
        risk_dashboard_cards,
        "Risk Dashboard Cards",
        max_cards=8,
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        show_chart("hermes_price_history.png", "Hermès price history")
        show_chart(
            "luxury_peer_normalized_performance.png",
            "Luxury peer normalized performance",
        )
        show_chart(
            "hermes_risk_metric_leaderboard.png",
            "Peer risk leaderboard",
        )

    with col2:
        show_chart("hermes_garch_style_volatility.png", "GARCH-style volatility")
        show_chart(
            "hermes_var_backtest.png",
            "VaR backtest: returns versus risk thresholds",
        )
        show_chart(
            "hermes_target_risk_profile.png",
            "Hermès target risk profile",
        )

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Risk/Return Metrics",
            "Risk Leaderboard",
            "Target Risk Profile",
            "VaR Summary",
        ]
    )

    with tab1:
        show_dataframe(metrics, "Risk/Return Metrics", height=360)

    with tab2:
        show_dataframe(
            risk_metric_leaderboard,
            "Risk Metric Leaderboard",
            height=360,
        )

    with tab3:
        show_dataframe(target_risk_profile, "Target Risk Profile", height=320)

    with tab4:
        show_dataframe(var_summary, "VaR Backtest Summary", height=260)


# =============================================================================
# Valuation
# =============================================================================

elif page == "Valuation":
    st.header("Valuation")

    show_card_grid(
        valuation_dashboard_cards,
        "Valuation Dashboard Cards",
        max_cards=7,
    )

    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        show_chart("hermes_fair_value_range.png", "Fair value range")
        show_chart("hermes_valuation_bridge.png", "Valuation bridge")
        show_dataframe(fair_value_summary, "Fair Value Summary", height=250)

    with col2:
        show_chart(
            "hermes_dcf_sensitivity.png",
            "DCF sensitivity: WACC versus terminal growth",
        )
        show_chart(
            "hermes_valuation_sensitivity_dashboard.png",
            "Valuation sensitivity dashboard",
        )
        show_dataframe(valuation_signal, "Valuation Signal", height=250)

    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Valuation Bridge",
            "DCF Valuation",
            "Peer Multiple Valuation",
            "DCF Sensitivity Table",
            "Sensitivity Dashboard",
        ]
    )

    with tab1:
        show_dataframe(valuation_bridge, "Valuation Bridge", height=420)

    with tab2:
        show_dataframe(dcf_valuation, "DCF Valuation", height=420)

    with tab3:
        show_dataframe(peer_valuation, "Peer Multiple Valuation", height=300)

    with tab4:
        show_dataframe(dcf_sensitivity, "DCF Sensitivity", height=300)

    with tab5:
        show_dataframe(
            valuation_sensitivity_dashboard,
            "Valuation Sensitivity Dashboard",
            height=420,
        )


# =============================================================================
# Shareholder Returns
# =============================================================================

elif page == "Shareholder Returns":
    st.header("Shareholder Returns")

    show_card_grid(
        shareholder_dashboard_cards,
        "Shareholder Return Dashboard Cards",
        max_cards=9,
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        show_chart("hermes_eps_vs_dividend.png", "EPS versus dividend per share")
        show_chart("hermes_dividend_yield.png", "Dividend yield")

    with col2:
        show_chart("hermes_market_cap_history.png", "Market capitalization history")
        show_chart(
            "hermes_dividend_policy_dashboard.png",
            "Dividend payout policy",
        )

    st.divider()

    show_chart(
        "hermes_shareholder_return_leaderboard.png",
        "EPS growth versus dividend growth",
    )

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Shareholder Summary",
            "Dividend Metrics",
            "Return Leaderboard",
            "Dividend Policy Dashboard",
        ]
    )

    with tab1:
        show_dataframe(shareholder_summary, "Shareholder Summary", height=260)

    with tab2:
        show_dataframe(dividend_metrics, "Dividend Metrics", height=420)

    with tab3:
        show_dataframe(
            shareholder_return_leaderboard,
            "Shareholder Return Leaderboard",
            height=420,
        )

    with tab4:
        show_dataframe(
            dividend_policy_dashboard,
            "Dividend Policy Dashboard",
            height=360,
        )


# =============================================================================
# Market Intelligence
# =============================================================================

elif page == "Market Intelligence":
    st.header("Market Intelligence")

    show_card_grid(
        intelligence_score_dashboard,
        "Luxury Intelligence Score Dashboard",
        max_cards=4,
    )

    show_card_grid(
        revenue_dashboard_cards,
        "Revenue Momentum Cards",
        max_cards=6,
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        show_dataframe(market_summary, "Market Intelligence Summary", height=250)
        show_dataframe(revenue_momentum, "Q1 2026 Revenue Momentum", height=320)
        show_chart(
            "hermes_revenue_segment_leaderboard.png",
            "Q1 revenue segment leaderboard",
        )

    with col2:
        show_chart(
            "hermes_market_intelligence_dashboard.png",
            "Numeric market intelligence signals",
        )
        show_chart(
            "hermes_intelligence_score.png",
            "Luxury Intelligence Score components",
        )
        show_chart(
            "hermes_revenue_fx_dashboard.png",
            "Revenue FX dashboard",
        )

    st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Luxury Intelligence Score",
            "Score Contribution",
            "Revenue Leaderboard",
            "Revenue FX Dashboard",
            "Market Signal Breakdown",
            "Market Watchlist",
        ]
    )

    with tab1:
        show_dataframe(intelligence_score, "Luxury Intelligence Score", height=360)

    with tab2:
        show_dataframe(
            intelligence_score_contribution,
            "Intelligence Score Contribution",
            height=360,
        )

    with tab3:
        show_dataframe(
            revenue_segment_leaderboard,
            "Revenue Segment Leaderboard",
            height=420,
        )

    with tab4:
        show_dataframe(revenue_fx_dashboard, "Revenue FX Dashboard", height=360)

    with tab5:
        show_dataframe(
            market_signal_breakdown,
            "Market Signal Breakdown",
            height=360,
        )

    with tab6:
        show_dataframe(market_watchlist, "Market Watchlist", height=420)


# =============================================================================
# News Intelligence
# =============================================================================

elif page == "News Intelligence":
    st.header("News Intelligence")

    show_card_grid(
        news_dashboard_cards,
        "News Dashboard Cards",
        max_cards=6,
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        show_dataframe(news_summary, "News Intelligence Summary", height=260)
        show_chart(
            "hermes_news_classification_summary.png",
            "News classification summary",
        )
        show_chart(
            "hermes_news_source_summary.png",
            "News source summary",
        )

    with col2:
        show_chart(
            "hermes_news_theme_summary.png",
            "News theme summary",
        )
        show_chart(
            "hermes_news_watchlist_classification.png",
            "News watchlist classification",
        )
        show_dataframe(news_theme_summary, "News Theme Summary", height=320)

    st.divider()

    tab1, tab2, tab3 = st.tabs(
        [
            "Risk/Catalyst Watchlist",
            "News Source Summary",
            "News Theme Summary",
        ]
    )

    with tab1:
        show_dataframe(
            news_risk_catalyst_watchlist,
            "News Risk/Catalyst Watchlist",
            height=500,
        )

    with tab2:
        show_dataframe(news_source_summary, "News Source Summary", height=360)

    with tab3:
        show_dataframe(news_theme_summary, "News Theme Summary", height=360)


# =============================================================================
# Geopolitical Risk
# =============================================================================

elif page == "Geopolitical Risk":
    st.header("Geopolitical Risk")

    show_card_grid(
        geopolitical_dashboard_summary,
        "Geopolitical Dashboard Summary",
        max_cards=6,
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        show_dataframe(geopolitical_summary, "Geopolitical Risk Summary", height=260)
        show_chart(
            "hermes_geopolitical_risk_factors.png",
            "Geopolitical risk factor scores",
        )

    with col2:
        show_chart(
            "hermes_geopolitical_category_summary.png",
            "Geopolitical risk by category",
        )
        show_chart(
            "hermes_geopolitical_scenario_overlay.png",
            "Geopolitical scenario stress overlay",
        )

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Risk Factors",
            "Risk Heatmap",
            "Scenario Overlay",
            "Scenario Dashboard",
        ]
    )

    with tab1:
        show_dataframe(geopolitical_risks, "Geopolitical Risk Factors", height=420)

    with tab2:
        show_dataframe(
            geopolitical_risk_heatmap,
            "Geopolitical Risk Heatmap",
            height=420,
        )

    with tab3:
        show_dataframe(
            geopolitical_scenarios,
            "Geopolitical Scenario Overlay",
            height=320,
        )

    with tab4:
        show_dataframe(
            geopolitical_scenario_dashboard,
            "Geopolitical Scenario Dashboard",
            height=420,
        )


# =============================================================================
# Strategy Overlay
# =============================================================================

elif page == "Strategy Overlay":
    st.header("Strategy Overlay")

    show_card_grid(
        strategy_dashboard_cards,
        "Strategy Dashboard Cards",
        max_cards=8,
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        show_chart(
            "hermes_strategy_vs_buy_hold.png",
            "Risk-control overlay versus buy & hold",
        )
        show_chart(
            "hermes_strategy_overlay_comparison.png",
            "Strategy overlay comparison",
        )
        show_chart(
            "hermes_strategy_comparison_leaderboard.png",
            "Strategy comparison leaderboard",
        )

    with col2:
        show_chart(
            "hermes_drawdown_analysis.png",
            "Drawdown analysis",
        )
        show_chart(
            "hermes_strategy_drawdown_comparison.png",
            "Strategy drawdown comparison",
        )
        show_chart(
            "hermes_strategy_signal_summary.png",
            "Strategy signal and exposure summary",
        )

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Strategy Overlay Summary",
            "Strategy Leaderboard",
            "Signal Summary",
            "Strategic Moat Scores",
        ]
    )

    with tab1:
        show_dataframe(strategy_summary, "Strategy Overlay Summary", height=320)

    with tab2:
        show_dataframe(
            strategy_comparison_leaderboard,
            "Strategy Comparison Leaderboard",
            height=420,
        )

    with tab3:
        show_dataframe(strategy_signal_summary, "Strategy Signal Summary", height=360)

    with tab4:
        show_dataframe(
            strategy_scores,
            "Strategic Moat and Qualitative Scores",
            height=360,
        )


# =============================================================================
# Export Center
# =============================================================================

elif page == "Export Center":
    st.header("Export Center")

    st.write(
        "Download the generated institutional-style PDF report or inspect the output folders."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        download_pdf_button(key="export_center_download_pdf")

    with col2:
        st.metric("Tables Folder", str(TABLE_DIR.relative_to(BASE_DIR)))

    with col3:
        st.metric("Charts Folder", str(CHART_DIR.relative_to(BASE_DIR)))

    st.divider()

    show_dataframe(data_source_status, "Data Source Status", height=260)
    show_dataframe(change_detection, "Run-to-Run Change Detection", height=260)

    st.divider()

    st.subheader("Available Output Files")

    output_files = []

    if OUTPUT_DIR.exists():
        for path in sorted(OUTPUT_DIR.rglob("*")):
            if path.is_file():
                output_files.append(
                    {
                        "File": str(path.relative_to(BASE_DIR)),
                        "Size KB": round(path.stat().st_size / 1024, 2),
                    }
                )

    if output_files:
        st.dataframe(pd.DataFrame(output_files), width="stretch")
    else:
        st.info("No output files found. Run `python main.py` first.")