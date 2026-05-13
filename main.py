"""Run the Hermès International market intelligence and equity research pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import (
    BENCHMARK_LABEL,
    BUSINESS_MODEL_FACTORS,
    CHART_DIR,
    END_DATE,
    FACTOR_LABELS,
    GEOPOLITICAL_RISK_FACTORS,
    GEOPOLITICAL_SCENARIO_ADJUSTMENTS,
    MONTE_CARLO,
    NEWS_MAX_ITEMS_PER_QUERY,
    NEWS_QUERIES,
    OUTPUT_DIR,
    PEER_LABELS,
    PROCESSED_DATA_DIR,
    QUALITATIVE_SIGNAL_WEIGHTS,
    RAW_DATA_DIR,
    REPORT_PATH,
    RISK_FREE_RATE,
    RISK_MODEL,
    SCENARIOS,
    SHAREHOLDER_GUIDE,
    SHARE_INFO,
    SNAPSHOT_PATH,
    SOURCE_CATEGORIES,
    SOURCE_URLS,
    START_DATE,
    STRATEGIC_MOAT_SCORE,
    STRATEGY,
    STRATEGY_PILLARS,
    TABLE_DIR,
    TARGET_LABEL,
    TICKERS,
)
from src.change_detection import (
    build_change_detection_table,
    build_snapshot,
    compare_snapshots,
    load_previous_snapshot,
    save_snapshot,
)
from src.company_profile import save_company_profile_outputs
from src.data_loader import download_adjusted_close
from src.factor_model import (
    build_factor_diagnostics_table,
    build_factor_exposure_table,
    build_scenario_contribution_table,
    estimate_scenarios,
    fit_factor_model,
)
from src.features import build_hermes_feature_set, compute_returns
from src.forecasting import (
    build_monte_carlo_dashboard_table,
    build_monte_carlo_probability_table,
    build_monte_carlo_summary_table,
    monte_carlo_forecast,
)
from src.fundamentals import save_fundamental_outputs
from src.garch_var import (
    build_garch_style_var_backtest,
    build_latest_risk_snapshot,
    build_var_dashboard_summary,
    build_var_model_comparison,
    summarize_var_backtest,
)
from src.geopolitical_risk import (
    build_geopolitical_category_summary,
    save_geopolitical_risk_outputs,
)
from src.intelligence_score import (
    build_intelligence_score,
    build_intelligence_score_contribution_table,
    build_intelligence_score_dashboard,
    build_intelligence_score_summary_table,
    signal_from_score,
)
from src.market_intelligence import save_market_intelligence_outputs
from src.news_intelligence import save_news_intelligence_outputs
from src.recommendation import save_recommendation_outputs
from src.report import build_pdf_report
from src.revenue_momentum import save_revenue_momentum_outputs
from src.risk import (
    build_risk_dashboard_cards,
    build_risk_metric_leaderboard,
    build_target_risk_profile,
    summarize_performance,
)
from src.shareholder_returns import save_shareholder_outputs
from src.strategy import (
    build_hermes_strategy_returns,
    build_strategy_comparison_leaderboard,
    build_strategy_dashboard_cards,
    build_strategy_signal_summary,
    build_strategy_summary,
)
from src.valuation import save_valuation_outputs
from src.visualization import (
    save_dcf_sensitivity_chart,
    save_dividend_chart,
    save_dividend_yield_chart,
    save_drawdown_chart,
    save_factor_exposure_chart,
    save_fair_value_range_chart,
    save_forecast_chart,
    save_fundamentals_chart,
    save_geopolitical_category_chart,
    save_geopolitical_risk_chart,
    save_geopolitical_scenario_chart,
    save_intelligence_score_chart,
    save_margin_profile_chart,
    save_market_cap_chart,
    save_market_intelligence_dashboard_chart,
    save_news_classification_chart,
    save_news_source_summary_chart,
    save_news_theme_chart,
    save_news_watchlist_classification_chart,
    save_peer_performance_chart,
    save_price_chart,
    save_q1_fx_gap_chart,
    save_q1_growth_by_region_chart,
    save_q1_growth_by_sector_chart,
    save_recommendation_framework_chart,
    save_revenue_fx_dashboard_chart,
    save_revenue_leaderboard_chart,
    save_risk_leaderboard_chart,
    save_scenario_chart,
    save_shareholder_policy_chart,
    save_shareholder_return_leaderboard_chart,
    save_strategy_chart,
    save_strategy_comparison_chart,
    save_strategy_drawdown_comparison_chart,
    save_strategy_leaderboard_chart,
    save_strategy_qualitative_score_chart,
    save_strategy_signal_summary_chart,
    save_target_risk_profile_chart,
    save_valuation_bridge_chart,
    save_valuation_sensitivity_dashboard_chart,
    save_var_chart,
    save_volatility_chart,
)


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    """Format numeric values for console output."""
    return df.map(
        lambda x: f"{x:.4f}" if isinstance(x, (int, float, np.floating)) else x
    )


def score_from_q1_revenue(revenue_momentum_summary: pd.DataFrame) -> float:
    """Convert Q1 revenue momentum into a simple 0-100 score."""
    total_growth = revenue_momentum_summary[
        revenue_momentum_summary["Insight"].eq("Constant-FX growth")
    ]

    if total_growth.empty:
        return 60.0

    growth = float(total_growth.iloc[0]["Value"])

    return float(np.clip(50 + growth * 5, 0, 100))


def main() -> None:
    """Run the full Hermès market intelligence and equity research pipeline."""
    print("Starting Hermès International market intelligence pipeline...")

    # -------------------------------------------------------------------------
    # Official company context and static investor-relations data
    # -------------------------------------------------------------------------

    _, fundamental_metrics = save_fundamental_outputs(RAW_DATA_DIR, TABLE_DIR)
    _, dividend_metrics = save_shareholder_outputs(RAW_DATA_DIR, TABLE_DIR)

    q1_revenue, revenue_momentum_summary = save_revenue_momentum_outputs(
        RAW_DATA_DIR,
        TABLE_DIR,
    )

    (
        sources,
        share_reference,
        shareholder_reference,
        business_model,
        strategy_pillars_table,
        strategy_scores,
        company_profile_summary,
    ) = save_company_profile_outputs(
        TABLE_DIR,
        SOURCE_URLS,
        SHARE_INFO,
        SHAREHOLDER_GUIDE,
        BUSINESS_MODEL_FACTORS,
        STRATEGY_PILLARS,
        STRATEGIC_MOAT_SCORE,
        QUALITATIVE_SIGNAL_WEIGHTS,
        SOURCE_CATEGORIES,
    )

    qualitative_score = float(strategy_scores["Score_0_to_100"].mean())
    revenue_momentum_score = score_from_q1_revenue(revenue_momentum_summary)

    # -------------------------------------------------------------------------
    # News and geopolitical market intelligence
    # -------------------------------------------------------------------------

    news_items, news_summary, news_theme_summary = save_news_intelligence_outputs(
        TABLE_DIR,
        NEWS_QUERIES,
        max_items_per_query=NEWS_MAX_ITEMS_PER_QUERY,
    )

    geopolitical_risks, geopolitical_summary, geopolitical_scenarios = (
        save_geopolitical_risk_outputs(
            TABLE_DIR,
            GEOPOLITICAL_RISK_FACTORS,
            GEOPOLITICAL_SCENARIO_ADJUSTMENTS,
        )
    )

    geopolitical_category_summary = build_geopolitical_category_summary(
        geopolitical_risks
    )

    geopolitical_category_summary.to_csv(
        TABLE_DIR / "hermes_geopolitical_category_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Market data and quantitative analytics
    # -------------------------------------------------------------------------

    prices = download_adjusted_close(
        TICKERS,
        START_DATE,
        END_DATE,
        table_dir=TABLE_DIR,
    )
    prices.to_csv(PROCESSED_DATA_DIR / "hermes_full_price_data.csv")

    peer_prices = prices[PEER_LABELS].dropna(how="all")
    peer_prices.to_csv(PROCESSED_DATA_DIR / "hermes_luxury_peer_prices.csv")

    returns = compute_returns(prices)
    returns.to_csv(PROCESSED_DATA_DIR / "hermes_full_returns.csv")

    peer_returns = returns[PEER_LABELS].dropna(how="all")
    peer_returns.to_csv(PROCESSED_DATA_DIR / "hermes_luxury_peer_returns.csv")

    hermes_price = prices[TARGET_LABEL]
    current_price_eur = float(hermes_price.dropna().iloc[-1])

    hermes_features = build_hermes_feature_set(hermes_price, STRATEGY)
    hermes_features.to_csv(PROCESSED_DATA_DIR / "hermes_features.csv")

    strategy_df = build_hermes_strategy_returns(hermes_features, STRATEGY)
    strategy_df.to_csv(PROCESSED_DATA_DIR / "hermes_strategy_results.csv")
    strategy_df.to_csv(TABLE_DIR / "hermes_strategy_returns.csv")

    strategy_summary = build_strategy_summary(strategy_df)
    strategy_summary.to_csv(TABLE_DIR / "hermes_strategy_summary.csv")

    strategy_dashboard_cards = build_strategy_dashboard_cards(strategy_summary)
    strategy_dashboard_cards.to_csv(
        TABLE_DIR / "hermes_strategy_dashboard_cards.csv",
        index=False,
    )

    strategy_comparison_leaderboard = build_strategy_comparison_leaderboard(
        strategy_summary
    )
    strategy_comparison_leaderboard.to_csv(
        TABLE_DIR / "hermes_strategy_comparison_leaderboard.csv",
        index=False,
    )

    strategy_signal_summary = build_strategy_signal_summary(strategy_df)
    strategy_signal_summary.to_csv(
        TABLE_DIR / "hermes_strategy_signal_summary.csv",
        index=False,
    )

    comparison_returns = peer_returns.copy()
    comparison_returns["Hermès Strategy"] = strategy_df["strategy_return"]
    comparison_returns = comparison_returns.dropna(how="all")

    metrics = summarize_performance(
        comparison_returns,
        risk_free_rate=RISK_FREE_RATE,
        confidence=RISK_MODEL["var_confidence_level"],
    )

    metrics.to_csv(TABLE_DIR / "hermes_risk_return_metrics.csv")

    risk_dashboard_cards = build_risk_dashboard_cards(
        performance_summary=metrics,
        target_label=TARGET_LABEL,
    )
    risk_dashboard_cards.to_csv(
        TABLE_DIR / "hermes_risk_dashboard_cards.csv",
        index=False,
    )

    risk_metric_leaderboard = build_risk_metric_leaderboard(metrics)
    risk_metric_leaderboard.to_csv(
        TABLE_DIR / "hermes_risk_metric_leaderboard.csv",
        index=False,
    )

    target_risk_profile = build_target_risk_profile(
        performance_summary=metrics,
        target_label=TARGET_LABEL,
    )
    target_risk_profile.to_csv(
        TABLE_DIR / "hermes_target_risk_profile.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # VaR backtesting
    # -------------------------------------------------------------------------

    var_backtest = build_garch_style_var_backtest(
        returns[TARGET_LABEL],
        confidence=RISK_MODEL["var_confidence_level"],
        lambda_=RISK_MODEL["ewma_lambda"],
        historical_window=RISK_MODEL["historical_var_window"],
    )

    var_backtest.to_csv(PROCESSED_DATA_DIR / "hermes_garch_style_var_backtest.csv")

    var_summary = summarize_var_backtest(
        var_backtest,
        confidence=RISK_MODEL["var_confidence_level"],
    )

    var_summary.to_csv(TABLE_DIR / "hermes_var_backtest_summary.csv")

    var_dashboard_summary = build_var_dashboard_summary(var_summary)
    var_dashboard_summary.to_csv(
        TABLE_DIR / "hermes_var_dashboard_summary.csv",
        index=False,
    )

    var_model_comparison = build_var_model_comparison(var_summary)
    var_model_comparison.to_csv(
        TABLE_DIR / "hermes_var_model_comparison.csv",
        index=False,
    )

    latest_risk_snapshot = build_latest_risk_snapshot(var_backtest)
    latest_risk_snapshot.to_csv(
        TABLE_DIR / "hermes_var_latest_risk_snapshot.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Factor model and scenario analysis
    # -------------------------------------------------------------------------

    factor_coefficients, fitted_returns, factor_r2 = fit_factor_model(
        returns,
        TARGET_LABEL,
        FACTOR_LABELS,
    )

    factor_coefficients.to_csv(
        TABLE_DIR / "hermes_factor_regression_coefficients.csv"
    )

    # Compatibility filename for Streamlit/app modules that expect this name.
    factor_coefficients.to_csv(
        TABLE_DIR / "hermes_factor_model_coefficients.csv"
    )

    fitted_returns.to_csv(
        PROCESSED_DATA_DIR / "hermes_factor_model_fitted_returns.csv"
    )

    factor_diagnostics = build_factor_diagnostics_table(factor_coefficients)
    factor_diagnostics.to_csv(
        TABLE_DIR / "hermes_factor_model_diagnostics.csv",
        index=False,
    )

    factor_exposures_clean = build_factor_exposure_table(factor_coefficients)
    factor_exposures_clean.to_csv(
        TABLE_DIR / "hermes_factor_exposures_clean.csv",
        index=False,
    )

    factor_model_rows_to_drop = [
        "Model R-Squared",
        "Adjusted R-Squared",
        "Residual Mean",
        "Residual Std Dev",
        "Observations",
    ]

    scenario_table = estimate_scenarios(
        factor_coefficients.drop(
            index=factor_model_rows_to_drop,
            errors="ignore",
        ),
        SCENARIOS,
    )

    scenario_table.to_csv(TABLE_DIR / "hermes_scenario_return_estimates.csv")

    # Compatibility filename for app/report modules that expect this name.
    scenario_table.to_csv(TABLE_DIR / "hermes_scenario_returns.csv")

    scenario_contribution_detail = build_scenario_contribution_table(scenario_table)
    scenario_contribution_detail.to_csv(
        TABLE_DIR / "hermes_scenario_contribution_detail.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Monte Carlo forecast
    # -------------------------------------------------------------------------

    monte_carlo_forecast_table, final_simulated_prices = monte_carlo_forecast(
        last_price=current_price_eur,
        returns=returns[TARGET_LABEL].dropna(),
        config=MONTE_CARLO,
        return_final_prices=True,
    )

    monte_carlo_forecast_table.to_csv(
        TABLE_DIR / "hermes_monte_carlo_forecast.csv",
        index=False,
    )

    final_simulated_prices.to_csv(
        TABLE_DIR / "hermes_monte_carlo_final_prices.csv",
        index=False,
    )

    monte_carlo_summary = build_monte_carlo_summary_table(
        forecast=monte_carlo_forecast_table,
        current_price=current_price_eur,
    )

    monte_carlo_summary.to_csv(
        TABLE_DIR / "hermes_monte_carlo_summary.csv",
        index=False,
    )

    monte_carlo_probabilities = build_monte_carlo_probability_table(
        final_prices=final_simulated_prices,
        current_price=current_price_eur,
    )

    monte_carlo_probabilities.to_csv(
        TABLE_DIR / "hermes_monte_carlo_probabilities.csv",
        index=False,
    )

    monte_carlo_dashboard = build_monte_carlo_dashboard_table(
        summary_table=monte_carlo_summary,
        probability_table=monte_carlo_probabilities,
    )

    monte_carlo_dashboard.to_csv(
        TABLE_DIR / "hermes_monte_carlo_dashboard.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Valuation model
    # -------------------------------------------------------------------------

    (
        valuation_forecast,
        dcf_valuation,
        peer_valuation,
        fair_value_summary,
        dcf_sensitivity,
        valuation_signal,
    ) = save_valuation_outputs(
        TABLE_DIR,
        fundamental_metrics=fundamental_metrics,
        dividend_metrics=dividend_metrics,
        current_price_eur=current_price_eur,
    )

    # -------------------------------------------------------------------------
    # Intelligence score
    # -------------------------------------------------------------------------

    intelligence_score = build_intelligence_score(
        metrics,
        factor_r2,
        var_summary,
        scenario_table,
        qualitative_score=qualitative_score,
        revenue_momentum_score=revenue_momentum_score,
    )

    intelligence_score.to_csv(TABLE_DIR / "hermes_luxury_intelligence_score.csv")

    intelligence_score_dashboard = build_intelligence_score_dashboard(
        intelligence_score
    )
    intelligence_score_dashboard.to_csv(
        TABLE_DIR / "hermes_intelligence_score_dashboard.csv",
        index=False,
    )

    intelligence_score_contribution = build_intelligence_score_contribution_table(
        intelligence_score
    )
    intelligence_score_contribution.to_csv(
        TABLE_DIR / "hermes_intelligence_score_contribution.csv",
        index=False,
    )

    intelligence_score_summary = build_intelligence_score_summary_table(
        intelligence_score
    )
    intelligence_score_summary.to_csv(
        TABLE_DIR / "hermes_intelligence_score_summary.csv",
        index=False,
    )

    final_score = float(
        intelligence_score.loc["Hermès Luxury Intelligence Score", "Score"]
    )

    signal = signal_from_score(final_score)

    # -------------------------------------------------------------------------
    # Market intelligence dashboard and recommendation framework
    # -------------------------------------------------------------------------

    market_dashboard, market_summary, catalyst_risk_register = (
        save_market_intelligence_outputs(
            TABLE_DIR,
            metrics=metrics,
            intelligence_score=intelligence_score,
            scenario_table=scenario_table,
            revenue_momentum_summary=revenue_momentum_summary,
            strategy_scores=strategy_scores,
            valuation_signal=valuation_signal,
            news_summary=news_summary,
            geopolitical_summary=geopolitical_summary,
            target_label=TARGET_LABEL,
        )
    )

    (
        recommendation_framework,
        recommendation_summary,
        recommendation_rules,
    ) = save_recommendation_outputs(
        TABLE_DIR,
        intelligence_score=intelligence_score,
        valuation_signal=valuation_signal,
        market_summary=market_summary,
        news_summary=news_summary,
        geopolitical_summary=geopolitical_summary,
    )

    # -------------------------------------------------------------------------
    # Change detection
    # -------------------------------------------------------------------------

    final_recommendation = recommendation_summary[
        recommendation_summary["Metric"].eq("Final Recommendation")
    ]

    final_recommendation_label = (
        str(final_recommendation.iloc[0]["Value"])
        if not final_recommendation.empty
        else None
    )

    risk_rating_row = recommendation_summary[
        recommendation_summary["Metric"].eq("Risk Rating")
    ]

    risk_rating_label = (
        str(risk_rating_row.iloc[0]["Value"])
        if not risk_rating_row.empty
        else None
    )

    valuation_signal_row = valuation_signal[
        valuation_signal["Metric"].eq("Valuation Signal")
    ]

    valuation_signal_label = (
        str(valuation_signal_row.iloc[0]["Value"])
        if not valuation_signal_row.empty
        else None
    )

    implied_upside_row = valuation_signal[
        valuation_signal["Metric"].eq("Implied Upside / Downside")
    ]

    implied_upside_value = (
        float(implied_upside_row.iloc[0]["Value"])
        if not implied_upside_row.empty
        else None
    )

    geopolitical_posture_row = geopolitical_summary[
        geopolitical_summary["Metric"].eq("Risk Posture")
    ]

    geopolitical_posture_label = (
        str(geopolitical_posture_row.iloc[0]["Value"])
        if not geopolitical_posture_row.empty
        else None
    )

    news_tone_row = news_summary[news_summary["Metric"].eq("Average News Score")]

    news_tone_label = (
        str(news_tone_row.iloc[0]["Interpretation"])
        if not news_tone_row.empty
        else None
    )

    market_view_row = market_summary[
        market_summary["Section"].eq("Overall Market Intelligence View")
    ]

    market_view_label = (
        str(market_view_row.iloc[0]["Summary"])
        if not market_view_row.empty
        else None
    )

    current_snapshot = build_snapshot(
        score=final_score,
        signal=signal,
        ann_return=float(metrics.loc[TARGET_LABEL, "Annualized Return"]),
        max_drawdown=float(metrics.loc[TARGET_LABEL, "Max Drawdown"]),
        var_breach_rate=float(var_summary["Breach Rate"].mean()),
        recommendation=final_recommendation_label,
        risk_rating=risk_rating_label,
        valuation_signal=valuation_signal_label,
        implied_upside=implied_upside_value,
        geopolitical_posture=geopolitical_posture_label,
        news_tone=news_tone_label,
        market_view=market_view_label,
    )

    previous_snapshot = load_previous_snapshot(SNAPSHOT_PATH)
    change_notes = compare_snapshots(previous_snapshot, current_snapshot)

    change_detection_table = build_change_detection_table(change_notes)
    change_detection_table.to_csv(
        TABLE_DIR / "hermes_change_detection.csv",
        index=False,
    )

    save_snapshot(SNAPSHOT_PATH, current_snapshot)

    # -------------------------------------------------------------------------
    # Dashboard tables generated inside module save functions
    # -------------------------------------------------------------------------

    news_source_summary = pd.read_csv(
        TABLE_DIR / "hermes_news_source_summary.csv"
    )

    news_risk_catalyst_watchlist = pd.read_csv(
        TABLE_DIR / "hermes_news_risk_catalyst_watchlist.csv"
    )

    revenue_segment_leaderboard = pd.read_csv(
        TABLE_DIR / "hermes_revenue_segment_leaderboard.csv"
    )

    revenue_fx_dashboard = pd.read_csv(
        TABLE_DIR / "hermes_revenue_fx_dashboard.csv"
    )

    shareholder_return_leaderboard = pd.read_csv(
        TABLE_DIR / "hermes_shareholder_return_leaderboard.csv"
    )

    dividend_policy_dashboard = pd.read_csv(
        TABLE_DIR / "hermes_dividend_policy_dashboard.csv"
    )

    valuation_bridge = pd.read_csv(
        TABLE_DIR / "hermes_valuation_bridge.csv"
    )

    valuation_sensitivity_dashboard = pd.read_csv(
        TABLE_DIR / "hermes_valuation_sensitivity_dashboard.csv"
    )

    # -------------------------------------------------------------------------
    # Charts
    # -------------------------------------------------------------------------

    chart_paths = [
        # Market price and peer charts
        save_price_chart(prices, CHART_DIR),
        save_peer_performance_chart(peer_returns, CHART_DIR),

        # Strategy and downside-risk charts
        save_strategy_chart(strategy_df, CHART_DIR),
        save_strategy_comparison_chart(strategy_df, CHART_DIR),
        save_drawdown_chart(strategy_df, CHART_DIR),
        save_strategy_drawdown_comparison_chart(strategy_df, CHART_DIR),

        # Volatility, VaR, factor, and scenario charts
        save_volatility_chart(var_backtest, CHART_DIR),
        save_var_chart(var_backtest, CHART_DIR),
        save_factor_exposure_chart(factor_coefficients, CHART_DIR),
        save_scenario_chart(scenario_table, CHART_DIR),
        save_intelligence_score_chart(intelligence_score, CHART_DIR),
        save_forecast_chart(monte_carlo_forecast_table, CHART_DIR),

        # Official fundamentals and shareholder-return charts
        save_fundamentals_chart(fundamental_metrics, CHART_DIR),
        save_margin_profile_chart(fundamental_metrics, CHART_DIR),
        save_dividend_chart(dividend_metrics, CHART_DIR),
        save_dividend_yield_chart(dividend_metrics, CHART_DIR),
        save_market_cap_chart(dividend_metrics, CHART_DIR),

        # Q1 2026 revenue momentum charts
        save_q1_growth_by_region_chart(q1_revenue, CHART_DIR),
        save_q1_growth_by_sector_chart(q1_revenue, CHART_DIR),
        save_q1_fx_gap_chart(q1_revenue, CHART_DIR),

        # Strategy / qualitative moat chart
        save_strategy_qualitative_score_chart(strategy_scores, CHART_DIR),

        # Valuation charts
        save_fair_value_range_chart(fair_value_summary, CHART_DIR),
        save_dcf_sensitivity_chart(dcf_sensitivity, CHART_DIR),
        save_valuation_bridge_chart(valuation_bridge, CHART_DIR),
        save_valuation_sensitivity_dashboard_chart(
            valuation_sensitivity_dashboard,
            CHART_DIR,
        ),

        # News and market-intelligence charts
        save_news_theme_chart(news_theme_summary, CHART_DIR),
        save_news_classification_chart(news_items, CHART_DIR),
        save_news_source_summary_chart(news_source_summary, CHART_DIR),
        save_news_watchlist_classification_chart(
            news_risk_catalyst_watchlist,
            CHART_DIR,
        ),
        save_market_intelligence_dashboard_chart(market_dashboard, CHART_DIR),

        # Geopolitical risk charts
        save_geopolitical_risk_chart(geopolitical_risks, CHART_DIR),
        save_geopolitical_category_chart(geopolitical_category_summary, CHART_DIR),
        save_geopolitical_scenario_chart(geopolitical_scenarios, CHART_DIR),

        # Recommendation charts
        save_recommendation_framework_chart(recommendation_framework, CHART_DIR),

        # Dashboard upgrade charts
        save_risk_leaderboard_chart(risk_metric_leaderboard, CHART_DIR),
        save_target_risk_profile_chart(target_risk_profile, CHART_DIR),
        save_revenue_leaderboard_chart(revenue_segment_leaderboard, CHART_DIR),
        save_revenue_fx_dashboard_chart(revenue_fx_dashboard, CHART_DIR),
        save_shareholder_policy_chart(dividend_policy_dashboard, CHART_DIR),
        save_shareholder_return_leaderboard_chart(
            shareholder_return_leaderboard,
            CHART_DIR,
        ),
        save_strategy_leaderboard_chart(strategy_comparison_leaderboard, CHART_DIR),
        save_strategy_signal_summary_chart(strategy_signal_summary, CHART_DIR),
    ]

    # -------------------------------------------------------------------------
    # PDF report
    # -------------------------------------------------------------------------

    build_pdf_report(
        REPORT_PATH,
        metrics,
        var_summary,
        factor_coefficients,
        scenario_table,
        intelligence_score,
        chart_paths,
        change_notes,
        fundamental_metrics=fundamental_metrics,
        dividend_metrics=dividend_metrics,
        revenue_momentum_summary=revenue_momentum_summary,
        strategy_scores=strategy_scores,
        company_profile_summary=company_profile_summary,
        sources_table=sources,
        valuation_signal=valuation_signal,
        fair_value_summary=fair_value_summary,
        dcf_valuation=dcf_valuation,
        peer_valuation=peer_valuation,
        dcf_sensitivity=dcf_sensitivity,
        news_summary=news_summary,
        news_theme_summary=news_theme_summary,
        geopolitical_summary=geopolitical_summary,
        geopolitical_risks=geopolitical_risks,
        geopolitical_scenarios=geopolitical_scenarios,
        market_dashboard=market_dashboard,
        market_summary=market_summary,
        catalyst_risk_register=catalyst_risk_register,
        recommendation_summary=recommendation_summary,
        recommendation_framework=recommendation_framework,
        recommendation_rules=recommendation_rules,
        strategy_summary=strategy_summary,
    )

    # -------------------------------------------------------------------------
    # Console output
    # -------------------------------------------------------------------------

    final_recommendation_label = final_recommendation_label or "Not Available"
    valuation_signal_label = valuation_signal_label or "Not Available"

    print("\nHermès market intelligence analysis complete.")
    print(f"Outputs saved in: {OUTPUT_DIR}")
    print(f"PDF report saved to: {REPORT_PATH}")
    print(f"Hermès Luxury Intelligence Score: {final_score:.2f} / 100")
    print(f"Signal: {signal}")
    print(f"Valuation Signal: {valuation_signal_label}")
    print(f"Recommendation: {final_recommendation_label}")

    print("\nRisk/return metrics:")
    print(format_table(metrics))

    print("\nMarket intelligence summary:")
    print(format_table(market_summary))

    print("\nRecommendation summary:")
    print(format_table(recommendation_summary))

    print("\nChange detection:")
    for note in change_notes:
        print(f"- {note}")


if __name__ == "__main__":
    main()