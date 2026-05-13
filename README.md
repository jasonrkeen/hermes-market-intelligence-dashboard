# Hermès International Market Intelligence & Equity Research Pipeline

A Python market intelligence and equity research project analyzing **Hermès International SCA** using market data, official investor-relations fundamentals, shareholder-return data, Q1 2026 revenue momentum, downside risk modeling, factor regression, scenario analysis, strategy overlays, valuation analysis, news intelligence, geopolitical risk monitoring, Monte Carlo forecasting, change detection, an interactive Streamlit dashboard, and a composite **Hermès Luxury Intelligence Score**.

The project is designed as a polished, GitHub-ready equity research and market intelligence pipeline. It connects Hermès' official company fundamentals with public market behavior, valuation sensitivity, macro/geopolitical risk, news flow, and a transparent recommendation framework.

---

## Key Takeaway

Hermès behaves less like a generic cyclical retailer and more like a high-margin luxury compounder. Its long-term value story is supported by craftsmanship, brand scarcity, pricing power, vertical integration, family-backed independence, resilient margins, and a strong balance sheet.

However, the stock can still face valuation drawdowns, FX pressure, China demand risk, luxury-sector re-rating risk, geopolitical risk, and broader market multiple compression. This project evaluates Hermès as both a luxury compounder and a publicly traded equity by combining official fundamentals, market analytics, valuation work, risk modeling, news intelligence, and geopolitical monitoring.

---

## Assumption Transparency

Several parts of this project are intentionally **rules-based and assumption-driven**. The historical market calculations are data-driven, but model inputs such as valuation multiples, DCF assumptions, geopolitical risk scores, qualitative moat scores, scenario assumptions, news-classification rules, and recommendation weights are subjective research assumptions.

The purpose of the project is not to produce a definitive investment rating. The purpose is to demonstrate a transparent, reproducible market intelligence framework where assumptions can be reviewed, challenged, adjusted, and sensitized.

Use the project language accordingly:

- **Model-implied** rather than guaranteed
- **Rules-based** rather than absolute
- **Scenario-based** rather than predictive
- **Configured risk score** rather than objective risk truth
- **Assumption-driven valuation** rather than definitive fair value

This framing is important because Hermès is a high-quality luxury compounder, but the final signal still depends on valuation assumptions, risk weights, market conditions, news flow, and geopolitical scoring inputs.

---

## Latest Run Snapshot

Run the project to refresh the latest score, signal, valuation view, recommendation, charts, tables, PDF report, and dashboard files:

```powershell
python main.py
```

The console prints the current outputs:

```text
Hermès Luxury Intelligence Score
Signal
Valuation Signal
Recommendation
Outputs folder
PDF report path
```

The generated PDF is saved to:

```text
outputs/reports/hermes_market_intelligence_report.pdf
```

The recommendation can remain **Neutral** even when business quality is strong because the framework also considers valuation sensitivity, drawdown pressure, risk-leaning news flow, geopolitical posture, and market intelligence risk signals.

---

## Final Project Status

The project is currently in a polished end-to-end state:

- `python main.py` runs the full market intelligence pipeline.
- The PDF report is generated at `outputs/reports/hermes_market_intelligence_report.pdf`.
- The Streamlit dashboard launches from `app.py`.
- Wide PDF tables are wrapped to fit the report page.
- Dashboard cards support long text values without truncating key messages.
- The Strategic Moat dashboard tab reads the correct `hermes_strategy_qualitative_score.csv` output.
- The final report and dashboard are suitable for GitHub, portfolio review, and LinkedIn presentation.

---

## Project Thesis

This project combines:

- **Market behavior**: price history, returns, drawdowns, volatility, and luxury peer comparison
- **Risk modeling**: downside volatility, Sharpe, Sortino, Calmar, historical VaR, expected shortfall, EWMA / GARCH-style volatility, and VaR backtesting
- **Factor sensitivity**: CAC 40, LVMH, EUR/USD, and China proxy exposure
- **Scenario analysis**: bear, base, bull, China slowdown, and luxury re-rating scenarios
- **Official fundamentals**: Hermès key figures, margins, free cash flow, net cash, and workforce data
- **Q1 2026 momentum**: region and sector revenue growth, including constant-FX growth and FX gap analysis
- **Shareholder returns**: EPS, dividend, payout ratio, dividend yield, implied P/E, market capitalization, and share-price range
- **Valuation model**: forward forecast, DCF valuation, peer multiple valuation, fair value range, and DCF sensitivity
- **News intelligence**: headline collection, theme tagging, catalyst/risk classification, source summary, and news score summary
- **Geopolitical risk**: risk factor scoring, category summary, risk heatmap, and scenario stress overlay
- **Market intelligence dashboard**: combined quality, risk, valuation, news, and geopolitical signal layer
- **Recommendation framework**: rules-based Outperform / Neutral / Underperform decision layer
- **Strategy research overlays**: risk-control, trend-following, momentum, defensive cash, and volatility-target overlays
- **Qualitative moat scoring**: craftsmanship, exclusivity, family independence, integrated model, brand desirability, and shareholder communication quality
- **Monte Carlo forecasting**: simulated forecast distribution using historical daily returns
- **Change detection**: compares the current run against the prior run
- **Automated PDF report**: executive summary, tables, charts, methodology notes, sources, and disclaimer
- **Streamlit dashboard**: interactive view of the full pipeline output

---

## Project Structure

```text
Hermès International Market Intelligence & Equity Research Pipeline/
├── app.py
├── main.py
├── README.md
├── requirements.txt
├── config/
│   ├── __init__.py
│   └── settings.py
├── data/
│   ├── raw/
│   │   ├── hermes_key_figures.csv
│   │   ├── hermes_q1_2026_revenue.csv
│   │   └── hermes_share_dividend_history.csv
│   └── processed/
├── outputs/
│   ├── charts/
│   ├── reports/
│   ├── snapshots/
│   └── tables/
└── src/
    ├── __init__.py
    ├── change_detection.py
    ├── company_profile.py
    ├── data_loader.py
    ├── factor_model.py
    ├── features.py
    ├── forecasting.py
    ├── fundamentals.py
    ├── garch_var.py
    ├── geopolitical_risk.py
    ├── intelligence_score.py
    ├── market_intelligence.py
    ├── news_intelligence.py
    ├── recommendation.py
    ├── report.py
    ├── revenue_momentum.py
    ├── risk.py
    ├── shareholder_returns.py
    ├── strategy.py
    ├── valuation.py
    └── visualization.py
```

---

## Complete Output Inventory

### Core Table Outputs

| Output | File | Purpose |
|---|---|---|
| Risk/return metrics | `outputs/tables/hermes_risk_return_metrics.csv` | Performance, volatility, drawdown, Sharpe, Sortino, Calmar, VaR, expected shortfall, skew, and kurtosis. |
| Risk dashboard cards | `outputs/tables/hermes_risk_dashboard_cards.csv` | Dashboard-ready target risk cards. |
| Risk metric leaderboard | `outputs/tables/hermes_risk_metric_leaderboard.csv` | Peer ranking by return, volatility, Sharpe, drawdown, VaR, and expected shortfall. |
| Target risk profile | `outputs/tables/hermes_target_risk_profile.csv` | Plain-English Hermès risk profile. |
| VaR backtest summary | `outputs/tables/hermes_var_backtest_summary.csv` | VaR breach statistics and Kupiec-style diagnostics. |
| VaR dashboard summary | `outputs/tables/hermes_var_dashboard_summary.csv` | Dashboard-ready VaR model cards. |
| VaR model comparison | `outputs/tables/hermes_var_model_comparison.csv` | GARCH-style VaR versus historical VaR comparison. |
| Latest risk snapshot | `outputs/tables/hermes_var_latest_risk_snapshot.csv` | Latest VaR and volatility values. |
| Factor coefficients | `outputs/tables/hermes_factor_regression_coefficients.csv` | Factor model coefficients and diagnostics. |
| Factor model compatibility output | `outputs/tables/hermes_factor_model_coefficients.csv` | Compatibility copy for dashboard/report modules. |
| Factor diagnostics | `outputs/tables/hermes_factor_model_diagnostics.csv` | Clean factor model diagnostic table. |
| Factor exposures | `outputs/tables/hermes_factor_exposures_clean.csv` | Dashboard-ready factor exposures. |
| Scenario estimates | `outputs/tables/hermes_scenario_return_estimates.csv` | Scenario return estimates. |
| Scenario compatibility output | `outputs/tables/hermes_scenario_returns.csv` | Compatibility copy for chart/dashboard modules. |
| Scenario contributions | `outputs/tables/hermes_scenario_contribution_detail.csv` | Scenario contribution detail by factor. |
| Monte Carlo forecast | `outputs/tables/hermes_monte_carlo_forecast.csv` | Forecast percentiles by simulated day. |
| Monte Carlo final prices | `outputs/tables/hermes_monte_carlo_final_prices.csv` | Final simulated price distribution. |
| Monte Carlo summary | `outputs/tables/hermes_monte_carlo_summary.csv` | Forecast summary. |
| Monte Carlo probabilities | `outputs/tables/hermes_monte_carlo_probabilities.csv` | Downside/upside probability table. |
| Monte Carlo dashboard | `outputs/tables/hermes_monte_carlo_dashboard.csv` | Dashboard-ready Monte Carlo cards. |
| Intelligence score | `outputs/tables/hermes_luxury_intelligence_score.csv` | Composite Hermès Luxury Intelligence Score. |
| Intelligence score dashboard | `outputs/tables/hermes_intelligence_score_dashboard.csv` | Score summary dashboard cards. |
| Intelligence score contribution | `outputs/tables/hermes_intelligence_score_contribution.csv` | Component contribution table. |
| Intelligence score summary | `outputs/tables/hermes_intelligence_score_summary.csv` | Plain-English score summary. |
| Change detection | `outputs/tables/hermes_change_detection.csv` | Current run versus prior run changes. |

### Fundamental, Revenue, Shareholder, and Company Tables

| Output | File | Purpose |
|---|---|---|
| Key figures | `outputs/tables/hermes_key_figures.csv` | Official Hermès financial figures. |
| Fundamental metrics | `outputs/tables/hermes_fundamental_metrics.csv` | Processed fundamental metrics. |
| Fundamental summary | `outputs/tables/hermes_fundamental_summary.csv` | Compact fundamental summary. |
| Fundamental dashboard cards | `outputs/tables/hermes_fundamental_dashboard_cards.csv` | Dashboard-ready fundamental cards. |
| Fundamental quality summary | `outputs/tables/hermes_fundamental_quality_summary.csv` | Quality and durability summary. |
| Q1 2026 revenue | `outputs/tables/hermes_q1_2026_revenue.csv` | Q1 2026 revenue by geography and sector. |
| Revenue momentum summary | `outputs/tables/hermes_revenue_momentum_summary.csv` | Q1 revenue momentum insight table. |
| Revenue momentum metrics | `outputs/tables/hermes_revenue_momentum_metrics.csv` | Derived Q1 revenue metrics. |
| Q1 FX impact summary | `outputs/tables/hermes_q1_fx_impact_summary.csv` | FX gap and reported versus constant-FX analysis. |
| Revenue dashboard cards | `outputs/tables/hermes_revenue_dashboard_cards.csv` | Dashboard-ready Q1 revenue cards. |
| Revenue segment leaderboard | `outputs/tables/hermes_revenue_segment_leaderboard.csv` | Region/sector momentum leaderboard. |
| Revenue FX dashboard | `outputs/tables/hermes_revenue_fx_dashboard.csv` | FX impact dashboard table. |
| Share/dividend history | `outputs/tables/hermes_share_dividend_history.csv` | Official share and dividend history. |
| Dividend metrics | `outputs/tables/hermes_dividend_metrics.csv` | EPS, dividend, payout, yield, implied P/E, and market cap metrics. |
| Shareholder summary | `outputs/tables/hermes_shareholder_summary.csv` | Latest shareholder-return summary. |
| Shareholder dashboard cards | `outputs/tables/hermes_shareholder_dashboard_cards.csv` | Dashboard-ready shareholder-return cards. |
| Shareholder return leaderboard | `outputs/tables/hermes_shareholder_return_leaderboard.csv` | Year-by-year EPS/dividend/market-cap growth table. |
| Dividend policy dashboard | `outputs/tables/hermes_dividend_policy_dashboard.csv` | Dividend policy, yield, earnings yield, and P/E dashboard. |
| Official sources | `outputs/tables/hermes_official_sources.csv` | Official Hermès source catalog. |
| Company profile summary | `outputs/tables/hermes_company_profile_summary.csv` | Company profile and interpretation table. |
| Dashboard company cards | `outputs/tables/hermes_dashboard_company_cards.csv` | Company profile cards for dashboard. |
| Strategy pillars | `outputs/tables/hermes_strategy_pillars.csv` | Hermès strategic pillars. |
| Business model factors | `outputs/tables/hermes_business_model_factors.csv` | Hermès business model and moat factors. |
| Strategy qualitative score | `outputs/tables/hermes_strategy_qualitative_score.csv` | Strategic moat score table. |

### Valuation, Market Intelligence, News, Geopolitical, and Recommendation Tables

| Output | File | Purpose |
|---|---|---|
| Forecast model | `outputs/tables/hermes_forecast_model.csv` | Forward revenue, margin, EPS, FCF, and dividend forecast. |
| DCF valuation | `outputs/tables/hermes_dcf_valuation.csv` | Explicit FCF, terminal value, equity value, and DCF fair value. |
| Peer multiple valuation | `outputs/tables/hermes_peer_multiple_valuation.csv` | Bear/base/bull P/E multiple valuation. |
| Fair value summary | `outputs/tables/hermes_fair_value_summary.csv` | Bear/base/bull and DCF fair value range. |
| DCF sensitivity | `outputs/tables/hermes_dcf_sensitivity.csv` | WACC and terminal growth sensitivity matrix. |
| Valuation signal | `outputs/tables/hermes_valuation_signal.csv` | Base fair value, current price, implied upside/downside, and valuation signal. |
| Valuation dashboard cards | `outputs/tables/hermes_valuation_dashboard_cards.csv` | Dashboard-ready valuation cards. |
| Valuation bridge | `outputs/tables/hermes_valuation_bridge.csv` | Forecast-to-fair-value bridge. |
| Valuation sensitivity dashboard | `outputs/tables/hermes_valuation_sensitivity_dashboard.csv` | Long-form valuation sensitivity table. |
| News items | `outputs/tables/hermes_news_intelligence_items.csv` | Scored news headline table. |
| News summary | `outputs/tables/hermes_news_intelligence_summary.csv` | News score and tone summary. |
| News theme summary | `outputs/tables/hermes_news_theme_summary.csv` | News theme count table. |
| News dashboard cards | `outputs/tables/hermes_news_dashboard_cards.csv` | Dashboard-ready news cards. |
| News risk/catalyst watchlist | `outputs/tables/hermes_news_risk_catalyst_watchlist.csv` | High-priority catalyst and risk headline watchlist. |
| News source summary | `outputs/tables/hermes_news_source_summary.csv` | Publisher/source summary. |
| Geopolitical risk factors | `outputs/tables/hermes_geopolitical_risk_factors.csv` | Geopolitical risk factor table. |
| Geopolitical risk summary | `outputs/tables/hermes_geopolitical_risk_summary.csv` | Average score, highest risk, and risk posture. |
| Geopolitical category summary | `outputs/tables/hermes_geopolitical_category_summary.csv` | Category-level geopolitical risk summary. |
| Geopolitical scenario overlay | `outputs/tables/hermes_geopolitical_scenario_overlay.csv` | Scenario stress overlay. |
| Geopolitical dashboard summary | `outputs/tables/hermes_geopolitical_dashboard_summary.csv` | Dashboard-ready geopolitical cards. |
| Geopolitical risk heatmap | `outputs/tables/hermes_geopolitical_risk_heatmap.csv` | Heatmap-ready risk table. |
| Geopolitical scenario dashboard | `outputs/tables/hermes_geopolitical_scenario_dashboard.csv` | Dashboard-ready scenario interpretation table. |
| Market intelligence dashboard | `outputs/tables/hermes_market_intelligence_dashboard.csv` | Combined numeric market-intelligence indicators. |
| Market intelligence summary | `outputs/tables/hermes_market_intelligence_summary.csv` | Overall market intelligence view. |
| Catalyst/risk register | `outputs/tables/hermes_catalyst_risk_register.csv` | Combined catalyst and risk register. |
| Market intelligence cards | `outputs/tables/hermes_market_intelligence_cards.csv` | Dashboard-ready market intelligence cards. |
| Market signal breakdown | `outputs/tables/hermes_market_signal_breakdown.csv` | Positive/negative/catalyst/risk signal breakdown. |
| Market watchlist | `outputs/tables/hermes_market_watchlist.csv` | Priority watchlist table. |
| Recommendation framework | `outputs/tables/hermes_recommendation_framework.csv` | Component-level recommendation point framework. |
| Recommendation summary | `outputs/tables/hermes_recommendation_summary.csv` | Final recommendation, risk rating, and conviction. |
| Recommendation rules | `outputs/tables/hermes_recommendation_rules.csv` | Transparent recommendation rules table. |
| Recommendation dashboard cards | `outputs/tables/hermes_recommendation_dashboard_cards.csv` | Dashboard-ready recommendation cards. |
| Recommendation component breakdown | `outputs/tables/hermes_recommendation_component_breakdown.csv` | Component contribution breakdown. |
| Recommendation action summary | `outputs/tables/hermes_recommendation_action_summary.csv` | Plain-English recommendation action summary. |

### Strategy Table Outputs

| Output | File | Purpose |
|---|---|---|
| Strategy results | `data/processed/hermes_strategy_results.csv` | Full strategy signals, returns, wealth, and drawdown series. |
| Strategy returns compatibility copy | `outputs/tables/hermes_strategy_returns.csv` | Dashboard/report-friendly strategy output. |
| Strategy summary | `outputs/tables/hermes_strategy_summary.csv` | Strategy return, volatility, Sharpe, drawdown, and exposure summary. |
| Strategy dashboard cards | `outputs/tables/hermes_strategy_dashboard_cards.csv` | Dashboard-ready strategy cards. |
| Strategy comparison leaderboard | `outputs/tables/hermes_strategy_comparison_leaderboard.csv` | Ranked strategy comparison table. |
| Strategy signal summary | `outputs/tables/hermes_strategy_signal_summary.csv` | Signal and exposure summary by strategy. |

### Chart Outputs

| Output | File | Purpose |
|---|---|---|
| Price history | `outputs/charts/hermes_price_history.png` | Shows Hermès' long-term price path. |
| Luxury peer performance | `outputs/charts/luxury_peer_normalized_performance.png` | Compares Hermès against luxury peers and CAC 40. |
| Strategy vs buy-and-hold | `outputs/charts/hermes_strategy_vs_buy_hold.png` | Compares the main risk-control overlay with buy-and-hold. |
| Strategy overlay comparison | `outputs/charts/hermes_strategy_overlay_comparison.png` | Compares all strategy overlays. |
| Drawdown analysis | `outputs/charts/hermes_drawdown_analysis.png` | Highlights drawdown severity. |
| Strategy drawdown comparison | `outputs/charts/hermes_strategy_drawdown_comparison.png` | Compares drawdown behavior across all strategy overlays. |
| GARCH-style volatility | `outputs/charts/hermes_garch_style_volatility.png` | Tracks annualized EWMA volatility. |
| VaR backtest | `outputs/charts/hermes_var_backtest.png` | Compares daily returns with VaR thresholds. |
| Factor exposure | `outputs/charts/hermes_factor_exposure.png` | Shows Hermès' factor exposure model. |
| Scenario returns | `outputs/charts/hermes_scenario_returns.png` | Summarizes estimated scenario returns. |
| Intelligence score | `outputs/charts/hermes_intelligence_score.png` | Breaks down Intelligence Score components. |
| Monte Carlo forecast | `outputs/charts/hermes_monte_carlo_forecast.png` | Shows simulated forecast percentiles. |
| Official fundamentals | `outputs/charts/hermes_official_fundamentals.png` | Compares official fundamentals. |
| Margin and growth profile | `outputs/charts/hermes_margin_growth_profile.png` | Compares operating margin and growth. |
| EPS vs dividend | `outputs/charts/hermes_eps_vs_dividend.png` | Shows EPS and dividend per share history. |
| Dividend yield | `outputs/charts/hermes_dividend_yield.png` | Shows dividend yield on year-end share price. |
| Market capitalization | `outputs/charts/hermes_market_cap_history.png` | Tracks market capitalization history. |
| Q1 growth by region | `outputs/charts/hermes_q1_growth_by_region.png` | Highlights Q1 2026 constant-FX growth by geography. |
| Q1 growth by sector | `outputs/charts/hermes_q1_growth_by_sector.png` | Highlights Q1 2026 constant-FX growth by product sector. |
| Q1 FX gap by region | `outputs/charts/hermes_q1_fx_gap_by_region.png` | Shows constant-FX growth minus reported growth. |
| Strategic moat score | `outputs/charts/hermes_strategy_qualitative_score.png` | Visualizes qualitative moat scoring. |
| Fair value range | `outputs/charts/hermes_fair_value_range.png` | Compares valuation cases with current price. |
| DCF sensitivity | `outputs/charts/hermes_dcf_sensitivity.png` | Shows DCF fair value sensitivity. |
| Valuation bridge | `outputs/charts/hermes_valuation_bridge.png` | Shows valuation bridge fair-value items. |
| Valuation sensitivity dashboard | `outputs/charts/hermes_valuation_sensitivity_dashboard.png` | Heatmap from long-form sensitivity data. |
| News theme summary | `outputs/charts/hermes_news_theme_summary.png` | Summarizes monitored news themes. |
| News classification summary | `outputs/charts/hermes_news_classification_summary.png` | Shows catalyst, risk, and neutral classifications. |
| News source summary | `outputs/charts/hermes_news_source_summary.png` | Summarizes tracked news sources. |
| News watchlist classification | `outputs/charts/hermes_news_watchlist_classification.png` | Shows news watchlist mix. |
| Market intelligence dashboard | `outputs/charts/hermes_market_intelligence_dashboard.png` | Visualizes numeric market-intelligence signals. |
| Geopolitical risk factors | `outputs/charts/hermes_geopolitical_risk_factors.png` | Scores individual geopolitical risk factors. |
| Geopolitical category summary | `outputs/charts/hermes_geopolitical_category_summary.png` | Summarizes geopolitical risk by category. |
| Geopolitical scenario overlay | `outputs/charts/hermes_geopolitical_scenario_overlay.png` | Shows geopolitical scenario stress overlay. |
| Recommendation framework | `outputs/charts/hermes_recommendation_framework.png` | Shows point contributions. |
| Risk leaderboard | `outputs/charts/hermes_risk_metric_leaderboard.png` | Visualizes peer risk leaderboard. |
| Target risk profile | `outputs/charts/hermes_target_risk_profile.png` | Visualizes Hermès target risk profile. |
| Revenue segment leaderboard | `outputs/charts/hermes_revenue_segment_leaderboard.png` | Visualizes Q1 revenue segment leaderboard. |
| Revenue FX dashboard | `outputs/charts/hermes_revenue_fx_dashboard.png` | Visualizes Q1 FX impact dashboard. |
| Dividend policy dashboard | `outputs/charts/hermes_dividend_policy_dashboard.png` | Visualizes dividend payout policy. |
| Shareholder return leaderboard | `outputs/charts/hermes_shareholder_return_leaderboard.png` | Shows EPS growth versus dividend growth. |
| Strategy leaderboard | `outputs/charts/hermes_strategy_comparison_leaderboard.png` | Visualizes strategy comparison leaderboard. |
| Strategy signal summary | `outputs/charts/hermes_strategy_signal_summary.png` | Visualizes average strategy exposure. |

---

## Selected Output Gallery

### Luxury Peer Performance

![Luxury Peer Normalized Performance](outputs/charts/luxury_peer_normalized_performance.png)

### Price History and Drawdown

![Hermès Price History](outputs/charts/hermes_price_history.png)

![Hermès Drawdown Analysis](outputs/charts/hermes_drawdown_analysis.png)

### Strategy Overlay Research

![Hermès Risk-Control Overlay vs Buy & Hold](outputs/charts/hermes_strategy_vs_buy_hold.png)

![Hermès Strategy Overlay Comparison](outputs/charts/hermes_strategy_overlay_comparison.png)

### Risk, VaR, and Volatility

![Hermès GARCH-Style Annualized Volatility](outputs/charts/hermes_garch_style_volatility.png)

![Hermès VaR Backtest](outputs/charts/hermes_var_backtest.png)

### Factor Exposure and Scenario Analysis

![Hermès Factor Exposure](outputs/charts/hermes_factor_exposure.png)

![Hermès Scenario Return Estimates](outputs/charts/hermes_scenario_returns.png)

### Intelligence Score and Forecast Range

![Hermès Luxury Intelligence Score Components](outputs/charts/hermes_intelligence_score.png)

![Hermès Monte Carlo Forecast Range](outputs/charts/hermes_monte_carlo_forecast.png)

### Official Fundamentals and Shareholder Returns

![Hermès Official Fundamentals](outputs/charts/hermes_official_fundamentals.png)

![Hermès Growth and Margin Profile](outputs/charts/hermes_margin_growth_profile.png)

![Hermès EPS vs Dividend per Share](outputs/charts/hermes_eps_vs_dividend.png)

![Hermès Dividend Yield](outputs/charts/hermes_dividend_yield.png)

![Hermès Market Capitalization History](outputs/charts/hermes_market_cap_history.png)

### Q1 2026 Revenue Momentum

![Hermès Q1 2026 Constant-FX Growth by Region](outputs/charts/hermes_q1_growth_by_region.png)

![Hermès Q1 2026 Constant-FX Growth by Sector](outputs/charts/hermes_q1_growth_by_sector.png)

![Hermès Q1 2026 FX Gap by Region](outputs/charts/hermes_q1_fx_gap_by_region.png)

### Valuation and Recommendation

![Hermès Fair Value Range](outputs/charts/hermes_fair_value_range.png)

![Hermès DCF Sensitivity](outputs/charts/hermes_dcf_sensitivity.png)

![Hermès Valuation Bridge](outputs/charts/hermes_valuation_bridge.png)

![Hermès Recommendation Framework](outputs/charts/hermes_recommendation_framework.png)

### News and Geopolitical Risk

![Hermès News Intelligence Theme Summary](outputs/charts/hermes_news_theme_summary.png)

![Hermès News Source Summary](outputs/charts/hermes_news_source_summary.png)

![Hermès Geopolitical Risk Factors](outputs/charts/hermes_geopolitical_risk_factors.png)

![Hermès Geopolitical Scenario Stress Overlay](outputs/charts/hermes_geopolitical_scenario_overlay.png)

### Strategic Moat Score

![Hermès Strategic Moat and Qualitative Signal Scores](outputs/charts/hermes_strategy_qualitative_score.png)

---

## Streamlit Dashboard

The project includes an interactive Streamlit dashboard in:

```text
app.py
```

Run the dashboard from the project root:

```powershell
python -m streamlit run app.py
```

Dashboard pages:

- Executive Overview
- Market Performance
- Valuation
- Shareholder Returns
- Market Intelligence
- News Intelligence
- Geopolitical Risk
- Strategy Overlay
- Export Center

The dashboard reads generated files from:

```text
outputs/tables/
outputs/charts/
outputs/reports/
```

Run `python main.py` before launching the dashboard so all tables, charts, and the PDF report exist.

The dashboard includes display safeguards for long text values such as `Constructive but monitored`, and the Strategic Moat tab reads from:

```text
outputs/tables/hermes_strategy_qualitative_score.csv
```

---

## Ticker Selection

The project uses Hermès International's primary Paris-listed ticker as the main analytical security:

```python
TARGET_TICKER = "RMS.PA"
US_OTC_TICKER = "HESAY"
```

`RMS.PA` is used for historical stock analysis because it is the primary listing on Euronext Paris. `HESAY` is included as a U.S. OTC investor reference.

---

## Peer Set and Factors

```python
TICKERS = {
    "Hermès International": "RMS.PA",
    "LVMH": "MC.PA",
    "Kering": "KER.PA",
    "Richemont": "CFR.SW",
    "CAC 40": "^FCHI",
    "EUR/USD": "EURUSD=X",
    "China Proxy ETF": "FXI",
}
```

The factor model estimates Hermès return sensitivity to:

- CAC 40 market exposure
- LVMH luxury-sector exposure
- EUR/USD FX exposure
- China demand proxy exposure

---

## Official Hermès Fundamentals

The project incorporates official Hermès key figures, including:

- Revenue
- Current and constant-FX growth
- Recurring operating income
- Recurring operating margin
- Net profit, Group share
- Operating cash flow
- Operating investments
- Adjusted free cash flow
- Net cash and restated net cash position
- Workforce

Stored in:

```text
data/raw/hermes_key_figures.csv
```

---

## Q1 2026 Revenue Momentum

The Q1 2026 revenue layer adds catalyst analysis by geography and sector.

It tracks:

- Q1 2026 revenue
- Published growth
- Constant-FX growth
- Regional revenue mix
- Sector-level momentum
- Reported growth versus constant-FX growth
- FX gap by region and segment

Stored in:

```text
data/raw/hermes_q1_2026_revenue.csv
```

---

## Share and Dividend Analysis

The shareholder-return layer tracks official Hermès share and dividend history.

It includes:

- Shares outstanding
- Market capitalization
- EPS
- Dividend per share
- Dividend payout ratio
- EPS growth
- Dividend growth
- Dividend yield
- Earnings yield
- Implied year-end P/E
- Market-cap-per-share check
- Average daily volume
- 12-month high and low share prices
- Dividend policy signal
- Valuation multiple signal

Stored in:

```text
data/raw/hermes_share_dividend_history.csv
```

---

## Valuation Model

The valuation layer adds a buy-side / sell-side style framework to the project.

It includes:

- Forward financial forecast
- Revenue growth assumptions
- Operating margin assumptions
- Net income margin assumptions
- Adjusted free cash flow conversion
- Dividend payout assumptions
- DCF valuation
- Peer-style P/E multiple valuation
- Fair value range
- DCF sensitivity by WACC and terminal growth
- Valuation signal
- Valuation bridge
- Dashboard-ready valuation cards

---

## News Intelligence

The news intelligence layer tracks recent Hermès, luxury-sector, and macro-relevant headlines through configured news queries.

It includes:

- News item table
- Headline classification
- Catalyst / risk / neutral scoring
- News theme summary
- News score summary
- News source summary
- Risk/catalyst watchlist
- Dashboard-ready news cards

---

## Geopolitical Risk Monitoring

The geopolitical risk layer converts configured risk assumptions into structured market-intelligence outputs.

It includes:

- China demand and policy exposure
- FX and macro risk
- Trade and tariff risk
- Europe consumer pressure
- Supply-chain and sourcing risk
- Geopolitical scenario adjustments
- Average geopolitical risk score
- Highest risk factor
- Overall risk posture
- Risk heatmap
- Scenario dashboard

---

## Market Intelligence Dashboard

The market intelligence module combines quantitative, qualitative, valuation, news, and geopolitical outputs into a dashboard and risk/catalyst register.

Generated outputs include:

```text
outputs/tables/hermes_market_intelligence_dashboard.csv
outputs/tables/hermes_market_intelligence_summary.csv
outputs/tables/hermes_catalyst_risk_register.csv
outputs/tables/hermes_market_intelligence_cards.csv
outputs/tables/hermes_market_signal_breakdown.csv
outputs/tables/hermes_market_watchlist.csv
outputs/charts/hermes_market_intelligence_dashboard.png
```

---

## Recommendation Framework

The recommendation framework is a rules-based layer that translates model outputs into an **Outperform**, **Neutral**, or **Underperform** recommendation.

It evaluates:

- Hermès Luxury Intelligence Score
- Valuation signal
- Implied upside/downside
- Market intelligence view
- News flow tone
- Geopolitical risk posture

Generated outputs include:

```text
outputs/tables/hermes_recommendation_framework.csv
outputs/tables/hermes_recommendation_summary.csv
outputs/tables/hermes_recommendation_rules.csv
outputs/tables/hermes_recommendation_dashboard_cards.csv
outputs/tables/hermes_recommendation_component_breakdown.csv
outputs/tables/hermes_recommendation_action_summary.csv
outputs/charts/hermes_recommendation_framework.png
```

---

## Risk Modeling

The risk layer includes:

- Daily returns
- Total return
- Annualized return
- Annualized volatility
- Downside volatility
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- Maximum drawdown
- Historical VaR
- Expected shortfall
- EWMA / GARCH-style volatility estimate
- Historical simulation VaR
- VaR breach rate
- Kupiec-style breach diagnostic
- Skew and kurtosis

---

## Factor Regression and Scenario Analysis

The factor model estimates Hermès exposure to market, luxury-sector, FX, and China proxy returns. Scenario analysis then applies bear, base, bull, China slowdown, and luxury re-rating assumptions to the estimated coefficients.

---

## Strategy Research Overlays

The project includes a research overlay module that tests several simple strategy variants against buy-and-hold. These strategies are not trading recommendations.

Included overlays:

- **Buy & Hold**
- **Risk-Control Overlay**: trend + RSI + volatility filter
- **Trend Following Overlay**: short EMA above long EMA
- **Momentum Overlay**: positive medium-term and longer-term momentum
- **Defensive Cash Overlay**: exits during downtrends or deeper drawdowns
- **Volatility Target Overlay**: scales exposure based on realized volatility

---

## Monte Carlo Forecasting

The Monte Carlo module simulates future Hermès price paths using historical daily returns. The forecast is a scenario-distribution tool, not a price prediction.

Generated outputs include:

```text
outputs/tables/hermes_monte_carlo_forecast.csv
outputs/tables/hermes_monte_carlo_final_prices.csv
outputs/tables/hermes_monte_carlo_summary.csv
outputs/tables/hermes_monte_carlo_probabilities.csv
outputs/tables/hermes_monte_carlo_dashboard.csv
outputs/charts/hermes_monte_carlo_forecast.png
```

---

## Hermès Luxury Intelligence Score

The **Hermès Luxury Intelligence Score** is a composite 0–100 score designed to summarize the project’s quantitative and qualitative findings in one signal. It is not a valuation model or investment recommendation.

The composite score combines:

| Component | What It Measures |
|---|---|
| Return Quality | Whether Hermès has generated strong annualized returns |
| Volatility Control | Whether realized volatility is relatively contained |
| Drawdown Resilience | How well the stock has avoided large peak-to-trough losses |
| Risk-Adjusted Return | Sharpe-style return quality after accounting for volatility |
| Factor Explainability | How much of Hermès' return behavior is explained by selected factors |
| VaR Calibration | Whether downside-risk models are reasonably aligned with actual breaches |
| Scenario Support | Whether base-case factor assumptions support positive estimated returns |
| Strategic Moat / Qualitative Signal | Craftsmanship, exclusivity, brand strength, family independence, and business-model quality |
| Recent Revenue Momentum | Q1 2026 constant-FX growth and regional / sector momentum |

Signals:

```text
75+   = Constructive
60–74 = Moderately Constructive
45–59 = Neutral
<45   = Cautious
```

---

## Change Detection

The pipeline saves a snapshot from each run and compares the current run against the prior run.

Tracked values include:

- Hermès Luxury Intelligence Score
- Signal
- Annualized return
- Maximum drawdown
- VaR breach rate
- Final recommendation
- Risk rating
- Valuation signal
- Implied upside/downside
- Geopolitical risk posture
- News tone
- Market intelligence view

Generated outputs:

```text
outputs/tables/hermes_change_detection.csv
outputs/snapshots/last_run_snapshot.json
```

---

## Automated PDF Report

The pipeline generates a PDF report at:

```text
outputs/reports/hermes_market_intelligence_report.pdf
```

The report includes:

- Executive summary
- Key takeaway
- Model signal
- Valuation signal
- Final recommendation
- Executive metric snapshot
- Change detection notes
- Recommendation summary
- Valuation signal table
- Market intelligence summary
- Market intelligence dashboard
- Catalyst and risk register
- News intelligence summary
- Geopolitical risk summary
- Risk/return metrics
- VaR backtest summary
- Factor exposure coefficients
- Scenario return estimates
- Intelligence score components
- Strategy overlay summary
- Official fundamentals
- Shareholder-return metrics
- Q1 2026 revenue momentum
- Strategic moat and qualitative score
- Company profile summary
- Fair value summary
- DCF valuation
- Peer multiple valuation
- DCF sensitivity
- News theme summary
- Geopolitical risk factors
- Geopolitical scenario overlay
- Recommendation framework
- Recommendation rules
- Charts
- Methodology notes
- Sources
- Educational disclaimer

---

## Offline Test Mode

The project can run in offline mode if Yahoo Finance is unavailable. Offline mode uses synthetic sample price data so the full pipeline can be tested end-to-end.

PowerShell:

```powershell
$env:HERMES_OFFLINE_MODE="1"
python main.py
```

Turn offline mode off:

```powershell
Remove-Item Env:\HERMES_OFFLINE_MODE
```

Offline sample data is for testing only and should not be used for investment interpretation.

---

## Requirements

Python 3.10 or newer is recommended.

```text
yfinance>=0.2.40
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
reportlab>=4.0.0
streamlit>=1.35.0
```

---

## How to Run

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the full pipeline:

```powershell
python main.py
```

Launch the Streamlit dashboard:

```powershell
python -m streamlit run app.py
```

The project will create output files in:

```text
outputs/charts/
outputs/tables/
outputs/reports/
outputs/snapshots/
data/processed/
```

---

## Main Data Sources

The project uses official and market-data references, including:

- Hermès Finance investor relations homepage
- Hermès digital activity report
- Hermès strategy page
- Hermès key figures page
- Hermès shareholder guide
- Hermès share and dividend page
- Hermès letter to shareholders
- Hermès Q1 revenue release
- Yahoo Finance ticker `RMS.PA`
- MarketBeat OTC reference ticker `HESAY`
- Configured RSS/news queries for Hermès and luxury-sector market intelligence

The official source URLs are stored in:

```python
SOURCE_URLS
```

inside:

```text
config/settings.py
```

---

## Methodology Notes

- Market data is downloaded through `yfinance` using the primary Hermès Paris ticker `RMS.PA`.
- Risk and return metrics are calculated from daily returns and annualized using 252 trading days.
- The GARCH-style VaR model uses EWMA volatility as a lightweight volatility-clustering approximation.
- Scenario estimates are based on factor-regression coefficients and user-defined annual factor assumptions.
- The strategy overlays are research experiments, not trading recommendations.
- The valuation layer uses a forward forecast, DCF analysis, peer-style P/E multiples, and sensitivity analysis.
- The news layer uses configured search/RSS queries and transparent keyword rules to classify news flow as catalyst, risk, or neutral.
- The geopolitical risk layer converts configured exposure assumptions into risk-factor, category, and scenario-overlay tables.
- The recommendation framework is rules-based and combines quality, valuation, news flow, market intelligence, and geopolitical posture.
- The Hermès Luxury Intelligence Score is a composite indicator, not a valuation model or investment recommendation.
- The Streamlit dashboard is a presentation layer over generated output files and should be refreshed by rerunning `python main.py`.
- Subjective inputs such as valuation multiples, DCF assumptions, geopolitical risk scores, qualitative moat scores, scenario assumptions, and recommendation weights are configured research assumptions and should be reviewed through sensitivity analysis.

---

## Disclaimer

This project is for educational and portfolio purposes only. It is not investment advice, a valuation opinion, or a recommendation to buy, sell, or hold any security.

The output depends on public market data, official Hermès disclosures, configured model assumptions, and rules-based scoring choices. Any valuation signal, risk score, intelligence score, or recommendation label should be interpreted as a framework output rather than a definitive investment conclusion.
