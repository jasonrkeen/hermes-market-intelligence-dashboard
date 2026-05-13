"""PDF report generation for the Hermès market intelligence pipeline.

This module creates a polished PDF report with an executive summary,
key takeaway, model signal, valuation view, recommendation framework,
core analytics tables, change detection, charts, methodology notes, and sources.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from config.settings import (
    KEY_TAKEAWAY,
    REPORT_TITLE,
    SOURCE_URLS,
    STRATEGY_DESCRIPTION,
)


def _format_number(value: Any) -> str:
    """Format numeric values for PDF tables."""
    if pd.isna(value):
        return ""

    if isinstance(value, (int, float)):
        abs_value = abs(float(value))

        if abs_value >= 1_000_000:
            return f"{value:,.0f}"
        if abs_value >= 1_000:
            return f"{value:,.2f}"
        return f"{value:.4f}"

    return str(value)


def _format_snapshot_value(metric: str, value: Any) -> str:
    """Format executive snapshot values for readability."""
    if pd.isna(value):
        return ""

    metric_lower = metric.lower()

    if isinstance(value, (int, float)):
        if any(term in metric_lower for term in ["return", "volatility", "drawdown"]):
            return f"{value:.2%}"

        if "score" in metric_lower:
            return f"{value:.2f}"

        if "sharpe" in metric_lower:
            return f"{value:.2f}"

        return _format_number(value)

    return str(value)


def _format_table_value(column: str, value: Any, row_context: str = "") -> str:
    """Format table values based on column name and row context."""
    if pd.isna(value):
        return ""

    column_lower = str(column).lower()
    row_lower = str(row_context).lower()
    context = f"{column_lower} {row_lower}"

    if not isinstance(value, (int, float)):
        return str(value)

    numeric_value = float(value)

    # ---------------------------------------------------------------------
    # Calendar years.
    # Keep years like 2021 instead of 2,021 or 2021.0000.
    # ---------------------------------------------------------------------
    if column_lower == "year":
        return f"{numeric_value:.0f}"

    # ---------------------------------------------------------------------
    # DCF sensitivity table:
    # WACC columns contain fair value outputs, not WACC percentages.
    # ---------------------------------------------------------------------
    if column_lower.startswith("wacc") and abs(numeric_value) > 10:
        return f"€{numeric_value:,.2f}"

    # ---------------------------------------------------------------------
    # Counts and whole-number quantities.
    # ---------------------------------------------------------------------
    count_terms = [
        "observations",
        "var breaches",
        "news items tracked",
        "catalyst items",
        "risk items",
        "neutral items",
        "headline count",
        "count",
        "shares outstanding",
        "workforce",
        "avg daily volume",
    ]

    if any(term in context for term in count_terms):
        return f"{numeric_value:,.0f}"

    # ---------------------------------------------------------------------
    # Score / model component rows.
    # These should stay as plain scores, not percentages or currency.
    # ---------------------------------------------------------------------
    score_context_terms = [
        "score",
        "score_1_to_5",
        "score_0_to_100",
        "weighted score",
        "risk score",
        "stress score",
        "points",
        "risk level",
        "qualitative moat score",
        "hermès luxury intelligence score",
        "recent revenue momentum",
        "volatility control",
        "drawdown resilience",
        "return quality",
        "risk-adjusted return",
        "factor explainability",
        "var calibration",
        "scenario support",
        "strategic moat / qualitative signal",
    ]

    if any(term in context for term in score_context_terms):
        return f"{numeric_value:.2f}"

    # ---------------------------------------------------------------------
    # Ratios and factor-model coefficients.
    # These should not become percentages.
    # ---------------------------------------------------------------------
    ratio_terms = [
        "beta",
        "sharpe",
        "sortino",
        "calmar",
        "r-squared",
        "r squared",
        "skew",
        "kurtosis",
        "coefficient",
        "alpha",
        "eur/usd",
        "cac 40 contribution",
        "lvmh contribution",
        "china proxy etf",
        "annualized alpha contribution",
    ]

    if any(term in context for term in ratio_terms):
        return f"{numeric_value:.4f}" if abs(numeric_value) < 1 else f"{numeric_value:.2f}"

    # ---------------------------------------------------------------------
    # Return-style decimals:
    # 2.9497 = 294.97%, 0.1786 = 17.86%.
    # ---------------------------------------------------------------------
    decimal_percent_terms = [
        "total return",
        "annualized return",
        "estimated hermès return",
        "scenario return",
        "base case scenario return",
        "implied upside",
        "implied downside",
        "upside / downside",
        "drawdown",
        "maximum drawdown",
        "max drawdown",
        "volatility",
        "breach rate",
        "average exposure",
    ]

    if any(term in context for term in decimal_percent_terms):
        return f"{numeric_value:.2%}"

    # ---------------------------------------------------------------------
    # Standard percentage fields:
    # 5.6 = 5.60%, 0.085 = 8.50%.
    # ---------------------------------------------------------------------
    percent_terms = [
        "growth",
        "margin",
        "yield",
        "wacc",
        "terminal growth",
        "rate",
        "constant-fx",
        "fx gap",
        "adjustment pct",
        "revenue growth adjustment",
        "margin adjustment",
        "valuation multiple adjustment",
        "percent",
        "percentage",
        "share of revenue",
        "weight",
    ]

    if any(term in context for term in percent_terms):
        if abs(numeric_value) <= 1:
            return f"{numeric_value:.2%}"
        return f"{numeric_value:.2f}%"

    # ---------------------------------------------------------------------
    # Currency-like values.
    # Currency comes after percent/score logic so revenue-growth and
    # implied-upside rows do not get formatted as euros.
    # ---------------------------------------------------------------------
    currency_terms = [
        "fair value",
        "current price",
        "price",
        "per share",
        "market cap",
        "revenue",
        "income",
        "cash",
        "dividend",
        "eps",
        "free cash flow",
        "fcf",
        "equity value",
        "enterprise value",
        "terminal value",
        "pv fcf",
        "present value",
        "value eurm",
        "eur millions",
        "eur bn",
        "eurm",
    ]

    non_currency_terms = [
        "implied upside",
        "implied downside",
        "upside / downside",
        "eur/usd",
        "wacc",
        "terminal growth",
        "growth",
        "margin",
        "percent change",
        "percentage-point",
        "share of revenue",
        "revenue growth adjustment",
    ]

    if any(term in context for term in currency_terms) and not any(
        term in context for term in non_currency_terms
    ):
        return f"€{numeric_value:,.2f}"

    if abs(numeric_value) >= 1_000_000:
        return f"{numeric_value:,.0f}"

    if abs(numeric_value) >= 1_000:
        return f"{numeric_value:,.2f}"

    return f"{numeric_value:.4f}"


def _clean_heading_from_path(path: Path) -> str:
    """Create a clean chart heading from a chart file path."""
    title = path.stem.replace("_", " ").title()

    replacements = {
        "Hermes": "Hermès",
        "Garch": "GARCH",
        "Var": "VaR",
        "Eps": "EPS",
        "Q1": "Q1",
        "Fx": "FX",
        "Vs": "vs",
        "Buy Hold": "Buy & Hold",
        "Pct": "pct.",
        "Dcf": "DCF",
        "Wacc": "WACC",
    }

    for old, new in replacements.items():
        title = title.replace(old, new)

    cleanup_map = {
        "Hermès Strategy vs Buy & Hold": "Hermès Risk-Control Overlay vs Buy & Hold",
        "Hermès GARCH Style Volatility": "Hermès GARCH-Style Volatility",
        "Hermès GARCH Style Annualized Volatility": "Hermès GARCH-Style Annualized Volatility",
        "Hermès Q1 FX Gap By Region": "Hermès Q1 2026 FX Gap by Region",
        "Hermès Q1 Growth By Region": "Hermès Q1 2026 Constant-FX Growth by Region",
        "Hermès Q1 Growth By Sector": "Hermès Q1 2026 Constant-FX Growth by Sector",
        "Hermès EPS vs Dividend": "Hermès EPS vs Dividend per Share",
        "Hermès Eps vs Dividend": "Hermès EPS vs Dividend per Share",
        "Hermès VaR Backtest": "Hermès VaR Backtest: Returns vs Risk Thresholds",
        "Hermès Strategy Overlay Comparison": "Hermès Strategy Overlay Comparison",
        "Hermès Strategy Drawdown Comparison": "Hermès Strategy Drawdown Comparison",
        "Luxury Peer Normalized Performance": "Luxury Peer Normalized Performance",
        "Hermès Fair Value Range": "Hermès Fair Value Range",
        "Hermès DCF Sensitivity": "Hermès DCF Sensitivity: WACC vs Terminal Growth",
        "Hermès News Theme Summary": "Hermès News Intelligence Theme Summary",
        "Hermès News Classification Summary": "Hermès News Classification Summary",
        "Hermès Market Intelligence Dashboard": "Hermès Market Intelligence Dashboard",
        "Hermès Geopolitical Risk Factors": "Hermès Geopolitical Risk Factors",
        "Hermès Geopolitical Category Summary": "Hermès Geopolitical Risk by Category",
        "Hermès Geopolitical Scenario Overlay": "Hermès Geopolitical Scenario Stress Overlay",
        "Hermès Recommendation Framework": "Hermès Recommendation Framework",
        "Hermès Recommendation Summary": "Hermès Recommendation Summary",
    }

    return cleanup_map.get(title, title)


def _paragraph_cell(value: Any, style: ParagraphStyle) -> Paragraph:
    """Convert table cell content into a wrapped ReportLab paragraph."""
    text = "" if value is None else str(value)

    text = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    return Paragraph(text, style)


def _build_column_widths(columns: list[str], available_width: float) -> list[float]:
    """Build proportional column widths for compact PDF tables."""
    weights: list[float] = []

    for column in columns:
        column_lower = str(column).lower()

        if column_lower in {"interpretation", "monitoring note", "exposure", "url"}:
            weights.append(3.0)
        elif column_lower in {"metric", "line item", "indicator", "risk factor"}:
            weights.append(1.8)
        elif column_lower in {"category", "component", "scenario", "case"}:
            weights.append(1.5)
        elif column_lower in {"value", "input", "assessment", "scoring role"}:
            weights.append(1.2)
        elif "fair value" in column_lower or "current price" in column_lower:
            weights.append(1.4)
        else:
            weights.append(1.0)

    total_weight = sum(weights)

    return [(weight / total_weight) * available_width for weight in weights]


def _table_from_df(
    df: pd.DataFrame,
    max_rows: int = 12,
    max_cols: int = 9,
    font_size: int = 7,
    include_index: bool | None = None,
) -> Table:
    """Convert a DataFrame into a compact ReportLab table with wrapped text."""
    display = df.copy().head(max_rows)

    if include_index is None:
        include_index = (
            isinstance(display.index, pd.MultiIndex)
            or display.index.name is not None
            or not isinstance(display.index, pd.RangeIndex)
        )

    if include_index:
        display = display.reset_index()
        display = display.rename(columns={"index": "", "level_0": ""})

    if len(display.columns) > max_cols:
        display = display.iloc[:, :max_cols].copy()

    context_columns = [
        col
        for col in display.columns
        if str(col).lower()
        in {
            "metric",
            "line item",
            "indicator",
            "scenario",
            "case",
            "component",
            "category",
            "insight",
            "factor",
            "risk factor",
            "section",
        }
    ]

    for col in display.columns:
        display[col] = display.apply(
            lambda row: _format_table_value(
                column=col,
                value=row[col],
                row_context=" ".join(
                    str(row[context_col]) for context_col in context_columns
                ),
            ),
            axis=1,
        )

    header_style = ParagraphStyle(
        name="TableHeader",
        fontName="Helvetica-Bold",
        fontSize=font_size,
        leading=font_size + 2,
        alignment=1,
    )

    body_style = ParagraphStyle(
        name="TableBody",
        fontName="Helvetica",
        fontSize=font_size,
        leading=font_size + 2,
    )

    data = [
        [_paragraph_cell(column, header_style) for column in display.columns]
    ]

    for _, row in display.iterrows():
        data.append(
            [_paragraph_cell(row[column], body_style) for column in display.columns]
        )

    available_width = 7.2 * inch
    column_widths = _build_column_widths(
        columns=list(display.columns),
        available_width=available_width,
    )

    table = Table(
        data,
        colWidths=column_widths,
        repeatRows=1,
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EFEFEF")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#222222")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#BDBDBD")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#FAFAFA")],
                ),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    return table

def _extract_table_value(
    table: pd.DataFrame | None,
    lookup_column: str,
    lookup_value: str,
    value_column: str,
    default: str = "Not Available",
) -> str:
    """Extract a value from a simple long-form table."""
    if table is None or table.empty:
        return default

    if lookup_column not in table.columns or value_column not in table.columns:
        return default

    rows = table[table[lookup_column].eq(lookup_value)]

    if rows.empty:
        return default

    return str(rows.iloc[0][value_column])


def _build_metric_snapshot(
    metrics: pd.DataFrame,
    intelligence_score: pd.DataFrame,
    valuation_signal: pd.DataFrame | None = None,
    recommendation_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create a compact executive snapshot table."""
    rows: list[dict[str, Any]] = []

    if "Hermès International" in metrics.index:
        hermes = metrics.loc["Hermès International"]

        metric_map = {
            "Annualized Return": "Annualized Return",
            "Annualized Volatility": "Annualized Volatility",
            "Sharpe Ratio": "Sharpe Ratio",
            "Max Drawdown": "Max Drawdown",
        }

        for label, column in metric_map.items():
            if column in hermes.index:
                rows.append({"Metric": label, "Value": hermes[column]})

    score_label = "Hermès Luxury Intelligence Score"

    if score_label in intelligence_score.index:
        rows.append(
            {
                "Metric": "Hermès Luxury Intelligence Score",
                "Value": intelligence_score.loc[score_label, "Score"],
            }
        )

    valuation_view = _extract_table_value(
        valuation_signal,
        lookup_column="Metric",
        lookup_value="Valuation Signal",
        value_column="Value",
    )

    if valuation_view != "Not Available":
        rows.append({"Metric": "Valuation Signal", "Value": valuation_view})

    recommendation = _extract_table_value(
        recommendation_summary,
        lookup_column="Metric",
        lookup_value="Final Recommendation",
        value_column="Value",
    )

    if recommendation != "Not Available":
        rows.append({"Metric": "Final Recommendation", "Value": recommendation})

    risk_rating = _extract_table_value(
        recommendation_summary,
        lookup_column="Metric",
        lookup_value="Risk Rating",
        value_column="Value",
    )

    if risk_rating != "Not Available":
        rows.append({"Metric": "Risk Rating", "Value": risk_rating})

    snapshot = pd.DataFrame(rows)

    if not snapshot.empty:
        snapshot["Value"] = snapshot.apply(
            lambda row: _format_snapshot_value(row["Metric"], row["Value"]),
            axis=1,
        )

    return snapshot


def _get_final_score_and_signal(
    intelligence_score: pd.DataFrame,
) -> tuple[float | None, str]:
    """Extract final score and convert it to a signal label."""
    score_label = "Hermès Luxury Intelligence Score"

    if score_label not in intelligence_score.index:
        return None, "Not Available"

    score = float(intelligence_score.loc[score_label, "Score"])

    if score >= 75:
        signal = "Constructive"
    elif score >= 60:
        signal = "Moderately Constructive"
    elif score >= 45:
        signal = "Neutral"
    else:
        signal = "Cautious"

    return score, signal


def _add_section_heading(story: list[Any], heading: str, styles) -> None:
    """Append a section heading to the PDF story."""
    story.append(Spacer(1, 8))
    story.append(Paragraph(heading, styles["CustomHeading2"]))
    story.append(Spacer(1, 6))


def _image_flowable(
    path: Path,
    max_width: float = 7.0 * inch,
    max_height: float = 3.8 * inch,
) -> Image:
    """Create a ReportLab image while preserving aspect ratio."""
    reader = ImageReader(str(path))
    image_width, image_height = reader.getSize()

    width_ratio = max_width / image_width
    height_ratio = max_height / image_height
    scale = min(width_ratio, height_ratio)

    final_width = image_width * scale
    final_height = image_height * scale

    return Image(str(path), width=final_width, height=final_height)


def _build_sources_table() -> pd.DataFrame:
    """Build a source table from settings.py source URLs."""
    return pd.DataFrame(
        [{"Source": source, "URL": url} for source, url in SOURCE_URLS.items()]
    )


def _build_methodology_notes() -> list[str]:
    """Create methodology notes for the report."""
    return [
        "Market data is downloaded through yfinance using the primary Hermès Paris ticker RMS.PA.",
        "Risk and return metrics are calculated from daily return series and annualized using 252 trading days.",
        "The GARCH-style VaR model uses EWMA volatility as a lightweight volatility-clustering approximation.",
        "Scenario estimates are based on factor-regression coefficients and user-defined annual factor assumptions.",
        "The Hermès Luxury Intelligence Score is a composite indicator, not an investment recommendation.",
        "The valuation layer uses a forward financial forecast, DCF analysis, peer-style P/E multiples, and sensitivity analysis.",
        "The news layer uses RSS headline monitoring and transparent keyword rules to classify news flow as catalyst, risk, or neutral.",
        "The geopolitical risk layer converts configured exposure assumptions into risk-factor, category, and scenario-overlay tables.",
        "The recommendation framework is rules-based and combines quality, valuation, news flow, market intelligence, and geopolitical posture.",
        STRATEGY_DESCRIPTION,
    ]


def _build_intelligence_score_explanation() -> list[str]:
    """Create plain-English explanation notes for the intelligence score."""
    return [
        (
            "The Hermès Luxury Intelligence Score is a composite 0–100 score designed "
            "to summarize the project’s quantitative and qualitative findings in one signal."
        ),
        (
            "It is not a valuation model or investment recommendation. It combines market "
            "performance, downside risk, factor-model support, scenario analysis, official "
            "Hermès fundamentals, strategic moat variables, and recent revenue momentum."
        ),
        (
            "Higher scores indicate stronger overall support across return quality, volatility "
            "control, drawdown resilience, risk-adjusted return, factor explainability, VaR "
            "calibration, scenario support, strategic moat, and recent revenue momentum."
        ),
        (
            "A Moderately Constructive signal means the model sees meaningful business and "
            "market support, but not enough to classify the setup as fully Constructive."
        ),
    ]


def build_pdf_report(
    report_path: Path,
    metrics: pd.DataFrame,
    var_summary: pd.DataFrame,
    factor_coefficients: pd.DataFrame,
    scenario_table: pd.DataFrame,
    intelligence_score: pd.DataFrame,
    chart_paths: list[Path],
    change_notes: list[str],
    fundamental_metrics: pd.DataFrame | None = None,
    dividend_metrics: pd.DataFrame | None = None,
    revenue_momentum_summary: pd.DataFrame | None = None,
    strategy_scores: pd.DataFrame | None = None,
    company_profile_summary: pd.DataFrame | None = None,
    sources_table: pd.DataFrame | None = None,
    valuation_signal: pd.DataFrame | None = None,
    fair_value_summary: pd.DataFrame | None = None,
    dcf_valuation: pd.DataFrame | None = None,
    peer_valuation: pd.DataFrame | None = None,
    dcf_sensitivity: pd.DataFrame | None = None,
    news_summary: pd.DataFrame | None = None,
    news_theme_summary: pd.DataFrame | None = None,
    geopolitical_summary: pd.DataFrame | None = None,
    geopolitical_risks: pd.DataFrame | None = None,
    geopolitical_scenarios: pd.DataFrame | None = None,
    market_dashboard: pd.DataFrame | None = None,
    market_summary: pd.DataFrame | None = None,
    catalyst_risk_register: pd.DataFrame | None = None,
    recommendation_summary: pd.DataFrame | None = None,
    recommendation_framework: pd.DataFrame | None = None,
    recommendation_rules: pd.DataFrame | None = None,
    strategy_summary: pd.DataFrame | None = None,
) -> None:
    """Build the Hermès market intelligence PDF report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(report_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="CustomHeading2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=14,
            spaceAfter=4,
            textColor=colors.HexColor("#222222"),
        )
    )

    styles.add(
        ParagraphStyle(
            name="SmallBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
        )
    )

    styles.add(
        ParagraphStyle(
            name="KeyTakeaway",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            backColor=colors.HexColor("#F7F7F7"),
            borderColor=colors.HexColor("#D0D0D0"),
            borderWidth=0.5,
            borderPadding=8,
            spaceAfter=10,
        )
    )

    story: list[Any] = []

    final_score, signal = _get_final_score_and_signal(intelligence_score)

    # -------------------------------------------------------------------------
    # Cover / executive summary
    # -------------------------------------------------------------------------

    story.append(Paragraph(REPORT_TITLE, styles["Title"]))
    story.append(Spacer(1, 10))

    intro = (
        "This report combines luxury peer performance, downside risk modeling, "
        "factor sensitivity, scenario analysis, official Hermès fundamentals, "
        "shareholder-return data, Q1 2026 revenue momentum, valuation analysis, "
        "news intelligence, geopolitical risk monitoring, and a composite Hermès "
        "Luxury Intelligence Score."
    )

    story.append(Paragraph(intro, styles["BodyText"]))
    story.append(Spacer(1, 10))

    story.append(
        Paragraph("<b>Key Takeaway:</b> " + KEY_TAKEAWAY, styles["KeyTakeaway"])
    )

    final_recommendation = _extract_table_value(
        recommendation_summary,
        lookup_column="Metric",
        lookup_value="Final Recommendation",
        value_column="Value",
    )

    valuation_view = _extract_table_value(
        valuation_signal,
        lookup_column="Metric",
        lookup_value="Valuation Signal",
        value_column="Value",
    )

    if final_score is not None:
        signal_text = (
            f"<b>Model Signal:</b> {signal}<br/>"
            f"<b>Hermès Luxury Intelligence Score:</b> {final_score:.2f} / 100"
        )
    else:
        signal_text = "<b>Model Signal:</b> Not Available"

    if valuation_view != "Not Available":
        signal_text += f"<br/><b>Valuation Signal:</b> {valuation_view}"

    if final_recommendation != "Not Available":
        signal_text += f"<br/><b>Final Recommendation:</b> {final_recommendation}"

    story.append(Paragraph(signal_text, styles["BodyText"]))
    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            "Interpretation: the model signal reflects a high-quality luxury compounder profile, "
            "supported by strategic moat, official fundamentals, and revenue momentum, but balanced "
            "against drawdown pressure, valuation sensitivity, FX effects, geopolitical risk, and "
            "current news flow.",
            styles["SmallBody"],
        )
    )

    story.append(
        Paragraph(
            "The recommendation can remain Neutral even when valuation is supportive because the "
            "framework also penalizes elevated geopolitical risk, risk-leaning news flow, and "
            "medium-high overall risk conditions.",
            styles["SmallBody"],
        )
    )

    story.append(Spacer(1, 12))

    snapshot = _build_metric_snapshot(
        metrics=metrics,
        intelligence_score=intelligence_score,
        valuation_signal=valuation_signal,
        recommendation_summary=recommendation_summary,
    )

    if not snapshot.empty:
        _add_section_heading(story, "Executive Metric Snapshot", styles)
        story.append(
            _table_from_df(
                snapshot,
                max_rows=12,
                max_cols=4,
                font_size=8,
                include_index=False,
            )
        )
        story.append(Spacer(1, 12))

    _add_section_heading(story, "Change Detection", styles)

    for note in change_notes:
        story.append(Paragraph(f"• {note}", styles["SmallBody"]))

    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # Market intelligence and recommendation tables
    # -------------------------------------------------------------------------

    front_tables: list[tuple[str, pd.DataFrame, int, int, bool | None]] = []

    if recommendation_summary is not None:
        front_tables.append(
            ("Recommendation Summary", recommendation_summary, 8, 4, False)
        )

    if valuation_signal is not None:
        front_tables.append(("Valuation Signal", valuation_signal, 8, 4, False))

    if market_summary is not None:
        front_tables.append(
            ("Market Intelligence Summary", market_summary, 8, 4, False)
        )

    if market_dashboard is not None:
        front_tables.append(
            ("Market Intelligence Dashboard", market_dashboard, 14, 5, False)
        )

    if catalyst_risk_register is not None:
        front_tables.append(
            ("Catalyst and Risk Register", catalyst_risk_register, 14, 5, False)
        )

    if news_summary is not None:
        front_tables.append(("News Intelligence Summary", news_summary, 8, 4, False))

    if geopolitical_summary is not None:
        front_tables.append(
            ("Geopolitical Risk Summary", geopolitical_summary, 8, 4, False)
        )

    for heading, df, max_rows, max_cols, include_index in front_tables:
        if df is None or df.empty:
            continue

        _add_section_heading(story, heading, styles)
        story.append(
            _table_from_df(
                df,
                max_rows=max_rows,
                max_cols=max_cols,
                font_size=7,
                include_index=include_index,
            )
        )
        story.append(Spacer(1, 12))

    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # Core analytics tables
    # -------------------------------------------------------------------------

    tables: list[tuple[str, pd.DataFrame, int, int]] = [
        ("Risk/Return Metrics", metrics, 12, 9),
        ("VaR Backtest Summary", var_summary, 12, 8),
        ("Factor Exposure Coefficients", factor_coefficients, 14, 6),
        ("Scenario Return Estimates", scenario_table, 12, 9),
        ("Hermès Luxury Intelligence Score", intelligence_score, 14, 6),
    ]

    if strategy_summary is not None:
        tables.append(("Strategy Overlay Summary", strategy_summary, 8, 8))

    if fundamental_metrics is not None:
        tables.append(("Official Hermès Fundamentals", fundamental_metrics, 14, 8))

    if dividend_metrics is not None:
        tables.append(
            ("Shareholder Return and Dividend Metrics", dividend_metrics, 8, 9)
        )

    if revenue_momentum_summary is not None:
        tables.append(("Q1 2026 Revenue Momentum", revenue_momentum_summary, 12, 7))

    if strategy_scores is not None:
        tables.append(
            ("Strategic Moat and Qualitative Score", strategy_scores, 14, 6)
        )

    if company_profile_summary is not None:
        tables.append(("Company Profile Summary", company_profile_summary, 12, 5))

    for heading, df, max_rows, max_cols in tables:
        if df is None or df.empty:
            continue

        _add_section_heading(story, heading, styles)
        story.append(_table_from_df(df, max_rows=max_rows, max_cols=max_cols))

        if heading == "Hermès Luxury Intelligence Score":
            story.append(Spacer(1, 8))
            story.append(Paragraph("<b>How to read this score:</b>", styles["SmallBody"]))

            for note in _build_intelligence_score_explanation():
                story.append(Paragraph(f"• {note}", styles["SmallBody"]))

        story.append(Spacer(1, 12))

    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # Valuation, news, geopolitical, and recommendation appendix tables
    # -------------------------------------------------------------------------

    if fair_value_summary is not None or dcf_valuation is not None:
        _add_section_heading(story, "Valuation Interpretation", styles)
        story.append(
            Paragraph(
                "The valuation section combines a conservative DCF view with peer-style "
                "multiple-based fair value cases. The DCF output is highly sensitive to WACC, "
                "terminal growth, and long-term margin assumptions, while the fair value range "
                "also reflects market-based luxury-sector valuation support.",
                styles["SmallBody"],
            )
        )
        story.append(Spacer(1, 12))

    appendix_tables: list[tuple[str, pd.DataFrame | None, int, int, bool | None]] = [
        ("Fair Value Summary", fair_value_summary, 8, 6, None),
        ("DCF Valuation", dcf_valuation, 18, 4, False),
        ("Peer Multiple Valuation", peer_valuation, 8, 6, None),
        ("DCF Sensitivity", dcf_sensitivity, 8, 8, None),
        ("News Theme Summary", news_theme_summary, 12, 4, False),
        ("Geopolitical Risk Factors", geopolitical_risks, 12, 6, False),
        ("Geopolitical Scenario Overlay", geopolitical_scenarios, 8, 6, False),
        ("Recommendation Framework", recommendation_framework, 14, 5, False),
        ("Recommendation Rules", recommendation_rules, 14, 4, False),
    ]

    for heading, df, max_rows, max_cols, include_index in appendix_tables:
        if df is None or df.empty:
            continue

        _add_section_heading(story, heading, styles)
        story.append(
            _table_from_df(
                df,
                max_rows=max_rows,
                max_cols=max_cols,
                font_size=7,
                include_index=include_index,
            )
        )
        story.append(Spacer(1, 12))

    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # Charts
    # -------------------------------------------------------------------------

    for path in chart_paths:
        chart_path = Path(path)

        if not chart_path.exists():
            continue

        heading = _clean_heading_from_path(chart_path)

        story.append(
            KeepTogether(
                [
                    Paragraph(heading, styles["CustomHeading2"]),
                    Spacer(1, 6),
                    _image_flowable(chart_path),
                    Spacer(1, 14),
                ]
            )
        )

    # -------------------------------------------------------------------------
    # Methodology and sources
    # -------------------------------------------------------------------------

    story.append(PageBreak())

    _add_section_heading(story, "Methodology Notes", styles)

    for note in _build_methodology_notes():
        story.append(Paragraph(f"• {note}", styles["SmallBody"]))

    story.append(Spacer(1, 12))

    _add_section_heading(story, "Sources", styles)

    final_sources_table = sources_table if sources_table is not None else _build_sources_table()

    story.append(
        _table_from_df(
            final_sources_table,
            max_rows=20,
            max_cols=3,
            font_size=6,
            include_index=False,
        )
    )

    story.append(Spacer(1, 10))

    disclaimer = (
        "This report is for educational and research purposes only. It is not financial advice, "
        "investment advice, or a recommendation to buy, sell, or hold any security."
    )

    story.append(Paragraph(disclaimer, styles["SmallBody"]))

    doc.build(story)