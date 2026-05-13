"""News and market intelligence tracking for the Hermès market intelligence pipeline.

This module pulls recent news headlines from RSS feeds, classifies each item as
a catalyst, risk, or neutral item, and creates market-intelligence summary
tables for the PDF report, README, and GitHub outputs.

The scoring is intentionally lightweight and transparent. It is not sentiment AI;
it is a rules-based news monitor for identifying current market themes.

The module also creates dashboard-ready card, watchlist, and source-summary
tables for Streamlit and other interactive reporting outputs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import pandas as pd


POSITIVE_KEYWORDS = {
    "beat",
    "beats",
    "growth",
    "grew",
    "strong",
    "resilient",
    "resilience",
    "record",
    "outperform",
    "outperformed",
    "positive",
    "rebound",
    "recovery",
    "pricing",
    "price increase",
    "margin",
    "high margin",
    "cash",
    "demand",
    "americas",
    "japan",
    "scarcity",
    "leather goods",
    "waiting list",
}

NEGATIVE_KEYWORDS = {
    "miss",
    "misses",
    "slowdown",
    "weak",
    "weaker",
    "weak demand",
    "decline",
    "declined",
    "drop",
    "drops",
    "fell",
    "falls",
    "war",
    "conflict",
    "tariff",
    "tariffs",
    "fx",
    "currency",
    "headwind",
    "china",
    "tourism",
    "middle east",
    "downgrade",
    "valuation",
    "drawdown",
    "re-rating",
    "pressure",
    "aspirational",
    "inventory",
    "destocking",
    "consumer confidence",
    "travel disruption",
}

RISK_THEME_KEYWORDS = {
    "Geopolitical / Middle East Risk": [
        "war",
        "conflict",
        "iran",
        "middle east",
        "dubai",
        "gulf",
    ],
    "China Demand Risk": [
        "china",
        "chinese",
        "asia",
        "asia-pacific",
        "hong kong",
        "consumer confidence",
        "weak demand",
    ],
    "FX / Currency Risk": [
        "fx",
        "currency",
        "euro",
        "exchange rate",
        "headwind",
    ],
    "Luxury Sector Re-rating": [
        "luxury",
        "sector",
        "lvmh",
        "kering",
        "gucci",
        "re-rating",
        "valuation",
    ],
    "Tourism / Travel Retail": [
        "tourism",
        "tourist",
        "airport",
        "travel retail",
        "travel disruption",
    ],
    "Inventory / Wholesale Risk": [
        "inventory",
        "destocking",
        "wholesale",
    ],
}

CATALYST_THEME_KEYWORDS = {
    "Brand Strength / Moat": [
        "resilient",
        "resilience",
        "outperform",
        "outperformed",
        "scarcity",
        "brand",
        "pricing",
        "price increase",
        "waiting list",
    ],
    "Revenue Momentum": [
        "growth",
        "sales",
        "revenue",
        "constant-fx",
        "constant currency",
    ],
    "Americas Strength": [
        "americas",
        "u.s.",
        "us",
        "united states",
    ],
    "Japan Strength": [
        "japan",
    ],
    "Margin / Cash Strength": [
        "margin",
        "high margin",
        "cash",
        "profit",
        "operating income",
    ],
    "Leather Goods Strength": [
        "leather goods",
        "saddlery",
        "birkin",
        "kelly",
    ],
}


def _google_news_rss_url(query: str) -> str:
    """Build a Google News RSS URL for a search query."""
    encoded_query = quote_plus(query)

    return (
        "https://news.google.com/rss/search?"
        f"q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    )


def _fetch_rss_xml(url: str, timeout: int = 15) -> bytes:
    """Fetch RSS XML using a basic browser-like user agent."""
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 HermèsMarketIntelligenceBot/1.0 "
                "(educational research project)"
            )
        },
    )

    with urlopen(request, timeout=timeout) as response:
        return response.read()


def _parse_pub_date(value: str | None) -> datetime | None:
    """Parse RSS publication date safely."""
    if not value:
        return None

    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def _clean_text(value: str | None) -> str:
    """Clean text fields from RSS."""
    if value is None:
        return ""

    return " ".join(str(value).replace("\n", " ").split())


def _safe_numeric(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_summary_value(
    news_summary: pd.DataFrame,
    metric: str,
    default: Any = None,
) -> Any:
    """Extract a metric value from the news summary table."""
    if news_summary.empty or "Metric" not in news_summary.columns:
        return default

    rows = news_summary[news_summary["Metric"].eq(metric)]

    if rows.empty:
        return default

    return rows.iloc[0].get("Value", default)


def _extract_summary_interpretation(
    news_summary: pd.DataFrame,
    metric: str,
    default: str = "Not Available",
) -> str:
    """Extract a metric interpretation from the news summary table."""
    if news_summary.empty or "Metric" not in news_summary.columns:
        return default

    rows = news_summary[news_summary["Metric"].eq(metric)]

    if rows.empty:
        return default

    return str(rows.iloc[0].get("Interpretation", default))


def _classify_watchlist_priority(
    classification: str,
    keyword_score: float,
    themes: str,
) -> str:
    """Assign priority to a news item for the dashboard watchlist."""
    classification_lower = str(classification).lower()
    themes_lower = str(themes).lower()
    score_abs = abs(float(keyword_score))

    high_risk_themes = {
        "geopolitical",
        "middle east",
        "china",
        "fx",
        "currency",
        "re-rating",
        "valuation",
    }

    high_catalyst_themes = {
        "brand strength",
        "moat",
        "revenue momentum",
        "margin",
        "cash",
        "leather goods",
    }

    if classification_lower == "risk":
        if score_abs >= 2 or any(theme in themes_lower for theme in high_risk_themes):
            return "High"
        return "Medium"

    if classification_lower == "catalyst":
        if score_abs >= 2 or any(theme in themes_lower for theme in high_catalyst_themes):
            return "Medium"
        return "Monitor"

    return "Monitor"


def _news_action_note(
    classification: str,
    themes: str,
) -> str:
    """Create a dashboard monitoring note for a news item."""
    classification_lower = str(classification).lower()
    themes_text = str(themes) if themes else "General"

    if classification_lower == "risk":
        return (
            f"Monitor whether this risk theme persists or spreads into valuation, "
            f"demand, or margin assumptions. Themes: {themes_text}."
        )

    if classification_lower == "catalyst":
        return (
            f"Monitor whether this catalyst continues to support revenue momentum, "
            f"brand strength, or market sentiment. Themes: {themes_text}."
        )

    return (
        f"Keep on watch as a neutral headline with possible context value. "
        f"Themes: {themes_text}."
    )


def fetch_news_for_query(
    query: str,
    max_items: int = 10,
) -> pd.DataFrame:
    """Fetch recent news items for a single query."""
    rss_url = _google_news_rss_url(query)
    xml_data = _fetch_rss_xml(rss_url)

    root = ET.fromstring(xml_data)

    fetched_utc = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []

    for item in root.findall(".//item")[:max_items]:
        title = _clean_text(item.findtext("title"))
        link = _clean_text(item.findtext("link"))
        source = _clean_text(item.findtext("source"))
        pub_date_raw = _clean_text(item.findtext("pubDate"))
        pub_date = _parse_pub_date(pub_date_raw)

        rows.append(
            {
                "Query": query,
                "Fetched UTC": fetched_utc,
                "Published UTC": pub_date.isoformat() if pub_date else "",
                "Source": source,
                "Title": title,
                "URL": link,
            }
        )

    return pd.DataFrame(rows)


def fetch_news_watchlist(
    queries: list[str],
    max_items_per_query: int = 10,
) -> pd.DataFrame:
    """Fetch news across all configured watchlist queries."""
    frames = []

    for query in queries:
        try:
            frames.append(
                fetch_news_for_query(
                    query=query,
                    max_items=max_items_per_query,
                )
            )
        except Exception as exc:
            frames.append(
                pd.DataFrame(
                    [
                        {
                            "Query": query,
                            "Fetched UTC": datetime.now(timezone.utc).isoformat(),
                            "Published UTC": "",
                            "Source": "RSS fetch error",
                            "Title": f"Unable to fetch news for query: {query}",
                            "URL": str(exc),
                        }
                    ]
                )
            )

    if not frames:
        return pd.DataFrame(
            columns=[
                "Query",
                "Fetched UTC",
                "Published UTC",
                "Source",
                "Title",
                "URL",
            ]
        )

    news = pd.concat(frames, ignore_index=True)
    news = news.drop_duplicates(subset=["Title"], keep="first")

    return news


def _keyword_score(text: str) -> int:
    """Score headline text using positive and negative keyword dictionaries."""
    lowered = text.lower()

    positive_hits = sum(1 for keyword in POSITIVE_KEYWORDS if keyword in lowered)
    negative_hits = sum(1 for keyword in NEGATIVE_KEYWORDS if keyword in lowered)

    return positive_hits - negative_hits


def _classify_news_item(title: str) -> str:
    """Classify a headline as Catalyst, Risk, or Neutral."""
    score = _keyword_score(title)

    if score >= 1:
        return "Catalyst"

    if score <= -1:
        return "Risk"

    return "Neutral"


def _extract_themes(title: str) -> str:
    """Extract catalyst and risk themes from a headline."""
    lowered = title.lower()
    themes: list[str] = []

    for theme, keywords in RISK_THEME_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            themes.append(theme)

    for theme, keywords in CATALYST_THEME_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            themes.append(theme)

    return "; ".join(themes) if themes else "General"


def score_news_items(news: pd.DataFrame) -> pd.DataFrame:
    """Add classification, keyword score, and theme columns to news items."""
    if news.empty:
        return news.copy()

    scored = news.copy()

    scored["Keyword Score"] = scored["Title"].map(_keyword_score)
    scored["Classification"] = scored["Title"].map(_classify_news_item)
    scored["Themes"] = scored["Title"].map(_extract_themes)

    return scored


def build_news_summary(scored_news: pd.DataFrame) -> pd.DataFrame:
    """Build a compact news intelligence summary."""
    if scored_news.empty:
        return pd.DataFrame(
            [
                {
                    "Metric": "News Items Tracked",
                    "Value": 0,
                    "Interpretation": "No news items were collected.",
                }
            ]
        )

    total_items = len(scored_news)
    catalyst_count = int(scored_news["Classification"].eq("Catalyst").sum())
    risk_count = int(scored_news["Classification"].eq("Risk").sum())
    neutral_count = int(scored_news["Classification"].eq("Neutral").sum())
    avg_score = float(scored_news["Keyword Score"].mean())

    if avg_score > 0.25:
        tone = "Positive / Catalyst-Leaning"
    elif avg_score < -0.25:
        tone = "Risk-Leaning"
    else:
        tone = "Balanced / Mixed"

    rows = [
        {
            "Metric": "News Items Tracked",
            "Value": total_items,
            "Interpretation": "Total unique headlines collected from the news watchlist.",
        },
        {
            "Metric": "Catalyst Items",
            "Value": catalyst_count,
            "Interpretation": "Headlines classified as positive or catalyst-oriented.",
        },
        {
            "Metric": "Risk Items",
            "Value": risk_count,
            "Interpretation": "Headlines classified as risk-oriented.",
        },
        {
            "Metric": "Neutral Items",
            "Value": neutral_count,
            "Interpretation": "Headlines without a clear rules-based direction.",
        },
        {
            "Metric": "Average News Score",
            "Value": avg_score,
            "Interpretation": tone,
        },
    ]

    if "Fetched UTC" in scored_news.columns:
        latest_fetch = scored_news["Fetched UTC"].dropna().astype(str)

        if not latest_fetch.empty:
            rows.append(
                {
                    "Metric": "Latest Fetch UTC",
                    "Value": latest_fetch.max(),
                    "Interpretation": "Most recent timestamp when the news watchlist was fetched.",
                }
            )

    return pd.DataFrame(rows)


def build_theme_summary(scored_news: pd.DataFrame) -> pd.DataFrame:
    """Summarize the most common themes in recent news."""
    if scored_news.empty or "Themes" not in scored_news.columns:
        return pd.DataFrame(columns=["Theme", "Count"])

    theme_counts: dict[str, int] = {}

    for theme_cell in scored_news["Themes"].dropna():
        for theme in str(theme_cell).split(";"):
            clean_theme = theme.strip()

            if not clean_theme:
                continue

            theme_counts[clean_theme] = theme_counts.get(clean_theme, 0) + 1

    if not theme_counts:
        return pd.DataFrame(columns=["Theme", "Count"])

    return (
        pd.DataFrame(
            [
                {
                    "Theme": theme,
                    "Count": count,
                }
                for theme, count in theme_counts.items()
            ]
        )
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )


def build_news_dashboard_cards(
    news_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build card-style news intelligence summary rows.

    Parameters
    ----------
    news_summary:
        Output from build_news_summary().

    Returns
    -------
    pd.DataFrame
        Dashboard-ready news summary card table.
    """
    tracked = _extract_summary_value(news_summary, "News Items Tracked", 0)
    catalysts = _extract_summary_value(news_summary, "Catalyst Items", 0)
    risks = _extract_summary_value(news_summary, "Risk Items", 0)
    neutral = _extract_summary_value(news_summary, "Neutral Items", 0)
    avg_score = _extract_summary_value(news_summary, "Average News Score", 0.0)
    tone = _extract_summary_interpretation(
        news_summary,
        "Average News Score",
        "Not Available",
    )
    latest_fetch = _extract_summary_value(news_summary, "Latest Fetch UTC", "")

    rows = [
        {
            "Card": "News Items Tracked",
            "Value": tracked,
            "Display Value": str(tracked),
            "Interpretation": "Total unique headlines collected from the configured watchlist.",
        },
        {
            "Card": "Catalyst Items",
            "Value": catalysts,
            "Display Value": str(catalysts),
            "Interpretation": "Headlines classified as positive or catalyst-oriented.",
        },
        {
            "Card": "Risk Items",
            "Value": risks,
            "Display Value": str(risks),
            "Interpretation": "Headlines classified as risk-oriented.",
        },
        {
            "Card": "Neutral Items",
            "Value": neutral,
            "Display Value": str(neutral),
            "Interpretation": "Headlines without a clear rules-based direction.",
        },
        {
            "Card": "Average News Score",
            "Value": avg_score,
            "Display Value": f"{_safe_numeric(avg_score):.2f}",
            "Interpretation": "Average positive-minus-negative keyword score across headlines.",
        },
        {
            "Card": "Overall News Tone",
            "Value": tone,
            "Display Value": tone,
            "Interpretation": "Rules-based tone derived from the average headline keyword score.",
        },
        {
            "Card": "Latest Fetch UTC",
            "Value": latest_fetch,
            "Display Value": str(latest_fetch),
            "Interpretation": "Most recent timestamp when the RSS watchlist was fetched.",
        },
    ]

    return pd.DataFrame(rows)


def build_news_risk_catalyst_watchlist(
    scored_news: pd.DataFrame,
    max_items: int = 25,
) -> pd.DataFrame:
    """Build a prioritized risk/catalyst watchlist from scored headlines.

    Parameters
    ----------
    scored_news:
        Output from score_news_items().
    max_items:
        Maximum number of rows to include in the dashboard watchlist.

    Returns
    -------
    pd.DataFrame
        Prioritized dashboard watchlist of important news items.
    """
    if scored_news.empty:
        return pd.DataFrame(
            columns=[
                "Priority",
                "Classification",
                "Keyword Score",
                "Themes",
                "Published UTC",
                "Source",
                "Title",
                "URL",
                "Monitoring Note",
            ]
        )

    required_columns = {
        "Classification",
        "Keyword Score",
        "Themes",
        "Published UTC",
        "Source",
        "Title",
        "URL",
    }

    missing_columns = required_columns.difference(scored_news.columns)

    if missing_columns:
        raise KeyError(
            "Scored news table is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    watchlist = scored_news.copy()

    watchlist["Priority"] = watchlist.apply(
        lambda row: _classify_watchlist_priority(
            classification=str(row["Classification"]),
            keyword_score=_safe_numeric(row["Keyword Score"]),
            themes=str(row["Themes"]),
        ),
        axis=1,
    )

    watchlist["Monitoring Note"] = watchlist.apply(
        lambda row: _news_action_note(
            classification=str(row["Classification"]),
            themes=str(row["Themes"]),
        ),
        axis=1,
    )

    priority_order = {
        "High": 0,
        "Medium": 1,
        "Monitor": 2,
    }

    classification_order = {
        "Risk": 0,
        "Catalyst": 1,
        "Neutral": 2,
    }

    watchlist["Priority Sort"] = watchlist["Priority"].map(priority_order).fillna(99)
    watchlist["Classification Sort"] = (
        watchlist["Classification"].map(classification_order).fillna(99)
    )
    watchlist["Abs Keyword Score"] = watchlist["Keyword Score"].abs()

    if "Published UTC" in watchlist.columns:
        watchlist["Published Sort"] = pd.to_datetime(
            watchlist["Published UTC"],
            errors="coerce",
            utc=True,
        )
    else:
        watchlist["Published Sort"] = pd.NaT

    output = (
        watchlist[
            [
                "Priority",
                "Classification",
                "Keyword Score",
                "Themes",
                "Published UTC",
                "Source",
                "Title",
                "URL",
                "Monitoring Note",
                "Priority Sort",
                "Classification Sort",
                "Abs Keyword Score",
                "Published Sort",
            ]
        ]
        .sort_values(
            [
                "Priority Sort",
                "Classification Sort",
                "Abs Keyword Score",
                "Published Sort",
            ],
            ascending=[True, True, False, False],
        )
        .head(max_items)
        .drop(
            columns=[
                "Priority Sort",
                "Classification Sort",
                "Abs Keyword Score",
                "Published Sort",
            ]
        )
        .reset_index(drop=True)
    )

    return output


def build_news_source_summary(
    scored_news: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize news classification by source.

    Parameters
    ----------
    scored_news:
        Output from score_news_items().

    Returns
    -------
    pd.DataFrame
        Source-level summary of headline count, tone, and classification mix.
    """
    if scored_news.empty:
        return pd.DataFrame(
            columns=[
                "Source",
                "Headline Count",
                "Average Keyword Score",
                "Catalyst Items",
                "Risk Items",
                "Neutral Items",
                "Dominant Tone",
            ]
        )

    required_columns = {
        "Source",
        "Keyword Score",
        "Classification",
    }

    missing_columns = required_columns.difference(scored_news.columns)

    if missing_columns:
        raise KeyError(
            "Scored news table is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    rows: list[dict[str, Any]] = []

    for source, group in scored_news.groupby("Source"):
        avg_score = float(group["Keyword Score"].mean())
        catalyst_count = int(group["Classification"].eq("Catalyst").sum())
        risk_count = int(group["Classification"].eq("Risk").sum())
        neutral_count = int(group["Classification"].eq("Neutral").sum())

        if catalyst_count > risk_count and catalyst_count > neutral_count:
            dominant_tone = "Catalyst-Leaning"
        elif risk_count > catalyst_count and risk_count > neutral_count:
            dominant_tone = "Risk-Leaning"
        elif neutral_count > catalyst_count and neutral_count > risk_count:
            dominant_tone = "Neutral"
        elif avg_score > 0.25:
            dominant_tone = "Catalyst-Leaning"
        elif avg_score < -0.25:
            dominant_tone = "Risk-Leaning"
        else:
            dominant_tone = "Balanced / Mixed"

        rows.append(
            {
                "Source": source if source else "Unknown",
                "Headline Count": int(len(group)),
                "Average Keyword Score": avg_score,
                "Catalyst Items": catalyst_count,
                "Risk Items": risk_count,
                "Neutral Items": neutral_count,
                "Dominant Tone": dominant_tone,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["Headline Count", "Risk Items", "Catalyst Items"],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def save_news_intelligence_outputs(
    table_dir: Path,
    queries: list[str],
    max_items_per_query: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch, score, summarize, and save news intelligence outputs."""
    table_dir.mkdir(parents=True, exist_ok=True)

    raw_news = fetch_news_watchlist(
        queries=queries,
        max_items_per_query=max_items_per_query,
    )

    scored_news = score_news_items(raw_news)
    news_summary = build_news_summary(scored_news)
    theme_summary = build_theme_summary(scored_news)

    news_dashboard_cards = build_news_dashboard_cards(news_summary)
    news_watchlist = build_news_risk_catalyst_watchlist(scored_news)
    news_source_summary = build_news_source_summary(scored_news)

    scored_news.to_csv(
        table_dir / "hermes_news_intelligence_items.csv",
        index=False,
    )

    news_summary.to_csv(
        table_dir / "hermes_news_intelligence_summary.csv",
        index=False,
    )

    theme_summary.to_csv(
        table_dir / "hermes_news_theme_summary.csv",
        index=False,
    )

    news_dashboard_cards.to_csv(
        table_dir / "hermes_news_dashboard_cards.csv",
        index=False,
    )

    news_watchlist.to_csv(
        table_dir / "hermes_news_risk_catalyst_watchlist.csv",
        index=False,
    )

    news_source_summary.to_csv(
        table_dir / "hermes_news_source_summary.csv",
        index=False,
    )

    return scored_news, news_summary, theme_summary