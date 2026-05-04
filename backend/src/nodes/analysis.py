"""
analysis.py — Node 5: LLM-driven insight extraction.

Sends aggregated product data + Amazon AI review insights to Gemini Pro
and returns a structured, fully-formatted Markdown market research report.

Review data comes from reviews_information.summary.insights (amazon_product engine):
  - Per-topic sentiment objects with mention counts and curated examples
  - Amazon-AI synthesised summary paragraph
  - Star-rating histogram
"""
from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from backend.src.config import get_llm, GEMINI_PRO
from backend.src.models import AgentState, AnalysisOutput

logger = logging.getLogger(__name__)

# Module-level LLM instance — avoids re-instantiation on every request
_llm = get_llm(temperature=0.2, model_name=GEMINI_PRO)
_structured_llm = _llm.with_structured_output(AnalysisOutput)

_SYSTEM = """You are a Senior Market Research Analyst at a top-tier e-commerce consulting firm.

You will receive:
1. A list of top Amazon products with pricing, ratings, badges, and feature bullets.
2. Amazon AI-generated review insights per product: structured per-topic sentiment data
   (mention counts, positive/negative splits, curated example snippets, and an overall
   summary paragraph synthesised across ALL reviews).

Your job is to produce:
- top_purchase_drivers: specific reasons customers buy these products (cite topics and mention counts)
- common_complaints: specific recurring complaints (cite the insight topic + negative mention count)
- sentiment_themes: general cross-product themes (e.g. "value for money dominates purchasing decisions")
- competitive_gaps: actionable market gaps NOT addressed by any current product
- answer: A comprehensive, beautifully formatted Markdown market research report tailored to the user's specific question

The `answer` MUST follow this exact structure:

## Direct Answer
**Start here.** In 2–4 sentences, directly and specifically answer the user's question using the data.
Do NOT write a generic intro — respond to exactly what was asked.
Examples:
  - If asked "what are the best budget options?", lead with which products offer the best value and why.
  - If asked "what do customers complain about?", lead with the top complaints and their mention counts.
  - If asked "compare X and Y", immediately state which is better and in what dimensions.

## Executive Summary
High-level synthesis: market size signals, pricing landscape, category leader, key trends.

## Comprehensive Product Comparison
A full Markdown table comparing ALL products.
Columns MUST include: Rank | Product (hyperlinked name + ASIN in parentheses) | Price | Rating | Reviews | Bought/Month | Key Strength | Biggest Weakness | Best For

## Customer Sentiment Analysis
For each sentiment insight topic (e.g. Taste, Value, Freshness):
- Topic name, overall sentiment, total mentions, positive vs negative split
- One-line synthesis from the insight summary
- 1–2 verbatim example snippets (quote them)
Group into: Positively Received Topics | Mixed / Negative Topics

## Competitive Landscape & Strategic Gaps
- How products are positioned (budget vs premium, feature trade-offs)
- Specific, actionable market gaps with opportunity sizing language

## Revenue & Opportunity Sizing
For each product with revenue data: monthly sales estimate, monthly revenue estimate, data basis.
Include a summary table: Product | Est. Monthly Sales | Est. Monthly Revenue | Basis

## Key Takeaways for Amazon Sellers
Bullet strategic recommendations for a seller entering or competing in this category.

---
RULES (follow strictly):
- ANSWER THE QUESTION FIRST: The ## Direct Answer section must directly address what the user asked.
  Adapt the depth and focus of the entire report to serve the user's actual intent.
- DATA GROUNDING: Every claim MUST come from the provided data. Do NOT invent facts, prices, or reviews.
- Always hyperlink product names using their `link` field and include ASIN in parentheses.
- When citing sentiment, always mention the mention count (e.g. "1,519 of 2,100 mentions positive").
- If data is sparse (< 3 products or no review insights), clearly state what data is available and
  provide the best analysis possible from what exists — do NOT hallucinate missing information.
- Use precise numbers, not vague language. Write like a McKinsey analyst, not a copywriter.
- Ensure flawless Markdown formatting throughout.
"""


def _format_insights(review_insights: list[dict]) -> str:
    """Format review_insights list into a compact, LLM-readable string."""
    if not review_insights:
        return "  No structured review insights available.\n"

    lines = []
    for ins in review_insights:
        title = ins.get("title", "Unknown")
        sentiment = ins.get("sentiment", "unknown")
        mentions = ins.get("mentions") or {}
        total = mentions.get("total", 0)
        pos = mentions.get("positive", 0)
        neg = mentions.get("negative", 0)
        summary = ins.get("summary", "")
        examples = ins.get("examples") or []

        lines.append(
            f"  [{title}] sentiment={sentiment} | mentions: {total} total, {pos} positive, {neg} negative\n"
            f"  Summary: {summary}"
        )
        # Include up to 2 example snippets
        for ex in examples[:2]:
            snippet = ex.get("snippet", "").strip()
            if snippet:
                lines.append(f'    • "{snippet}"')
    return "\n".join(lines)


def _build_context(state: AgentState) -> str:
    products = state.get("aggregated_products") or []
    query = state.get("message", "")
    intent = state.get("intent", "")

    lines: list[str] = [
        f"## User Query\n{query}\n",
        f"## Intent Classified As: {intent}\n",
    ]

    if not products:
        lines.append("## Data\nNo products were found for this query.\n")
        return "\n".join(lines)

    lines.append(f"## Top {len(products)} Products\n")

    for p in products:
        asin = p.get("asin", "")
        title = p.get("title", "Unknown")
        link = p.get("link", "N/A")
        price = p.get("price", "N/A")
        rating = p.get("rating", "N/A")
        reviews = p.get("reviews", "N/A")
        blm = p.get("bought_last_month", "N/A")
        badges = ", ".join(p.get("badges") or []) or "none"
        about = "; ".join((p.get("about_item") or [])[:5])
        rev_est = p.get("revenue_estimate")
        review_summary = p.get("review_summary_text", "")
        review_insights = p.get("review_insights") or []

        # Revenue estimate block
        rev_block = ""
        if rev_est:
            rev_block = (
                f"- Revenue Estimate: ~{rev_est.get('monthly_sales_est', 0):,} units/month × "
                f"${rev_est.get('price', 0):.2f} = "
                f"~${rev_est.get('monthly_revenue_est', 0):,}/month (basis: {rev_est.get('basis', '')})\n"
            )

        product_block = (
            f"### [{p.get('rank', '?')}] [{title}]({link}) (ASIN: {asin})\n"
            f"- Price: {price} | Rating: {rating} | Reviews: {reviews} | Bought last month: {blm}\n"
            f"- Badges: {badges}\n"
            f"- Feature Bullets: {about}\n"
            f"{rev_block}"
            f"- Amazon AI Review Summary: {review_summary}\n"
            f"- Structured Review Insights:\n"
            f"{_format_insights(review_insights)}\n"
        )
        lines.append(product_block)

    # Market stats
    market = (state.get("analysis") or {}).get("market_stats") or {}
    pr = market.get("price_range") or {}
    lines.append(
        f"## Market Statistics\n"
        f"- Price Range: ${pr.get('min')} – ${pr.get('max')} (avg ${pr.get('avg')})\n"
        f"- Average Rating: {market.get('avg_rating')}\n"
        f"- Total Reviews Analyzed: {market.get('total_reviews_analyzed')}\n"
        f"- Total Products Analyzed: {market.get('total_products_analyzed')}\n"
    )

    return "\n".join(lines)


def analysis_node(state: AgentState) -> dict:
    context = _build_context(state)
    logger.info("Sending analysis context to Gemini Pro (%d chars)", len(context))

    try:
        result: AnalysisOutput = _structured_llm.invoke(
            [SystemMessage(content=_SYSTEM), HumanMessage(content=context)]
        )
    except Exception:
        logger.exception("LLM analysis failed")
        # Return a graceful degradation response
        existing = state.get("analysis") or {}
        return {
            "analysis": {
                **existing,
                "top_purchase_drivers": [],
                "common_complaints": [],
                "sentiment_themes": [],
                "competitive_gaps": [],
                "answer": " Analysis could not be completed due to an internal error. Please try again.",
            }
        }

    existing = state.get("analysis") or {}
    return {
        "analysis": {
            **existing,
            "top_purchase_drivers": result.top_purchase_drivers,
            "common_complaints":    result.common_complaints,
            "sentiment_themes":     result.sentiment_themes,
            "competitive_gaps":     result.competitive_gaps,
            "answer":               result.answer,
        }
    }
