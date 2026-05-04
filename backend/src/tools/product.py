"""
product.py — SerpApi Amazon product detail tool.

Uses the `amazon_product` engine which returns:
  - Full product metadata (price, rating, badges, about_item, etc.)
  - `reviews_information.summary` with Amazon-AI-generated review insights:
      * summary text  (paragraph synthesising all reviews)
      * insights[]    (per-topic: sentiment, mention counts, example snippets)
      * customer_reviews histogram (star ratings)

No separate `amazon_reviews` engine call is needed — the insights above are
already aggregated across ALL reviews by Amazon and are far richer than the
20-review slice the reviews engine returns.
"""
from __future__ import annotations

import logging

from backend.src.config import get_serpapi_client

logger = logging.getLogger(__name__)


def get_product_details(asin: str) -> dict:
    """
    Fetch full product details for `asin` via the amazon_product engine.

    Returns a flat dict containing pricing, ratings, feature bullets,
    item specifications, and structured review insights (Amazon AI-synthesised).
    Raises on any API-level failure so the caller can decide how to handle it.
    """
    client = get_serpapi_client()
    raw = client.search({"engine": "amazon_product", "asin": asin, "amazon_domain": "amazon.com"})

    pr: dict = raw.get("product_results") or {}
    ri: dict = raw.get("reviews_information") or {}
    ri_summary: dict = ri.get("summary") or {}

    # ── Review insights (Amazon AI-generated, per topic) ───────────────────────
    # Each insight: { title, sentiment, mentions: {total, positive, negative},
    #                 summary, examples: [{position, snippet, link}] }
    review_insights: list[dict] = ri_summary.get("insights") or []

    # Star histogram — keys are "5_star", "4_star", … (integers as percent)
    cr_block: dict = ri_summary.get("customer_reviews") or {}
    reviews_histogram: dict = {
        "5_star": cr_block.get("5_star"),
        "4_star": cr_block.get("4_star"),
        "3_star": cr_block.get("3_star"),
        "2_star": cr_block.get("2_star"),
        "1_star": cr_block.get("1_star"),
    }

    return {
        "asin": asin,
        "title": pr.get("title", ""),
        "description": pr.get("description", ""),
        "brand": pr.get("brand", ""),
        "link": pr.get("link_clean") or pr.get("link", ""),
        "price": pr.get("price", ""),
        "extracted_price": pr.get("extracted_price"),
        "old_price": pr.get("old_price"),
        "extracted_old_price": pr.get("extracted_old_price"),
        "discount": pr.get("discount"),
        "rating": pr.get("rating"),
        "reviews": pr.get("reviews"),
        "bought_last_month": pr.get("bought_last_month"),
        "badges": pr.get("badges") or [],
        "tags": pr.get("tags") or [],
        "thumbnails": pr.get("thumbnails") or [],
        "stock": pr.get("stock", ""),
        "delivery": pr.get("delivery") or [],
        # Feature bullets (top-level key in the response, not under product_results)
        "about_item": raw.get("about_item") or [],
        "item_specifications": raw.get("item_specifications") or {},
        "product_details": raw.get("product_details") or {},
        # Amazon AI review intelligence
        "review_summary_text": ri_summary.get("text", ""),
        "review_insights": review_insights,
        "reviews_histogram": reviews_histogram,
        # Comparison data
        "compare_with_similar": raw.get("compare_with_similar") or [],
    }
