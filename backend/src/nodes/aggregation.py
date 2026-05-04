"""
aggregation.py — Node 4: normalise raw tool outputs into scored product records.

Merges search-level data with the richer product-detail data (which now includes
Amazon AI review insights), recomputes scores using enriched data, and computes
market-level stats.
"""
from __future__ import annotations

import logging
import math
import statistics
from typing import Optional

from backend.src.models import AgentState
from backend.src.tools.search import parse_bought_last_month, score_product

logger = logging.getLogger(__name__)


# ── Revenue heuristic ──────────────────────────────────────────────────────────

def _estimate_revenue(detail: dict) -> Optional[dict]:
    """
    Estimate monthly revenue from product detail data.
    Returns None if there is no price or review data to base it on.
    """
    if not detail:
        return None

    price: float = detail.get("extracted_price") or 0.0
    reviews: int = detail.get("reviews") or 0
    blm_raw: Optional[str] = detail.get("bought_last_month")
    blm = parse_bought_last_month(blm_raw)

    # If we have no signal at all, skip the estimate
    if price == 0.0 and reviews == 0 and blm == 0:
        return None

    if blm > 0:
        monthly_sales = blm
        basis = "bought_last_month"
    else:
        monthly_sales = int(reviews * 0.5)   # midpoint of empirical 0.3–0.7 range
        basis = "review_heuristic"

    return {
        "asin": detail.get("asin", ""),
        "title": detail.get("title", ""),
        "price": price,
        "monthly_sales_est": monthly_sales,
        "monthly_revenue_est": int(monthly_sales * price),
        "basis": basis,
    }


# ── Main node ──────────────────────────────────────────────────────────────────

def aggregation_node(state: AgentState) -> dict:
    search_products_list = (state.get("raw_search_results") or {}).get("products") or []
    details_list: list[dict] = state.get("raw_product_details") or []

    if not search_products_list and not details_list:
        logger.warning("aggregation_node: no products to aggregate")
        return {
            "aggregated_products": [],
            "analysis": {"market_stats": {"price_range": {}, "avg_rating": None,
                                          "total_reviews_analyzed": 0, "total_products_analyzed": 0}},
        }

    # Build lookup: asin → detail
    detail_by_asin: dict[str, dict] = {d["asin"]: d for d in details_list if d.get("asin")}

    aggregated: list[dict] = []
    seen_asins: set[str] = set()

    # Use search products as the base (preserves original ranking order)
    base_list = search_products_list or [
        {"asin": d["asin"], "rank": i + 1} for i, d in enumerate(details_list)
    ]

    for sp in base_list:
        asin = sp.get("asin", "")
        if not asin or asin in seen_asins:
            continue
        seen_asins.add(asin)

        detail = detail_by_asin.get(asin, {})

        # Merge: detail wins on overlapping fields (richer data)
        merged = {
            **sp,
            "title":           detail.get("title") or sp.get("title", ""),
            "brand":           detail.get("brand") or sp.get("brand", ""),
            "link":            detail.get("link") or sp.get("link", ""),
            "extracted_price": detail.get("extracted_price") or sp.get("extracted_price"),
            "price":           detail.get("price") or sp.get("price", ""),
            "rating":          detail.get("rating") or sp.get("rating"),
            "reviews":         detail.get("reviews") or sp.get("reviews"),
            "bought_last_month": detail.get("bought_last_month") or sp.get("bought_last_month"),
            "badges":          list(set((detail.get("badges") or []) + (sp.get("badges") or []))),
            "about_item":      detail.get("about_item") or [],
            "item_specifications": detail.get("item_specifications") or {},
            # Amazon AI-generated review intelligence
            "review_summary_text": detail.get("review_summary_text", ""),
            "review_insights":     detail.get("review_insights") or [],
            "reviews_histogram":   detail.get("reviews_histogram") or {},
            # Comparison & revenue
            "compare_with_similar": detail.get("compare_with_similar") or [],
            "revenue_estimate":     _estimate_revenue(detail) if detail else None,
        }

        # Recompute score using the enriched detail data (better rating/review counts)
        merged["score"] = score_product(merged)

        aggregated.append(merged)

    # Add any detail-only products not in search (compare / deep_dive flows)
    for asin, detail in detail_by_asin.items():
        if asin not in seen_asins:
            record = {
                **detail,
                "rank": len(aggregated) + 1,
                "revenue_estimate": _estimate_revenue(detail),
            }
            record["score"] = score_product(record)
            aggregated.append(record)
            seen_asins.add(asin)

    # ── Market-level stats ─────────────────────────────────────────────────────
    prices = [r["extracted_price"] for r in aggregated if r.get("extracted_price")]
    ratings = [r["rating"] for r in aggregated if r.get("rating")]
    reviews_counts = [r["reviews"] for r in aggregated if r.get("reviews")]

    market_stats = {
        "price_range": {
            "min": round(min(prices), 2) if prices else None,
            "max": round(max(prices), 2) if prices else None,
            "avg": round(statistics.mean(prices), 2) if prices else None,
        },
        "avg_rating":            round(statistics.mean(ratings), 2) if ratings else None,
        "total_reviews_analyzed": sum(reviews_counts),
        "total_products_analyzed": len(aggregated),
    }

    logger.info(
        "Aggregated %d products; price range $%s–$%s, avg rating %.2f",
        len(aggregated),
        market_stats["price_range"].get("min"),
        market_stats["price_range"].get("max"),
        market_stats["avg_rating"] or 0,
    )

    return {
        "aggregated_products": aggregated,
        "analysis": {"market_stats": market_stats},
    }
