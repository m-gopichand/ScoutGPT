"""
search.py — SerpApi Amazon search tool.

Prioritises organic (non-sponsored) results, scores by rating × log(reviews+1)
plus a bought_last_month bonus, and returns the top-N normalised products.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Optional

from backend.src.config import get_serpapi_client

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_bought_last_month(blm: Optional[str]) -> int:
    """'20K+ bought in past month' → 20000."""
    if not blm:
        return 0
    text = blm.lower()
    m = re.search(r"([\d,]+\.?\d*)\s*k", text)
    if m:
        return int(float(m.group(1).replace(",", "")) * 1000)
    m = re.search(r"([\d,]+)", text)
    if m:
        return int(m.group(1).replace(",", ""))
    return 0


def score_product(p: dict) -> float:
    """rating × log(reviews+1) + log(blm+1)×0.5"""
    rating = p.get("rating") or 0.0
    reviews = p.get("reviews") or 0
    blm = parse_bought_last_month(p.get("bought_last_month"))
    score = rating * math.log(reviews + 1)
    if blm:
        score += math.log(blm + 1) * 0.5
    return round(score, 4)


def _normalize(p: dict, rank: int) -> dict:
    return {
        "rank": rank,
        "asin": p.get("asin", ""),
        "title": p.get("title", ""),
        "link": p.get("link_clean") or p.get("link", ""),
        "thumbnail": p.get("thumbnail", ""),
        "brand": p.get("brand", ""),
        "price": p.get("price", ""),
        "extracted_price": p.get("extracted_price"),
        "rating": p.get("rating"),
        "reviews": p.get("reviews"),
        "bought_last_month": p.get("bought_last_month"),
        "badges": p.get("badges") or [],
        "prime": p.get("prime", False),
        "sponsored": p.get("sponsored", False),
        "score": score_product(p),
    }


# ── Tool ───────────────────────────────────────────────────────────────────────

def search_products(query: str, n: int = 6) -> dict:
    """
    Search Amazon for `query` and return the top `n` products scored by
    relevance (rating × log(reviews+1) + bought_last_month bonus).
    Organic non-sponsored results are preferred over ads.

    Returns a dict with keys: query, total_results, products, related_searches.
    Raises on API failure — callers should catch and handle gracefully.
    """
    if not query or not query.strip():
        logger.warning("search_products called with empty query")
        return {"query": query, "total_results": 0, "products": [], "related_searches": []}

    client = get_serpapi_client()
    logger.info("Searching Amazon: %r (top %d)", query, n)
    raw = client.search({"engine": "amazon", "k": query, "amazon_domain": "amazon.com"})

    organic: list[dict] = raw.get("organic_results") or []

    if not organic:
        logger.warning("No organic results for query: %r", query)
        return {
            "query": query,
            "total_results": 0,
            "products": [],
            "related_searches": [r.get("query") for r in raw.get("related_searches", [])[:5]],
        }

    # Prefer non-sponsored; fall back to all organic if not enough
    non_sponsored = [p for p in organic if not p.get("sponsored")]
    candidates = non_sponsored if len(non_sponsored) >= n else organic

    top = sorted(candidates, key=score_product, reverse=True)[:n]
    products = [_normalize(p, i + 1) for i, p in enumerate(top)]

    return {
        "query": query,
        "total_results": raw.get("search_information", {}).get("total_results"),
        "products": products,
        "related_searches": [r.get("query") for r in raw.get("related_searches", [])[:5]],
    }
