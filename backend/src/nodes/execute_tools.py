"""
execute_tools.py — Node 3: execute planned tool calls.

Two-phase pattern:
  1. search_products  →  get ranked ASINs
  2. get_product_details per ASIN (parallel)  →  full data including review insights

Review insights are now embedded inside each product detail response
(via reviews_information.summary from the amazon_product engine).
No separate reviews API call is needed.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend.src.models import AgentState
from backend.src.tools.search import search_products
from backend.src.tools.product import get_product_details

logger = logging.getLogger(__name__)

# Max parallel threads for product detail fetches
_MAX_WORKERS = 6


def _fetch_detail_safe(asin: str) -> dict | None:
    """Fetch product detail for a single ASIN; returns None on failure."""
    try:
        return get_product_details(asin)
    except Exception:
        logger.exception("Failed to fetch details for ASIN %s", asin)
        return None


def _fetch_details_parallel(asins: list[str]) -> list[dict]:
    """Fetch product details for multiple ASINs in parallel."""
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(asins))) as executor:
        future_to_asin = {executor.submit(_fetch_detail_safe, asin): asin for asin in asins}
        for future in as_completed(future_to_asin):
            result = future.result()
            if result is not None:
                results.append(result)
    return results


def execute_tools_node(state: AgentState) -> dict:
    plan: list[dict] = state.get("tool_plan") or []

    raw_search: dict = {}
    raw_details: list[dict] = []

    for step in plan:
        t = step["type"]

        try:
            if t == "search":
                raw_search = search_products(step["query"], step.get("n", 6))
                logger.info(
                    "Search returned %d products for %r",
                    len(raw_search.get("products") or []),
                    step["query"],
                )

            elif t == "fetch_details":
                # Run after search — fetch details for ASINs from raw_search
                products = raw_search.get("products") or []
                limit = step.get("limit")  # used by deep_dive to cap at 1
                if limit:
                    products = products[:limit]

                asins = [p["asin"] for p in products if p.get("asin")]
                if not asins:
                    logger.warning("fetch_details step found no ASINs from search results")
                    continue

                logger.info("Fetching details for %d ASINs (parallel)", len(asins))
                raw_details = _fetch_details_parallel(asins)

            elif t == "get_details":
                asin = step["asin"]
                detail = _fetch_detail_safe(asin)
                if detail:
                    raw_details.append(detail)

        except Exception:
            logger.exception("Unhandled error in execute_tools_node for step %s", step)
            # Continue — partial data is better than no data

    logger.info(
        "execute_tools complete: %d search products, %d detailed records",
        len((raw_search.get("products") or [])),
        len(raw_details),
    )

    return {
        "raw_search_results": raw_search,
        "raw_product_details": raw_details,
    }
