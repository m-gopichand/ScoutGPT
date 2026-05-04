"""
react_agent.py — Node 2: Autonomous tool-calling agent (replaces planning + execute_tools).

Uses LangGraph's create_react_agent so the LLM decides:
  - Which tools to call (search_products, get_product_details)
  - In what order
  - With what arguments

This naturally handles cases like "deep dive + find competitors":
  1. LLM calls get_product_details(ASIN) → learns the product title
  2. LLM calls search_products(title) → discovers competitors
  3. LLM calls get_product_details for each competitor → full data

The node extracts all tool results from the agent's message history and
stores them into state so aggregation_node can process them as before.
"""
from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from backend.src.config import get_llm, GEMINI_FLASH
from backend.src.models import AgentState
from backend.src.tools.search import search_products as _search_products
from backend.src.tools.product import get_product_details as _get_product_details

logger = logging.getLogger(__name__)

# ── Tool definitions (wrapped so the ReAct agent can call them) ────────────────

@tool
def search_products(query: str, n: int = 6) -> str:
    """
    Search Amazon for products matching the given query.
    Returns the top N products with ranking, pricing, ratings, and sales data.
    Use this to find competitors, discover a category, or search by product type.

    Args:
        query: Amazon search keywords (e.g. "thick yoga mat NBR")
        n: Number of top products to return (default 6, max 10)
    """
    n = min(n, 10)
    result = _search_products(query, n)
    return json.dumps(result)


@tool
def get_product_details(asin: str) -> str:
    """
    Fetch full details for a specific Amazon product by ASIN.
    Returns pricing, ratings, feature bullets, and Amazon AI-synthesised review insights
    (per-topic sentiment, mention counts, example snippets).
    Use this for any specific ASIN the user mentions, or for top competitors found via search.

    Args:
        asin: The Amazon Standard Identification Number (e.g. "B01LP0U5X0")
    """
    result = _get_product_details(asin)
    return json.dumps(result)


# ── Agent setup ────────────────────────────────────────────────────────────────

_AGENT_SYSTEM = """You are an Amazon product intelligence agent. Your job is to gather data
from Amazon to answer the user's research query using the available tools.

You have two tools:
  1. search_products(query, n) — search Amazon for products by keyword
  2. get_product_details(asin) — get full details + review insights for a specific ASIN

Strategy based on the user's intent:
  - "search": call search_products, then get_product_details for each ASIN returned.
  - "analyze": call search_products with category keywords, then get_product_details for all results.
  - "compare": call get_product_details for each specific ASIN. If no ASINs given, search first.
  - "deep_dive" on a specific ASIN:
      1. Call get_product_details(asin) for the user's product.
      2. If the user asks for competitors or comparison: use the product title to
         call search_products, then call get_product_details for the top competitor ASINs
         (excluding the user's own ASIN).
      3. If no competitors requested, just fetch the single product.

IMPORTANT:
  - Always call get_product_details after search_products to get review insights.
  - Do NOT call get_product_details for more than 8 ASINs total to stay efficient.
  - When you have gathered sufficient data, stop calling tools and output a final summary.
  - Your final response should be a plain JSON object summarising what data you collected.
    Example: {"collected": ["B01LP0U5X0", "B08XYZ"], "search_queries": ["yoga mat"]}
"""

_llm = get_llm(temperature=0.0, model_name=GEMINI_FLASH)
_tools = [search_products, get_product_details]
_agent = create_react_agent(_llm, _tools)


# ── Extraction helpers ─────────────────────────────────────────────────────────

def _extract_tool_results(messages: list) -> tuple[dict, list[dict]]:
    """
    Walk the agent's message history and extract all tool call results.
    Returns (raw_search_results, raw_product_details).
    """
    raw_search: dict = {}
    raw_details: list[dict] = []
    seen_asins: set[str] = set()

    for msg in messages:
        # ToolMessage carries the result of a tool call as a string
        if msg.__class__.__name__ != "ToolMessage":
            continue

        try:
            data = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            continue

        # Distinguish search results from product details by key presence
        if "products" in data:
            # Merge search results — keep the one with the most products
            if len(data.get("products") or []) > len(raw_search.get("products") or []):
                raw_search = data

        elif "asin" in data and data["asin"]:
            asin = data["asin"]
            if asin not in seen_asins:
                raw_details.append(data)
                seen_asins.add(asin)

    return raw_search, raw_details


# ── Node ───────────────────────────────────────────────────────────────────────

def react_agent_node(state: AgentState) -> dict:
    """
    Runs the ReAct agent and extracts all tool results into state fields
    compatible with aggregation_node.
    """
    intent = state.get("intent", "search")
    message = state.get("message", "")
    keywords = state.get("keywords") or []
    asins = state.get("asins") or []
    top_n = state.get("top_n") or 6

    # Build a rich prompt so the agent understands exactly what to do
    context_parts = [
        f"User query: {message}",
        f"Classified intent: {intent}",
    ]
    if keywords:
        context_parts.append(f"Extracted keywords: {', '.join(keywords)}")
    if asins:
        context_parts.append(f"Explicit ASINs mentioned: {', '.join(asins)}")
    context_parts.append(f"Top N requested: {top_n}")

    prompt = "\n".join(context_parts)
    logger.info("Invoking ReAct agent | intent=%s | asins=%s | keywords=%s", intent, asins, keywords)

    agent_input = {
        "messages": [
            SystemMessage(content=_AGENT_SYSTEM),
            HumanMessage(content=prompt),
        ]
    }

    result = _agent.invoke(agent_input, config={"recursion_limit": 25})
    messages = result.get("messages", [])

    raw_search, raw_details = _extract_tool_results(messages)

    logger.info(
        "ReAct agent complete: %d search products, %d detailed records",
        len(raw_search.get("products") or []),
        len(raw_details),
    )

    return {
        "raw_search_results": raw_search,
        "raw_product_details": raw_details,
    }
