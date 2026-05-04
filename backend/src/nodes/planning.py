"""
planning.py — Node 2: build a deterministic tool-call plan from intent.

No LLM call — pure logic so the pipeline stays deterministic.

Step types:
  search        → run Amazon keyword search, store top-N ASINs
  fetch_details → for each ASIN from the search result, fetch full product details
                  (review insights are embedded inside the product detail response)
  get_details   → fetch details for a single explicit ASIN
"""
from __future__ import annotations

from backend.src.models import AgentState


def planning_node(state: AgentState) -> dict:
    intent = state["intent"]
    keywords = state.get("keywords") or []
    asins = state.get("asins") or []
    top_n = state.get("top_n") or 6

    steps: list[dict] = []

    if intent in ("search", "analyze"):
        query = " ".join(keywords)
        steps.append({"type": "search", "query": query, "n": top_n})
        # Details (including review insights) fetched after search results arrive
        steps.append({"type": "fetch_details", "from_search": True})

    elif intent == "compare":
        if asins:
            for asin in asins:
                steps.append({"type": "get_details", "asin": asin})
        else:
            # No explicit ASINs — search first then fetch details
            query = " ".join(keywords)
            steps.append({"type": "search", "query": query, "n": top_n})
            steps.append({"type": "fetch_details", "from_search": True})

    elif intent == "deep_dive":
        if asins:
            # Explicit ASINs: fetch details for each
            for asin in asins:
                steps.append({"type": "get_details", "asin": asin})
        else:
            # No explicit ASINs: search for top-1, then deep-dive it
            query = " ".join(keywords)
            steps.append({"type": "search", "query": query, "n": 1})
            steps.append({"type": "fetch_details", "from_search": True, "limit": 1})

    return {"tool_plan": steps}
