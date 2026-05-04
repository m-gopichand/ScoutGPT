"""
agent.py — LangGraph graph wiring.

Graph:  query_understanding → react_agent
        → aggregation → analysis → response_formatter → END

The react_agent node replaces the old planning + execute_tools nodes.
It uses LangGraph's create_react_agent to autonomously decide which tools
to call (search_products, get_product_details) and in what order.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from backend.src.models import AgentState
from backend.src.nodes import (
    aggregation_node,
    analysis_node,
    query_understanding_node,
    react_agent_node,
    response_formatter_node,
)

# ── Build graph ────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("query_understanding", query_understanding_node)
    builder.add_node("react_agent", react_agent_node)
    builder.add_node("aggregation", aggregation_node)
    builder.add_node("analysis", analysis_node)
    builder.add_node("response_formatter", response_formatter_node)

    builder.set_entry_point("query_understanding")
    builder.add_edge("query_understanding", "react_agent")
    builder.add_edge("react_agent", "aggregation")
    builder.add_edge("aggregation", "analysis")
    builder.add_edge("analysis", "response_formatter")
    builder.add_edge("response_formatter", END)

    return builder.compile()


# Singleton compiled graph (imported by langgraph.json)
agent_graph = build_graph()
