"""
agent.py — LangGraph graph wiring.

Graph:  query_understanding → planning → execute_tools
        → aggregation → analysis → response_formatter → END
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from backend.src.models import AgentState
from backend.src.nodes import (
    aggregation_node,
    analysis_node,
    execute_tools_node,
    planning_node,
    query_understanding_node,
    response_formatter_node,
)

# ── Build graph ────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("query_understanding", query_understanding_node)
    builder.add_node("planning", planning_node)
    builder.add_node("execute_tools", execute_tools_node)
    builder.add_node("aggregation", aggregation_node)
    builder.add_node("analysis", analysis_node)
    builder.add_node("response_formatter", response_formatter_node)

    builder.set_entry_point("query_understanding")
    builder.add_edge("query_understanding", "planning")
    builder.add_edge("planning", "execute_tools")
    builder.add_edge("execute_tools", "aggregation")
    builder.add_edge("aggregation", "analysis")
    builder.add_edge("analysis", "response_formatter")
    builder.add_edge("response_formatter", END)

    return builder.compile()


# Singleton compiled graph (imported by api.py)
agent_graph = build_graph()
