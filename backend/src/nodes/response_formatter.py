"""
response_formatter.py — Node 6: assemble the final response dict.

The LLM in analysis_node produces the complete Markdown report as `analysis["answer"]`.
This node is a thin pass-through that packages it into the ChatResponse schema.
"""
from __future__ import annotations

from backend.src.models import AgentState, ChatResponse


def response_formatter_node(state: AgentState) -> dict:
    analysis = state.get("analysis") or {}
    response = ChatResponse(
        answer=analysis.get("answer") or "No analysis could be produced for this query."
    )
    return {"response": response.model_dump()}
