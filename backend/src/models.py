from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ── LangGraph State ────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Input
    message: str
    # Query Understanding
    intent: str            # search | analyze | compare | deep_dive
    keywords: list[str]
    asins: list[str]
    top_n: int
    # Tool Outputs (populated by ReAct agent)
    raw_search_results: dict
    raw_product_details: list[dict]   # each item now embeds review_insights
    # Processed
    aggregated_products: list[dict]
    # Analysis
    analysis: dict
    # Final response
    response: dict


# ── API Schemas ────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., description="User research query")


class ChatResponse(BaseModel):
    answer: str


# ── Structured LLM Outputs ─────────────────────────────────────────────────────

class QueryIntent(BaseModel):
    intent: Literal["search", "analyze", "compare", "deep_dive"]
    keywords: list[str] = Field(description="Amazon search keywords extracted from the query (NOT the whole sentence)")
    asins: list[str] = Field(default_factory=list, description="Explicit ASINs mentioned by user (e.g. B07XXXXX)")
    top_n: int = Field(default=6, description="Number of top products to analyse; defaults to 6 unless user specifies")
    reasoning: str = Field(description="One-line reasoning for the intent classification")


class AnalysisOutput(BaseModel):
    top_purchase_drivers: list[str] = Field(description="Top 5 specific reasons customers buy these products, grounded in the provided review insights")
    common_complaints: list[str] = Field(description="Top 5 recurring complaints grounded in the provided review insights")
    sentiment_themes: list[str] = Field(description="General sentiment themes across all products (e.g. 'value for money', 'durability concerns')")
    competitive_gaps: list[str] = Field(description="Specific market gaps / opportunities not addressed by any current product")
    answer: str = Field(description="A comprehensive, beautifully formatted Markdown market research report following the exact structure defined in the system prompt.")
