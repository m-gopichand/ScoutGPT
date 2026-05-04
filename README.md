# Pixii — Amazon Research Agent

A Perplexity-style conversational agent for Amazon product research and competitive analysis. It uses **LangGraph** to orchestrate a multi-step research pipeline, **Google Gemini** for reasoning, and **SerpApi** for real-time Amazon data.

## Setup

1. **Environment Variables**:
   Ensure your `.env` file contains:
   ```env
   SERPAPI_KEY=your_serpapi_key
   GEMINIAPI_KEY=your_gemini_api_key
   ```

2. **Install Dependencies**:
   ```bash
   uv sync
   ```

## Running the Server

Start the backend API server:
```bash
uv run python main.py
```
The server will be available at `http://localhost:8000`.

## API Usage

### `POST /api/v1/chat`
Stateless research endpoint that returns an SSE (Server-Sent Events) stream.

**Request:**
```json
{
  "message": "find top ergonomic keyboards under $100"
}
```

**SSE Events:**
- `progress`: Real-time updates on what the agent is currently doing (Searching, Analyzing, etc.)
- `data`: The final structured research report (JSON).
- `done`: Signal that the research is complete.

## Features
- **Deterministic Planning**: Classified intent (search, analyze, compare) drives tool usage.
- **Deep Sentiment**: Analyzes up to 20 deep customer reviews per product.
- **Revenue Estimates**: Heuristic-based monthly revenue calculation grounded in sales data.
- **Competitive Analysis**: Identifies market gaps and customer pain points.
