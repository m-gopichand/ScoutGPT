# ScoutGPT — Amazon Research Agent

A Perplexity-style conversational agent for Amazon product research and competitive analysis. It uses **LangGraph** to orchestrate a multi-step research pipeline, **Google Gemini** for reasoning, and **SerpApi** for real-time Amazon data.

## Try It Out

👉 [Open ScoutGPT Studio](https://smith.langchain.com/studio/?baseUrl=https://scoutgpt.gopichand.dev)

---

## Setup Instructions

### Step 1: Configure Connection

Click on **"Configure connection"**.

![Step 1 - Configure Connection](https://github.com/user-attachments/assets/7a294fba-8293-453c-8d71-84c0ec805fe1)

---

### Step 2: Allow Domain Access

Click on **"Add to allowed domains"**, then click on connect. 

![Step 2 - Add to Allowed Domains](https://github.com/user-attachments/assets/c1d0ee66-840f-47a8-9169-5fc2c181fb06)

### Step 3: Login with your langsmith account and reopen the link. 
## Local Setup

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

Start the LangGraph development server:
```bash
uv run langgraph dev
```

The server and LangGraph studio will be available locally.


## Features
- **Deterministic Planning**: Classified intent (search, analyze, compare) drives tool usage.
- **Deep Sentiment**: Analyzes up to 20 deep customer reviews per product.
- **Revenue Estimates**: Heuristic-based monthly revenue calculation grounded in sales data.
- **Competitive Analysis**: Identifies market gaps and customer pain points.


