import os
from dotenv import load_dotenv
import serpapi
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
GEMINI_API_KEY = os.getenv("GEMINIAPI_KEY")

# Valid Gemini model identifiers
GEMINI_FLASH = "gemini-3-flash-preview"   # fast — used for classification
GEMINI_PRO   = "gemini-3.1-pro-preview"     # deep reasoning — used for analysis


def get_serpapi_client() -> serpapi.Client:
    return serpapi.Client(api_key=SERPAPI_KEY)


def get_llm(
    temperature: float = 0.1,
    streaming: bool = False,
    model_name: str = GEMINI_FLASH,
) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=GEMINI_API_KEY,
        temperature=temperature,
        streaming=streaming,
    )
