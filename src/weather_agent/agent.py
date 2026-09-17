import os
from typing import Literal

from deepagents import create_deep_agent
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def search_weather(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
):
    """Search the web for current weather conditions or forecasts."""
    return tavily_client.search(
        query,
        max_results=max_results,
        topic=topic,
        include_raw_content=False,
    )


weather_instructions = """You are a weather assistant.

Use search_weather for every weather question that needs current or forecast data.
Search with the location, date or date range, and the word weather or forecast.
Prefer authoritative weather sources when the search results provide them.
Never invent weather details. State when a result is unavailable or uncertain.
Include the location, forecast time, useful conditions such as temperature and precipitation,
and source links in your answer. If a location is ambiguous, ask the user to clarify it
before searching.
"""

agent = create_deep_agent(
    model=os.getenv("WEATHER_AGENT_MODEL", "openai:gpt-5.5"),
    tools=[search_weather],
    system_prompt=weather_instructions,
).with_config({"recursion_limit": 20})
