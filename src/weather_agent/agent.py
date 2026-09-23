from deepagents import create_deep_agent
from tavily import TavilyClient

from .search import create_weather_search


weather_instructions = """You are a weather assistant.

Use search_weather for every weather question that needs current or forecast data.
Search with the location, date or date range, and the word weather or forecast.
Call search_weather at most once.
Prefer authoritative weather sources when the search results provide them.
Never invent weather details. State when a result is unavailable or uncertain.
Include the location, forecast time, useful conditions such as temperature and precipitation,
and source links in a concise answer. If a location is ambiguous, ask the user to clarify
it before searching.
"""


def create_weather_agent(*, model: str, tavily_api_key: str):
    return create_deep_agent(
        model=model,
        tools=[create_weather_search(TavilyClient(api_key=tavily_api_key))],
        system_prompt=weather_instructions,
    ).with_config({"recursion_limit": 20})
