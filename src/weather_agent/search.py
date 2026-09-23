from collections.abc import Callable

from tavily import TavilyClient


MAX_RESULTS = 4
MAX_TITLE_CHARS = 120
MAX_URL_CHARS = 300
MAX_EXCERPT_CHARS = 350


def _bounded(value: object, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def compact_search_results(response: dict) -> list[dict[str, str]]:
    return [
        {
            "title": _bounded(result.get("title"), MAX_TITLE_CHARS),
            "url": _bounded(result.get("url"), MAX_URL_CHARS),
            "excerpt": _bounded(result.get("content"), MAX_EXCERPT_CHARS),
        }
        for result in response.get("results", [])[:MAX_RESULTS]
    ]


def create_weather_search(client: TavilyClient) -> Callable[[str], list[dict[str, str]]]:
    def search_weather(query: str) -> list[dict[str, str]]:
        """Search once for current weather or forecast evidence."""
        return compact_search_results(
            client.search(
                query,
                max_results=MAX_RESULTS,
                topic="general",
                include_raw_content=False,
            )
        )

    return search_weather
