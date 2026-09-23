import os

from dotenv import load_dotenv

from .agent import create_weather_agent


def main() -> None:
    load_dotenv()
    os.environ["LANGSMITH_GATEWAY"] = "true"
    agent = create_weather_agent(
        model=os.getenv("WEATHER_AGENT_MODEL", "openai:gpt-5.5"),
        tavily_api_key=os.environ["TAVILY_API_KEY"],
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is the weather in Seattle today?"}]}
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
