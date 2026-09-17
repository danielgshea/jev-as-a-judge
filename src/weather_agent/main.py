from .agent import agent


if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is the weather in Seattle today?"}]}
    )
    print(result["messages"][-1].content)
