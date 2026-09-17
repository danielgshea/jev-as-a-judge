# JEV as a Judge

This project evaluates a DeepAgents weather agent that uses Tavily to find current conditions and forecasts. The evaluator is powered by [Jev](https://docs.typesafe.ai/introduction), TypeSafe's flagship System One model.

## Why Jev for evals?

Traditional LLM judges generate text that an application must interpret. Jev is designed to make structured decisions directly: it evaluates typed questions against structured state and returns typed answers without an explanation-parsing step.

That makes Jev a natural fit for evaluator logic:

- `Noul` returns the probability that a yes/no judgment is true.
- `Score` rates an answer against an ordered rubric and returns probabilities and confidence.
- `Choice` selects one option and returns probabilities and confidence.
- Multiple atomic questions can be evaluated in parallel against the same state.

This project sends Jev the weather question, the agent's final answer, Tavily evidence, tool calls, and expected behavior. One Jev call checks whether the answer is grounded, whether search behavior matches the example, and whether the response is useful. The evaluator averages those Noul probabilities into a LangSmith score.

The point is not that one judge is universally better. Jev gives this evaluator typed, composable signals that are easy to combine, threshold, and inspect.

## Quick start

Requires Python 3.13+, an OpenAI API key, a Tavily API key, a TypeSafe API key, and a LangSmith API key.

```bash
cp .env.example .env
# Add your API keys to .env
uv sync
```

Create the LangSmith dataset:

```bash
uv run python src/evals/dataset.py
```

Run the local weather-agent evaluation. This uses the examples in `src/evals/dataset.py`, pretty-prints each JEV result, and does not require the dataset to exist in LangSmith:

```bash
uv run python main.py
```

To upload the dataset and record an evaluation experiment in LangSmith, run:

```bash
uv run python src/evals/offline_evals.py
```

The evaluation runner creates the `weather-agent` dataset if it is missing and reuses it on later runs. Run `src/evals/dataset.py` separately only when you want to create or inspect the dataset without running an experiment.

Run only the sample weather agent directly:

```bash
uv run python main.py
```

## Configuration

The main settings are in `.env`:

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Model used by the DeepAgents weather agent |
| `TAVILY_API_KEY` | Web search used by `search_weather` |
| `TYPESAFE_API_KEY` | Jev evaluator access |
| `LANGSMITH_API_KEY` | Dataset and evaluation results |
| `LANGSMITH_TRACING` | Enables LangSmith traces; set to `true` |
| `LANGSMITH_PROJECT` | LangSmith project for traces |
| `WEATHER_AGENT_MODEL` | Model string, defaulting to `openai:gpt-5.5` |

## How the evaluation is wired

```text
weather question
      |
      v
DeepAgent -- search_weather --> Tavily evidence
      |
      v
final answer + evidence + tool calls + expectations
      |
      v
Jev: grounded? searched correctly? useful?
      |
      v
LangSmith score and trace
```

The JEV interaction is wrapped with LangSmith's `@traceable` decorator as `jev_weather_judge`, so each judge call appears as a nested trace when tracing is enabled.

## Project layout

- `src/weather_agent/agent.py` — DeepAgents weather agent and Tavily tool.
- `src/evals/dataset.py` — Creates the `weather-agent` LangSmith dataset.
- `src/evals/judges.py` — JEV-based evaluator and typed questions.
- `src/evals/offline_evals.py` — Runs the agent over the dataset and uploads results.
- `langgraph.json` — Registers the weather agent for LangGraph tooling.

## Official documentation

- [TypeSafe introduction](https://docs.typesafe.ai/introduction)
- [TypeSafe Python quick start](https://docs.typesafe.ai/introduction/quickstart)
- [TypeSafe primitives](https://docs.typesafe.ai/primitives)
- [Noul](https://docs.typesafe.ai/primitives/noul)
- [Score](https://docs.typesafe.ai/primitives/score)
- [Confidence](https://docs.typesafe.ai/confidence)
- [TypeSafe API reference](https://docs.typesafe.ai/api)
- [LangSmith evaluation quickstart](https://docs.langchain.com/langsmith/evaluation-quickstart)
- [LangSmith `@traceable`](https://docs.langchain.com/langsmith/annotate-code)
- [DeepAgents quickstart](https://docs.langchain.com/oss/python/deepagents/quickstart)
