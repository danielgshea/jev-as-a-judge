import json
import unittest
from types import SimpleNamespace

from evals.datasets.recorded import parse_example
from evals.reliability.experiment import recorded_target
from weather_agent.search import (
    MAX_EXCERPT_CHARS,
    MAX_RESULTS,
    MAX_TITLE_CHARS,
    MAX_URL_CHARS,
    compact_search_results,
)


class RecordedDatasetTest(unittest.TestCase):
    def test_compacts_search_cards(self) -> None:
        cards = compact_search_results(
            {
                "results": [
                    {"title": "t" * 500, "url": "u" * 500, "content": "e" * 500}
                    for _ in range(10)
                ]
            }
        )

        self.assertEqual(len(cards), MAX_RESULTS)
        self.assertEqual(set(cards[0]), {"title", "url", "excerpt"})
        self.assertLessEqual(len(cards[0]["title"]), MAX_TITLE_CHARS)
        self.assertLessEqual(len(cards[0]["url"]), MAX_URL_CHARS)
        self.assertLessEqual(len(cards[0]["excerpt"]), MAX_EXCERPT_CHARS)

    def test_returns_recorded_response_unchanged(self) -> None:
        response = {"answer": "Sunny", "tool_calls": [], "evidence": []}

        self.assertIs(recorded_target({"recorded_response": response}), response)

    def test_parses_nested_expected_behavior(self) -> None:
        response = {"answer": "Sunny", "tool_calls": [], "evidence": []}
        example = SimpleNamespace(
            inputs={"question": "Weather?", "recorded_response": response},
            outputs={
                "expected_behavior": {"location": "Seattle", "search_required": True},
                "oracle": {"does_pass": 1},
            },
            metadata={"category": "current", "generator_model": "gpt", "state_size": 123},
        )

        parsed = parse_example(example)

        self.assertEqual(parsed["expected_behavior"], example.outputs["expected_behavior"])
        self.assertEqual(json.dumps(parsed["recorded_response"]), json.dumps(response))
