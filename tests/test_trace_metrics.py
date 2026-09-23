import unittest
from types import SimpleNamespace

from evals.analysis.visualize import _latency_comparison
from evals.analysis.traces import _cost, _summarize, _token_counts


class TraceMetricsTest(unittest.TestCase):
    def test_compares_mean_latency(self) -> None:
        svg = _latency_comparison({"judges": {
            "Jev": {"average_latency_seconds": 0.5},
            "SemIf": {"average_latency_seconds": 0.55},
        }})

        self.assertIn("Mean evaluator latency", svg)
        self.assertIn("0.500s", svg)
        self.assertIn("0.550s", svg)

    def test_summarizes_latency_by_repetition(self) -> None:
        records = [
            {"judge": "Jev", "status": "success", "input_tokens": 1, "output_tokens": 1,
             "latency_seconds": latency, "total_cost": 0.0, "model": "Jev", "repetition": repetition}
            for repetition, latency in [(0, 0.1), (0, 0.3), (1, 0.5)]
        ]
        records.append(
            {"judge": "Jev", "status": "success", "input_tokens": None, "output_tokens": None,
             "latency_seconds": 0.7, "total_cost": 0.0, "model": "Jev", "repetition": 1}
        )

        self.assertEqual(_summarize(records, 2)["Jev"]["latency_by_repetition_seconds"], [0.2, 0.6])

    def test_estimates_semif_tokens_and_cost_from_trace_payload(self) -> None:
        run = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            inputs={
                "inputs": {"question": "Weather?"},
                "outputs": {"answer": "Sunny", "tool_calls": [], "evidence": []},
                "reference_outputs": {"expected_behavior": {"search_required": True}},
            },
            outputs={"score": 0.9},
        )

        input_tokens, output_tokens, source = _token_counts("semif", run)
        cost, cost_range, cost_source = _cost(
            "semif", None, input_tokens, output_tokens
        )

        self.assertGreater(input_tokens, 0)
        self.assertGreater(output_tokens, 0)
        self.assertEqual(source, "estimated_from_trace_payload")
        self.assertAlmostEqual(
            cost,
            (input_tokens * 0.135 + output_tokens * 0.20) / 1_000_000,
        )
        self.assertLess(cost_range[0], cost)
        self.assertGreater(cost_range[1], cost)
        self.assertEqual(cost_source, "estimated_qwen3.5_9b_proxy")

    def test_prefers_reported_semif_usage(self) -> None:
        run = SimpleNamespace(prompt_tokens=12, completion_tokens=3)

        self.assertEqual(_token_counts("semif", run), (12, 3, "reported"))


if __name__ == "__main__":
    unittest.main()
