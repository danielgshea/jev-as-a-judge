import unittest

from evals.reliability.statistics import JUDGES, summarize


class StatisticsTest(unittest.TestCase):
    def test_summarizes_without_langsmith(self) -> None:
        records = [
            {
                "case": 0,
                **{
                    judge: {
                        "quality": 0.5 if judge == "semif" and repetition == 0 else 1.0,
                        "does_pass": 1,
                        "choice": "answered",
                    }
                    for judge in JUDGES
                },
            }
            for repetition in range(2)
        ]

        result = summarize(records, 1)

        self.assertEqual(result["summary"]["jev"]["quality"]["mean"], 1.0)
        self.assertEqual(result["analysis"]["bootstrap_unit"], "recorded case")
