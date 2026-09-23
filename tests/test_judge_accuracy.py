import unittest

from evals.analysis.common import score_oracle


class JudgeAccuracyTest(unittest.TestCase):
    def test_scores_all_repetitions_and_quality_tolerance(self) -> None:
        data = {
            "judges": [["judge", "Judge"]],
            "values": {
                "quality": {"Judge": [[1.0, 0.9]]},
                "does_pass": {"Judge": [[1, 0]]},
            },
        }
        recorded_cases = [{"inputs": {"question": "Question"}}]
        oracle = {
            "labels": [
                {
                    "question": "Question",
                    "is_grounded": 1,
                    "matches_search_expectation": 1,
                    "is_useful": 1,
                    "does_pass": 1,
                }
            ]
        }

        result = score_oracle(data, recorded_cases, oracle, tolerance=0.1)["judges"]["Judge"]

        self.assertEqual(result["predictions"], 2)
        self.assertEqual(result["does_pass_accuracy"], 0.5)
        self.assertAlmostEqual(result["quality_mae"], 0.05)
        self.assertEqual(result["quality_within_tolerance"], 1.0)


if __name__ == "__main__":
    unittest.main()
