import os
import unittest
from unittest.mock import patch

from evals.judges.state import judge_state
from evals.judges.system_one import QUALITY_QUESTIONS, _classifier
from evals.config import DECISION_GATEWAY_BASE_URL, DECISION_JUDGES


class SystemOneJudgesTest(unittest.TestCase):
    def test_includes_complete_evidence_but_excludes_oracle(self) -> None:
        evidence = [{"title": "Weather", "url": "https://example.com", "excerpt": "x" * 500}]
        state = judge_state(
            {"question": "Weather?"},
            {"answer": "Sunny", "tool_calls": ["search_weather"], "evidence": evidence},
            {
                "expected_behavior": {"search_required": True},
                "oracle": {"does_pass": 1},
            },
        )

        self.assertEqual(state["search_evidence"], evidence)
        self.assertEqual(state["expected_behavior"], {"search_required": True})
        self.assertNotIn("oracle", state)

    def test_uses_gateway_for_both_decision_models(self) -> None:
        with patch.dict(os.environ, {"LS_LLM_GATEWAY_KEY": "test-key"}):
            classifiers = {
                name: _classifier(config, QUALITY_QUESTIONS)
                for name, config in DECISION_JUDGES.items()
            }

        self.assertEqual(
            {name: classifier.model for name, classifier in classifiers.items()},
            {"jev": "typesafe/jev-1.13.0", "semif": "semif-qwen3.5-4b"},
        )
        self.assertTrue(
            all(
                classifier.base_url == DECISION_GATEWAY_BASE_URL
                and classifier.api_key.get_secret_value() == "test-key"
                and classifier.questions == QUALITY_QUESTIONS
                for classifier in classifiers.values()
            )
        )

if __name__ == "__main__":
    unittest.main()
