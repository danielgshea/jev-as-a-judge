import os
import time

from langsmith import traceable
from langchain_typesafe import Choice, Noul, TypeSafeClassifier
from langchain_typesafe.client import (
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeRateLimitError,
)

from ..config import DECISION_GATEWAY_BASE_URL, DECISION_JUDGES, ModelConfig
from .state import CHOICE_CRITERIA, judge_state


QUALITY_QUESTIONS = {
    "is_grounded": Noul(
        instructions=(
            "Does the final answer stay grounded in the supplied search evidence "
            "and avoid inventing weather details?"
        ),
    ),
    "matches_search_expectation": Noul(
        instructions=(
            "Does the agent's search behavior match expected_behavior.search_required? "
            "Unambiguous weather requests should search; ambiguous locations should "
            "ask for clarification without searching."
        ),
    ),
    "is_useful": Noul(
        instructions=(
            "Is the final answer useful for the user's weather question by naming the "
            "location, addressing the requested time, and giving weather details and "
            "source links, or by clearly asking for clarification when the location is ambiguous?"
        ),
    ),
}

DOES_PASS_QUESTIONS = {
    "does_pass": Noul(instructions="Does the final answer pass?"),
}

CHOICE_QUESTIONS = {
    "outcome": Choice(
        instructions="What outcome best describes the final weather response?",
        criteria=CHOICE_CRITERIA,
    ),
}


def _classifier(config: ModelConfig, questions: dict) -> TypeSafeClassifier:
    return TypeSafeClassifier(
        questions=questions,
        model=config.model,
        api_key=os.environ[config.api_key_env],
        base_url=DECISION_GATEWAY_BASE_URL,
        timeout=180,
    )


def _invoke(config: ModelConfig, questions: dict, state):
    for attempt in range(6):
        try:
            return _classifier(config, questions).invoke(state)
        except TypeSafeRateLimitError as error:
            if attempt == 5:
                raise RuntimeError(f"{config.label}: {error}") from None
            time.sleep(min((error.retry_after_ms or 2_000) / 1_000, 60))
        except TypeSafeAPIConnectionError as error:
            if attempt == 5:
                raise RuntimeError(f"{config.label}: {error}") from None
            time.sleep(2)
        except TypeSafeAPIError as error:
            detail = error.body.get("detail") if isinstance(error.body, dict) else None
            raise RuntimeError(f"{config.label}: {detail or error}") from None


def verify_decision_models() -> None:
    failures = []
    for config in DECISION_JUDGES.values():
        try:
            _invoke(config, DOES_PASS_QUESTIONS, "Gateway preflight")
        except RuntimeError as error:
            failures.append(str(error))
    if failures:
        raise RuntimeError(f"Gateway preflight failed: {'; '.join(failures)}")


def _build_evaluators(prefix: str, config: ModelConfig) -> tuple:
    @traceable(name=f"{prefix}_weather_judge")
    def run_quality(state: dict) -> dict:
        response = _invoke(config, QUALITY_QUESTIONS, state)
        return {name: answer.noul for name, answer in response.nouls.items()}

    def quality(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        scores = run_quality(judge_state(inputs, outputs, reference_outputs))
        return {
            "key": f"{prefix}_weather_quality",
            "score": sum(scores.values()) / len(scores),
            "comment": ", ".join(
                f"{name}={score:.3f}" for name, score in scores.items()
            ),
        }

    @traceable(name=f"{prefix}_weather_does_pass")
    def run_does_pass(state: dict) -> float:
        return _invoke(config, DOES_PASS_QUESTIONS, state).nouls["does_pass"].noul

    def does_pass(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        probability = run_does_pass(
            judge_state(inputs, outputs, reference_outputs)
        )
        return {
            "key": f"{prefix}_weather_does_pass",
            "score": int(probability >= 0.5),
            "comment": f"probability={probability:.3f}",
        }

    @traceable(name=f"{prefix}_weather_choice")
    def run_choice(state: dict) -> dict:
        answer = _invoke(config, CHOICE_QUESTIONS, state).choices["outcome"]
        return {
            "choice": answer.choice,
            "confidence": answer.confidence,
            "probabilities": answer.probabilities,
        }

    def choice(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        answer = run_choice(judge_state(inputs, outputs, reference_outputs))
        return {
            "key": f"{prefix}_weather_outcome",
            "value": answer["choice"],
            "comment": (
                f"confidence={answer['confidence']:.3f}, "
                f"probabilities={answer['probabilities']}"
            ),
        }

    return quality, does_pass, choice


DECISION_EVALUATORS = {
    prefix: _build_evaluators(prefix, config)
    for prefix, config in DECISION_JUDGES.items()
}

jev_weather_quality, jev_weather_does_pass, jev_weather_choice = (
    DECISION_EVALUATORS["jev"]
)
