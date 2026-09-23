import json
from typing import Literal

from langsmith import traceable
from pydantic import BaseModel

from ..config import LLM_JUDGES, ModelConfig, create_chat_model
from .state import judge_state


class LLMQualityResult(BaseModel):
    is_grounded: float
    matches_search_expectation: float
    is_useful: float


class LLMDoesPassResult(BaseModel):
    does_pass: Literal[0, 1]


class LLMChoiceResult(BaseModel):
    choice: Literal["answered", "clarification_needed", "poor"]


def _llm_prompt(state: dict, instructions: str) -> str:
    return f"{instructions}\n\nEvaluate this JSON state:\n{json.dumps(state, sort_keys=True)}"


def _build_evaluators(prefix: str, config: ModelConfig) -> tuple:

    @traceable(name=f"{prefix}_weather_quality")
    def run_quality(state: dict) -> LLMQualityResult:
        llm = create_chat_model(config)
        judge = llm.with_structured_output(LLMQualityResult, method="json_schema")
        return judge.invoke(
            _llm_prompt(
                state,
                """Score each statement from 0 to 1. Grounded means the answer is supported by
the supplied search evidence. Search expectation means the agent searched for an
unambiguous location and did not search an ambiguous one. Useful means it answers
the requested weather question with the location, timing, details, and source links,
or asks for clarification when appropriate.""",
            )
        )

    def quality(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        answer = run_quality(judge_state(inputs, outputs, reference_outputs))
        scores = answer.model_dump()
        return {
            "key": f"{prefix}_weather_quality",
            "score": sum(scores.values()) / len(scores),
            "comment": ", ".join(
                f"{name}={score:.3f}" for name, score in scores.items()
            ),
        }

    @traceable(name=f"{prefix}_weather_does_pass")
    def run_does_pass(state: dict) -> LLMDoesPassResult:
        llm = create_chat_model(config)
        judge = llm.with_structured_output(LLMDoesPassResult, method="json_schema")
        return judge.invoke(
            _llm_prompt(
                state,
                "Return exactly 1 if the final answer passes and 0 if it fails.",
            )
        )

    def does_pass(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        answer = run_does_pass(judge_state(inputs, outputs, reference_outputs))
        return {
            "key": f"{prefix}_weather_does_pass",
            "score": answer.does_pass,
            "comment": "",
        }

    @traceable(name=f"{prefix}_weather_choice")
    def run_choice(state: dict) -> LLMChoiceResult:
        llm = create_chat_model(config)
        judge = llm.with_structured_output(LLMChoiceResult, method="json_schema")
        return judge.invoke(
            _llm_prompt(
                state,
                """Classify the final response as exactly one of these outcomes:
answered: a grounded weather answer with useful timing and source details;
clarification_needed: correctly asks for clarification before answering an ambiguous
location; poor: fails to answer usefully or makes unsupported weather claims.""",
            )
        )

    def choice(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        answer = run_choice(judge_state(inputs, outputs, reference_outputs))
        return {
            "key": f"{prefix}_weather_outcome",
            "value": answer.choice,
            "comment": "",
        }

    return quality, does_pass, choice


def build_llm_evaluators() -> dict:
    return {
        prefix: _build_evaluators(prefix, config)
        for prefix, config in LLM_JUDGES.items()
    }
