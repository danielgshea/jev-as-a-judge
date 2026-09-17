import json
import os
from typing import Literal

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient
from langsmith import traceable
from pydantic import BaseModel, Field


load_dotenv()


SCORE_CRITERIA = [
    "Poor: missing, unsupported, or misleading",
    "Adequate: partially answers the question or has minor omissions",
    "Excellent: grounded, useful, and complete for the requested weather information",
]

CHOICE_CRITERIA = {
    "answered": "Provides a grounded weather answer with useful timing and source details",
    "clarification_needed": "Correctly asks for clarification before answering an ambiguous location",
    "poor": "Fails to answer usefully or makes unsupported weather claims",
}


def judge_state(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    return {
        "user_question": inputs["question"],
        "expected_behavior": reference_outputs,
        "final_answer": outputs["answer"],
        "tool_calls": outputs["tool_calls"],
        "search_evidence": outputs.get("evidence", []),
    }


class LLMQualityResult(BaseModel):
    is_grounded: float = Field(ge=0, le=1)
    matches_search_expectation: float = Field(ge=0, le=1)
    is_useful: float = Field(ge=0, le=1)


class LLMScoreResult(BaseModel):
    score: int = Field(ge=0, le=len(SCORE_CRITERIA) - 1)


class LLMChoiceResult(BaseModel):
    choice: Literal["answered", "clarification_needed", "poor"]


_llm_model = os.getenv("LLM_JUDGE_MODEL", "gpt-5.6-luna")
if not _llm_model.startswith("openai/"):
    _llm_model = f"openai/{_llm_model.removeprefix('openai:')}"

_llm = ChatOpenAI(
    model=_llm_model,
    base_url="https://gateway.smith.langchain.com/v1",
    api_key=os.environ["LANGSMITH_API_KEY"],
    timeout=180,
    max_retries=0,
)


def _llm_prompt(state: dict, instructions: str) -> str:
    return f"{instructions}\n\nEvaluate this JSON state:\n{json.dumps(state, sort_keys=True)}"


@traceable(name="jev_weather_judge")
def _run_jev_judge(state: dict) -> dict:
    with TypeSafeClient() as client:
        response = client.system_one(
            state=state,
            questions={
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
            },
        )
    return {name: answer.noul for name, answer in response.answers.items()}


def jev_weather_quality(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Use Jev to score groundedness, tool behavior, and usefulness together."""
    scores = _run_jev_judge(judge_state(inputs, outputs, reference_outputs))
    return {
        "key": "jev_weather_quality",
        "score": sum(scores.values()) / len(scores),
        "comment": ", ".join(f"{name}={score:.3f}" for name, score in scores.items()),
    }


@traceable(name="jev_weather_score")
def _run_jev_score(state: dict) -> dict:
    with TypeSafeClient() as client:
        response = client.system_one(
            state=state,
            questions={
                "answer_quality": Score(
                    instructions="How well does the final answer satisfy the weather request?",
                    criteria=SCORE_CRITERIA,
                ),
            },
        )
    answer = response.answers["answer_quality"]
    return {
        "score": answer.score,
        "confidence": answer.confidence,
    }


def jev_weather_score(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Use Jev's ordered Score primitive for response quality."""
    answer = _run_jev_score(judge_state(inputs, outputs, reference_outputs))
    maximum = len(SCORE_CRITERIA) - 1
    return {
        "key": "jev_weather_score",
        "score": answer["score"] / maximum,
        "comment": (
            f"raw_score={answer['score']:.3f}, "
            f"confidence={answer['confidence']:.3f}"
        ),
    }


@traceable(name="jev_weather_choice")
def _run_jev_choice(state: dict) -> dict:
    with TypeSafeClient() as client:
        response = client.system_one(
            state=state,
            questions={
                "outcome": Choice(
                    instructions="What outcome best describes the final weather response?",
                    criteria=CHOICE_CRITERIA,
                ),
            },
        )
    answer = response.answers["outcome"]
    return {
        "choice": answer.choice,
        "confidence": answer.confidence,
        "probabilities": answer.probabilities,
    }


def jev_weather_choice(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Use Jev's Choice primitive to classify the response outcome."""
    answer = _run_jev_choice(judge_state(inputs, outputs, reference_outputs))
    return {
        "key": "jev_weather_outcome",
        "value": answer["choice"],
        "comment": (
            f"confidence={answer['confidence']:.3f}, "
            f"probabilities={answer['probabilities']}"
        ),
    }


@traceable(name="llm_weather_quality")
def _run_llm_quality(state: dict) -> LLMQualityResult:
    judge = _llm.with_structured_output(LLMQualityResult, method="json_schema")
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


def llm_weather_quality(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    answer = _run_llm_quality(judge_state(inputs, outputs, reference_outputs))
    scores = answer.model_dump()
    return {
        "key": "llm_weather_quality",
        "score": sum(scores.values()) / len(scores),
        "comment": ", ".join(f"{name}={score:.3f}" for name, score in scores.items()),
    }


@traceable(name="llm_weather_score")
def _run_llm_score(state: dict) -> LLMScoreResult:
    judge = _llm.with_structured_output(LLMScoreResult, method="json_schema")
    return judge.invoke(
        _llm_prompt(
            state,
            """Rate the final answer using this ordered rubric. Return 0 for poor: missing,
unsupported, or misleading; 1 for adequate: partially answers the question or has
minor omissions; 2 for excellent: grounded, useful, and complete for the requested
weather information.""",
        )
    )


def llm_weather_score(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    answer = _run_llm_score(judge_state(inputs, outputs, reference_outputs))
    return {
        "key": "llm_weather_score",
        "score": answer.score / (len(SCORE_CRITERIA) - 1),
        "comment": f"raw_score={answer.score}",
    }


@traceable(name="llm_weather_choice")
def _run_llm_choice(state: dict) -> LLMChoiceResult:
    judge = _llm.with_structured_output(LLMChoiceResult, method="json_schema")
    return judge.invoke(
        _llm_prompt(
            state,
            """Classify the final response as exactly one of these outcomes:
answered: a grounded weather answer with useful timing and source details;
clarification_needed: correctly asks for clarification before answering an ambiguous
location; poor: fails to answer usefully or makes unsupported weather claims.""",
        )
    )


def llm_weather_choice(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    answer = _run_llm_choice(judge_state(inputs, outputs, reference_outputs))
    return {
        "key": "llm_weather_outcome",
        "value": answer.choice,
        "comment": "",
    }
