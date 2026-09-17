from langsmith import traceable
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient


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


def _state(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    return {
        "user_question": inputs["question"],
        "expected_behavior": reference_outputs,
        "final_answer": outputs["answer"],
        "tool_calls": outputs["tool_calls"],
        "search_evidence": outputs.get("evidence", []),
    }


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
    scores = _run_jev_judge(_state(inputs, outputs, reference_outputs))
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
    answer = _run_jev_score(_state(inputs, outputs, reference_outputs))
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
    answer = _run_jev_choice(_state(inputs, outputs, reference_outputs))
    return {
        "key": "jev_weather_outcome",
        "value": answer["choice"],
        "comment": (
            f"confidence={answer['confidence']:.3f}, "
            f"probabilities={answer['probabilities']}"
        ),
    }
