from dotenv import load_dotenv
from langsmith import traceable
from langchain_typesafe import Choice, Noul, TypeSafeClassifier

from . import CHOICE_CRITERIA, judge_state


load_dotenv()


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


@traceable(name="jev_weather_judge")
def _run_jev_judge(state: dict) -> dict:
    response = TypeSafeClassifier(questions=QUALITY_QUESTIONS).invoke(state)
    return {name: answer.noul for name, answer in response.nouls.items()}


def jev_weather_quality(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Use Jev to score groundedness, tool behavior, and usefulness together."""
    scores = _run_jev_judge(judge_state(inputs, outputs, reference_outputs))
    return {
        "key": "jev_weather_quality",
        "score": sum(scores.values()) / len(scores),
        "comment": ", ".join(f"{name}={score:.3f}" for name, score in scores.items()),
    }


@traceable(name="jev_weather_does_pass")
def _run_jev_does_pass(state: dict) -> float:
    return TypeSafeClassifier(questions=DOES_PASS_QUESTIONS).invoke(state).nouls[
        "does_pass"
    ].noul


def jev_weather_does_pass(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Use Jev's does_pass judgment for an overall pass/fail result."""
    probability = _run_jev_does_pass(judge_state(inputs, outputs, reference_outputs))
    return {
        "key": "jev_weather_does_pass",
        "score": int(probability >= 0.5),
        "comment": f"probability={probability:.3f}",
    }


@traceable(name="jev_weather_choice")
def _run_jev_choice(state: dict) -> dict:
    answer = TypeSafeClassifier(questions=CHOICE_QUESTIONS).invoke(state).choices[
        "outcome"
    ]
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
