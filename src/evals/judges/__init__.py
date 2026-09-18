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


from ..model import LLM_JUDGES
from .llm import LLM_EVALUATORS
from .system_one import (
    jev_weather_choice,
    jev_weather_does_pass,
    jev_weather_quality,
)
