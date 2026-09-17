from langsmith import traceable
from typesafe_sdk import Noul, TypeSafeClient


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
    state = {
        "user_question": inputs["question"],
        "expected_behavior": reference_outputs,
        "final_answer": outputs["answer"],
        "tool_calls": outputs["tool_calls"],
        "search_evidence": outputs.get("evidence", []),
    }

    scores = _run_jev_judge(state)
    return {
        "key": "jev_weather_quality",
        "score": sum(scores.values()) / len(scores),
        "comment": ", ".join(f"{name}={score:.3f}" for name, score in scores.items()),
    }
