import json


CHOICE_CRITERIA = {
    "answered": "Provides a grounded weather answer with useful timing and source details",
    "clarification_needed": "Correctly asks for clarification before answering an ambiguous location",
    "poor": "Fails to answer usefully or makes unsupported weather claims",
}


def judge_state(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    return {
        "user_question": inputs["question"],
        "expected_behavior": reference_outputs["expected_behavior"],
        "final_answer": outputs["answer"],
        "tool_calls": outputs["tool_calls"],
        "search_evidence": outputs.get("evidence", []),
    }


def judge_state_json(inputs: dict, outputs: dict, reference_outputs: dict) -> str:
    return json.dumps(
        judge_state(inputs, outputs, reference_outputs),
        separators=(",", ":"),
        sort_keys=True,
    )
