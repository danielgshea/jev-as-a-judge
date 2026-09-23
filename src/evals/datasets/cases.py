CASES = [
    {
        "question": "What is the weather in Seattle today?",
        "expected_behavior": {"location": "Seattle", "search_required": True},
        "category": "current_conditions",
    },
    {
        "question": "What will the weather be like in Austin, Texas this weekend?",
        "expected_behavior": {"location": "Austin", "search_required": True},
        "category": "multi_day_forecast",
    },
    {
        "question": "Will I need an umbrella in Dublin tomorrow?",
        "expected_behavior": {"location": "Dublin", "search_required": True},
        "category": "weather_decision",
    },
    {
        "question": "What is the forecast for Tokyo next week?",
        "expected_behavior": {"location": "Tokyo", "search_required": True},
        "category": "extended_forecast",
    },
    {
        "question": "What is the weather in Springfield today?",
        "expected_behavior": {"location": "Springfield", "search_required": False},
        "category": "ambiguous_location",
    },
]


ORACLE_BY_QUESTION = {
    "What is the weather in Seattle today?": {
        "is_grounded": 1,
        "matches_search_expectation": 1,
        "is_useful": 1,
        "does_pass": 1,
        "reasoning": {
            "is_grounded": "The reported conditions are supported by the recorded search evidence.",
            "matches_search_expectation": "The agent searched before answering the unambiguous Seattle request.",
            "is_useful": "It provides current weather details and source links.",
            "does_pass": "The response is grounded, follows the search requirement, and is useful.",
        },
    },
    "What will the weather be like in Austin, Texas this weekend?": {
        "is_grounded": 1,
        "matches_search_expectation": 1,
        "is_useful": 1,
        "does_pass": 1,
        "reasoning": {
            "is_grounded": "The weekend forecast is supported by the recorded search evidence.",
            "matches_search_expectation": "The agent searched before answering the unambiguous Austin request.",
            "is_useful": "It addresses the requested weekend and provides source links.",
            "does_pass": "The response is grounded, follows the search requirement, and is useful.",
        },
    },
    "Will I need an umbrella in Dublin tomorrow?": {
        "is_grounded": 1,
        "matches_search_expectation": 0,
        "is_useful": 1,
        "does_pass": 0,
        "reasoning": {
            "is_grounded": "The clarification makes no unsupported weather claim.",
            "matches_search_expectation": "The expected behavior requires a search, but the recorded response asks for location clarification.",
            "is_useful": "It identifies the location ambiguity and asks the user to resolve it.",
            "does_pass": "The useful clarification still fails the experiment's required-search behavior.",
        },
    },
    "What is the forecast for Tokyo next week?": {
        "is_grounded": 1,
        "matches_search_expectation": 1,
        "is_useful": 1,
        "does_pass": 1,
        "reasoning": {
            "is_grounded": "The forecast is supported by the recorded search evidence.",
            "matches_search_expectation": "The agent searched before answering the unambiguous Tokyo request.",
            "is_useful": "It addresses the requested week and provides source links.",
            "does_pass": "The response is grounded, follows the search requirement, and is useful.",
        },
    },
    "What is the weather in Springfield today?": {
        "is_grounded": 1,
        "matches_search_expectation": 1,
        "is_useful": 1,
        "does_pass": 1,
        "reasoning": {
            "is_grounded": "The clarification makes no unsupported weather claim.",
            "matches_search_expectation": "The ambiguous location correctly triggers clarification without a search.",
            "is_useful": "It asks for the missing state or country.",
            "does_pass": "Clarification is the expected response for the ambiguous request.",
        },
    },
}
