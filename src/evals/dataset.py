from langsmith import Client

DATASET_NAME = "weather-agent"

EXAMPLES = [
    {
        "inputs": {"question": "What is the weather in Seattle today?"},
        "outputs": {"location": "Seattle", "search_required": True},
        "metadata": {"category": "current_conditions"},
    },
    {
        "inputs": {"question": "What will the weather be like in Austin, Texas this weekend?"},
        "outputs": {"location": "Austin", "search_required": True},
        "metadata": {"category": "multi_day_forecast"},
    },
    {
        "inputs": {"question": "Will I need an umbrella in Dublin tomorrow?"},
        "outputs": {"location": "Dublin", "search_required": True},
        "metadata": {"category": "weather_decision"},
    },
    {
        "inputs": {"question": "What is the forecast for Tokyo next week?"},
        "outputs": {"location": "Tokyo", "search_required": True},
        "metadata": {"category": "extended_forecast"},
    },
    {
        "inputs": {"question": "What is the weather in Springfield today?"},
        "outputs": {"location": "Springfield", "search_required": False},
        "metadata": {"category": "ambiguous_location"},
    },
]


def ensure_dataset(client: Client | None = None):
    client = client or Client()
    dataset = next(client.list_datasets(dataset_name=DATASET_NAME, limit=1), None)
    if dataset:
        return dataset, False

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Current-weather and forecast requests for the DeepAgents weather agent.",
    )
    client.create_examples(dataset_id=dataset.id, examples=EXAMPLES)
    return dataset, True


def main() -> None:
    _, created = ensure_dataset()
    print(f"{'Created' if created else 'Using existing'} dataset: {DATASET_NAME}")


if __name__ == "__main__":
    main()
