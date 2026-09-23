import os
from dataclasses import dataclass

from langchain_openai import ChatOpenAI


GATEWAY_BASE_URL = "https://gateway.smith.langchain.com/v1"
DECISION_GATEWAY_BASE_URL = "https://gateway.smith.langchain.com"


@dataclass(frozen=True)
class ModelConfig:
    label: str
    model: str
    api_key_env: str


LLM_JUDGES = {
    "gpt_luna": ModelConfig("GPT-5.6 Luna", "openai/gpt-5.6-luna", "LANGSMITH_API_KEY"),
    "gpt_terra": ModelConfig("GPT-5.6 Terra", "openai/gpt-5.6-terra", "LANGSMITH_API_KEY"),
    "claude_sonnet": ModelConfig("Claude Sonnet 4.6", "anthropic/claude-sonnet-4-6", "LS_LLM_GATEWAY_KEY"),
}

DECISION_JUDGES = {
    "jev": ModelConfig("Jev", "typesafe/jev-1.13.0", "LS_LLM_GATEWAY_KEY"),
    "semif": ModelConfig("SemIf", "semif-qwen3.5-4b", "LS_LLM_GATEWAY_KEY"),
}


def create_chat_model(config: ModelConfig) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.model,
        base_url=GATEWAY_BASE_URL,
        api_key=os.environ[config.api_key_env],
        timeout=180,
        max_retries=2,
    )
