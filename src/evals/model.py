import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


GATEWAY_BASE_URL = "https://gateway.smith.langchain.com/v1"


@dataclass(frozen=True)
class ModelConfig:
    label: str
    model: str
    api_key_env: str


LLM_JUDGES = {
    "gpt_luna": ModelConfig(
        label="GPT-5.6 Luna",
        model="openai/gpt-5.6-luna",
        api_key_env="LANGSMITH_API_KEY",
    ),
    "gpt_terra": ModelConfig(
        label="GPT-5.6 Terra",
        model="openai/gpt-5.6-terra",
        api_key_env="LANGSMITH_API_KEY",
    ),
    "claude_sonnet": ModelConfig(
        label="Claude Sonnet 4.6",
        model="anthropic/claude-sonnet-4-6",
        api_key_env="LS_LLM_GATEWAY_KEY",
    ),
}


def create_chat_model(config: ModelConfig) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.model,
        base_url=GATEWAY_BASE_URL,
        api_key=os.environ[config.api_key_env],
        timeout=180,
        max_retries=2,
    )
