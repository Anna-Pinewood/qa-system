"""Settings for the project."""
import os
from pathlib import Path
from dotenv import dotenv_values

PROJECT_PATH = Path(__file__).parent.parent
ENV_PATH = PROJECT_PATH / ".env"
config_env = dotenv_values(ENV_PATH)

FAISS_INDEX_PATH = PROJECT_PATH / "faiss-index"
EMBEDDINGS_MODEL_NAME = os.environ.get(
    "EMBEDDINGS_MODEL_NAME", default=config_env.get("EMBEDDINGS_MODEL_NAME"))
TOP_K_DOCUMENTS = int(os.environ.get(
    "TOP_K_DOCUMENTS", default=config_env.get("TOP_K_DOCUMENTS")))

SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT", default=config_env.get("SYSTEM_PROMPT"))
CONTEXT_PROMPT = os.environ.get(
    "CONTEXT_PROMPT", default=config_env.get("CONTEXT_PROMPT"))

MODEL_URL = os.environ.get("MODEL_URL", default=config_env.get("MODEL_URL"))
MAX_TOKENS_ANSWER = int(os.environ.get(
    "MAX_TOKENS_ANSWER", default=config_env.get("MAX_TOKENS_ANSWER")))
TEMPERATURE = int(os.environ.get(
    "TEMPERATURE", default=config_env.get("TEMPERATURE")))


MODEL_PIPELINE_NAME = os.environ.get(
    "MODEL_PIPELINE_NAME", default=config_env.get("MODEL_PIPELINE_NAME"))
OPENAI_TOKEN = os.environ.get(
    "OPENAI_TOKEN", default=config_env.get("OPENAI_TOKEN"))
