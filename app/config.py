"""Runtime settings, read from environment variables or a .env file."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # The language model is used only when a real key is set. A placeholder
    # (for example "dummy-key") switches the answer step to extractive mode.
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-5-5"

    # "hashing" needs no download. "bge-m3" loads BAAI/bge-m3 (needs requirements-ml.txt).
    embedder: str = "hashing"
    # "lexical" needs no download. "bge" loads BAAI/bge-reranker-v2-m3.
    reranker: str = "lexical"

    top_k: int = 20
    top_n: int = 4
    relevance_threshold: float = 0.5
    max_upload_mb: int = 20

    @property
    def llm_enabled(self) -> bool:
        key = self.anthropic_api_key.strip()
        return key.startswith("sk-ant-") and "dummy" not in key


@lru_cache
def get_settings() -> Settings:
    return Settings()
