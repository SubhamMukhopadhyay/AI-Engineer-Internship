"""
config.py
---------
Centralised, typed configuration loaded from environment variables.
Nothing sensitive is ever hardcoded — every value here has a safe
default or comes from the environment (see .env.example).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM (generation)
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_base_url: str = Field(default="", alias="LLM_BASE_URL")
    llm_model: str = Field(default="", alias="LLM_MODEL")

    # Embeddings
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")

    # Reranking
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL"
    )
    use_reranker: bool = Field(default=True, alias="USE_RERANKER")

    # Chunking
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=180, alias="CHUNK_OVERLAP")

    # Retrieval
    retrieval_top_k: int = Field(default=8, alias="RETRIEVAL_TOP_K")
    rerank_top_k: int = Field(default=4, alias="RERANK_TOP_K")

    # Storage
    index_dir: str = Field(default="./data/index", alias="INDEX_DIR")
    upload_dir: str = Field(default="./data/uploads", alias="UPLOAD_DIR")
    max_file_size_mb: int = Field(default=25, alias="MAX_FILE_SIZE_MB")

    # API
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    def is_llm_configured(self) -> bool:
        return bool(self.llm_api_key and self.llm_base_url and self.llm_model)


@lru_cache
def get_settings() -> Settings:
    return Settings()
