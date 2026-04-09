"""Configuration loading for the local debate app."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    facilitator_model: str = Field(default="llama3.1:8b", alias="FACILITATOR_MODEL")
    debater_a_model: str = Field(default="gemma4:8b", alias="DEBATER_A_MODEL")
    debater_b_model: str = Field(default="qwen3.5:8b", alias="DEBATER_B_MODEL")
    validator_model: str = Field(default="rnj-1:latest", alias="VALIDATOR_MODEL")
    model_keep_alive: str = Field(default="10m", alias="MODEL_KEEP_ALIVE")

    max_turns: int = Field(default=8, alias="MAX_TURNS", ge=1, le=50)
    search_command_template: str = Field(
        default='ddgs text "{query}" --max-results 5',
        alias="SEARCH_COMMAND_TEMPLATE",
    )
    search_timeout_seconds: int = Field(default=20, alias="SEARCH_TIMEOUT_SECONDS", ge=1)
    ollama_timeout_seconds: int = Field(default=60, alias="OLLAMA_TIMEOUT_SECONDS", ge=5)

    markdown_log_dir: str = Field(default="./logs", alias="MARKDOWN_LOG_DIR")
    input_dir: str = Field(default="./in", alias="INPUT_DIR")
    output_dir: str = Field(default="./out", alias="OUTPUT_DIR")

    langfuse_enabled: bool = Field(default=False, alias="LANGFUSE_ENABLED")
    langfuse_host: str = Field(default="", alias="LANGFUSE_HOST")
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")

    summary_recent_turns: int = Field(default=6, ge=2, le=20)
    recent_context_turns: int = Field(default=4, ge=2, le=12)

    preload_models_on_start: bool = True
    warmup_models_on_preload: bool = True

    @field_validator("search_command_template")
    @classmethod
    def validate_search_template(cls, value: str) -> str:
        """Ensure template has a query placeholder."""
        if "{query}" not in value:
            raise ValueError("SEARCH_COMMAND_TEMPLATE must include '{query}'.")
        return value

    @property
    def markdown_log_dir_path(self) -> Path:
        """Return markdown log directory as a Path."""
        return Path(self.markdown_log_dir).expanduser().resolve()

    @property
    def input_dir_path(self) -> Path:
        """Return default input markdown directory as a Path."""
        return Path(self.input_dir).expanduser().resolve()

    @property
    def output_dir_path(self) -> Path:
        """Return output directory for final result markdown as a Path."""
        return Path(self.output_dir).expanduser().resolve()


def load_config() -> Config:
    """Load configuration from .env and environment variables."""
    load_dotenv(override=False)
    return Config()
