"""Service configuration, from environment variables or a .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AI_", extra="ignore")

    # Largest file this service will read.
    #
    # The .NET API enforces its own limit before anything reaches here. This one
    # exists because a service should not depend on its caller being careful —
    # if the API is ever misconfigured, or something else calls this directly,
    # the failure should be a clean 413 rather than the process being killed by
    # the OOM reaper mid-parse.
    max_upload_bytes: int = 20 * 1024 * 1024

    # Ollama, for embeddings in Phase 5. Unused today, declared here so the
    # deployment surface does not change when that lands.
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"


settings = Settings()
