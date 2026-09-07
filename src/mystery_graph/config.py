"""Explicit configuration; no model loading or network access at import time."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """Actionable configuration problem safe to display to the operator."""


@dataclass(frozen=True)
class Settings:
    backend: str = "hf"
    hf_token: str = field(default="", repr=False)
    hf_model: str = "openai/gpt-oss-120b"
    hf_provider: str = "auto"
    timeout: int = 90
    local_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    local_device: str = "cpu"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    retrieval_k: int = 3
    max_tokens: int = 3500
    corpus_path: str = ""
    host: str = "127.0.0.1"
    port: int = 7860
    username: str = ""
    password: str = field(default="", repr=False)

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(override=False)
        try:
            settings = cls(
                backend=os.getenv("MODEL_BACKEND", "hf"),
                hf_token=os.getenv("HF_TOKEN", ""),
                hf_model=os.getenv("HF_MODEL_ID", cls.hf_model),
                hf_provider=os.getenv("HF_PROVIDER", "auto"),
                timeout=int(os.getenv("HF_TIMEOUT_SECONDS", "90")),
                local_model=os.getenv("LOCAL_MODEL_ID", cls.local_model),
                local_device=os.getenv("LOCAL_DEVICE", "cpu"),
                embedding_model=os.getenv("EMBEDDING_MODEL_ID", cls.embedding_model),
                embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
                retrieval_k=int(os.getenv("RETRIEVAL_K", "3")),
                max_tokens=int(os.getenv("MAX_GENERATION_TOKENS", "3500")),
                corpus_path=os.getenv("CORPUS_PATH", ""),
                host=os.getenv("HOST", "127.0.0.1"),
                port=int(os.getenv("PORT", "7860")),
                username=os.getenv("APP_USERNAME", ""),
                password=os.getenv("APP_PASSWORD", ""),
            )
        except ValueError as exc:
            raise ConfigurationError(
                "Port, timeouts, token limit and retrieval count must be integers."
            ) from exc
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.backend not in {"hf", "local"}:
            raise ConfigurationError("MODEL_BACKEND must be hf or local.")
        if not 1 <= self.retrieval_k <= 10 or not 256 <= self.max_tokens <= 8192:
            raise ConfigurationError("RETRIEVAL_K must be 1–10; MAX_GENERATION_TOKENS 256–8192.")
        if not 1 <= self.port <= 65535 or not 1 <= self.timeout <= 600:
            raise ConfigurationError("Invalid PORT or HF_TIMEOUT_SECONDS.")
        if bool(self.username) != bool(self.password):
            raise ConfigurationError("Set APP_USERNAME and APP_PASSWORD together.")
        if self.host not in {"127.0.0.1", "localhost", "::1"} and not self.password:
            raise ConfigurationError(
                "Set APP_USERNAME and APP_PASSWORD before binding a public interface."
            )

    def require_generation_credentials(self) -> None:
        if self.backend == "hf" and not self.hf_token.strip():
            raise ConfigurationError(
                "Set HF_TOKEN for hosted inference, or choose MODEL_BACKEND=local. "
                "No model request has been made."
            )
