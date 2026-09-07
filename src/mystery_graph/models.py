"""Hugging Face generation adapted to a LangChain Runnable interface."""

from threading import Lock
from typing import Protocol

from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableLambda

from mystery_graph.config import Settings


class ModelError(RuntimeError):
    """Sanitized inference failure; provider payloads can contain private prompts."""


class ChatBackend(Protocol):
    def complete(self, messages: list[BaseMessage]) -> str: ...


def chat_messages(messages: list[BaseMessage]) -> list[dict[str, str]]:
    roles = {"system": "system", "human": "user", "ai": "assistant"}
    return [{"role": roles[m.type], "content": str(m.content)} for m in messages]


class HuggingFaceBackend:
    def __init__(self, settings: Settings):
        from huggingface_hub import InferenceClient

        settings.require_generation_credentials()
        self.settings = settings
        self.client = InferenceClient(
            token=settings.hf_token, provider=settings.hf_provider, timeout=settings.timeout
        )

    def complete(self, messages: list[BaseMessage]) -> str:
        try:
            response = self.client.chat_completion(
                model=self.settings.hf_model,
                messages=chat_messages(messages),
                max_tokens=self.settings.max_tokens,
                temperature=0.5,
            )
            content = response.choices[0].message.content
            if not content:
                raise ModelError(
                    "The model returned no text. Try a larger token budget or another model."
                )
            return content
        except ModelError:
            raise
        except Exception:
            raise ModelError(
                "Hugging Face inference failed. Check the token's inference permissions, "
                "model/provider availability, quota and connection."
            ) from None


class LocalTransformersBackend:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipeline = None
        self._lock = Lock()

    def complete(self, messages: list[BaseMessage]) -> str:
        with self._lock:
            try:
                if self._pipeline is None:
                    from transformers import pipeline

                    self._pipeline = pipeline(
                        "text-generation",
                        model=self.settings.local_model,
                        device=self.settings.local_device,
                        trust_remote_code=False,
                    )
                result = self._pipeline(
                    chat_messages(messages),
                    max_new_tokens=self.settings.max_tokens,
                    do_sample=True,
                    temperature=0.5,
                    return_full_text=False,
                )
                content = result[0]["generated_text"]
                if isinstance(content, list):
                    content = content[-1]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ModelError("The local model returned no text.")
                return content
            except ModelError:
                raise
            except Exception:
                raise ModelError(
                    "Local generation failed. Check the local dependencies, model access, "
                    "device setting, available memory and model cache."
                ) from None


def as_runnable(backend: ChatBackend) -> RunnableLambda:
    def generate(prompt: PromptValue) -> str:
        return backend.complete(prompt.to_messages())

    return RunnableLambda(generate)


def create_backend(settings: Settings) -> ChatBackend:
    if settings.backend == "local":
        return LocalTransformersBackend(settings)
    return HuggingFaceBackend(settings)
