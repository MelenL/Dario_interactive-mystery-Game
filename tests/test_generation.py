import pytest

from mystery_graph.config import ConfigurationError, Settings
from mystery_graph.generation import Generator, StructuredGenerationError
from mystery_graph.models import ModelError


class ScriptedBackend:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_schema_failure_retries_with_bounded_regeneration(case):
    backend = ScriptedBackend(["not JSON", case.model_dump_json()])
    result = Generator(backend).story(
        theme="archive", difficulty="medium", seed="test", references="[]"
    )
    assert result == case and backend.calls == 2


def test_schema_failures_stop_after_two_attempts():
    backend = ScriptedBackend(["invalid", "still invalid"])
    with pytest.raises(StructuredGenerationError):
        Generator(backend).answer(scene="scene", evidence="[]", history="[]", question="Cold?")
    assert backend.calls == 2


def test_provider_errors_are_not_retried_as_schema_errors():
    backend = ScriptedBackend([ModelError("Quota exceeded")])
    with pytest.raises(ModelError, match="Quota"):
        Generator(backend).answer(scene="scene", evidence="[]", history="[]", question="Cold?")
    assert backend.calls == 1


def test_missing_token_is_local_configuration_failure():
    with pytest.raises(ConfigurationError, match="HF_TOKEN"):
        Settings(hf_token="").require_generation_credentials()
    Settings(backend="local").require_generation_credentials()


def test_public_host_requires_authentication():
    with pytest.raises(ConfigurationError, match="APP_USERNAME"):
        Settings(host="0.0.0.0").validate()


def test_secrets_are_not_in_settings_repr():
    settings = Settings(hf_token="private-value", password="private-password")
    assert "private-value" not in repr(settings)
    assert "private-password" not in repr(settings)
