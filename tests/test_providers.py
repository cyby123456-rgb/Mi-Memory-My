from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from mimemory.providers import (
    Endpoint,
    OpenAICompatibleClient,
    PaperModelRoles,
    ProviderConfigurationError,
    RemoteCallNotApproved,
    json_from_completion,
)


class ProviderConfigurationTests(unittest.TestCase):
    def test_missing_remote_opt_in_is_rejected(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ProviderConfigurationError, "REMOTE_MODELS_ENABLED"):
                PaperModelRoles.from_environment()

    def test_roles_read_only_the_expected_environment_variables(self) -> None:
        values = {
            "MIMEMORY_REMOTE_MODELS_ENABLED": "1",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_API_BASE": "https://generation.example/v1/",
            "OPENAI_MODEL": "generation-model",
            "OPENAI_EMBEDDING_API_KEY": "embedding-key",
            "OPENAI_EMBEDDING_API_BASE": "https://embedding.example/v1",
            "OPENAI_EMBEDDING_MODEL": "embedding-model",
            "OPENAI_EMBEDDING_DIMENSIONS": "1024",
            "EVALUATOR_API_KEY": "evaluator-key",
            "EVALUATOR_API_BASE": "https://evaluator.example/v1",
            "EVALUATOR_MODEL": "evaluator-model",
        }
        with patch.dict(os.environ, values, clear=True):
            roles = PaperModelRoles.from_environment()
        self.assertEqual(roles.extraction.base_url, "https://generation.example/v1")
        self.assertEqual(roles.embedding.model, "embedding-model")
        self.assertEqual(roles.evaluator.model, "evaluator-model")
        self.assertEqual(roles.embedding_dimensions, 1024)

    def test_json_parser_accepts_fenced_json_only(self) -> None:
        self.assertEqual(json_from_completion("```json\n{\"facts\": []}\n```"), {"facts": []})

    def test_provider_refuses_unapproved_network_calls(self) -> None:
        endpoint = Endpoint("https://provider.example/v1", "test-key", "test-model")
        with patch.dict(os.environ, {"MIMEMORY_LIVE_PROVIDER_APPROVED": "0"}, clear=True):
            with self.assertRaises(RemoteCallNotApproved):
                OpenAICompatibleClient(endpoint).complete([{"role": "user", "content": "test"}])


if __name__ == "__main__":
    unittest.main()
