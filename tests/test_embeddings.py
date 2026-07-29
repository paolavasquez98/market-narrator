"""Tests for the Embedder wrapper, using a stubbed fastembed model so no
network call or model download is required. This tests that our wrapper
correctly adapts fastembed's numpy-array output to plain lists (JSON- and
type-hint-friendly, and easy to pass around outside the knowledge_base
package) and preserves order -- not the embedding model's actual output
quality.

Note: plain `list[float]` still needs an explicit `::vector` SQL cast
wherever it's compared against a `vector` column (see vector_store.py) --
psycopg's default array dumper, not pgvector's, handles raw Python lists.
"""

import sys
import types

import numpy as np
import pytest


@pytest.fixture
def stub_fastembed(monkeypatch):
    """Install a fake `fastembed` module before Embedder imports it."""
    fake_module = types.ModuleType("fastembed")

    class FakeTextEmbedding:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def embed(self, texts: list[str]):
            # Deterministic fake vectors: length matches input, values encode
            # the text's index so ordering can be asserted.
            for i, _ in enumerate(texts):
                yield np.array([float(i)] * 4)

    fake_module.TextEmbedding = FakeTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", fake_module)
    return fake_module


def test_embed_preserves_order_and_converts_to_lists(stub_fastembed):
    from finrag.knowledge_base.embeddings import Embedder

    embedder = Embedder()
    vectors = embedder.embed(["first", "second", "third"])

    assert vectors == [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]
    assert all(isinstance(v, list) for v in vectors)
