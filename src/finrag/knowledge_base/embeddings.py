"""Text embedding wrapper.

Embeddings turn text into a fixed-length vector such that semantically
similar sentences land close together (by cosine distance) -- this is what
lets retrieval match "how did Apple do in 2022" against a document that
never uses those exact words but says "AAPL fell 27% over the year".

We use `fastembed` (ONNX Runtime under the hood) rather than
`sentence-transformers` (PyTorch) or an API-based embedding model
(OpenAI/Cohere): it's free, runs fully offline after the one-time model
download, produces the same vector for the same text every time (important
for reproducibility), and it doesn't drag torch into the Docker image.

Model: BAAI/bge-small-en-v1.5, 384 dimensions -- matches the
`VECTOR(384)` column in schema.sql. If the model ever changes, that column
width has to change too.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384


class Embedder:
    """Thin wrapper around fastembed so the rest of the app depends on this
    interface, not directly on the fastembed package -- swapping the
    embedding backend later (e.g. to an API-based model) means changing
    this one file.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed a batch of texts. Order is preserved."""
        return [vector.tolist() for vector in self._model.embed(list(texts))]


@lru_cache
def get_embedder() -> Embedder:
    """Process-wide cached Embedder instance. Loading the ONNX model has a
    real one-time cost (reading model files, initializing the runtime);
    the RAG pipeline embeds one query per request and shouldn't pay that
    cost again on every call. Same `@lru_cache`-as-singleton pattern as
    `config.get_settings()`.
    """
    return Embedder()
