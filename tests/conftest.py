from __future__ import annotations

import os
import warnings

import httpx


# Unit/evaluation tests must be deterministic and must not depend on local .env,
# GPU, embedding model cache, Neo4j, Qdrant, Ollama or API keys.
os.environ["DIRECT_QDRANT_PROJECTION"] = "false"
os.environ["ENABLE_LLM"] = "false"
os.environ["ENABLE_LOCAL_EMBEDDINGS"] = "false"
os.environ["KG_BACKEND"] = "fallback"
os.environ["RETRIEVAL_MODE"] = "bm25"

warnings.filterwarnings(
    "ignore",
    message=r"Please use `import python_multipart` instead\.",
    category=PendingDeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"The 'app' shortcut is now deprecated.*",
    category=DeprecationWarning,
    module=r"httpx\._client",
)

_HTTPX_CLIENT_INIT = httpx.Client.__init__


def _httpx_client_init_without_testclient_warning(self, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The 'app' shortcut is now deprecated.*",
            category=DeprecationWarning,
        )
        return _HTTPX_CLIENT_INIT(self, *args, **kwargs)


httpx.Client.__init__ = _httpx_client_init_without_testclient_warning
