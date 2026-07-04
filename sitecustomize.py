"""Project-local warning policy for deterministic test runs.

This file is imported by Python's site machinery before test modules.  It keeps
`python -m pytest -q -W error` usable while filtering two known third-party
warnings emitted during FastAPI/TestClient import.
"""

from __future__ import annotations

import warnings


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
