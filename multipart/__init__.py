"""Compatibility shim for Starlette's legacy `multipart` import.

The installed `multipart` package emits a PendingDeprecationWarning and then
re-exports `python_multipart`.  This local shim performs the same re-export
without warning so `pytest -W error` remains usable.
"""
import python_multipart as _python_multipart

from python_multipart import *  # noqa: F403

__all__ = getattr(_python_multipart, "__all__", [])
__version__ = getattr(_python_multipart, "__version__", "")
