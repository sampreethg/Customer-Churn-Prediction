"""
Centralized Authentication Dependency for AI Portfolio Hub API.
"""

from src.auth import get_configured_api_keys, require_api_key

__all__ = ["require_api_key", "get_configured_api_keys"]
