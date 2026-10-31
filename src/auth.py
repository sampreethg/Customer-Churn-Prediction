"""
API Key and Bearer Token Authentication Dependency for FastAPI.

Provides constant-time digest comparison, multi-key allowlisting,
support for X-API-Key and Authorization: Bearer headers, and test-mode bypass.
"""

import os
import secrets
from typing import Optional, Set

from fastapi import HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Optional Bearer security scheme for OpenAPI swagger documentation
bearer_scheme = HTTPBearer(auto_error=False)

DEFAULT_KEYS = "dev-secret-key-2026,test-api-key,prod-churn-key-99a"


def get_configured_api_keys() -> Set[str]:
    """Retrieve set of authorized API keys from environment."""
    raw_keys = os.getenv("API_KEYS", DEFAULT_KEYS)
    return {k.strip() for k in raw_keys.split(",") if k.strip()}


async def require_api_key(
    request: Request,
    bearer_token: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> str:
    """
    Validate incoming request credentials against authorized API keys.
    Accepts X-API-Key header or Authorization: Bearer token.
    """
    auth_enabled = os.getenv("AUTH_ENABLED", "true").lower() in ("true", "1", "yes")
    if not auth_enabled:
        return "bypass"

    candidate_key: Optional[str] = None

    # 1. Check X-API-Key header
    header_key = request.headers.get("X-API-Key")
    if header_key:
        candidate_key = header_key.strip()

    # 2. Check Bearer token if X-API-Key was not provided
    if not candidate_key and bearer_token:
        candidate_key = bearer_token.credentials.strip()

    if not candidate_key:
        # Check raw Authorization header fallback
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            candidate_key = auth_header[7:].strip()

    if not candidate_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide X-API-Key or Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    authorized_keys = get_configured_api_keys()

    # Constant-time comparison across all authorized keys to prevent timing attacks
    is_valid = any(
        secrets.compare_digest(candidate_key, valid_key)
        for valid_key in authorized_keys
    )

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key supplied.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return candidate_key
