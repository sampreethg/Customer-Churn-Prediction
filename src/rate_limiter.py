"""
Sliding-Window In-Memory Tiered Rate Limiter for FastAPI.

Thread-safe and async-safe request rate regulation with automatic stale bucket pruning,
proxy-aware IP extraction, and Retry-After HTTP 429 response enforcement.
"""

import asyncio
import os
import time
from collections import defaultdict
from typing import Callable, Dict, List

from fastapi import HTTPException, Request, status


class SlidingWindowRateLimiter:
    """
    Sliding-window in-memory rate limiter regulated by asyncio.Lock.
    """

    def __init__(self, window_seconds: float = 60.0):
        self.window_seconds: float = window_seconds
        self._buckets: Dict[str, List[float]] = defaultdict(list)
        self._lock: asyncio.Lock = asyncio.Lock()
        self._check_counter: int = 0
        self.enabled: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() in ("true", "1", "yes")
        self.trust_proxies: bool = os.getenv("TRUST_PROXIES", "false").lower() in ("true", "1", "yes")

    def get_client_ip(self, request: Request) -> str:
        """
        Extract client IP address. Parses X-Forwarded-For only if TRUST_PROXIES is active.
        """
        if self.trust_proxies:
            forwarded_header = request.headers.get("X-Forwarded-For")
            if forwarded_header:
                client_ip = forwarded_header.split(",")[0].strip()
                if client_ip:
                    return client_ip
        if request.client and request.client.host:
            return request.client.host
        return "127.0.0.1"

    async def _prune_stale(self, now: float) -> None:
        """Remove empty or expired IP buckets to prevent memory leaks."""
        stale_keys = []
        cutoff = now - self.window_seconds
        for key, timestamps in self._buckets.items():
            valid_ts = [t for t in timestamps if t > cutoff]
            if valid_ts:
                self._buckets[key] = valid_ts
            else:
                stale_keys.append(key)
        for key in stale_keys:
            self._buckets.pop(key, None)

    async def check(self, request: Request, max_requests: int) -> None:
        """
        Enforce rate limit for client IP against a specific max_requests quota.
        """
        if not self.enabled:
            return

        client_ip = self.get_client_ip(request)
        endpoint_key = f"{client_ip}:{request.url.path}"
        now = time.monotonic()

        async with self._lock:
            self._check_counter += 1
            if self._check_counter % 50 == 0:
                await self._prune_stale(now)

            cutoff = now - self.window_seconds
            timestamps = [t for t in self._buckets[endpoint_key] if t > cutoff]
            self._buckets[endpoint_key] = timestamps

            if len(timestamps) >= max_requests:
                oldest_ts = timestamps[0]
                retry_after = max(1, int(self.window_seconds - (now - oldest_ts)))
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "Rate limit exceeded. Try again later.",
                        "retry_after": retry_after,
                    },
                    headers={"Retry-After": str(retry_after)},
                )

            self._buckets[endpoint_key].append(now)

    def limit(self, max_requests: int) -> Callable:
        """
        FastAPI dependency factory for route-level rate limiting.
        """
        async def dependency(request: Request) -> None:
            await self.check(request, max_requests)

        return dependency


# Global singleton instance
global_rate_limiter = SlidingWindowRateLimiter(window_seconds=60.0)
rate_limiter = global_rate_limiter


def rate_limit(max_requests: int) -> Callable:
    """Convenience dependency function for route definitions."""
    return global_rate_limiter.limit(max_requests)
