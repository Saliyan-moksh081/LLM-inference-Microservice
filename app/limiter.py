"""Rate-limit configuration for the chat endpoint."""

from fastapi import FastAPI
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .config import RATE_LIMIT

# Use the caller's IP address as the rate-limit key.
# The in-memory store is appropriate while running one FastAPI process.
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="memory://",
)


def configure_rate_limiting(app: FastAPI) -> None:
    """Attach SlowAPI's limiter and 429 handler to the FastAPI app."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
