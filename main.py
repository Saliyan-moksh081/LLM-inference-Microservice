"""Compatibility entry point for the existing `uvicorn main:app` command."""

from app.main import app

__all__ = ["app"]
