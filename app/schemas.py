"""Pydantic models used to validate chat API requests."""

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    # Keep the first version focused on ordinary text chat roles.
    role: Literal["system", "user", "assistant"]

    # The text content for this message.
    content: str


class ChatCompletionRequest(BaseModel):
    # A conversation must contain at least one message.
    messages: list[ChatMessage] = Field(min_length=1)

    # Accept the usual API field, but the service uses its configured model ID.
    model: str | None = None

    # Default to a modest response length to limit inference work.
    max_tokens: int = Field(default=256, ge=1, le=1024)

    # Restrict temperature to the common supported range.
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    # Enable SSE output when the caller requests token streaming.
    stream: bool = False
