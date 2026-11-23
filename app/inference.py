"""Async HTTPX helpers for calling the existing vLLM server."""

import httpx

from .config import VLLM_BASE_URL, VLLM_TIMEOUT


async def check_vllm_health() -> httpx.Response:
    """Return vLLM's health response; callers handle request errors."""
    async with httpx.AsyncClient(timeout=2.0) as client:
        return await client.get(f"{VLLM_BASE_URL}/health")


async def request_chat_completion(payload: dict) -> httpx.Response:
    """Send a non-streaming chat completion request to vLLM."""
    async with httpx.AsyncClient(timeout=VLLM_TIMEOUT) as client:
        return await client.post(
            f"{VLLM_BASE_URL}/v1/chat/completions",
            json=payload,
        )


async def open_chat_completion_stream(
    payload: dict,
) -> tuple[httpx.AsyncClient, httpx.Response]:
    """Open a streaming response and return its client and response objects."""
    client = httpx.AsyncClient(timeout=VLLM_TIMEOUT)
    request = client.build_request(
        "POST",
        f"{VLLM_BASE_URL}/v1/chat/completions",
        json=payload,
    )

    try:
        response = await client.send(request, stream=True)
    except BaseException:
        # Close the client if opening the upstream response fails or is cancelled.
        await client.aclose()
        raise

    return client, response


async def relay_sse_events(
    upstream_response: httpx.Response,
    streaming_client: httpx.AsyncClient,
):
    """Relay vLLM's SSE chunks and close both upstream resources afterward."""
    try:
        async for chunk in upstream_response.aiter_bytes():
            yield chunk

    except httpx.HTTPError:
        # The HTTP status cannot be changed after streaming has begun.
        # Send an SSE error event so the client knows the stream failed.
        yield (
            b'data: {"error":{"message":'
            b'"vLLM connection was interrupted during streaming."}}\n\n'
        )

    finally:
        # Close both connections when the stream ends or the client disconnects.
        await upstream_response.aclose()
        await streaming_client.aclose()
