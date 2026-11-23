"""FastAPI routes for the LLM inference gateway."""

import json

import httpx
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from .config import RATE_LIMIT, VLLM_BASE_URL, VLLM_MODEL
from .inference import (
    check_vllm_health,
    open_chat_completion_stream,
    relay_sse_events,
    request_chat_completion,
)
from .limiter import configure_rate_limiting, limiter
from .schemas import ChatCompletionRequest

# Create and configure the FastAPI application.
app = FastAPI(title="LLM Inference Microservice")
configure_rate_limiting(app)


@app.get("/health")
async def health():
    # Liveness check: confirms the FastAPI process itself is running.
    # This remains healthy even if vLLM is offline.
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    # The API is not ready for inference until both settings are configured.
    if not VLLM_BASE_URL or not VLLM_MODEL:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="vLLM endpoint is offline or not fully configured.",
        )

    try:
        # Check the configured vLLM server with a short timeout.
        response = await check_vllm_health()

    except httpx.RequestError:
        # Connection errors and timeouts mean the inference endpoint is offline.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="vLLM endpoint is offline.",
        )

    # Treat an unsuccessful health response as the backend being unavailable.
    if not response.is_success:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="vLLM endpoint is offline.",
        )

    # Both FastAPI and the inference backend are reachable.
    return {"status": "ready", "inference_backend": "online"}


@app.post("/v1/chat/completions")
@limiter.limit(RATE_LIMIT)
async def create_chat_completion(
    request: Request,
    chat_request: ChatCompletionRequest,
):
    # Return a clear service error if the temporary chat model is not configured.
    if not VLLM_BASE_URL or not VLLM_MODEL:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="vLLM endpoint is offline or not fully configured.",
        )

    # Convert the validated request into JSON for vLLM.
    # Always use the model configured by the service rather than a caller-provided ID.
    payload = chat_request.model_dump(exclude_none=True)
    payload["model"] = VLLM_MODEL

    # Handle streaming separately so connection errors can be returned before
    # FastAPI begins sending the response headers.
    if chat_request.stream:
        payload["stream"] = True

        try:
            # Open the vLLM response without buffering its full body.
            streaming_client, upstream_response = await open_chat_completion_stream(
                payload
            )

        except httpx.TimeoutException:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Timed out waiting for vLLM to start the response.",
            )

        except httpx.RequestError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="vLLM endpoint is offline.",
            )

        # Do not expose internal inference-server errors to the caller.
        if upstream_response.status_code >= 500:
            await upstream_response.aclose()
            await streaming_client.aclose()
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The vLLM inference server returned an error.",
            )

        # Return vLLM's client/request error before starting the SSE response.
        if not upstream_response.is_success:
            try:
                error_bytes = await upstream_response.aread()

                # Preserve vLLM's JSON error when possible.
                try:
                    error_body = json.loads(error_bytes)
                except ValueError:
                    error_body = {"detail": "vLLM rejected the request."}
            finally:
                # Release the upstream response and its HTTP connection.
                await upstream_response.aclose()
                await streaming_client.aclose()

            return JSONResponse(
                status_code=upstream_response.status_code,
                content=error_body,
            )

        # Tell clients and proxies that this response is an unbuffered event stream.
        return StreamingResponse(
            relay_sse_events(upstream_response, streaming_client),
            status_code=upstream_response.status_code,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Keep regular requests as non-streaming JSON responses.
    payload["stream"] = False

    try:
        # Forward the request to vLLM's OpenAI-compatible chat endpoint.
        response = await request_chat_completion(payload)

    except httpx.TimeoutException:
        # The request took longer than the configured timeout.
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Timed out waiting for a response from vLLM.",
        )

    except httpx.RequestError:
        # The endpoint could not be reached.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="vLLM endpoint is offline.",
        )

    # Avoid exposing internal vLLM server failures to callers.
    if response.status_code >= 500:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The vLLM inference server returned an error.",
        )

    # Forward client/request errors from vLLM, such as an invalid request.
    if not response.is_success:
        try:
            error_body = response.json()
        except ValueError:
            error_body = {"detail": "vLLM rejected the request."}

        return JSONResponse(
            status_code=response.status_code,
            content=error_body,
        )

    # Return vLLM's successful completion as JSON.
    try:
        return response.json()
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="vLLM returned an invalid JSON response.",
        )
