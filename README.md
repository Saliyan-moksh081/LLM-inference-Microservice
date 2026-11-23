# LLM Inference Microservice

## Project goal

Build a small Python HTTP service that accepts chat requests, applies request
limits, and forwards valid requests to a vLLM server on the in-house Ubuntu GPU
machine. FastAPI does not load model weights or run GPU inference itself.

## Current deployment context

The provided `nvidia-smi` screenshot shows an NVIDIA RTX A6000 with about
49.1 GiB total GPU memory, with roughly 18.7 GiB and 1.0 GiB allocated by two
existing vLLM engine processes at the time of the snapshot. The GPU is already
serving production workloads. This project must not change or restart those
processes. A separate temporary general-chat model is used for development and
verification, with an explicitly agreed GPU memory budget that leaves headroom
for the existing services. It shares the GPU while running, so it can create
some resource contention; do not treat all remaining VRAM as available.
The general-chat model will be temporary for project development and
verification. After that phase, stop only its dedicated vLLM process so its GPU
memory is released and the two existing services return to their current state.
Do not stop or reconfigure the existing embedding or translation processes.

The currently reported vLLM services are:

| Port | Served model | Current launch memory setting | Role |
| --- | --- | --- | --- |
| `8001` | `google/embeddinggemma-300m` | `--gpu-memory-utilization 0.1` | Embeddings |
| `8002` | `Infomaniak-AI/vllm-translategemma-4b-it` | `--gpu-memory-utilization 0.35` | Translation/generation |

These are existing workloads and are not to be modified by this project. The
temporary general-chat model will have its own reviewed memory budget, process,
and API endpoint; the FastAPI gateway should target that endpoint while it is
running.

## Local development runbook

This setup runs FastAPI on a Mac and a temporary vLLM model on the Ubuntu GPU
server. The vLLM endpoint binds to the server's loopback interface; an SSH
tunnel forwards it to the Mac. Keep the temporary vLLM process and SSH tunnel
running while testing FastAPI.

### 1. Start the temporary general-chat model on Ubuntu

Check that port `8003` is free:

```bash
sudo ss -ltnp 'sport = :8003'
```

In the Ubuntu terminal, start the temporary model using the existing vLLM
virtual environment:

```bash
source /home/manoj/vllm-env/bin/activate
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 8003 \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.1 \
  --enforce-eager
```

The first launch downloads the model into the Hugging Face cache and then loads
it into GPU memory. `VLLM_USE_FLASHINFER_SAMPLER=0` selects vLLM's fallback
sampling path; it avoids the FlashInfer sampling-kernel compilation error found
in this server's current toolchain. The `0.1` setting is a per-instance memory
target, not strict GPU isolation. The current test run logged 1.12 GiB for model
weights and 3.31 GiB for KV cache. Monitor the existing workloads while this
temporary process is running.

Leave this terminal open. vLLM is bound to `127.0.0.1` and listens only on the
Ubuntu host's loopback interface.

### 2. Create an SSH tunnel from the Mac

On Ubuntu, find its IP addresses:

```bash
hostname -I
```

From a separate Mac terminal, use the private IP reachable from the Mac (or the
SSH host/alias you normally use):

```bash
ssh -N -L 8003:127.0.0.1:8003 manoj@<UBUNTU_PRIVATE_IP>
```

Replace `<UBUNTU_PRIVATE_IP>` with the actual address; `RB-A6000-01` may not
resolve through the Mac's DNS. Leave the SSH command running. It forwards the
Mac's `127.0.0.1:8003` to the Ubuntu server's `127.0.0.1:8003` without exposing
the vLLM port on the network.

Verify the tunnel from the Mac in another terminal:

```bash
curl -sS http://127.0.0.1:8003/v1/models | python3 -m json.tool
```

The response should include the model ID `Qwen/Qwen3-0.6B`.

### 3. Install and run FastAPI on the Mac

From the repository root, create/activate the Python environment and install
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The project-root `.env` file is loaded automatically. For a fresh checkout,
create it from the template once:

```bash
cp .env.example .env
```

The local `.env` should contain:

```dotenv
VLLM_BASE_URL=http://127.0.0.1:8003
VLLM_MODEL=Qwen/Qwen3-0.6B
VLLM_TIMEOUT=60
RATE_LIMIT=10/minute
```

`.env` is ignored by Git; `.env.example` is the shareable template. Because the
URL uses the local end of the SSH tunnel, keep the tunnel open. Start FastAPI
without passing model configuration on the command line:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8080
```

The root `main.py` compatibility entry point also allows
`uvicorn app.main:app --reload --host 127.0.0.1 --port 8080`.

Confirm the API process and vLLM backend are ready:

```bash
curl -i http://127.0.0.1:8080/health
curl -i http://127.0.0.1:8080/ready
```

`/health` should return `200` when FastAPI is running. `/ready` should return
`200` while the tunnel and vLLM endpoint are reachable, or `503` when the
backend is offline.

## Proposed architecture

```text
Client
  │ HTTP (chat request)
  ▼
FastAPI microservice
  ├─ Validate request and enforce rate limit
  ├─ Forward request asynchronously
  └─ Return vLLM response, relay token stream, or a clear service error
       │ HTTP on the private network
       ▼
Existing vLLM OpenAI-compatible server
  └─ Existing model and GPU processes (managed separately)
```

FastAPI acts as a small API gateway. vLLM remains responsible for model
loading, tokenization, scheduling, batching, and generation.

## Technology stack

| Area | Choice | Purpose |
| --- | --- | --- |
| Language | Python 3.11+ | Application implementation |
| Web API | FastAPI | Routes, request validation, and API documentation |
| ASGI server | Uvicorn | Run the FastAPI application |
| HTTP client | HTTPX (`AsyncClient`) | Forward requests to vLLM without blocking the event loop |
| Rate limiting | SlowAPI | Apply a configurable per-client request limit |
| Shared rate-limit store | Redis (production, if running multiple app workers/replicas) | Share rate-limit counters across processes |
| Inference runtime | Existing vLLM server | Perform inference on the already allocated GPU |
| Configuration | Environment variables | Configure vLLM URL, model, timeout, and rate limits without hardcoding deployment values |

For local development or a single process, an in-memory rate-limit store can
be used. It is process-local and resets on restart, so use a shared Redis store
when multiple workers or replicas must enforce one common limit. The first
implementation should stay single-process unless the deployment needs more.

## High-level design (HLD)

### Components

1. **Client** sends a chat-completion request to this service.
2. **FastAPI application** validates its shape and applies a rate limit before
   it makes any upstream call.
3. **HTTPX client** forwards the request to the configured vLLM OpenAI-compatible
   chat-completions endpoint, using a bounded timeout.
4. **vLLM** produces the response using the model already hosted on the GPU.
5. **FastAPI application** returns the response to the caller and translates
   upstream connection/time-out failures into clear HTTP errors.

## FastAPI API reference

The service exposes these application routes:

| Method and path | Purpose | Backend required? |
| --- | --- | --- |
| `GET /health` | Liveness check for the FastAPI process | No |
| `GET /ready` | Readiness check against the configured vLLM `/health` endpoint | Yes |
| `POST /v1/chat/completions` | OpenAI-style chat completion; JSON response or SSE when `stream: true` | Yes |
| `GET /docs` | FastAPI Swagger UI | No |
| `GET /redoc` | FastAPI ReDoc API documentation | No |
| `GET /openapi.json` | OpenAPI schema for the FastAPI routes | No |

The gateway currently proxies chat completions only. It does not expose the
existing embedding or translation model endpoints.

### Chat request fields

- `messages` (required): non-empty list of text messages. Supported roles are
  `system`, `user`, and `assistant`; each message has a string `content`.
- `model` (optional): accepted for API compatibility, but the gateway always
  forwards the configured `VLLM_MODEL` value.
- `max_tokens` (optional): defaults to `256`; allowed range is `1`–`1024`.
- `temperature` (optional): defaults to `0.7`; allowed range is `0`–`2`.
- `stream` (optional): defaults to `false`; when `true`, the gateway relays
  vLLM's Server-Sent Events (SSE) stream.

### Example calls

Non-streaming response:

```bash
curl -sS -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64
  }'
```

Stream tokens as SSE (`-N` prevents curl from buffering the stream):

```bash
curl -N -sS -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64,
    "stream": true
  }'
```

### Limits and errors

- Chat requests are limited to `10/minute` per client IP by default; set
  `RATE_LIMIT` in `.env` to change the limit. Exceeded requests receive `429`.
- Invalid request bodies receive FastAPI/Pydantic validation errors (`422`).
- An offline or unconfigured vLLM backend returns `503`; upstream timeouts
  return `504`; upstream server failures return `502`.
- `/health` continues to report FastAPI as alive when vLLM is offline. `/ready`
  reports `503` until the inference backend is reachable again.

The first version includes both non-streaming and streaming chat responses.
FastAPI relays the SSE token stream from vLLM; generation and GPU scheduling
remain the responsibility of the separate vLLM process.

The temporary general-chat vLLM process is needed while exercising the chat
endpoint. Once verification is complete, stopping that process releases its
GPU memory. FastAPI may remain running, but `/ready` will report the inference
backend unavailable and chat requests cannot succeed until an inference
endpoint is started/configured again. Stopping the process does not necessarily
delete downloaded model files from disk; disk-cache cleanup is separate and
should only happen if explicitly wanted.

When the vLLM endpoint is offline, FastAPI remains running: `/health` reports
that the API process is alive, while `/ready` and chat requests return
`503 Service Unavailable` with a clear message that the vLLM endpoint is
offline. The API should not expose an internal traceback or misreport this as
a successful inference response.

## Stop the temporary model after testing

After chat and streaming tests are complete:

1. In the Ubuntu terminal running the temporary `vllm serve` command, press
   `Ctrl+C`. This stops only the Qwen process on port `8003` and releases its
   GPU memory.
2. In the Mac terminal running the SSH tunnel, press `Ctrl+C` to close the
   tunnel.
3. Optionally check `nvidia-smi` and the listening ports. Existing services on
   `8001` and `8002` should remain running.

The downloaded Qwen files remain in the Hugging Face cache on disk. To remove
them and reclaim disk space as well, first stop the model process, then preview
and confirm the cache deletion:

```bash
/home/manoj/vllm-env/bin/hf cache rm model/Qwen/Qwen3-0.6B --dry-run
/home/manoj/vllm-env/bin/hf cache rm model/Qwen/Qwen3-0.6B
```

Cache deletion frees disk space, not GPU memory; stopping the vLLM process is
what releases VRAM. Do not delete other model caches or change the services on
ports `8001` and `8002`.

### Operational boundaries

- The FastAPI app can run on the GPU server or a separate host with private
  network access to vLLM; it does not need GPU access itself.
- The temporary general-chat model must have a deliberate GPU memory cap or
  budget that preserves headroom for existing production vLLM workloads. Do not
  launch a new engine based only on a point-in-time free-memory reading from
  `nvidia-smi`.
- Configure the vLLM base URL and served model name via environment variables.
- Do not expose the vLLM port publicly; expose the FastAPI service through the
  existing network/reverse-proxy setup.
- Apply conservative input/output token limits and an upstream timeout so the
  gateway cannot accidentally send unbounded work to the production engine.
- Avoid logging full prompts or generated content by default.
- Rate-limit by client IP initially. If deployed behind a reverse proxy, only
  trust forwarded client-IP headers from that known proxy. Per-API-key quotas
  can be added if caller authentication is introduced.

## Low-level design (LLD)

### Suggested simple source layout

```text
app/
  __init__.py
  main.py          # FastAPI app and routes
  config.py        # Environment-backed settings
  schemas.py       # Pydantic request/response models or request validation
  inference.py     # Async HTTPX calls to the existing vLLM endpoint
  limiter.py       # Rate-limit configuration and client identity function
main.py            # Compatibility entry point for `uvicorn main:app`
```

Keep the first version small: these are logical responsibilities and can be
combined into fewer files if that makes the initial implementation clearer.

The repository-root `main.py` is a small compatibility entry point, so the
existing `uvicorn main:app` command continues to work. The package entry point
is `uvicorn app.main:app`.

### Request flow: chat completion

1. Client submits `POST /v1/chat/completions` with messages and generation
   options.
2. FastAPI parses and validates the JSON request. Invalid requests receive
   `400`/`422` without reaching vLLM.
3. SlowAPI checks the configured limit for the caller. Exceeded limits receive
   `429 Too Many Requests`.
4. The route forwards the request to vLLM using an asynchronous HTTPX
   client, the configured upstream URL, and a finite timeout.
5. On success, the gateway returns the vLLM JSON result, or relays its SSE
   events to the client when streaming was requested.
6. On upstream timeout or connection failure before the response starts, return
   a clear `504`, `502`, or `503` response rather than hanging or exposing an
   internal traceback. If the upstream fails after streaming has begun, close
   the stream cleanly; an HTTP status can no longer be changed after headers
   have been sent.

### Request flow: readiness check

1. Orchestrator or operator calls `GET /ready`.
2. The app checks the configured vLLM health endpoint with a short timeout.
3. Return success only when the upstream is reachable; otherwise return a
   service-unavailable response. `GET /health` remains a process-liveness
   check and does not depend on vLLM.

### Configuration to provide at implementation/deployment time

```text
VLLM_BASE_URL       # Private URL of the existing vLLM HTTP server
VLLM_MODEL          # Model/served-model identifier expected by vLLM
VLLM_TIMEOUT        # Upstream request timeout in seconds
RATE_LIMIT          # Initial per-client limit, for example requests per minute
REDIS_URL           # Optional for one local process; use for shared production limits
```

The app loads these settings from the project-root `.env` file automatically.
Copy `.env.example` to `.env` and edit the values for your environment. The
local `.env` file is ignored by Git. The current temporary test setup uses the
SSH tunnel URL `http://127.0.0.1:8003` and model ID `Qwen/Qwen3-0.6B`.

For deployment, use the correct private URL/model ID and an approved request
rate; do not guess production values.

## Implementation sequence

1. Inspect current vLLM endpoints and served model IDs; confirm the temporary
   general-chat model, deployment approach, and approved GPU memory budget with
   the service owner.
2. Confirm the vLLM private URL and network access for the FastAPI service.
3. Create the minimal FastAPI app with `/health` and `/ready`.
4. Add validated chat request forwarding to the chosen vLLM endpoint.
5. Add configurable rate limiting and decide whether the app needs Redis-backed
   shared counters.
6. Verify normal and streamed responses, invalid input, rate-limit responses,
   and the offline `503` behavior without restarting or changing existing vLLM
   services.
7. Stop only the temporary general-chat vLLM process and verify its GPU memory
   is released while ports `8001` and `8002` remain available and unchanged.
8. Keep this runbook updated when the API host, model endpoint, or deployment
   setup changes.

## Inspect currently served vLLM models

On the Ubuntu GPU server, first find the vLLM process arguments and listening
ports (these commands only inspect running services):

```bash
ps -eo pid,args | grep -E '[v]llm serve|[v]llm\.entrypoints\.openai\.api_server'
sudo ss -ltnp | grep -E 'python|vllm'
```

For each vLLM port, ask its OpenAI-compatible API which model IDs it serves. If
the server uses the default port `8000`:

```bash
curl -sS http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

Replace `8000` with each port discovered above. If the endpoint requires an API
key, add `-H 'Authorization: Bearer <API_KEY>'` to the `curl` command. The
process arguments help identify configured model paths/options; `/v1/models`
reports the API-visible model IDs. Do not share API keys in chat.

## Out of scope for the first version

- Starting, stopping, or reconfiguring the existing production vLLM services.
- Model loading or GPU-specific inference code in FastAPI.
- Distributed deployment, Kubernetes, queues, or databases.
- Authentication and per-user quotas until the caller/deployment requirements
  are specified.
