"""Environment-backed configuration for the inference gateway."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load the project-root .env file regardless of the directory Uvicorn is started from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# Base URL for the temporary or configured vLLM server.
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "").rstrip("/")

# Model ID reported by the target server's /v1/models endpoint.
VLLM_MODEL = os.getenv("VLLM_MODEL", "").strip()

# Timeout for requests sent to the vLLM server.
VLLM_TIMEOUT = float(os.getenv("VLLM_TIMEOUT", "60"))

# Per-client request limit, adjustable without changing application code.
RATE_LIMIT = os.getenv("RATE_LIMIT", "10/minute")
