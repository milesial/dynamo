# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker for the HTTP-status-propagation e2e test.

Always raises a duck-typed `.status=415` exception from `generate`. The
test sends one chat-completions request and asserts the frontend's HTTP
response is 415, proving that the status survives the Python→Rust
boundary AND the wire transport — not just the in-process pipeline that
``test_http_server.py`` covers.
"""

from __future__ import annotations

import asyncio

import uvloop

from dynamo.llm import ModelInput, ModelType, register_model
from dynamo.runtime import DistributedRuntime
from tests.frontend.test_http_status_propagation import (
    ENDPOINT_PATH,
    EXPECTED_MESSAGE,
    EXPECTED_STATUS,
    MODEL_NAME,
)
from tests.utils.constants import QWEN


class _StatusLikeError(Exception):
    """Duck-typed `.status` + `.message` — same shape as
    `dynamo.common.http.HttpStatusError`."""

    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


async def generate(request, context):
    raise _StatusLikeError(status=EXPECTED_STATUS, message=EXPECTED_MESSAGE)
    yield  # pragma: no cover — unreachable; declares this as an async generator


async def main():
    runtime = DistributedRuntime(asyncio.get_running_loop(), "etcd", "tcp")
    endpoint = runtime.endpoint(ENDPOINT_PATH)
    await register_model(
        ModelInput.Tokens,
        ModelType.Chat,
        endpoint,
        QWEN,
        model_name=MODEL_NAME,
    )
    await endpoint.serve_endpoint(generate)


if __name__ == "__main__":
    uvloop.run(main())
