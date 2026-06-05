# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end check that an HTTP status raised inside a Python engine
propagates through the wire transport to the frontend's HTTP response.

A launched worker subprocess raises a duck-typed `.status=415` exception
from ``generate``; the test sends one chat-completions request through a
launched frontend subprocess and asserts the response is 415 — proving
that the boundary fix in
``lib/bindings/python/rust/{engine,backend}.rs`` plus the JSON-shaped
DynamoError message both survive the TCP/etcd request plane.

Coverage of the other status paths (`.code`, plain `ValueError` → 500)
already lives in the in-process ``test_http_server.py`` test — this file
specifically pins down the wire path.
"""

from __future__ import annotations

import logging
from typing import Generator

import pytest
import requests

from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
from tests.utils.port_utils import ServicePorts

logger = logging.getLogger(__name__)

MODEL_NAME = "test-http-status-prop"
ENDPOINT_PATH = "test.http_status_prop.generate"
EXPECTED_STATUS = 415
EXPECTED_MESSAGE = "unsupported-media-via-wire"

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.gpu_0,
]


class _WorkerProcess(ManagedProcess):
    def __init__(self, request, *, frontend_port: int) -> None:
        super().__init__(
            command=["python3", "-m", "tests.frontend.http_status_propagation_worker"],
            health_check_urls=[
                (f"http://localhost:{frontend_port}/v1/models", self._model_listed)
            ],
            timeout=60,
            display_output=True,
            terminate_all_matching_process_names=False,
            straggler_commands=["-m tests.frontend.http_status_propagation_worker"],
            log_dir=f"{request.node.name}_worker",
        )

    @staticmethod
    def _model_listed(response: requests.Response) -> bool:
        try:
            if response.status_code != 200:
                return False
            data = response.json()
        except (ValueError, KeyError):
            return False
        return any(m.get("id") == MODEL_NAME for m in data.get("data", []))


@pytest.fixture(scope="function")
def services(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports: ServicePorts,
) -> Generator[int, None, None]:
    _ = runtime_services_dynamic_ports
    frontend_port = dynamo_dynamic_ports.frontend_port
    with DynamoFrontendProcess(
        request,
        frontend_port=frontend_port,
        extra_args=["--discovery-backend", "etcd", "--request-plane", "tcp"],
        terminate_all_matching_process_names=False,
    ):
        with _WorkerProcess(request, frontend_port=frontend_port):
            yield frontend_port


def test_http_status_propagates_through_wire(services: int) -> None:
    response = requests.post(
        f"http://localhost:{services}/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1,
        },
        timeout=30,
    )
    assert response.status_code == EXPECTED_STATUS, response.text
    assert EXPECTED_MESSAGE in response.text
