# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the brokered sandbox path.

The stub broker below serves the real wire contract over a real socket, so these exercise the HTTP
client end to end rather than a mocked session. What it does not check is the broker's own
behaviour -- that lives in NeMo-RL, and the contract module is what keeps the two honest.
"""

import asyncio
import base64
from typing import Any

import pytest
from aiohttp import web

from nemo_gym.sandbox import AsyncSandbox, Sandbox, SandboxCreateError, SandboxSpec, SandboxStatus
from nemo_gym.sandbox.broker.wire import (
    BROKER_AUTH_HEADER,
    BROKER_PROTOCOL_VERSION,
    BROKER_TOKEN_ENV,
    BROKER_URL_ENV,
    BrokerErrorCode,
)
from nemo_gym.sandbox.providers.broker import BrokerCreateError, BrokerError, BrokerProvider


pytestmark = pytest.mark.sandbox

TOKEN = "job-scoped-token"  # pragma: allowlist secret


class StubBroker:
    """Minimal in-process stand-in for ``SandboxEpisodeBrokerActor``'s HTTP surface."""

    def __init__(self) -> None:
        self.protocol_version = BROKER_PROTOCOL_VERSION
        self.requests: list[tuple[str, str]] = []
        self.create_bodies: list[dict[str, Any]] = []
        self.uploads: list[dict[str, str]] = []
        self.auth_headers: list[str | None] = []
        self.files: dict[str, bytes] = {}
        self.episodes: set[str] = set()
        self.next_episode = 0
        # (status, code, message) applied to the next matching route, then cleared.
        self.fail: dict[str, tuple[int, str | None, str]] = {}
        self._runner: web.AppRunner | None = None
        self.url = ""

    def _check(self, request: web.Request) -> web.Response | None:
        self.requests.append((request.method, request.path))
        self.auth_headers.append(request.headers.get(BROKER_AUTH_HEADER))
        if request.headers.get(BROKER_AUTH_HEADER) != TOKEN:
            return web.json_response({"error": "bad token", "code": BrokerErrorCode.UNAUTHORIZED.value}, status=401)
        planned = self.fail.pop(request.method + " " + request.match_info.route.resource.canonical, None)
        if planned is not None:
            status, code, message = planned
            if code is None:
                return web.Response(status=status, text=message)
            return web.json_response({"error": message, "code": code}, status=status)
        return None

    async def health(self, request: web.Request) -> web.Response:
        return self._check(request) or web.json_response(
            {"status": "ok", "job_id": "job-test", "protocol_version": self.protocol_version}
        )

    async def create(self, request: web.Request) -> web.Response:
        refusal = self._check(request)
        if refusal is not None:
            return refusal
        self.create_bodies.append(await request.json())
        self.next_episode += 1
        episode_id = f"ep_{self.next_episode}"
        self.episodes.add(episode_id)
        return web.json_response({"episode_id": episode_id, "status": "running"}, status=201)

    async def status(self, request: web.Request) -> web.Response:
        refusal = self._check(request)
        if refusal is not None:
            return refusal
        if request.match_info["episode_id"] not in self.episodes:
            return self._not_found()
        return web.json_response({"status": "running"})

    async def exec_command(self, request: web.Request) -> web.Response:
        refusal = self._check(request)
        if refusal is not None:
            return refusal
        body = await request.json()
        return web.json_response(
            {"stdout": f"ran {body['command']} as {body['user']}", "stderr": "", "return_code": 0}
        )

    async def upload(self, request: web.Request) -> web.Response:
        refusal = self._check(request)
        if refusal is not None:
            return refusal
        body = await request.json()
        self.uploads.append(body)
        self.files[body["path"]] = base64.b64decode(body["content_b64"])
        return web.Response(status=204)

    async def download(self, request: web.Request) -> web.Response:
        refusal = self._check(request)
        if refusal is not None:
            return refusal
        content = self.files.get(request.query["path"])
        if content is None:
            return self._not_found()
        return web.json_response({"content_b64": base64.b64encode(content).decode()})

    async def close_episode(self, request: web.Request) -> web.Response:
        refusal = self._check(request)
        if refusal is not None:
            return refusal
        if request.match_info["episode_id"] not in self.episodes:
            return self._not_found()
        self.episodes.discard(request.match_info["episode_id"])
        return web.json_response({"closed": True})

    @staticmethod
    def _not_found() -> web.Response:
        return web.json_response(
            {"error": "unknown episode", "code": BrokerErrorCode.EPISODE_NOT_FOUND.value}, status=404
        )

    async def start(self) -> "StubBroker":
        app = web.Application()
        app.router.add_get("/health", self.health)
        app.router.add_post("/episodes", self.create)
        app.router.add_get("/episodes/{episode_id}", self.status)
        app.router.add_delete("/episodes/{episode_id}", self.close_episode)
        app.router.add_post("/episodes/{episode_id}/exec", self.exec_command)
        app.router.add_put("/episodes/{episode_id}/files", self.upload)
        app.router.add_get("/episodes/{episode_id}/files", self.download)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{self._runner.addresses[0][1]}"
        return self

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    def paths(self, method: str) -> list[str]:
        return [path for verb, path in self.requests if verb == method]


@pytest.fixture
async def broker():
    stub = await StubBroker().start()
    try:
        yield stub
    finally:
        await stub.stop()


@pytest.fixture
async def provider(broker):
    instance = BrokerProvider(base_url=broker.url, token=TOKEN)
    try:
        yield instance
    finally:
        await instance.aclose()


@pytest.fixture
def brokered_env(broker, monkeypatch):
    monkeypatch.setenv(BROKER_URL_ENV, broker.url)
    monkeypatch.setenv(BROKER_TOKEN_ENV, TOKEN)
    return broker


# ── provider: create ──────────────────────────────────────────────────


async def test_create_sends_the_spec_and_returns_a_handle(broker, provider):
    handle = await provider.create(
        SandboxSpec(
            image="python:3.12-slim",
            ttl_s=600,
            workdir="/workspace",
            env={"FOO": "bar"},
            metadata={"harness": "test"},
            resources={"cpu": 2, "memory_mib": 4096},
            entrypoint=["sleep", "infinity"],
        )
    )

    assert handle.sandbox_id == "ep_1"
    assert handle.provider_name == "broker"
    body = broker.create_bodies[0]
    assert body["image"] == "python:3.12-slim"
    assert body["ttl_s"] == 600
    assert body["workdir"] == "/workspace"
    assert body["env"] == {"FOO": "bar"}
    assert body["metadata"] == {"harness": "test"}
    assert body["resources"]["cpu"] == 2
    assert body["resources"]["memory_mib"] == 4096
    assert body["entrypoint"] == ["sleep", "infinity"]


async def test_every_request_carries_the_job_token(broker, provider):
    await provider.create(SandboxSpec(image="python:3.12-slim"))
    assert broker.auth_headers
    assert set(broker.auth_headers) == {TOKEN}


async def test_create_checks_the_protocol_version_once(broker, provider):
    await provider.create(SandboxSpec(image="python:3.12-slim"))
    await provider.create(SandboxSpec(image="python:3.12-slim"))
    # Probed on the first create and remembered, not re-probed per episode.
    assert broker.paths("GET").count("/health") == 1


async def test_protocol_mismatch_fails_before_creating_anything(broker, provider):
    broker.protocol_version = "999"
    with pytest.raises(BrokerError, match="protocol version"):
        await provider.create(SandboxSpec(image="python:3.12-slim"))
    assert broker.create_bodies == []


async def test_provider_options_are_refused_without_a_round_trip(broker, provider):
    with pytest.raises(BrokerCreateError, match="platform"):
        await provider.create(SandboxSpec(image="python:3.12-slim", provider_options={"platform": {"os": "linux"}}))
    assert broker.requests == []


async def test_missing_image_is_refused_without_a_round_trip(broker, provider):
    with pytest.raises(BrokerCreateError, match="explicit image"):
        await provider.create(SandboxSpec(ttl_s=600))
    assert broker.requests == []


async def test_create_does_not_send_files(broker, provider):
    """``AsyncSandbox.start`` stages files for every provider; sending them here would double-write."""
    await provider.create(SandboxSpec(image="python:3.12-slim", files={"/tmp/a.txt": "hello"}))
    assert broker.create_bodies[0]["files_b64"] == {}


async def test_create_failure_is_also_a_sandbox_create_error(broker, provider):
    broker.fail["POST /episodes"] = (403, BrokerErrorCode.IMAGE_NOT_APPROVED.value, "image not approved")
    with pytest.raises(SandboxCreateError) as excinfo:
        await provider.create(SandboxSpec(image="docker.io/attacker/evil"))
    assert isinstance(excinfo.value, BrokerCreateError)
    assert excinfo.value.code is BrokerErrorCode.IMAGE_NOT_APPROVED
    assert excinfo.value.status_code == 403
    assert "image not approved" in str(excinfo.value)


async def test_spec_the_wire_cannot_express_fails_locally(broker, provider):
    with pytest.raises(BrokerCreateError, match="cannot be expressed"):
        await provider.create(SandboxSpec(image="python:3.12-slim", workdir="relative/path"))
    assert broker.requests == []


# ── provider: operations ──────────────────────────────────────────────


async def test_exec_round_trip(provider):
    handle = await provider.create(SandboxSpec(image="python:3.12-slim"))
    result = await provider.exec(handle, "whoami", user="root", timeout_s=30)
    assert result.return_code == 0
    assert result.stdout == "ran whoami as root"


async def test_file_round_trip(broker, provider, tmp_path):
    handle = await provider.create(SandboxSpec(image="python:3.12-slim"))
    source = tmp_path / "payload.bin"
    source.write_bytes(b"\x00\x01binary")
    await provider.upload_file(handle, source, "/workspace/payload.bin")
    assert broker.files["/workspace/payload.bin"] == b"\x00\x01binary"

    target = tmp_path / "nested" / "roundtrip.bin"
    await provider.download_file(handle, "/workspace/payload.bin", target)
    assert target.read_bytes() == b"\x00\x01binary"


async def test_status_reports_running(provider):
    handle = await provider.create(SandboxSpec(image="python:3.12-slim"))
    assert await provider.status(handle) is SandboxStatus.RUNNING


async def test_status_of_a_forgotten_episode_is_stopped(provider):
    handle = await provider.create(SandboxSpec(image="python:3.12-slim"))
    await provider.close(handle)
    assert await provider.status(handle) is SandboxStatus.STOPPED


async def test_close_is_idempotent(provider):
    handle = await provider.create(SandboxSpec(image="python:3.12-slim"))
    await provider.close(handle)
    await provider.close(handle)


async def test_close_surfaces_failures_that_are_not_a_missing_episode(broker, provider):
    handle = await provider.create(SandboxSpec(image="python:3.12-slim"))
    broker.fail["DELETE /episodes/{episode_id}"] = (502, BrokerErrorCode.BACKEND_ERROR.value, "backend down")
    with pytest.raises(BrokerError, match="backend down"):
        await provider.close(handle)


async def test_a_non_broker_error_body_still_raises_with_its_text(broker, provider):
    broker.fail["POST /episodes"] = (502, None, "<html>gateway</html>")
    with pytest.raises(BrokerCreateError, match="gateway"):
        await provider.create(SandboxSpec(image="python:3.12-slim"))


async def test_a_wrong_token_is_rejected(broker):
    provider = BrokerProvider(base_url=broker.url, token="wrong")  # pragma: allowlist secret
    try:
        with pytest.raises(BrokerError, match="unauthorized"):
            await provider.create(SandboxSpec(image="python:3.12-slim"))
    finally:
        await provider.aclose()


async def test_an_unreachable_broker_reports_that_rather_than_hanging():
    provider = BrokerProvider(base_url="http://127.0.0.1:1", token=TOKEN, request_timeout_s=5.0)
    try:
        with pytest.raises(BrokerError, match="could not be reached"):
            await provider.create(SandboxSpec(image="python:3.12-slim"))
    finally:
        await provider.aclose()


def test_construction_requires_a_url_and_token():
    with pytest.raises(ValueError, match="base_url"):
        BrokerProvider(base_url="", token=TOKEN)
    with pytest.raises(ValueError, match="token"):
        BrokerProvider(base_url="http://broker", token="")


# ── routing: existing call sites, unchanged ───────────────────────────


async def test_brokered_mode_overrides_a_configured_provider(brokered_env):
    sandbox = AsyncSandbox({"docker": {}}, SandboxSpec(image="python:3.12-slim"))
    async with await sandbox.start():
        assert isinstance(sandbox._provider, BrokerProvider)
    assert brokered_env.create_bodies[0]["image"] == "python:3.12-slim"


async def test_brokered_mode_overrides_a_provider_instance(brokered_env):
    class HandRolledProvider:
        name = "hand-rolled"

    sandbox = AsyncSandbox(HandRolledProvider(), SandboxSpec(image="python:3.12-slim"))
    assert isinstance(sandbox._provider, BrokerProvider)


async def test_brokered_mode_ignores_an_unknown_provider_name(brokered_env):
    # A config naming a provider this NeMo-Gym has never heard of still runs: in brokered mode the
    # name is not consulted at all, so an environment cannot be broken by naming the wrong one.
    sandbox = AsyncSandbox({"not-a-real-provider": {}}, SandboxSpec(image="python:3.12-slim"))
    assert isinstance(sandbox._provider, BrokerProvider)


def test_a_url_without_a_token_is_a_configuration_error(monkeypatch):
    monkeypatch.setenv(BROKER_URL_ENV, "http://broker")
    monkeypatch.setenv(BROKER_TOKEN_ENV, "")
    with pytest.raises(ValueError, match=BROKER_TOKEN_ENV):
        AsyncSandbox({"docker": {}})


def test_without_the_env_the_normal_provider_path_is_used(monkeypatch):
    monkeypatch.delenv(BROKER_URL_ENV, raising=False)
    with pytest.raises(ValueError, match="Unknown sandbox provider"):
        AsyncSandbox({"not-a-real-provider": {}})


async def test_full_async_lifecycle_through_the_broker(brokered_env, tmp_path):
    spec = SandboxSpec(image="python:3.12-slim", workdir="/workspace", files={"/workspace/seed.txt": "seeded"})
    async with await AsyncSandbox({"opensandbox": {}}, spec).start() as sandbox:
        assert await sandbox.status() is SandboxStatus.RUNNING

        result = await sandbox.exec("whoami", user="root")
        assert result.stdout == "ran whoami as root"

        local = tmp_path / "upload.txt"
        local.write_text("uploaded")
        await sandbox.upload(local, "/workspace/upload.txt")

        target = tmp_path / "downloaded.txt"
        await sandbox.download("/workspace/seed.txt", target)
        assert target.read_text() == "seeded"

    # Staged exactly once each, through the same PUT path every provider uses.
    assert [upload["path"] for upload in brokered_env.uploads] == [
        "/workspace/seed.txt",
        "/workspace/upload.txt",
    ]
    assert brokered_env.episodes == set()


async def test_sync_sandbox_works_through_the_broker(brokered_env):
    """The blocking wrapper drives the broker from its own loop thread.

    Run in a worker thread rather than as a sync test: the stub broker serves from this test's
    event loop, which has to stay running while ``Sandbox`` blocks on its own.
    """

    def use_sandbox() -> int:
        with Sandbox({"apptainer": {}}).start(SandboxSpec(image="python:3.12-slim")) as sandbox:
            return sandbox.exec("whoami", user="root").return_code

    assert await asyncio.to_thread(use_sandbox) == 0
    assert brokered_env.episodes == set()


async def test_a_failed_start_closes_the_episode(brokered_env, monkeypatch):
    """Staging failure must not leave the episode behind; that is existing ``start`` behaviour."""
    brokered_env.fail["PUT /episodes/{episode_id}/files"] = (
        413,
        BrokerErrorCode.PAYLOAD_TOO_LARGE.value,
        "too big",
    )
    spec = SandboxSpec(image="python:3.12-slim", files={"/workspace/seed.txt": "seeded"})
    with pytest.raises(BrokerError, match="too big"):
        await AsyncSandbox({"docker": {}}, spec).start()
    assert brokered_env.episodes == set()


def test_broker_is_a_registered_provider_name():
    from nemo_gym.sandbox import create_provider, list_providers

    assert "broker" in list_providers()
    assert isinstance(create_provider({"broker": {"base_url": "http://broker", "token": TOKEN}}), BrokerProvider)


def test_construction_does_not_require_a_running_loop():
    """The sync ``Sandbox`` builds its provider outside the loop, so the session must be lazy."""
    assert BrokerProvider(base_url="http://broker", token=TOKEN)._session is None
