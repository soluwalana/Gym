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

"""Sandbox provider that asks a trusted episode broker instead of a backend SDK.

A GRPO job running a user-authored environment executes inside an isolated job sandbox that holds
**no** backend credential. Agents in that sandbox still need per-episode sandboxes, so they ask a
broker (NeMo-RL's ``SandboxEpisodeBrokerActor``) over HTTP. This provider is that HTTP client.

It deliberately does **no** security work. The broker is the trust boundary: it re-derives every
field from an allowlist, applies the image policy and resource caps, and owns the egress policy.
Anything this client checked could be skipped by a caller that simply did not use this client, so
checking here would buy a false sense of enforcement. The one thing it does do is *fail early and
legibly* -- rejecting a request the broker is certain to refuse, before a round trip, so a
misconfigured environment fails at ``start()`` with an actionable message rather than mid-rollout
with an HTTP status. A hostile caller that skips those checks still meets the broker's refusal, and
the broker still audits it.

Transport is aiohttp, matching the rest of NeMo-Gym: httpx's connection pooling is quadratic in
pool size and hangs at the concurrency rollouts reach (see the note in ``pyproject.toml``).
"""

import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any

import aiohttp
from pydantic import ValidationError

from nemo_gym.sandbox.broker.wire import (
    BROKER_AUTH_HEADER,
    BROKER_PROTOCOL_VERSION,
    EPISODE_EXEC_PATH,
    EPISODE_FILES_PATH,
    EPISODE_PATH,
    EPISODES_PATH,
    HEALTH_PATH,
    BrokerErrorCode,
    BrokerErrorResponse,
    BrokerHealthResponse,
    EpisodeCreateRequest,
    EpisodeCreateResponse,
    EpisodeExecRequest,
    EpisodeExecResponse,
    EpisodeFileDownloadResponse,
    EpisodeFileUploadRequest,
    EpisodeResources,
    EpisodeStatusResponse,
)
from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxExecResult,
    SandboxHandle,
    SandboxResources,
    SandboxSpec,
    SandboxStatus,
)


LOGGER = logging.getLogger(__name__)

DEFAULT_REQUEST_TIMEOUT_S = 300.0


class BrokerError(RuntimeError):
    """A broker rejection, or a request this client knows the broker would reject.

    ``code`` is the machine-readable :class:`BrokerErrorCode` when the broker supplied one, and
    ``None`` when the failure was detected locally or the response was not a broker error body.
    """

    def __init__(
        self,
        message: str,
        *,
        code: BrokerErrorCode | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class BrokerCreateError(BrokerError, SandboxCreateError):
    """A create-time broker failure.

    Also a :class:`SandboxCreateError` so environments that already catch provider create failures
    keep catching this one -- brokering an existing call site should not change which excepts fire.
    """


def _error_from_response(status: int, body: bytes, method: str, path: str) -> BrokerError:
    """Turn a non-2xx broker response into an exception, preserving its error code when present."""
    detail = body.decode("utf-8", errors="replace").strip()
    code: BrokerErrorCode | None = None
    try:
        parsed = BrokerErrorResponse.model_validate_json(body)
    except (ValidationError, ValueError):
        # Not a broker error body: a proxy, a crash, or an auth rejection from something in front
        # of the broker. Keep the raw text -- it is the only diagnostic available.
        message = f"broker {method} {path} failed with HTTP {status}: {detail or '<empty body>'}"
    else:
        code = parsed.code
        message = f"broker {method} {path} refused ({parsed.code.value}): {parsed.error}"
    return BrokerError(message, code=code, status_code=status)


class BrokerProvider:
    """``SandboxProvider`` backed by the episode broker's HTTP API.

    Holds only the job-scoped broker token, which is readable by environment code by design and
    identifies the owning job rather than granting anything. One instance belongs to one event
    loop: it keeps a single ``aiohttp`` session, created lazily on first use so the provider can be
    constructed outside a running loop (which is what the synchronous ``Sandbox`` wrapper does).
    """

    name = "broker"

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    ) -> None:
        if not base_url:
            raise ValueError("BrokerProvider requires a non-empty base_url")
        if not token:
            raise ValueError("BrokerProvider requires a non-empty token")
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = aiohttp.ClientTimeout(total=request_timeout_s)
        self._session: aiohttp.ClientSession | None = None
        self._protocol_checked = False

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self._timeout,
                headers={BROKER_AUTH_HEADER: self._token},
            )
        return self._session

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        session = await self._get_session()
        url = f"{self._base_url}{path}"
        try:
            async with session.request(method, url, json=json_body, params=params) as response:
                body = await response.read()
                status = response.status
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise BrokerError(f"broker {method} {path} could not be reached: {e}") from e

        if status >= 400:
            raise _error_from_response(status, body, method, path)
        if not body:
            return {}
        try:
            decoded = json.loads(body)
        except ValueError as e:
            raise BrokerError(f"broker {method} {path} returned a non-JSON body") from e
        if not isinstance(decoded, dict):
            raise BrokerError(f"broker {method} {path} returned {type(decoded).__name__}, expected an object")
        return decoded

    async def _check_protocol(self) -> None:
        """Confirm the broker speaks the wire version this client was built against.

        A job-sandbox image and the broker's NeMo-RL image are built from different revisions, so
        they can disagree. Checking once per provider turns that into one clear failure at
        ``start()`` instead of a field quietly going unread for a whole rollout.
        """
        if self._protocol_checked:
            return
        health = BrokerHealthResponse.model_validate(await self._request("GET", HEALTH_PATH))
        if health.protocol_version != BROKER_PROTOCOL_VERSION:
            raise BrokerError(
                f"episode broker speaks protocol version {health.protocol_version!r} but this "
                f"NeMo-Gym expects {BROKER_PROTOCOL_VERSION!r}. The job-sandbox image and the "
                f"broker's NeMo-RL image were built from incompatible revisions."
            )
        self._protocol_checked = True

    def _build_create_request(self, spec: SandboxSpec) -> EpisodeCreateRequest:
        """Render a ``SandboxSpec`` onto the wire, refusing what the broker cannot accept."""
        if not spec.image:
            raise BrokerCreateError(
                "brokered sandboxes require an explicit image: SandboxSpec.image is unset, and the "
                "broker evaluates its approved-image policy against that field. Set `image` on the "
                "agent's sandbox_spec."
            )
        if spec.provider_options:
            # Refused here rather than forwarded so the failure names the offending keys and costs
            # no round trip. The broker refuses provider_options unconditionally too, and audits
            # the attempt -- this check is for the honest misconfiguration, not the hostile one.
            raise BrokerCreateError(
                "provider_options are not available to brokered sandboxes; the broker builds its "
                "create request from named fields only. Remove: "
                f"{', '.join(sorted(spec.provider_options))}."
            )

        resources = (
            spec.resources
            if isinstance(spec.resources, SandboxResources)
            else SandboxResources.from_mapping(spec.resources)
        )
        try:
            return EpisodeCreateRequest(
                image=spec.image,
                ttl_s=spec.ttl_s,
                ready_timeout_s=spec.ready_timeout_s,
                workdir=spec.workdir,
                env=dict(spec.env),
                metadata=dict(spec.metadata),
                resources=EpisodeResources(
                    cpu=resources.cpu,
                    memory_mib=resources.memory_mib,
                    disk_gib=resources.disk_gib,
                    gpu=resources.gpu,
                    gpu_type=resources.gpu_type,
                ),
                entrypoint=list(spec.entrypoint) if spec.entrypoint is not None else None,
                # ``spec.files`` is deliberately not sent, even though the wire carries
                # ``files_b64`` and the broker stages it. No provider honours ``spec.files`` in
                # ``create``; ``AsyncSandbox.start`` uploads them afterwards for every provider
                # alike. Sending them here would write each file twice and make this the one
                # provider whose ``create`` has different semantics from the rest. Per-file uploads
                # also each get their own transfer budget, where one create body shares a single
                # cap across the whole set.
            )
        except ValidationError as e:
            raise BrokerCreateError(f"sandbox spec cannot be expressed as a broker episode: {e}") from e

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """Create one episode through the broker and return a handle to it."""
        request = self._build_create_request(spec)
        try:
            await self._check_protocol()
            created = EpisodeCreateResponse.model_validate(
                await self._request("POST", EPISODES_PATH, json_body=request.model_dump(mode="json"))
            )
        except BrokerCreateError:
            raise
        except BrokerError as e:
            raise BrokerCreateError(str(e), code=e.code, status_code=e.status_code) from e
        except ValidationError as e:
            raise BrokerCreateError(f"broker returned an unreadable create response: {e}") from e
        return SandboxHandle(
            sandbox_id=created.episode_id,
            provider_name=self.name,
            raw={"episode_id": created.episode_id, "status": created.status},
        )

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        """Run one command inside an episode."""
        try:
            request = EpisodeExecRequest(
                command=command,
                cwd=cwd,
                env=env,
                user=user,
                timeout_s=timeout_s,
            )
        except ValidationError as e:
            raise BrokerError(f"exec request cannot be expressed as a broker call: {e}") from e
        path = EPISODE_EXEC_PATH.format(episode_id=handle.sandbox_id)
        result = EpisodeExecResponse.model_validate(
            await self._request("POST", path, json_body=request.model_dump(mode="json"))
        )
        return SandboxExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.return_code,
            error_type=result.error_type,
        )

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        """Write one local file into an episode."""
        content = Path(source_path).read_bytes()
        try:
            request = EpisodeFileUploadRequest(
                path=target_path,
                content_b64=base64.b64encode(content).decode("ascii"),
            )
        except ValidationError as e:
            raise BrokerError(f"upload to {target_path!r} is not a valid broker request: {e}") from e
        await self._request(
            "PUT",
            EPISODE_FILES_PATH.format(episode_id=handle.sandbox_id),
            json_body=request.model_dump(mode="json"),
        )

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        """Read one file out of an episode onto the local filesystem."""
        response = EpisodeFileDownloadResponse.model_validate(
            await self._request(
                "GET",
                EPISODE_FILES_PATH.format(episode_id=handle.sandbox_id),
                params={"path": source_path},
            )
        )
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(response.content_b64))

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        """Report the episode's lifecycle status.

        An episode the broker no longer knows about reports ``STOPPED``: from a caller's view a
        forgotten episode and a terminated one are the same thing, and both mean "do not use it".
        """
        try:
            response = await self._request("GET", EPISODE_PATH.format(episode_id=handle.sandbox_id))
        except BrokerError as e:
            if e.code is BrokerErrorCode.EPISODE_NOT_FOUND:
                return SandboxStatus.STOPPED
            raise
        return EpisodeStatusResponse.model_validate(response).status

    async def close(self, handle: SandboxHandle) -> None:
        """Terminate the episode. Idempotent, so repeated or late teardown is not an error."""
        try:
            await self._request("DELETE", EPISODE_PATH.format(episode_id=handle.sandbox_id))
        except BrokerError as e:
            if e.code is BrokerErrorCode.EPISODE_NOT_FOUND:
                return
            raise

    async def aclose(self) -> None:
        """Close the HTTP session. Episodes are not touched; ``close`` ends those."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
