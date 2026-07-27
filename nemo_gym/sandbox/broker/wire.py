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

"""Frozen HTTP contract between an untrusted job sandbox and the trusted episode broker.

A GRPO job that runs a user-authored environment executes the whole NeMo-Gym stack inside an
isolated job sandbox. Agents in that sandbox still need per-episode sandboxes (SWE-style grading),
but the job sandbox holds **no** backend credential: it asks a trusted broker
(``SandboxEpisodeBrokerActor``, hosted by NeMo-RL on the training leader) over HTTP instead.

This module is the single source of truth for that wire format. It lives in NeMo-Gym because both
sides need it and NeMo-Gym is the dependency NeMo-RL already has, never the reverse.

Trust posture
-------------
* The caller is **untrusted**. The job-scoped token in :data:`BROKER_AUTH_HEADER` is readable by
  user environment code by design, so it only identifies the owning job -- it is not a capability.
  Every operation reachable with it must be escalation-free.
* Validation here is enforced **server-side** (the broker parses requests with these models), so it
  is real. It is still only the first layer: the broker applies its own policy (image allowlist,
  resource caps, platform-owned mounts) on top and never forwards a field it did not read.
* Request models are ``extra="forbid"``: an unrecognized field is a hard error rather than
  something that might reach a backend SDK. Response models are permissive on read so an older
  client keeps working against a newer broker.
"""

import base64
import binascii
from enum import Enum
from pathlib import PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nemo_gym.sandbox.providers.base import SandboxStatus


# Bump when a change is not backward compatible for an already-deployed client. Reported by
# ``GET /health`` so a job-sandbox image built against a different NeMo-Gym revision than the
# broker's NeMo-RL image fails loudly instead of drifting.
BROKER_PROTOCOL_VERSION = "1"

BROKER_AUTH_HEADER = "OPENSANDBOX-EPISODE-BROKER-AUTH"

HEALTH_PATH = "/health"
EPISODES_PATH = "/episodes"
EPISODE_PATH = "/episodes/{episode_id}"
EPISODE_EXEC_PATH = "/episodes/{episode_id}/exec"
EPISODE_FILES_PATH = "/episodes/{episode_id}/files"


class BrokerErrorCode(str, Enum):
    """Machine-readable reason for a broker rejection.

    Clients map these onto NeMo-Gym sandbox exceptions so an environment fails fast with an
    actionable message instead of a generic HTTP error mid-rollout.
    """

    UNAUTHORIZED = "unauthorized"
    INVALID_REQUEST = "invalid_request"
    FIELD_NOT_ALLOWED = "field_not_allowed"
    IMAGE_NOT_APPROVED = "image_not_approved"
    EPISODE_NOT_FOUND = "episode_not_found"
    QUOTA_EXCEEDED = "quota_exceeded"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    UNSUPPORTED_OPERATION = "unsupported_operation"
    BACKEND_ERROR = "backend_error"


def validate_base64(value: str) -> str:
    """Return ``value`` if it is standard base64, else raise ``ValueError``."""
    try:
        base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError("content must be standard base64") from e
    return value


def validate_absolute_path(value: str) -> str:
    """Return ``value`` if it is an absolute POSIX path with no traversal, else raise ``ValueError``.

    Traversal inside an episode is not a cluster-boundary escape (the episode is itself isolated),
    but the broker refuses to relay ambiguous paths so a backend can never resolve one differently
    than the caller intended.
    """
    if "\x00" in value:
        raise ValueError("path must not contain NUL bytes")
    if not value.startswith("/"):
        raise ValueError("path must be absolute")
    if any(part == ".." for part in PurePosixPath(value).parts):
        raise ValueError("path must not contain '..' segments")
    return value


class EpisodeResources(BaseModel):
    """Resource request for one episode; mirrors :class:`nemo_gym.sandbox.SandboxResources`.

    Typed rather than a free-form mapping so unknown keys cannot ride through to a backend SDK and
    so the broker has a fixed set of fields to cap.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    cpu: float | None = Field(default=None, gt=0)
    memory_mib: int | None = Field(default=None, gt=0)
    disk_gib: int | None = Field(default=None, gt=0)
    gpu: int | None = Field(default=None, ge=0)
    gpu_type: str | None = None


class EpisodeCreateRequest(BaseModel):
    """``POST /episodes`` -- the job sandbox asking for one episode sandbox.

    ``image`` is required even though :class:`nemo_gym.sandbox.SandboxSpec` allows ``None``: it is
    what the broker's approved-image policy is evaluated against, and a request the broker cannot
    evaluate is a request it must not forward.

    ``ttl_s`` left unset means "use the broker's default"; lifetime policy belongs to the trusted
    side. ``provider_options`` is accepted so a client never has to guess what is permitted, but
    the broker rejects any key it has not explicitly allowed rather than silently dropping it.
    """

    model_config = ConfigDict(extra="forbid")

    image: str = Field(min_length=1)
    ttl_s: float | None = Field(default=None, gt=0)
    ready_timeout_s: float | None = Field(default=None, gt=0)
    workdir: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)
    resources: EpisodeResources = Field(default_factory=EpisodeResources)
    entrypoint: list[str] | None = None
    files_b64: dict[str, str] = Field(default_factory=dict)
    provider_options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("workdir")
    @classmethod
    def _check_workdir(cls, value: str | None) -> str | None:
        return value if value is None else validate_absolute_path(value)

    @field_validator("files_b64")
    @classmethod
    def _check_files(cls, value: dict[str, str]) -> dict[str, str]:
        for path, content in value.items():
            validate_absolute_path(path)
            validate_base64(content)
        return value


class EpisodeCreateResponse(BaseModel):
    """``POST /episodes`` result. ``episode_id`` is a broker-owned opaque handle."""

    model_config = ConfigDict(frozen=True)

    episode_id: str
    status: SandboxStatus = SandboxStatus.RUNNING


class EpisodeStatusResponse(BaseModel):
    """``GET /episodes/{episode_id}`` result."""

    model_config = ConfigDict(frozen=True)

    status: SandboxStatus


class EpisodeExecRequest(BaseModel):
    """``POST /episodes/{episode_id}/exec`` -- run one command inside an episode.

    ``user`` defaults to ``None`` (the backend's own default) rather than ``"root"``: privileged
    execution is something a call site opts into explicitly. Backends that cannot honour a
    requested user reject the call with ``UNSUPPORTED_OPERATION`` instead of quietly downgrading.
    """

    model_config = ConfigDict(extra="forbid")

    command: str = Field(min_length=1)
    cwd: str | None = None
    env: dict[str, str] | None = None
    user: str | int | None = None
    timeout_s: float | None = Field(default=None, gt=0)

    @field_validator("cwd")
    @classmethod
    def _check_cwd(cls, value: str | None) -> str | None:
        return value if value is None else validate_absolute_path(value)


class EpisodeExecResponse(BaseModel):
    """``POST /episodes/{episode_id}/exec`` result; mirrors ``SandboxExecResult``."""

    model_config = ConfigDict(frozen=True)

    stdout: str | None
    stderr: str | None
    return_code: int
    error_type: str | None = None


class EpisodeFileUploadRequest(BaseModel):
    """``PUT /episodes/{episode_id}/files`` -- write one file into an episode.

    A successful upload answers ``204 No Content``; there is no response body to parse.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    content_b64: str

    @field_validator("path")
    @classmethod
    def _check_path(cls, value: str) -> str:
        return validate_absolute_path(value)

    @field_validator("content_b64")
    @classmethod
    def _check_content(cls, value: str) -> str:
        return validate_base64(value)


class EpisodeFileDownloadRequest(BaseModel):
    """``GET /episodes/{episode_id}/files`` query parameters."""

    model_config = ConfigDict(extra="forbid")

    path: str

    @field_validator("path")
    @classmethod
    def _check_path(cls, value: str) -> str:
        return validate_absolute_path(value)


class EpisodeFileDownloadResponse(BaseModel):
    """``GET /episodes/{episode_id}/files`` result."""

    model_config = ConfigDict(frozen=True)

    content_b64: str


class EpisodeCloseResponse(BaseModel):
    """``DELETE /episodes/{episode_id}`` result."""

    model_config = ConfigDict(frozen=True)

    closed: bool = True


class BrokerHealthResponse(BaseModel):
    """``GET /health`` result. Authenticated like every other route -- an unauthenticated readiness
    endpoint is a free oracle for anything that can reach the leader pod.
    """

    model_config = ConfigDict(frozen=True)

    status: str = "ok"
    job_id: str
    protocol_version: str = BROKER_PROTOCOL_VERSION


class BrokerErrorResponse(BaseModel):
    """Error body returned for every non-2xx broker response."""

    model_config = ConfigDict(frozen=True)

    error: str
    code: BrokerErrorCode
