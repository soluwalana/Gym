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

import base64

import pytest
from pydantic import ValidationError

from nemo_gym.sandbox.broker import (
    BROKER_AUTH_HEADER,
    BROKER_PROTOCOL_VERSION,
    BrokerErrorCode,
    BrokerErrorResponse,
    BrokerHealthResponse,
    EpisodeCreateRequest,
    EpisodeCreateResponse,
    EpisodeExecRequest,
    EpisodeExecResponse,
    EpisodeFileDownloadRequest,
    EpisodeFileUploadRequest,
    EpisodeResources,
    EpisodeStatusResponse,
)
from nemo_gym.sandbox.providers.base import SandboxStatus


B64_HELLO = base64.b64encode(b"hello").decode()


def test_auth_header_and_protocol_version_are_stable():
    # Both sides key off these constants; changing them is a breaking wire change.
    assert BROKER_AUTH_HEADER == "OPENSANDBOX-EPISODE-BROKER-AUTH"
    assert BROKER_PROTOCOL_VERSION == "1"


def test_create_request_minimal_defaults():
    request = EpisodeCreateRequest(image="registry.example.com/swe:1")

    assert request.image == "registry.example.com/swe:1"
    # Unset lifetime means "broker decides"; policy stays on the trusted side.
    assert request.ttl_s is None
    assert request.ready_timeout_s is None
    assert request.env == {}
    assert request.files_b64 == {}
    assert request.provider_options == {}
    assert request.resources == EpisodeResources()


def test_create_request_requires_image():
    # No image means the approved-image policy cannot be evaluated, so it cannot be forwarded.
    with pytest.raises(ValidationError):
        EpisodeCreateRequest()

    with pytest.raises(ValidationError):
        EpisodeCreateRequest(image="")


@pytest.mark.parametrize(
    "unknown_field",
    ["volumes", "platform", "extensions", "snapshot_id", "mounts", "network_policy"],
)
def test_create_request_forbids_unknown_fields(unknown_field):
    # These are exactly the escalation levers a backend SDK would accept; they must not be
    # expressible on the wire at all.
    with pytest.raises(ValidationError):
        EpisodeCreateRequest(image="img", **{unknown_field: {"anything": True}})


def test_create_request_rejects_non_positive_ttl():
    with pytest.raises(ValidationError):
        EpisodeCreateRequest(image="img", ttl_s=0)

    with pytest.raises(ValidationError):
        EpisodeCreateRequest(image="img", ttl_s=-1)


@pytest.mark.parametrize("bad_path", ["relative/path", "/nested/../escape", "/nul\x00byte", ""])
def test_create_request_rejects_bad_file_paths(bad_path):
    with pytest.raises(ValidationError):
        EpisodeCreateRequest(image="img", files_b64={bad_path: B64_HELLO})


def test_create_request_rejects_non_base64_file_content():
    with pytest.raises(ValidationError):
        EpisodeCreateRequest(image="img", files_b64={"/work/f.txt": "not base64!!"})


def test_create_request_accepts_valid_files_and_workdir():
    request = EpisodeCreateRequest(
        image="img",
        workdir="/workspace",
        files_b64={"/workspace/input.txt": B64_HELLO},
    )

    assert request.workdir == "/workspace"
    assert base64.b64decode(request.files_b64["/workspace/input.txt"]) == b"hello"


def test_create_request_rejects_relative_workdir():
    with pytest.raises(ValidationError):
        EpisodeCreateRequest(image="img", workdir="workspace")


def test_create_request_json_round_trip():
    request = EpisodeCreateRequest(
        image="img",
        ttl_s=120.0,
        env={"FOO": "bar"},
        metadata={"suite": "swe"},
        resources=EpisodeResources(cpu=2, memory_mib=4096),
        entrypoint=["/bin/sh", "-c", "sleep infinity"],
    )

    assert EpisodeCreateRequest.model_validate_json(request.model_dump_json()) == request


def test_resources_forbid_unknown_keys_and_require_positive_values():
    with pytest.raises(ValidationError):
        EpisodeResources(cpus=4)

    with pytest.raises(ValidationError):
        EpisodeResources(memory_mib=0)


def test_exec_request_user_defaults_to_none():
    # Privileged execution is opt-in per call site rather than a wire default.
    assert EpisodeExecRequest(command="ls").user is None


def test_exec_request_accepts_explicit_user_forms():
    assert EpisodeExecRequest(command="ls", user="root").user == "root"
    assert EpisodeExecRequest(command="ls", user=1000).user == 1000


def test_exec_request_validation():
    with pytest.raises(ValidationError):
        EpisodeExecRequest(command="")

    with pytest.raises(ValidationError):
        EpisodeExecRequest(command="ls", shell=True)

    with pytest.raises(ValidationError):
        EpisodeExecRequest(command="ls", cwd="relative")

    with pytest.raises(ValidationError):
        EpisodeExecRequest(command="ls", timeout_s=0)


def test_file_upload_request_validation():
    upload = EpisodeFileUploadRequest(path="/workspace/f.txt", content_b64=B64_HELLO)
    assert base64.b64decode(upload.content_b64) == b"hello"

    with pytest.raises(ValidationError):
        EpisodeFileUploadRequest(path="/workspace/../etc/passwd", content_b64=B64_HELLO)

    with pytest.raises(ValidationError):
        EpisodeFileUploadRequest(path="/workspace/f.txt", content_b64="!!!")


def test_file_download_request_validation():
    assert EpisodeFileDownloadRequest(path="/workspace/f.txt").path == "/workspace/f.txt"

    with pytest.raises(ValidationError):
        EpisodeFileDownloadRequest(path="../escape")


def test_status_response_uses_sandbox_status_vocabulary():
    response = EpisodeStatusResponse(status="running")

    assert response.status is SandboxStatus.RUNNING
    assert response.model_dump(mode="json") == {"status": "running"}


def test_create_response_defaults_to_running():
    assert EpisodeCreateResponse(episode_id="ep_abc").status is SandboxStatus.RUNNING


def test_health_response_carries_job_and_protocol_version():
    response = BrokerHealthResponse(job_id="job-1")

    assert response.model_dump(mode="json") == {
        "status": "ok",
        "job_id": "job-1",
        "protocol_version": BROKER_PROTOCOL_VERSION,
    }


def test_error_response_serializes_machine_readable_code():
    response = BrokerErrorResponse(error="episode image not approved", code=BrokerErrorCode.IMAGE_NOT_APPROVED)

    assert response.model_dump(mode="json") == {
        "error": "episode image not approved",
        "code": "image_not_approved",
    }


@pytest.mark.parametrize(
    "response, field_name",
    [
        (EpisodeCreateResponse(episode_id="ep_abc"), "episode_id"),
        (EpisodeStatusResponse(status=SandboxStatus.RUNNING), "status"),
        (EpisodeExecResponse(stdout="hi", stderr=None, return_code=0), "return_code"),
        (BrokerHealthResponse(job_id="job-1"), "job_id"),
        (BrokerErrorResponse(error="nope", code=BrokerErrorCode.UNAUTHORIZED), "error"),
    ],
)
def test_responses_are_frozen(response, field_name):
    with pytest.raises(ValidationError):
        setattr(response, field_name, "mutated")


def test_responses_tolerate_unknown_fields_for_forward_compatibility():
    # An older client must keep working against a newer broker that adds response fields.
    response = EpisodeExecResponse.model_validate(
        {"stdout": "hi", "stderr": None, "return_code": 0, "future_field": 123}
    )

    assert response.return_code == 0
    assert not hasattr(response, "future_field")
