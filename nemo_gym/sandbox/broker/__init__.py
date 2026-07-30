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

"""Episode provisioning broker contract shared by NeMo-Gym and NeMo-RL."""

from nemo_gym.sandbox.broker.wire import (
    BROKER_AUTH_HEADER,
    BROKER_PROTOCOL_VERSION,
    BROKER_TOKEN_ENV,
    BROKER_URL_ENV,
    EPISODE_EXEC_PATH,
    EPISODE_FILES_PATH,
    EPISODE_PATH,
    EPISODES_PATH,
    HEALTH_PATH,
    BrokerErrorCode,
    BrokerErrorResponse,
    BrokerHealthResponse,
    EpisodeCloseResponse,
    EpisodeCreateRequest,
    EpisodeCreateResponse,
    EpisodeExecRequest,
    EpisodeExecResponse,
    EpisodeFileDownloadRequest,
    EpisodeFileDownloadResponse,
    EpisodeFileUploadRequest,
    EpisodeResources,
    EpisodeStatusResponse,
    validate_absolute_path,
    validate_base64,
)


__all__ = [
    "BROKER_AUTH_HEADER",
    "BROKER_PROTOCOL_VERSION",
    "BROKER_TOKEN_ENV",
    "BROKER_URL_ENV",
    "EPISODES_PATH",
    "EPISODE_EXEC_PATH",
    "EPISODE_FILES_PATH",
    "EPISODE_PATH",
    "HEALTH_PATH",
    "BrokerErrorCode",
    "BrokerErrorResponse",
    "BrokerHealthResponse",
    "EpisodeCloseResponse",
    "EpisodeCreateRequest",
    "EpisodeCreateResponse",
    "EpisodeExecRequest",
    "EpisodeExecResponse",
    "EpisodeFileDownloadRequest",
    "EpisodeFileDownloadResponse",
    "EpisodeFileUploadRequest",
    "EpisodeResources",
    "EpisodeStatusResponse",
    "validate_absolute_path",
    "validate_base64",
]
