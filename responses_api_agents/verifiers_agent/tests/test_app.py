# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from unittest.mock import MagicMock

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.verifiers_agent.app import (
    VerifiersAgent,
    VerifiersAgentConfig,
    VerifiersAgentRunRequest,
)


class _CapturingEnv:
    """Stands in for a verifiers Environment, recording the sampling args it is given."""

    def __init__(self) -> None:
        self.sampling_args: dict | None = None

    async def run_group(self, *, group_inputs, client, model, sampling_args, state_columns):
        self.sampling_args = sampling_args
        return [
            {
                "reward": 1.0,
                "completion": [{"role": "assistant", "content": "answer"}],
                "trajectory": [
                    {
                        "completion": [{"role": "assistant", "content": "answer"}],
                        "tokens": {"prompt_ids": [1], "completion_ids": [2], "completion_logprobs": [-0.1]},
                    }
                ],
            }
        ]


def _agent(monkeypatch, env: _CapturingEnv, **config_overrides) -> VerifiersAgent:
    config = VerifiersAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        model_server=ModelServerRef(type="responses_api_models", name=""),
        vf_env_id="env",
        **config_overrides,
    )
    # Patched on the class: VerifiersAgent is a pydantic model, so leading-underscore
    # instance attributes are treated as private attrs and cannot be assigned.
    monkeypatch.setattr(VerifiersAgent, "_get_env", lambda self, vf_env_id: env)
    monkeypatch.setattr(VerifiersAgent, "_get_client", lambda self: MagicMock())
    return VerifiersAgent(config=config, server_client=MagicMock(spec=ServerClient))


def _request(**params) -> VerifiersAgentRunRequest:
    return VerifiersAgentRunRequest(
        task_idx=0,
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "q"}],
            **params,
        ),
    )


class TestApp:
    def test_sanity(self) -> None:
        config = VerifiersAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(type="responses_api_models", name=""),
        )
        VerifiersAgent(config=config, server_client=MagicMock(spec=ServerClient))

    def test_convert_completion_keeps_tool_outputs_as_response_items(self) -> None:
        config = VerifiersAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(type="responses_api_models", name=""),
        )
        agent = VerifiersAgent(config=config, server_client=MagicMock(spec=ServerClient))

        rollout_output = {
            "prompt": [{"role": "user", "content": "q"}],
            "completion": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        json.dumps(
                            {
                                "id": "call_1",
                                "name": "python",
                                "arguments": json.dumps({"expr": "2+2"}),
                            }
                        )
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "4"},
                {"role": "assistant", "content": "answer"},
            ],
            "trajectory": [
                {
                    "completion": [
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "name": "python",
                                    "arguments": json.dumps({"expr": "2+2"}),
                                }
                            ],
                        }
                    ],
                    "tokens": {
                        "prompt_ids": [1],
                        "completion_ids": [2],
                        "completion_logprobs": [0.0],
                        "routed_experts": [[[0, 1]], [[2, 3]]],
                    },
                },
                {
                    "completion": [{"role": "assistant", "content": "answer"}],
                    "tokens": {
                        "prompt_ids": [3],
                        "completion_ids": [4],
                        "completion_logprobs": [-0.1],
                    },
                },
            ],
        }

        output = agent._convert_trajectory_to_output(rollout_output)

        assert [item["type"] for item in output] == ["function_call", "function_call_output", "message"]
        assert output[0]["call_id"] == "call_1"
        assert output[0]["name"] == "python"
        assert output[0]["arguments"] == json.dumps({"expr": "2+2"})
        assert output[0]["prompt_token_ids"] == [1]
        assert output[0]["routed_experts"] == [[[0, 1]], [[2, 3]]]
        assert output[1]["call_id"] == "call_1"
        assert output[1]["output"] == "4"
        assert output[2]["content"][0]["text"] == "answer"
        assert output[2]["prompt_token_ids"] == [3]

    async def test_sampling_args_prefer_the_row_over_the_agent_config(self, monkeypatch) -> None:
        """NeMo RL sets max_output_tokens/temperature/top_p per row; all three must win.

        max_output_tokens used to be ignored here while the other two were honoured, so a
        job's max_new_tokens silently did nothing and the package's max_tokens governed
        every rollout instead.
        """
        env = _CapturingEnv()
        agent = _agent(monkeypatch, env, max_tokens=8192, temperature=1.0, top_p=1.0)

        await agent.responses(
            MagicMock(),
            MagicMock(),
            _request(max_output_tokens=512, temperature=0.7, top_p=0.9),
        )

        assert env.sampling_args == {"max_tokens": 512, "temperature": 0.7, "top_p": 0.9}

    async def test_sampling_args_fall_back_to_the_agent_config(self, monkeypatch) -> None:
        """Standalone Gym runs send no sampling params, so the config still applies."""
        env = _CapturingEnv()
        agent = _agent(monkeypatch, env, max_tokens=4096, temperature=0.5, top_p=0.8)

        await agent.responses(MagicMock(), MagicMock(), _request())

        assert env.sampling_args == {"max_tokens": 4096, "temperature": 0.5, "top_p": 0.8}
