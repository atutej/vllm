# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert parallelism of the draft worker follows the draft model, not the target.

Regression: a dense draft model (for example a DSpark drafter with dense Grug
layers) under a MoE target served with --enable-expert-parallel failed
``ModelConfig.verify_with_parallel_config`` because the draft parallel config
copied ``enable_expert_parallel`` from the target.
"""

import pytest
from transformers import PretrainedConfig

from vllm.config import ParallelConfig, SpeculativeConfig
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)


@pytest.mark.parametrize(
    "draft_is_moe, expected_expert_parallel", [(False, False), (True, True)]
)
def test_draft_expert_parallel_follows_draft_model(
    draft_is_moe: bool, expected_expert_parallel: bool
):
    target = ParallelConfig(tensor_parallel_size=1, enable_expert_parallel=True)

    draft = SpeculativeConfig.create_draft_parallel_config(
        target, 1, draft_is_moe=draft_is_moe
    )

    assert draft.enable_expert_parallel is expected_expert_parallel
    assert draft.tensor_parallel_size == 1


@pytest.mark.parametrize(
    "config_kwargs, expected_experts",
    [
        # A Grug DSpark draft with routed experts is labelled MoE by its own key.
        ({"grug_num_experts": 8}, 8),
        # The usual (dense) Grug DSpark draft.
        ({"grug_num_experts": 0}, 0),
        # A target-style key takes precedence over the Grug key.
        ({"num_experts": 256, "grug_num_experts": 0}, 256),
    ],
)
def test_grug_draft_expert_count_is_read_from_config(
    config_kwargs: dict, expected_experts: int
):
    hf_config = PretrainedConfig(**config_kwargs)

    convertor = ModelArchConfigConvertorBase(hf_config, hf_config)

    assert convertor.get_num_experts() == expected_experts
