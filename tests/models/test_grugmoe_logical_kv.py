# SPDX-License-Identifier: Apache-2.0
# Copyright The Marin Authors

import pytest
import torch

from vllm import envs
from vllm.platforms import current_platform

HEAD_SIZE = 128
KV_WIDTH = 2 * HEAD_SIZE


def _paged_history(
    history: torch.Tensor, heads: int, page_size: int
) -> torch.Tensor:
    length = history.shape[0]
    pages = (length + page_size - 1) // page_size
    repeated = history.repeat_interleave(heads // history.shape[1], dim=1)
    padded = torch.zeros(
        pages * page_size, heads, KV_WIDTH, dtype=torch.bfloat16, device="cuda"
    )
    padded[:length] = repeated
    return (
        padded.reshape(pages, page_size, heads, KV_WIDTH).transpose(1, 2).contiguous()
    )


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(100),
    reason="Grouped GrugMoE decode is qualified on SM100 CUDA",
)
@pytest.mark.parametrize("batch,length", [(1, 2048), (4, 8208), (32, 262144)])
@torch.inference_mode()
def test_grouped_decode_preserves_repeated_heads(
    batch: int, length: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("flashinfer")
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache

    from vllm.v1.attention.backends.grug_logical_kv import (
        trtllm_decode_with_query_groups,
    )

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    generator = torch.Generator(device="cuda").manual_seed(42)
    histories = torch.randn(
        length + 32 * batch,
        6,
        KV_WIDTH,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    query = torch.randn(
        batch, 48, HEAD_SIZE, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    lengths = [length - (index % 8) * 37 for index in range(batch)]
    workspace = torch.zeros(
        envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE, dtype=torch.uint8, device="cuda"
    )
    outputs = []
    for heads, page_size in [(12, 16), (6, 32)]:
        cache = _paged_history(histories, heads, page_size)
        pages = (length + page_size - 1) // page_size
        block_tables = torch.arange(pages, dtype=torch.int32, device="cuda")[None]
        block_tables = block_tables + (
            torch.arange(batch, dtype=torch.int32, device="cuda")[:, None]
            * (32 // page_size)
        )
        output = torch.empty_like(query)
        options = dict(
            query=query,
            kv_cache=cache.split(HEAD_SIZE, dim=-1),
            workspace_buffer=workspace,
            block_tables=block_tables,
            seq_lens=torch.tensor(lengths, dtype=torch.int32, device="cuda"),
            max_seq_len=length,
            bmm1_scale=HEAD_SIZE**-0.5,
            bmm2_scale=1.0,
            window_left=-1,
            sinks=None,
            o_sf_scale=None,
            out=output,
            kv_layout="HND",
            backend="trtllm-gen",
            q_len_per_req=1,
            max_q_len=None,
            cum_seq_lens_q=None,
            kv_cache_sf=None,
            lse=None,
            return_lse=False,
        )
        if heads == 12:
            trtllm_batch_decode_with_kv_cache(**options)
        else:
            trtllm_decode_with_query_groups(**options, groups=2)
        outputs.append(output)
        del options, cache
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    head = torch.arange(48, device="cuda") // 8
    for index, seq_len in enumerate(lengths):
        history = histories[index * 32 : index * 32 + seq_len]
        keys = history[:, head, :HEAD_SIZE].float().transpose(0, 1)
        scores = query[index].float()[:, None] @ keys.transpose(1, 2)
        del keys
        values = history[:, head, HEAD_SIZE:].float().transpose(0, 1)
        expected = (
            torch.softmax(scores * (HEAD_SIZE**-0.5), dim=-1) @ values
        ).squeeze(1)
        torch.testing.assert_close(
            outputs[1][index].float(), expected, atol=0.01, rtol=0.01
        )
        del values, scores, expected
