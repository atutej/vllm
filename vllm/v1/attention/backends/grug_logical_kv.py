# SPDX-License-Identifier: Apache-2.0
# Copyright The Marin Authors

"""Preserve GrugMoE's original GQA grouping with deduplicated KV heads."""

from typing import Any

import torch
from flashinfer.decode import trtllm_batch_decode_with_kv_cache


def trtllm_decode_with_query_groups(
    *,
    query: torch.Tensor,
    kv_cache: tuple[torch.Tensor, torch.Tensor],
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    groups: int,
    **kwargs: Any,
) -> torch.Tensor:
    """Batch independent query groups against the same native KV pages."""
    assert query.dtype == out.dtype == torch.bfloat16
    assert all(cache.dtype == torch.bfloat16 for cache in kv_cache)
    assert kwargs["backend"] == "trtllm-gen"
    assert kwargs["q_len_per_req"] == 1
    assert kwargs["max_q_len"] is None
    assert kwargs["cum_seq_lens_q"] is None
    assert kwargs["window_left"] == -1
    assert kwargs["sinks"] is None
    assert kwargs["o_sf_scale"] is None
    assert kwargs["kv_cache_sf"] is None
    assert kwargs["lse"] is None and not kwargs["return_lse"]
    layout = kwargs["kv_layout"]
    assert layout in ("HND", "NHD")
    kv_heads = kv_cache[0].shape[1 if layout == "HND" else 2]
    batch, query_heads, head_size = query.shape
    assert groups > 1 and query_heads % (kv_heads * groups) == 0
    assert batch == seq_lens.numel() == block_tables.shape[0]
    queries_per_stored_head = query_heads // (kv_heads * groups)
    grouped_query = (
        query.reshape(batch, kv_heads, groups, queries_per_stored_head, head_size)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch * groups, query_heads // groups, head_size)
        .contiguous()
    )
    grouped_output = torch.empty_like(grouped_query)
    trtllm_batch_decode_with_kv_cache(
        query=grouped_query,
        kv_cache=kv_cache,
        block_tables=block_tables.repeat_interleave(groups, dim=0),
        seq_lens=seq_lens.repeat_interleave(groups),
        out=grouped_output,
        **kwargs,
    )
    out.copy_(
        grouped_output.reshape(
            batch, groups, kv_heads, queries_per_stored_head, head_size
        )
        .permute(0, 2, 1, 3, 4)
        .reshape(batch, query_heads, head_size)
    )
    return out
