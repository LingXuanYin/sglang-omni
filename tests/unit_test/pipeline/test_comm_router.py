# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch

from sglang_omni.comm.data_ref import TransportKind
from sglang_omni.comm.router import CommRouter


def test_comm_router_uses_cuda_ipc_for_same_node_gpu_payload_edges() -> None:
    router = CommRouter(
        stage_name="thinker",
        gpu_id=0,
        same_process_targets={"local"},
        gpu_stage_names={"decode"},
        comm_config={},
    )

    assert router.outbound("local") is TransportKind.LOCAL_OBJECT
    assert router.outbound("decode") is TransportKind.CUDA_IPC
    with pytest.raises(ValueError, match="selected cuda_ipc"):
        router.outbound_stream("decode", torch.empty(1))


def test_comm_router_uses_mooncake_only_for_remote_edges() -> None:
    router = CommRouter(
        stage_name="thinker",
        gpu_id=0,
        same_process_targets=set(),
        gpu_stage_names={"decode"},
        remote_stage_names={"remote_decode"},
        comm_config={},
    )

    assert router.outbound("decode") is TransportKind.CUDA_IPC
    assert router.outbound("cpu_decode") is TransportKind.SHM
    assert router.outbound("remote_decode") is TransportKind.MOONCAKE
    assert router.outbound_stream("remote_decode", torch.empty(1)) is (
        TransportKind.MOONCAKE
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_comm_router_uses_cuda_ipc_for_cuda_stream_chunks_only() -> None:
    router = CommRouter(
        stage_name="thinker",
        gpu_id=0,
        same_process_targets=set(),
        gpu_stage_names={"decode"},
        comm_config={},
    )

    assert (
        router.outbound_stream("decode", torch.empty(1, device="cuda:0"))
        is TransportKind.CUDA_IPC
    )
