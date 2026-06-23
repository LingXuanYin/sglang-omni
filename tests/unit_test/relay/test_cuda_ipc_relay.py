# SPDX-License-Identifier: Apache-2.0
"""GPU-direct round-trip tests for the CUDA-IPC relay.

The relay shares a CUDA buffer across processes via an IPC handle; the receiver
opens it and copies into its own buffer (a peer/NVLink copy when the GPUs
differ). These run two real processes because a process cannot open its own CUDA
IPC handle.
"""
from __future__ import annotations

import asyncio
import multiprocessing as mp

import pytest
import torch

from sglang_omni.relay.cuda_ipc import CudaIpcRelay

_N = 1024 * 1024  # 1 MiB payload


def _expected(n: int) -> torch.Tensor:
    return (torch.arange(n, dtype=torch.int64) % 251).to(torch.uint8)


def _sender(src_gpu: int, meta_q: mp.Queue, done_q: mp.Queue) -> None:
    torch.cuda.set_device(src_gpu)
    relay = CudaIpcRelay(engine_id="sender", device=f"cuda:{src_gpu}")
    buf = _expected(_N).to(f"cuda:{src_gpu}")

    async def run() -> None:
        op = await relay.put_async(buf, request_id="r")
        meta_q.put(op.metadata)
        await op.wait_for_completion()

    asyncio.run(run())
    done_q.get(timeout=60)  # keep the source buffer alive until the receiver copies


def _receiver(
    dst_gpu: int, meta_q: mp.Queue, done_q: mp.Queue, result_q: mp.Queue
) -> None:
    try:
        torch.cuda.set_device(dst_gpu)
        relay = CudaIpcRelay(engine_id="receiver", device=f"cuda:{dst_gpu}")
        metadata = meta_q.get(timeout=60)

        async def run() -> torch.Tensor:
            size = metadata["transfer_info"]["size"]
            dest = torch.zeros(size, dtype=torch.uint8, device=f"cuda:{dst_gpu}")
            op = await relay.get_async(metadata, dest, request_id="r")
            await op.wait_for_completion()
            return dest

        dest = asyncio.run(run())
        expected = _expected(_N).to(f"cuda:{dst_gpu}")
        result_q.put(("ok", bool(torch.equal(dest, expected))))
    except BaseException as exc:  # noqa: BLE001
        result_q.put(("err", repr(exc)))
    finally:
        done_q.put(True)


def _run_case(src_gpu: int, dst_gpu: int) -> None:
    ctx = mp.get_context("spawn")
    meta_q, done_q, result_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    sender = ctx.Process(target=_sender, args=(src_gpu, meta_q, done_q))
    receiver = ctx.Process(target=_receiver, args=(dst_gpu, meta_q, done_q, result_q))
    sender.start()
    receiver.start()
    try:
        status, value = result_q.get(timeout=120)
    finally:
        sender.join(timeout=30)
        receiver.join(timeout=30)
        for proc in (sender, receiver):
            if proc.is_alive():
                proc.terminate()
    assert status == "ok", value
    assert value is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_ipc_same_gpu_round_trip() -> None:
    _run_case(0, 0)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires >= 2 GPUs for cross-GPU transfer"
)
def test_cuda_ipc_cross_gpu_round_trip() -> None:
    _run_case(0, 1)
