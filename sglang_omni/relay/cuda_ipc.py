# SPDX-License-Identifier: Apache-2.0
"""CUDA IPC relay: intra-node, GPU-direct tensor transfer over NVLink.

This backend moves a CUDA buffer between two processes on the *same host*
without a host round-trip. The sender shares its GPU buffer as a CUDA IPC
handle (via torch's ``ForkingPickler`` reduction); the receiver opens the
handle (a tensor that aliases the sender's GPU memory) and copies it into its
own destination buffer. When sender and receiver are on different GPUs the copy
is a peer-to-peer transfer over NVLink (``cudaMemcpyPeer``); when they share a
GPU it is a same-device copy.

It is the same-node GPU cell of the transport ladder (see ``TransportRouter``):

    same process            -> local object reference
    same node, GPU buffer   -> cuda_ipc   (this backend)
    same node, CPU buffer   -> shm
    different node          -> nixl / mooncake (RDMA)

The small IPC handle rides the control-plane ``DataReadyMessage`` (like NIXL's
agent metadata), so there is no extra fetch round-trip. Cross-process lifetime
of the shared block is managed by torch's CUDA IPC reference counting, the same
mechanism the previous inline stream-chunk path relied on.
"""
from __future__ import annotations

import asyncio
import io
import logging
import pickle
from multiprocessing.reduction import ForkingPickler
from typing import Any

import torch

from .base import Relay, RelayOperation, register_relay

logger = logging.getLogger(__name__)

# Pairs of GPUs for which peer access has already been ensured this process.
_PEER_ENABLED: set[tuple[int, int]] = set()


def _parse_device_id(device: str) -> int:
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    return 0


def _ensure_peer_access(src_index: int, dst_index: int) -> None:
    """Best-effort enable of P2P access so the copy stays on the NVLink fabric.

    torch enables peer access lazily on first cross-device copy; doing it
    explicitly lets us warn loudly when two GPUs cannot reach each other over
    the fabric (the copy would silently stage through host memory, defeating the
    point of this relay).
    """
    if src_index == dst_index:
        return
    key = (dst_index, src_index)
    if key in _PEER_ENABLED:
        return
    if not torch.cuda.can_device_access_peer(dst_index, src_index):
        logger.warning(
            "cuda_ipc: GPU %d cannot peer-access GPU %d; cross-GPU copy will "
            "stage through host memory (no NVLink fast path)",
            dst_index,
            src_index,
        )
    _PEER_ENABLED.add(key)


class CudaIpcPutOperation(RelayOperation):
    """Sender-side handle. The transfer is the receiver's READ, so there is no
    sender-side completion to wait on; torch's IPC cache keeps the shared block
    alive until the receiver releases it."""

    def __init__(self, metadata: dict[str, Any], tensor: torch.Tensor):
        self._metadata = metadata
        # Hold a reference so the source buffer is not collected before the
        # control message is even sent. Long-term liveness is torch's job.
        self._tensor = tensor

    @property
    def metadata(self) -> Any:
        return self._metadata

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        return


class CudaIpcGetOperation(RelayOperation):
    """Receiver-side handle. Performs the (peer) copy and completes when the
    copy's CUDA event is done — event-scoped, never a device-wide sync."""

    def __init__(self, event: torch.cuda.Event, ipc_tensor: torch.Tensor):
        self._event = event
        # Keep the IPC-mapped source alive until the copy has completed.
        self._ipc_tensor: torch.Tensor | None = ipc_tensor
        self._completed = False

    @property
    def metadata(self) -> Any:
        return None

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        if self._completed:
            return
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while not self._event.query():
            if loop.time() > deadline:
                raise TimeoutError("cuda_ipc copy did not complete in time")
            await asyncio.sleep(0)
        self._completed = True
        self._ipc_tensor = None


@register_relay("cuda_ipc")
class CudaIpcRelay(Relay):
    def __init__(
        self,
        engine_id: str,
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:
        self.engine_id = engine_id
        if device == "cpu":
            raise ValueError(
                "cuda_ipc relay requires a CUDA device; got 'cpu'. Use the shm "
                "relay for host-memory stages."
            )
        self.device = device
        self.device_id = _parse_device_id(device)

    async def put_async(
        self,
        tensor: torch.Tensor,
        request_id: str | None = None,
        dst_rank: int | None = None,
    ) -> CudaIpcPutOperation:
        if not tensor.is_cuda:
            raise ValueError(
                "cuda_ipc relay can only transfer CUDA tensors; "
                f"got tensor on {tensor.device}"
            )
        buf = io.BytesIO()
        ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(tensor)
        metadata = {
            "transfer_info": {"size": int(tensor.numel())},
            "src_device_id": int(tensor.device.index or 0),
            "ipc_handle": buf.getvalue(),
        }
        return CudaIpcPutOperation(metadata, tensor)

    async def get_async(
        self,
        metadata: Any,
        dest_tensor: torch.Tensor,
        request_id: str | None = None,
    ) -> CudaIpcGetOperation:
        if not dest_tensor.is_cuda:
            raise ValueError(
                "cuda_ipc relay can only receive into CUDA tensors; "
                f"dest is on {dest_tensor.device}"
            )
        ipc_tensor = pickle.loads(metadata["ipc_handle"])
        src_index = int(metadata.get("src_device_id", ipc_tensor.device.index or 0))
        dst_index = int(dest_tensor.device.index or 0)
        _ensure_peer_access(src_index, dst_index)

        size = int(metadata["transfer_info"]["size"])
        src = ipc_tensor.view(torch.uint8).reshape(-1)[:size]
        dst = dest_tensor.view(torch.uint8).reshape(-1)
        copy_len = min(dst.numel(), size)

        stream = torch.cuda.current_stream(dest_tensor.device)
        with torch.cuda.stream(stream):
            dst[:copy_len].copy_(src[:copy_len])
        event = torch.cuda.Event()
        event.record(stream)
        return CudaIpcGetOperation(event, ipc_tensor)

    def cleanup(self, request_id: str) -> None:
        pass

    def close(self) -> None:
        pass
