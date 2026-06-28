# SPDX-License-Identifier: Apache-2.0
"""Adapters between stage objects and data-plane refs."""
from __future__ import annotations

import base64
import pickle
from typing import Any

import torch

from sglang_omni.comm.data_ref import (
    BackendRef,
    DataKind,
    DataLayout,
    DataRef,
    MetadataTensorRef,
    TensorMeta,
    TransportKind,
)
from sglang_omni.proto import DataReadyMessage, StagePayload
from sglang_omni.relay.base import Relay

_TORCH_DTYPES: dict[str, torch.dtype] = {
    "torch.bool": torch.bool,
    "torch.uint8": torch.uint8,
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.complex64": torch.complex64,
    "torch.complex128": torch.complex128,
}


def relay_device(relay: Relay) -> str:
    device = relay.device
    if not isinstance(device, str):
        raise TypeError(
            f"{type(relay).__name__}.device must be str, got "
            f"{type(device).__name__}"
        )
    return device


def extract_tensors(obj: Any, path: str = "") -> tuple[Any, dict[str, torch.Tensor]]:
    if isinstance(obj, torch.Tensor):
        return {
            "_tensor_placeholder": path,
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "device": str(obj.device),
        }, {path: obj}
    if isinstance(obj, dict):
        out, tensors = {}, {}
        for key, value in obj.items():
            child_path = f"{path}.{key}" if path else key
            out[key], child_tensors = extract_tensors(value, child_path)
            tensors.update(child_tensors)
        return out, tensors
    if isinstance(obj, (list, tuple)):
        out, tensors = [], {}
        for idx, value in enumerate(obj):
            child, child_tensors = extract_tensors(value, f"{path}[{idx}]")
            out.append(child)
            tensors.update(child_tensors)
        return type(obj)(out), tensors
    return obj, {}


def restore_tensors(obj: Any, tensors: dict[str, torch.Tensor]) -> Any:
    if isinstance(obj, dict):
        if "_tensor_placeholder" in obj:
            path = obj["_tensor_placeholder"]
            if path not in tensors:
                raise KeyError(f"missing tensor payload for placeholder {path!r}")
            return tensors[path]
        return {key: restore_tensors(value, tensors) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(restore_tensors(value, tensors) for value in obj)
    return obj


async def write_payload(
    relay: Relay,
    request_id: str,
    payload: StagePayload,
    *,
    transport: TransportKind,
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> tuple[DataRef, Any]:
    data_without_tensors, tensors = extract_tensors(payload.data)
    packed, entries = _pack_tensors(tensors, device=relay_device(relay))
    header = StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=data_without_tensors,
    )
    op = await relay.put_async(packed, request_id=request_id)
    data_ref = DataRef(
        version=1,
        object_id=f"{request_id}:payload:{from_stage or ''}:{to_stage or ''}",
        kind=DataKind.STAGE_PAYLOAD,
        transport=transport,
        layout=DataLayout.PACKED_TENSORS,
        buffer=BackendRef.from_relay_info(
            transport=transport,
            relay_info=op.metadata,
        ),
        header=base64.b64encode(pickle.dumps(header)).decode("ascii"),
        tensors=tuple(entries),
    )
    return data_ref, op


async def read_payload(
    relay: Relay,
    request_id: str,
    data_ref: DataRef,
) -> StagePayload:
    if data_ref.kind is not DataKind.STAGE_PAYLOAD:
        raise ValueError(f"expected stage_payload, got {data_ref.kind.value}")
    if data_ref.header is None:
        raise ValueError("stage_payload data_ref is missing header")
    header = pickle.loads(base64.b64decode(data_ref.header))
    transfer_buf = await _read_transfer_buffer(relay, request_id, data_ref)
    tensors = {
        entry.path: transfer_buf[entry.offset : entry.offset + entry.size]
        .view(_torch_dtype(entry.dtype))
        .reshape(entry.shape)
        for entry in data_ref.tensors
    }
    relay.cleanup(request_id)
    return StagePayload(
        request_id=header.request_id,
        request=header.request,
        data=restore_tensors(header.data, tensors),
    )


async def write_tensor(
    relay: Relay,
    object_id: str,
    tensor: torch.Tensor,
    *,
    transport: TransportKind,
    kind: DataKind = DataKind.STREAM_CHUNK,
    request_id: str | None = None,
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> tuple[DataRef, Any]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"write_tensor requires torch.Tensor, got {type(tensor).__name__}"
        )
    packed = tensor.contiguous().view(torch.uint8).reshape(-1)
    target_device = torch.device(relay_device(relay))
    if packed.device != target_device:
        packed = packed.to(device=target_device)
    offset = _pad_offset(0, _dtype_alignment(tensor.dtype))
    if offset:
        packed = torch.cat(
            [torch.zeros(offset, dtype=torch.uint8, device=target_device), packed]
        )
    op = await relay.put_async(packed, request_id=object_id)
    return (
        DataRef(
            version=1,
            object_id=object_id,
            kind=kind,
            transport=transport,
            layout=DataLayout.RAW_TENSOR,
            buffer=BackendRef.from_relay_info(
                transport=transport,
                relay_info=op.metadata,
            ),
            shape=tuple(int(dim) for dim in tensor.shape),
            dtype=str(tensor.dtype),
            offset=offset,
        ),
        op,
    )


async def read_tensor(
    relay: Relay,
    data_ref: DataRef,
) -> torch.Tensor:
    if data_ref.layout is not DataLayout.RAW_TENSOR:
        raise ValueError(f"expected raw_tensor layout, got {data_ref.layout.value}")
    if data_ref.shape is None:
        raise ValueError("raw tensor data_ref is missing shape")
    if data_ref.dtype is None:
        raise ValueError("raw tensor data_ref is missing dtype")
    if data_ref.offset is None:
        raise ValueError("raw tensor data_ref is missing offset")
    transfer_buf = await _read_transfer_buffer(relay, data_ref.object_id, data_ref)
    return (
        transfer_buf[data_ref.offset :]
        .view(_torch_dtype(data_ref.dtype))
        .reshape(data_ref.shape)
    )


async def send_stream_chunk(
    relay: Relay,
    control_plane: Any,
    *,
    request_id: str,
    data: torch.Tensor,
    target_stage: str,
    target_endpoint: str,
    from_stage: str,
    chunk_id: int,
    metadata: dict | None = None,
    transport: TransportKind,
) -> None:
    object_id = f"{request_id}:stream:{from_stage}:{target_stage}:{chunk_id}"
    data_ref, op = await write_tensor(
        relay,
        object_id,
        data,
        transport=transport,
        kind=DataKind.STREAM_CHUNK,
        request_id=request_id,
        from_stage=from_stage,
        to_stage=target_stage,
    )
    pending_ops = [op]
    data_ref = await _with_stream_metadata(
        relay, data_ref, metadata, transport, pending_ops
    )
    await control_plane.send_to_stage(
        target_stage,
        target_endpoint,
        DataReadyMessage(
            request_id=request_id,
            from_stage=from_stage,
            to_stage=target_stage,
            data_ref=data_ref.to_dict(),
            chunk_id=chunk_id,
        ),
    )
    for pending_op in pending_ops:
        await pending_op.wait_for_completion()


async def read_stream_chunk(
    relay: Relay,
    data_ref: DataRef,
) -> tuple[torch.Tensor, dict[str, Any] | None]:
    data = await read_tensor(relay, data_ref)
    metadata = dict(data_ref.metadata or {})
    if data_ref.metadata_tensors:
        tensors = {
            ref.path: await read_tensor(relay, ref.ref)
            for ref in data_ref.metadata_tensors
        }
        metadata = restore_tensors(metadata, tensors)
    return data, metadata or None


async def send_stream_signal(
    control_plane: Any,
    *,
    request_id: str,
    target_stage: str,
    target_endpoint: str,
    from_stage: str,
    is_done: bool = False,
    error: str | None = None,
) -> None:
    await control_plane.send_to_stage(
        target_stage,
        target_endpoint,
        DataReadyMessage(
            request_id=request_id,
            from_stage=from_stage,
            to_stage=target_stage,
            data_ref=None,
            is_done=is_done,
            error=error,
        ),
    )


async def _with_stream_metadata(
    relay: Relay,
    data_ref: DataRef,
    metadata: dict | None,
    transport: TransportKind,
    pending_ops: list[Any],
) -> DataRef:
    if metadata is None:
        return data_ref
    metadata_without_tensors, tensors = extract_tensors(metadata)
    tensor_refs = []
    for idx, (path, tensor) in enumerate(tensors.items()):
        ref, op = await write_tensor(
            relay,
            f"{data_ref.object_id}:meta:{idx}",
            tensor,
            transport=transport,
            kind=DataKind.STREAM_METADATA_TENSOR,
        )
        tensor_refs.append(MetadataTensorRef(path=path, ref=ref))
        pending_ops.append(op)
    return DataRef(
        version=data_ref.version,
        object_id=data_ref.object_id,
        kind=data_ref.kind,
        transport=data_ref.transport,
        layout=data_ref.layout,
        buffer=data_ref.buffer,
        shape=data_ref.shape,
        dtype=data_ref.dtype,
        offset=data_ref.offset,
        metadata=metadata_without_tensors,
        metadata_tensors=tuple(tensor_refs),
    )


def _pack_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    device: str,
) -> tuple[torch.Tensor, list[TensorMeta]]:
    target_device = torch.device(device)
    entries, chunks, offset = [], [], 0
    for path, tensor in tensors.items():
        flat = tensor.contiguous().view(torch.uint8).reshape(-1)
        if flat.device != target_device:
            flat = flat.to(device=target_device)
        padding = _pad_offset(offset, _dtype_alignment(tensor.dtype))
        if padding:
            chunks.append(torch.zeros(padding, dtype=torch.uint8, device=target_device))
            offset += padding
        chunks.append(flat)
        entries.append(
            TensorMeta(
                path=path,
                shape=tuple(int(dim) for dim in tensor.shape),
                dtype=str(tensor.dtype),
                offset=offset,
                size=int(flat.numel()),
            )
        )
        offset += int(flat.numel())
    if not chunks:
        chunks.append(torch.zeros(1, dtype=torch.uint8, device=target_device))
    return torch.cat(chunks), entries


async def _read_transfer_buffer(
    relay: Relay,
    request_id: str,
    data_ref: DataRef,
) -> torch.Tensor:
    buf = torch.zeros(
        data_ref.buffer.length,
        dtype=torch.uint8,
        device=relay_device(relay),
    )
    op = await relay.get_async(
        metadata=data_ref.buffer.info,
        dest_tensor=buf,
        request_id=request_id,
    )
    await op.wait_for_completion()
    return buf


def _dtype_alignment(dtype: torch.dtype) -> int:
    return max(torch.empty((), dtype=dtype).element_size(), 1)


def _pad_offset(offset: int, alignment: int) -> int:
    return (-offset) % alignment


def _torch_dtype(dtype_str: str) -> torch.dtype:
    dtype = _TORCH_DTYPES.get(dtype_str)
    if dtype is None:
        raise ValueError(f"unsupported tensor dtype metadata: {dtype_str!r}")
    return dtype
