# SPDX-License-Identifier: Apache-2.0
"""Relay IO utilities for inter-stage data transfer."""
from __future__ import annotations

import base64
import pickle
from typing import Any

import torch

from sglang_omni.profiler.comm_trace import elapsed_ms as _comm_elapsed_ms
from sglang_omni.profiler.comm_trace import emit as _comm_trace
from sglang_omni.profiler.comm_trace import now_ns as _comm_now_ns
from sglang_omni.proto import DataReadyMessage, StagePayload
from sglang_omni.relay.base import Relay


def _dtype_alignment(dtype: torch.dtype) -> int:
    return max(torch.empty((), dtype=dtype).element_size(), 1)


def _pad_offset(offset: int, alignment: int) -> int:
    return (-offset) % alignment


def _relay_device(relay: Relay) -> str:
    try:
        device = relay.device
    except AttributeError as exc:
        raise TypeError(
            f"{type(relay).__name__} must expose a string 'device' attribute"
        ) from exc
    if not isinstance(device, str):
        raise TypeError(
            f"{type(relay).__name__}.device must be a string, got "
            f"{type(device).__name__}"
        )
    return device


def _torch_dtype(dtype_str: str) -> torch.dtype:
    prefix = "torch."
    if not isinstance(dtype_str, str) or not dtype_str.startswith(prefix):
        raise ValueError(f"invalid tensor dtype metadata: {dtype_str!r}")
    dtype_name = dtype_str[len(prefix) :]
    dtype = vars(torch).get(dtype_name)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported tensor dtype metadata: {dtype_str!r}")
    return dtype


def extract_tensors(obj: Any, path: str = "") -> tuple[Any, dict[str, torch.Tensor]]:
    """Replace tensors in nested data with placeholders."""
    tensors = {}

    if isinstance(obj, torch.Tensor):
        placeholder = {
            "_tensor_placeholder": path,
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "device": str(obj.device),
        }
        tensors[path] = obj
        return placeholder, tensors

    elif isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            new_value, sub_tensors = extract_tensors(value, new_path)
            new_dict[key] = new_value
            tensors.update(sub_tensors)
        return new_dict, tensors

    elif isinstance(obj, (list, tuple)):
        new_list = []
        for i, item in enumerate(obj):
            new_path = f"{path}[{i}]"
            new_item, sub_tensors = extract_tensors(item, new_path)
            new_list.append(new_item)
            tensors.update(sub_tensors)
        return (type(obj)(new_list), tensors)

    else:
        return obj, tensors


def restore_tensors(obj: Any, tensor_dict: dict[str, torch.Tensor]) -> Any:
    """Restore tensor placeholders in nested data."""
    if isinstance(obj, dict):
        if "_tensor_placeholder" in obj:
            path = obj["_tensor_placeholder"]
            if path not in tensor_dict:
                raise KeyError(f"missing tensor payload for placeholder {path!r}")
            return tensor_dict[path]
        else:
            return {
                key: restore_tensors(value, tensor_dict) for key, value in obj.items()
            }
    elif isinstance(obj, (list, tuple)):
        return type(obj)(restore_tensors(item, tensor_dict) for item in obj)
    else:
        return obj


async def write_payload(
    relay: Relay,
    request_id: str,
    payload: StagePayload,
) -> tuple[dict[str, Any], Any]:
    """Write a StagePayload to relay."""
    device = _relay_device(relay)
    transport_device = torch.device(device)

    modified_data, tensor_dict = extract_tensors(payload.data)
    payload_no_tensors = StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=modified_data,
    )
    metadata_bytes = pickle.dumps(payload_no_tensors)

    if tensor_dict:
        tensor_buffers = []
        tensor_info = []
        offset = 0
        for path, tensor in tensor_dict.items():
            flat = tensor.contiguous().view(torch.uint8).reshape(-1)
            if flat.device != transport_device:
                flat = flat.to(device=transport_device)
            padding = _pad_offset(offset, _dtype_alignment(tensor.dtype))
            if padding:
                tensor_buffers.append(
                    torch.zeros(padding, dtype=torch.uint8, device=transport_device)
                )
                offset += padding
            tensor_buffers.append(flat)
            tensor_info.append(
                {
                    "path": path,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "offset": offset,
                    "size": flat.numel(),
                }
            )
            offset += flat.numel()
        all_tensors = torch.cat(tensor_buffers)
    else:
        all_tensors = torch.zeros(1, dtype=torch.uint8, device=device)
        tensor_info = []

    op = await relay.put_async(all_tensors, request_id=request_id)

    return {
        "relay_info": op.metadata,
        "payload_pickle": base64.b64encode(metadata_bytes).decode("ascii"),
        "tensor_info": tensor_info,
    }, op


async def read_payload(
    relay: Relay,
    request_id: str,
    metadata: dict[str, Any],
) -> StagePayload:
    """Read a StagePayload from relay."""
    device = _relay_device(relay)

    payload_bytes = base64.b64decode(metadata["payload_pickle"])
    payload_no_tensors = pickle.loads(payload_bytes)

    relay_info = metadata["relay_info"]
    tensor_info = metadata["tensor_info"]
    if not isinstance(tensor_info, list):
        raise TypeError(f"tensor_info must be a list, got {type(tensor_info).__name__}")
    tensor_dict = {}

    data_size = relay_info["transfer_info"]["size"]
    recv_tensor = torch.zeros(data_size, dtype=torch.uint8, device=device)
    op = await relay.get_async(
        metadata=relay_info, dest_tensor=recv_tensor, request_id=request_id
    )
    await op.wait_for_completion()

    if tensor_info:
        for info in tensor_info:
            path = info["path"]
            shape = info["shape"]
            dtype_str = info["dtype"]
            offset = info["offset"]
            size = info["size"]
            tensor_bytes = recv_tensor[offset : offset + size]
            dtype = _torch_dtype(dtype_str)
            tensor = tensor_bytes.view(dtype).reshape(shape)
            tensor_dict[path] = tensor

    restored_data = restore_tensors(payload_no_tensors.data, tensor_dict)
    payload = StagePayload(
        request_id=payload_no_tensors.request_id,
        request=payload_no_tensors.request,
        data=restored_data,
    )
    relay.cleanup(request_id)
    return payload


async def write_blob(
    relay: Relay,
    key: str,
    tensor: torch.Tensor,
) -> tuple[dict[str, Any], Any]:
    """Write a raw tensor to relay."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"write_blob requires torch.Tensor, got {type(tensor).__name__}"
        )
    start = _comm_now_ns()
    flat = tensor.contiguous().view(torch.uint8).reshape(-1)
    contiguous_ms = _comm_elapsed_ms(start)
    transport_device = torch.device(_relay_device(relay))
    move_start = _comm_now_ns()
    source_device = str(flat.device)
    if flat.device != transport_device:
        flat = flat.to(device=transport_device)
    move_ms = _comm_elapsed_ms(move_start)
    padding = _pad_offset(0, _dtype_alignment(tensor.dtype))
    if padding:
        flat = torch.cat(
            [
                torch.zeros(padding, dtype=torch.uint8, device=transport_device),
                flat,
            ]
        )
    put_start = _comm_now_ns()
    op = await relay.put_async(flat, request_id=key)
    put_ms = _comm_elapsed_ms(put_start)
    _comm_trace(
        "relay_write_blob",
        key=key,
        source_device=source_device,
        transport_device=str(transport_device),
        tensor_device=str(tensor.device),
        tensor_shape=list(tensor.shape),
        tensor_dtype=str(tensor.dtype),
        bytes=int(flat.numel()),
        padding=int(padding),
        contiguous_ms=round(contiguous_ms, 6),
        device_move_ms=round(move_ms, 6),
        put_async_ms=round(put_ms, 6),
        elapsed_ms=round(_comm_elapsed_ms(start), 6),
    )
    metadata = {
        "relay_info": op.metadata,
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.dtype),
        "tensor_offset": padding,
    }
    return metadata, op


async def read_blob(
    relay: Relay,
    key: str,
    metadata: dict[str, Any],
) -> torch.Tensor:
    """Read a raw tensor from relay."""
    start = _comm_now_ns()
    device = _relay_device(relay)
    relay_info = metadata["relay_info"]
    shape = metadata["tensor_shape"]
    dtype_str = metadata["tensor_dtype"]
    offset = int(metadata["tensor_offset"])

    data_size = relay_info["transfer_info"]["size"]
    alloc_start = _comm_now_ns()
    recv_buf = torch.zeros(data_size, dtype=torch.uint8, device=device)
    alloc_ms = _comm_elapsed_ms(alloc_start)
    get_start = _comm_now_ns()
    op = await relay.get_async(
        metadata=relay_info, dest_tensor=recv_buf, request_id=key
    )
    get_ms = _comm_elapsed_ms(get_start)
    wait_start = _comm_now_ns()
    await op.wait_for_completion()
    wait_ms = _comm_elapsed_ms(wait_start)

    dtype = _torch_dtype(dtype_str)
    result = recv_buf[offset:].view(dtype).reshape(shape)
    _comm_trace(
        "relay_read_blob",
        key=key,
        device=str(device),
        tensor_shape=list(shape),
        tensor_dtype=dtype_str,
        bytes=int(data_size),
        offset=offset,
        alloc_ms=round(alloc_ms, 6),
        get_async_ms=round(get_ms, 6),
        wait_completion_ms=round(wait_ms, 6),
        elapsed_ms=round(_comm_elapsed_ms(start), 6),
    )
    return result


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
    transport_kind: str | None = None,
) -> None:
    """Send a relay-backed stream tensor chunk."""
    if not isinstance(data, torch.Tensor):
        raise TypeError(
            f"send_stream_chunk requires torch.Tensor, got {type(data).__name__}"
        )
    start = _comm_now_ns()
    blob_key = f"{request_id}:stream:{from_stage}:{target_stage}:{chunk_id}"

    pending_ops = []
    write_start = _comm_now_ns()
    relay_metadata, op = await write_blob(relay, blob_key, data)
    write_ms = _comm_elapsed_ms(write_start)
    if transport_kind is not None:
        relay_metadata["_transport"] = transport_kind
    pending_ops.append(op)

    if metadata:
        cleaned_meta, tensor_dict = extract_tensors(metadata)
        relay_metadata["chunk_metadata"] = cleaned_meta
        if tensor_dict:
            metadata_refs: dict[str, Any] = {}
            for meta_idx, (tkey, tensor) in enumerate(tensor_dict.items()):
                meta_blob_key = f"{blob_key}:meta:{meta_idx}"
                meta_relay_info, meta_op = await write_blob(
                    relay, meta_blob_key, tensor
                )
                pending_ops.append(meta_op)
                metadata_refs[tkey] = {
                    "blob_key": meta_blob_key,
                    "relay_metadata": meta_relay_info,
                }
            relay_metadata["chunk_metadata_tensors"] = metadata_refs

    # Notify before waiting; relay credits are released by receiver progress.
    msg = DataReadyMessage(
        request_id=request_id,
        from_stage=from_stage,
        to_stage=target_stage,
        shm_metadata=relay_metadata,
        chunk_id=chunk_id,
    )
    send_start = _comm_now_ns()
    await control_plane.send_to_stage(target_stage, target_endpoint, msg)
    send_ms = _comm_elapsed_ms(send_start)

    wait_start = _comm_now_ns()
    for pending_op in pending_ops:
        await pending_op.wait_for_completion()
    wait_ms = _comm_elapsed_ms(wait_start)
    _comm_trace(
        "relay_send_stream_chunk",
        request_id=request_id,
        blob_key=blob_key,
        from_stage=from_stage,
        target_stage=target_stage,
        chunk_id=chunk_id,
        data_device=str(data.device),
        data_shape=list(data.shape),
        metadata_tensor_count=len(relay_metadata.get("chunk_metadata_tensors", {})),
        write_blob_ms=round(write_ms, 6),
        control_send_ms=round(send_ms, 6),
        sender_wait_completion_ms=round(wait_ms, 6),
        elapsed_ms=round(_comm_elapsed_ms(start), 6),
    )


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
    """Send stream done/error signal to downstream stage."""
    msg = DataReadyMessage(
        request_id=request_id,
        from_stage=from_stage,
        to_stage=target_stage,
        shm_metadata={},
        is_done=is_done,
        error=error,
    )
    await control_plane.send_to_stage(target_stage, target_endpoint, msg)
