# SPDX-License-Identifier: Apache-2.0
"""Typed references to data-plane buffers carried by control messages."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class TransportKind(str, Enum):
    LOCAL_OBJECT = "local_object"
    CUDA_IPC = "cuda_ipc"
    SHM = "shm"
    MOONCAKE = "mooncake"


class DataKind(str, Enum):
    STAGE_PAYLOAD = "stage_payload"
    STREAM_CHUNK = "stream_chunk"
    STREAM_METADATA_TENSOR = "stream_metadata_tensor"
    KV_PAGES = "kv_pages"
    WEIGHT_BUCKET = "weight_bucket"
    MOE_EXPERT_PAYLOAD = "moe_expert_payload"


class DataLayout(str, Enum):
    PACKED_TENSORS = "packed_tensors"
    RAW_TENSOR = "raw_tensor"
    PAGED = "paged"
    BUCKETED = "bucketed"
    SCATTER = "scatter"


@dataclass(frozen=True)
class TensorMeta:
    path: str
    shape: tuple[int, ...]
    dtype: str
    offset: int
    size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "offset": self.offset,
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TensorMeta":
        return cls(
            path=_str(value, "path"),
            shape=_ints(value, "shape"),
            dtype=_str(value, "dtype"),
            offset=_int(value, "offset"),
            size=_int(value, "size"),
        )


@dataclass(frozen=True)
class BackendRef:
    transport: TransportKind
    info: dict[str, Any]
    length: int

    @classmethod
    def from_relay_info(
        cls, *, transport: TransportKind, relay_info: dict[str, Any]
    ) -> "BackendRef":
        transfer_info = _dict(relay_info, "transfer_info")
        return cls(transport=transport, info=relay_info, length=_int(transfer_info, "size"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "transport": self.transport.value,
            "info": self.info,
            "length": self.length,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BackendRef":
        return cls(
            transport=TransportKind(_str(value, "transport")),
            info=_dict(value, "info"),
            length=_int(value, "length"),
        )


@dataclass(frozen=True)
class MetadataTensorRef:
    path: str
    ref: "DataRef"

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "ref": self.ref.to_dict()}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MetadataTensorRef":
        return cls(path=_str(value, "path"), ref=DataRef.from_dict(_dict(value, "ref")))


@dataclass(frozen=True)
class DataRef:
    """Control-plane pointer to one data-plane object."""

    version: int
    kind: DataKind
    object_id: str
    transport: TransportKind
    layout: DataLayout
    buffer: BackendRef
    header: str | None = None
    tensors: tuple[TensorMeta, ...] = ()
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    offset: int | None = None
    metadata: dict[str, Any] | None = None
    metadata_tensors: tuple[MetadataTensorRef, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "_type": "DataRef",
            "version": self.version,
            "kind": self.kind.value,
            "object_id": self.object_id,
            "transport": self.transport.value,
            "layout": self.layout.value,
            "buffer": self.buffer.to_dict(),
            "tensors": [tensor.to_dict() for tensor in self.tensors],
            "metadata_tensors": [
                tensor_ref.to_dict() for tensor_ref in self.metadata_tensors
            ],
        }
        if self.header is not None:
            value["header"] = self.header
        if self.shape is not None:
            value["shape"] = list(self.shape)
        if self.dtype is not None:
            value["dtype"] = self.dtype
        if self.offset is not None:
            value["offset"] = self.offset
        if self.metadata is not None:
            value["metadata"] = self.metadata
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DataRef":
        if _str(value, "_type") != "DataRef":
            raise ValueError("data_ref must have _type='DataRef'")
        version = _int(value, "version")
        if version != 1:
            raise ValueError(f"unsupported DataRef version {version}")
        return cls(
            version=version,
            kind=DataKind(_str(value, "kind")),
            object_id=_str(value, "object_id"),
            transport=TransportKind(_str(value, "transport")),
            layout=DataLayout(_str(value, "layout")),
            buffer=BackendRef.from_dict(_dict(value, "buffer")),
            header=_optional_str(value, "header"),
            tensors=tuple(TensorMeta.from_dict(item) for item in _list(value, "tensors")),
            shape=_ints(value, "shape") if "shape" in value else None,
            dtype=_optional_str(value, "dtype"),
            offset=_int(value, "offset") if "offset" in value else None,
            metadata=_dict(value, "metadata") if "metadata" in value else None,
            metadata_tensors=tuple(
                MetadataTensorRef.from_dict(item)
                for item in _list(value, "metadata_tensors")
            ),
        )


def _dict(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value[key]
    if not isinstance(item, dict):
        raise TypeError(f"{key} must be dict, got {type(item).__name__}")
    return item


def _list(value: dict[str, Any], key: str) -> list[Any]:
    item = value[key]
    if not isinstance(item, list):
        raise TypeError(f"{key} must be list, got {type(item).__name__}")
    return item


def _str(value: dict[str, Any], key: str) -> str:
    item = value[key]
    if not isinstance(item, str):
        raise TypeError(f"{key} must be str, got {type(item).__name__}")
    return item


def _optional_str(value: dict[str, Any], key: str) -> str | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, str):
        raise TypeError(f"{key} must be str or None, got {type(item).__name__}")
    return item


def _int(value: dict[str, Any], key: str) -> int:
    item = value[key]
    if not isinstance(item, int):
        raise TypeError(f"{key} must be int, got {type(item).__name__}")
    return item


def _ints(value: dict[str, Any], key: str) -> tuple[int, ...]:
    items = _list(value, key)
    if not all(isinstance(item, int) for item in items):
        raise TypeError(f"{key} must be list[int]")
    return tuple(items)
