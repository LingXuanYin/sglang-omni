# SPDX-License-Identifier: Apache-2.0
"""Locality classification and relay ownership for Omni communication."""
from __future__ import annotations

from contextlib import suppress
from typing import Any

import torch

from sglang_omni.comm.data_ref import TransportKind
from sglang_omni.relay.base import Relay, create_relay


class CommRouter:
    """Maps an edge to the physical mover and owns relay instances.

    The router classifies locality. It does not define the stage protocol and it
    does not expose Mooncake-specific handles to stage code.
    """

    def __init__(
        self,
        *,
        stage_name: str,
        gpu_id: int | None,
        same_process_targets: set[str] | None,
        gpu_stage_names: set[str] | None,
        remote_stage_names: set[str] | None = None,
        relay_config: dict[str, Any] | None = None,
        injected_relay: Relay | None = None,
    ) -> None:
        self.stage_name = stage_name
        self.gpu_id = gpu_id
        self.same_process_targets = set(same_process_targets or ())
        self.gpu_stage_names = set(gpu_stage_names or ())
        self.remote_stage_names = set(remote_stage_names or ())
        self.relay_config = dict(relay_config or {})
        self.injected_relay = injected_relay
        self.default_relay_transport = self._default_relay_transport()
        self._relays: dict[TransportKind, Relay] = {}

    @property
    def self_is_gpu(self) -> bool:
        return self.gpu_id is not None

    def is_local_object(self, target: str) -> bool:
        return target in self.same_process_targets

    def outbound(self, target: str) -> TransportKind:
        if target in self.same_process_targets:
            return TransportKind.LOCAL_OBJECT
        if target in self.remote_stage_names:
            return TransportKind.MOONCAKE
        if self.self_is_gpu and target in self.gpu_stage_names:
            return TransportKind.CUDA_IPC
        return self.default_relay_transport

    def outbound_stream(self, target: str, data: torch.Tensor) -> TransportKind:
        if not isinstance(data, torch.Tensor):
            raise TypeError(
                "relay-backed stream chunks must be torch.Tensor, got "
                f"{type(data).__name__}"
            )
        kind = self.outbound(target)
        if kind is TransportKind.CUDA_IPC and not data.is_cuda:
            return self.default_relay_transport
        return kind

    def inbound(self, from_stage: str) -> TransportKind:
        if from_stage in self.remote_stage_names:
            return TransportKind.MOONCAKE
        if self.self_is_gpu and from_stage in self.gpu_stage_names:
            return TransportKind.CUDA_IPC
        return self.default_relay_transport

    def relay(self, kind: TransportKind) -> Relay:
        if kind is TransportKind.LOCAL_OBJECT:
            raise ValueError("local_object has no relay")
        if self.injected_relay is not None:
            return self.injected_relay
        relay = self._relays.get(kind)
        if relay is None:
            relay = self._build_relay(kind)
            self._relays[kind] = relay
        return relay

    def relay_for(self, target: str) -> tuple[TransportKind, Relay]:
        kind = self.outbound(target)
        if kind is TransportKind.LOCAL_OBJECT:
            kind = self.default_relay_transport
        return kind, self.relay(kind)

    def relay_for_stream(
        self, target: str, data: torch.Tensor
    ) -> tuple[TransportKind, Relay]:
        kind = self.outbound_stream(target, data)
        if kind is TransportKind.LOCAL_OBJECT:
            kind = self.default_relay_transport
        return kind, self.relay(kind)

    def inbound_relay(self, from_stage: str) -> Relay:
        return self.relay(self.inbound(from_stage))

    def _default_relay_transport(self) -> TransportKind:
        relay_type = (
            self.relay_config["relay_type"]
            if "relay_type" in self.relay_config
            else "shm"
        )
        if relay_type == "mooncake":
            return TransportKind.MOONCAKE
        return TransportKind.SHM

    def _build_relay(self, kind: TransportKind) -> Relay:
        cfg = self.relay_config
        engine_id = (
            cfg["worker_id"] if "worker_id" in cfg else f"{self.stage_name}_relay"
        )
        slot_size_mb = cfg["slot_size_mb"] if "slot_size_mb" in cfg else 512
        credits = cfg["credits"] if "credits" in cfg else 2
        if kind is TransportKind.CUDA_IPC:
            if self.gpu_id is None:
                raise ValueError(
                    f"cuda_ipc relay requested for non-GPU stage "
                    f"{self.stage_name!r}"
                )
            return create_relay(
                "cuda_ipc",
                engine_id=engine_id,
                device=f"cuda:{self.gpu_id}",
                slot_size_mb=slot_size_mb,
                credits=credits,
            )
        if kind is TransportKind.MOONCAKE:
            device = f"cuda:{self.gpu_id}" if self.gpu_id is not None else "cpu"
            return create_relay(
                "mooncake",
                engine_id=engine_id,
                device=device,
                slot_size_mb=slot_size_mb,
                credits=credits,
                protocol=cfg["protocol"] if "protocol" in cfg else "rdma",
                hostname=cfg["hostname"] if "hostname" in cfg else None,
                device_name=cfg["device_name"] if "device_name" in cfg else "",
            )
        if kind is TransportKind.SHM:
            return create_relay(
                "shm",
                engine_id=engine_id,
                device="cpu",
                slot_size_mb=slot_size_mb,
                credits=credits,
            )
        raise ValueError(f"CommRouter cannot build a relay for {kind}")

    def cleanup(self, request_id: str) -> None:
        for relay in self._active_relays():
            with suppress(Exception):
                relay.cleanup(request_id)

    def close(self) -> None:
        for relay in self._active_relays():
            with suppress(Exception):
                relay.close()
        self._relays.clear()

    def _active_relays(self) -> list[Relay]:
        if self.injected_relay is not None:
            return [self.injected_relay]
        return list(self._relays.values())
