# SPDX-License-Identifier: Apache-2.0
"""Centralized transport selection for inter-stage data movement.

Every decision about *how* two stages exchange a payload lives here, so the
logic is not duplicated across the send and receive paths. The choice is derived
purely from static placement (which stage runs where), following one ladder:

    same process              -> LOCAL     (python object reference, zero copy)
    same node, both on GPU    -> CUDA_IPC  (NVLink / cudaMemcpyPeer)
    same node, otherwise      -> SHM       (host shared memory)
    different node            -> (future) nixl / mooncake (RDMA)

Because the decision is a function of placement alone, the sender and the
receiver independently compute the *same* transport for an edge — the sender
from its target, the receiver from ``msg.from_stage`` — so no per-message
transport tag is needed. ``gpu_stage_names`` is the set of GPU-resident stages,
which both sides share.

This is also the single seam for cross-node unification: when multi-node lands,
the only change is to return a nixl/mooncake relay for cross-node edges here.
"""
from __future__ import annotations

import logging
from contextlib import suppress
from enum import Enum
from typing import Any

from sglang_omni.relay.base import Relay, create_relay

logger = logging.getLogger(__name__)


class TransportKind(str, Enum):
    LOCAL = "local"
    CUDA_IPC = "cuda_ipc"
    SHM = "shm"
    # Future cross-node kinds: NIXL / MOONCAKE.


class TransportRouter:
    """Maps an inter-stage edge to its transport and owns the backing relays."""

    def __init__(
        self,
        *,
        stage_name: str,
        gpu_id: int | None,
        same_process_targets: set[str] | None,
        gpu_stage_names: set[str] | None,
        local_dispatcher: Any | None,
        relay_config: dict[str, Any] | None = None,
        injected_relay: Relay | None = None,
    ) -> None:
        self.stage_name = stage_name
        self.gpu_id = gpu_id
        self._same_process = set(same_process_targets or ())
        self._gpu_stages = set(gpu_stage_names or ())
        self.local_dispatcher = local_dispatcher
        self._relay_config = dict(relay_config or {})
        # An explicitly supplied relay (tests / future cross-node override) is
        # used for every relay kind, bypassing locality-based construction.
        self._injected = injected_relay
        self._relays: dict[TransportKind, Relay] = {}

    @property
    def _self_is_gpu(self) -> bool:
        return self.gpu_id is not None

    def is_local(self, target: str) -> bool:
        return target in self._same_process

    def outbound(self, target: str) -> TransportKind:
        """Transport for data this stage sends to ``target``."""
        if target in self._same_process:
            return TransportKind.LOCAL
        if self._self_is_gpu and target in self._gpu_stages:
            return TransportKind.CUDA_IPC
        return TransportKind.SHM

    def inbound(self, from_stage: str) -> TransportKind:
        """Transport for relay data this stage receives from ``from_stage``.

        Same-process traffic never reaches the relay (it arrives via the local
        dispatcher), so only the relay-backed kinds are returned here.
        """
        if self._self_is_gpu and from_stage in self._gpu_stages:
            return TransportKind.CUDA_IPC
        return TransportKind.SHM

    def relay(self, kind: TransportKind) -> Relay:
        if self._injected is not None:
            return self._injected
        relay = self._relays.get(kind)
        if relay is None:
            relay = self._build_relay(kind)
            self._relays[kind] = relay
        return relay

    def relay_for(self, target: str) -> tuple[TransportKind, Relay]:
        kind = self.outbound(target)
        return kind, self.relay(kind)

    def inbound_relay(self, from_stage: str) -> Relay:
        return self.relay(self.inbound(from_stage))

    def _build_relay(self, kind: TransportKind) -> Relay:
        cfg = self._relay_config
        engine_id = cfg.get("worker_id", f"{self.stage_name}_relay")
        if kind is TransportKind.CUDA_IPC:
            if self.gpu_id is None:
                raise ValueError(
                    f"cuda_ipc relay requested for non-GPU stage "
                    f"{self.stage_name!r}"
                )
            return create_relay(
                "cuda_ipc", engine_id=engine_id, device=f"cuda:{self.gpu_id}"
            )
        if kind is TransportKind.SHM:
            return create_relay(
                "shm",
                engine_id=engine_id,
                device="cpu",
                slot_size_mb=cfg.get("slot_size_mb", 512),
                credits=cfg.get("credits", 2),
            )
        raise ValueError(f"TransportRouter cannot build a relay for {kind}")

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
        if self._injected is not None:
            return [self._injected]
        return list(self._relays.values())
