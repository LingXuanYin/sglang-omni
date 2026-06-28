# SPDX-License-Identifier: Apache-2.0
"""Omni communication engine facade used by pipeline stages."""
from __future__ import annotations

from typing import Any

import torch

from sglang_omni.comm import stage_io
from sglang_omni.comm.router import CommRouter
from sglang_omni.comm.data_ref import DataRef, TransportKind
from sglang_omni.proto import StagePayload
from sglang_omni.relay.base import Relay


class CommEngine:
    """Stage-owned communication engine.

    It owns locality classification and data_ref-based relay IO. Stages keep
    routing semantics; the engine owns byte movement mechanics.
    """

    def __init__(self, router: CommRouter) -> None:
        self.router = router

    def outbound(self, target: str) -> TransportKind:
        return self.router.outbound(target)

    def outbound_stream(self, target: str, data: torch.Tensor) -> TransportKind:
        return self.router.outbound_stream(target, data)

    def relay(self, kind: TransportKind) -> Relay:
        return self.router.relay(kind)

    def inbound_relay(self, from_stage: str) -> Relay:
        return self.router.inbound_relay(from_stage)

    async def write_payload(
        self,
        *,
        relay: Relay,
        request_id: str,
        payload: StagePayload,
        transport: TransportKind,
        from_stage: str,
        to_stage: str,
    ) -> tuple[DataRef, Any]:
        return await stage_io.write_payload(
            relay,
            request_id,
            payload,
            transport=transport,
            from_stage=from_stage,
            to_stage=to_stage,
        )

    async def read_payload(
        self,
        *,
        relay: Relay,
        request_id: str,
        data_ref: DataRef,
    ) -> StagePayload:
        return await stage_io.read_payload(relay, request_id, data_ref)

    async def send_stream_chunk(
        self,
        *,
        relay: Relay,
        control_plane: Any,
        request_id: str,
        data: torch.Tensor,
        target_stage: str,
        target_endpoint: str,
        from_stage: str,
        chunk_id: int,
        metadata: dict[str, Any] | None,
        transport: TransportKind,
    ) -> None:
        await stage_io.send_stream_chunk(
            relay,
            control_plane,
            request_id=request_id,
            data=data,
            target_stage=target_stage,
            target_endpoint=target_endpoint,
            from_stage=from_stage,
            chunk_id=chunk_id,
            metadata=metadata,
            transport=transport,
        )

    async def read_stream_chunk(
        self,
        *,
        relay: Relay,
        data_ref: DataRef,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        return await stage_io.read_stream_chunk(relay, data_ref)

    def cleanup(self, request_id: str) -> None:
        self.router.cleanup(request_id)

    def close(self) -> None:
        self.router.close()
