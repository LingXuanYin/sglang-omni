from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.pipeline_state import (
    PipelineStateBase,
    build_usage,
    load_state,
    store_state,
)


@dataclass
class _DummyState(PipelineStateBase):
    value: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = {"value": self.value, "sample_rate": self.sample_rate}
        self.append_usage_fields(data)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_DummyState":
        return cls(
            value=data.get("value", ""),
            sample_rate=int(data.get("sample_rate", 24000)),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            engine_time_s=float(data.get("engine_time_s", 0.0)),
        )


def test_build_usage_omits_empty_usage() -> None:
    assert build_usage(_DummyState()) is None


def test_build_usage_includes_total_and_rounded_engine_time() -> None:
    state = _DummyState(prompt_tokens=3, completion_tokens=5, engine_time_s=1.23456789)

    assert build_usage(state) == {
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
        "engine_time_s": 1.234568,
    }


def test_load_and_store_state_round_trip_stage_payload() -> None:
    payload = StagePayload(
        request_id="req",
        request=OmniRequest(inputs={}),
        data={"value": "ok", "prompt_tokens": 2},
    )

    state = load_state(payload, _DummyState)
    state.completion_tokens = 4
    stored = store_state(payload, state)

    assert stored is payload
    assert payload.data == {
        "value": "ok",
        "sample_rate": 24000,
        "prompt_tokens": 2,
        "completion_tokens": 4,
    }


def test_serialize_value_detaches_tensor_to_cpu() -> None:
    tensor = torch.tensor([1, 2], requires_grad=False)

    value = PipelineStateBase.serialize_value(tensor)

    assert isinstance(value, torch.Tensor)
    assert value.device.type == "cpu"
    assert value.tolist() == [1, 2]


def test_tts_pipeline_states_share_base_usage_contract() -> None:
    from sglang_omni.models.fishaudio_s2_pro.payload_types import S2ProState
    from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
    from sglang_omni.models.moss_tts.payload_types import MossTTSState
    from sglang_omni.models.moss_tts_local.payload_types import MossTTSLocalState
    from sglang_omni.models.qwen3_tts.payload_types import Qwen3TTSState

    state_classes = (
        S2ProState,
        HiggsTtsState,
        MossTTSState,
        MossTTSLocalState,
        Qwen3TTSState,
    )

    for state_cls in state_classes:
        assert issubclass(state_cls, PipelineStateBase)
