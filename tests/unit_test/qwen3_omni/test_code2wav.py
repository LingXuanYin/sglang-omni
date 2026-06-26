# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch

from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    Code2WavScheduler,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.scheduling.messages import IncomingMessage
from tests.unit_test.fixtures.qwen_fakes import FakeCode2WavModel, make_qwen_payload


def test_qwen_code2wav_streams_incrementally_and_abort_clears_state() -> None:
    """Preserves incremental waveform emission and request-state cleanup on abort."""
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        sample_rate=24000,
    )
    scheduler._payloads["req-1"] = make_qwen_payload(request_id="req-1")
    scheduler._ensure_request_state("req-1")

    chunk_meta = {"stream": False}  # non-streaming: final result carries full PCM
    scheduler._on_chunk(
        "req-1",
        StreamItem(0, torch.tensor([1, 10]), "talker", metadata=chunk_meta),
    )
    scheduler._on_chunk(
        "req-1",
        StreamItem(1, torch.tensor([2, 20]), "talker", metadata=chunk_meta),
    )
    scheduler._on_chunk(
        "req-1",
        StreamItem(2, torch.tensor([3, 30]), "talker", metadata=chunk_meta),
    )
    scheduler._on_done("req-1")

    message = scheduler.outbox.get_nowait()
    audio = np.frombuffer(message.data.data["audio_waveform"], dtype=np.float32)
    assert model.calls == [(1, 2, 2), (1, 2, 2)]
    assert audio.shape == (6,)

    scheduler._payloads["req-2"] = make_qwen_payload(request_id="req-2")
    scheduler._ensure_request_state("req-2")
    scheduler._pending_done.add("req-2")
    scheduler.abort("req-2")
    assert "req-2" not in scheduler._code_chunks
    assert "req-2" not in scheduler._payloads
    assert "req-2" not in scheduler._pending_done


def _chunk(chunk_id: int, code_a: int, code_b: int, *, stream: bool = False):
    return StreamItem(
        chunk_id,
        torch.tensor([code_a, code_b], dtype=torch.long),
        "talker",
        metadata={"stream": stream},
    )


def _seed_payloads(scheduler: Code2WavScheduler, request_ids: list[str]) -> None:
    for request_id in request_ids:
        scheduler._payloads[request_id] = make_qwen_payload(request_id=request_id)
        scheduler._ensure_request_state(request_id)


def _drain_final_audio(scheduler: Code2WavScheduler) -> dict[str, np.ndarray]:
    audio_by_request: dict[str, np.ndarray] = {}
    while not scheduler.outbox.empty():
        message = scheduler.outbox.get_nowait()
        if message.type != "result":
            continue
        audio_by_request[message.request_id] = np.frombuffer(
            message.data.data["audio_waveform"], dtype=np.float32
        )
    return audio_by_request


def test_qwen_code2wav_batches_ready_stream_windows_without_audio_changes() -> None:
    batched_model = FakeCode2WavModel(total_upsample=2)
    batched = Code2WavScheduler(
        batched_model,
        device="cpu",
        stream_chunk_size=2,
        left_context_size=0,
        sample_rate=24000,
        max_batch_wait_ms=0,
    )
    _seed_payloads(batched, ["req-1", "req-2"])

    batched._on_chunk("req-1", _chunk(0, 1, 10))
    batched._on_chunk("req-2", _chunk(0, 2, 20))
    batched.inbox.put(IncomingMessage("req-2", "stream_chunk", _chunk(1, 4, 40)))
    batched._on_chunk("req-1", _chunk(1, 3, 30))
    batched._on_done("req-1")
    batched._on_done("req-2")
    batched_audio = _drain_final_audio(batched)

    single_model = FakeCode2WavModel(total_upsample=2)
    single = Code2WavScheduler(
        single_model,
        device="cpu",
        stream_chunk_size=2,
        left_context_size=0,
        sample_rate=24000,
        enable_batched_decode=False,
    )
    _seed_payloads(single, ["req-1", "req-2"])
    for request_id, chunks in {
        "req-1": [_chunk(0, 1, 10), _chunk(1, 3, 30)],
        "req-2": [_chunk(0, 2, 20), _chunk(1, 4, 40)],
    }.items():
        for item in chunks:
            single._on_chunk(request_id, item)
        single._on_done(request_id)
    single_audio = _drain_final_audio(single)

    assert batched_model.calls == [(2, 2, 2)]
    assert single_model.calls == [(1, 2, 2), (1, 2, 2)]
    assert set(batched_audio) == {"req-1", "req-2"}
    for request_id in batched_audio:
        np.testing.assert_array_equal(
            batched_audio[request_id], single_audio[request_id]
        )


def test_qwen_code2wav_batches_same_length_final_partials() -> None:
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=10,
        left_context_size=0,
        sample_rate=24000,
        max_batch_wait_ms=0,
    )
    _seed_payloads(scheduler, ["req-1", "req-2", "req-3"])

    scheduler._on_chunk("req-1", _chunk(0, 1, 10))
    scheduler._on_chunk("req-1", _chunk(1, 2, 20))
    scheduler._on_chunk("req-2", _chunk(0, 3, 30))
    scheduler._on_chunk("req-2", _chunk(1, 4, 40))
    scheduler._on_chunk("req-3", _chunk(0, 5, 50))
    scheduler.inbox.put(IncomingMessage("req-2", "stream_done"))
    scheduler.inbox.put(IncomingMessage("req-3", "stream_done"))

    scheduler._on_done("req-1")
    _drain_final_audio(scheduler)

    assert model.calls == [(2, 2, 2), (1, 2, 1)]


def test_qwen_code2wav_streaming_first_window_can_emit_early() -> None:
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=4,
        first_stream_chunk_size=2,
        left_context_size=0,
        sample_rate=24000,
        max_batch_wait_ms=0,
    )
    _seed_payloads(scheduler, ["req-1"])

    scheduler._on_chunk("req-1", _chunk(0, 1, 10, stream=True))
    assert model.calls == []
    scheduler._on_chunk("req-1", _chunk(1, 2, 20, stream=True))
    assert model.calls == [(1, 2, 2)]

    stream_messages = []
    while not scheduler.outbox.empty():
        stream_messages.append(scheduler.outbox.get_nowait())
    assert any(message.type == "stream" for message in stream_messages)

    for idx in range(2, 5):
        scheduler._on_chunk("req-1", _chunk(idx, idx + 1, idx + 10, stream=True))
    assert model.calls == [(1, 2, 2)]
    scheduler._on_chunk("req-1", _chunk(5, 6, 15, stream=True))
    assert model.calls == [(1, 2, 2), (1, 2, 4)]


def test_qwen_code2wav_non_streaming_keeps_full_window_for_first_decode() -> None:
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=4,
        first_stream_chunk_size=2,
        left_context_size=0,
        sample_rate=24000,
        max_batch_wait_ms=0,
    )
    _seed_payloads(scheduler, ["req-1"])

    for idx in range(2):
        scheduler._on_chunk("req-1", _chunk(idx, idx + 1, idx + 10, stream=False))
    assert model.calls == []

    for idx in range(2, 4):
        scheduler._on_chunk("req-1", _chunk(idx, idx + 1, idx + 10, stream=False))
    assert model.calls == [(1, 2, 4)]


def test_qwen_code2wav_first_window_env_can_disable(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_OMNI_QWEN3_CODE2WAV_FIRST_CHUNK_SIZE", "0")
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=4,
        first_stream_chunk_size=2,
        left_context_size=0,
        sample_rate=24000,
        max_batch_wait_ms=0,
    )
    _seed_payloads(scheduler, ["req-1"])

    for idx in range(2):
        scheduler._on_chunk("req-1", _chunk(idx, idx + 1, idx + 10, stream=True))
    assert model.calls == []

    for idx in range(2, 4):
        scheduler._on_chunk("req-1", _chunk(idx, idx + 1, idx + 10, stream=True))
    assert model.calls == [(1, 2, 4)]


def test_qwen_code2wav_skips_eos_chunks_before_batching() -> None:
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=1,
        left_context_size=0,
        sample_rate=24000,
        codec_eos_token_id=2150,
        max_batch_wait_ms=0,
    )
    _seed_payloads(scheduler, ["req-1"])

    scheduler._on_chunk(
        "req-1",
        StreamItem(
            0,
            torch.tensor([2150, 2150], dtype=torch.long),
            "talker",
            metadata={"stream": False},
        ),
    )

    assert model.calls == []
    assert scheduler._code_chunks["req-1"] == []


def test_qwen_code2wav_tensor_buffer_builds_context_windows() -> None:
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=10,
        left_context_size=2,
        sample_rate=24000,
        max_batch_wait_ms=0,
    )
    _seed_payloads(scheduler, ["req-1"])

    for idx in range(5):
        scheduler._append_stream_chunk("req-1", _chunk(idx, idx + 1, idx + 10))
    scheduler._emitted["req-1"] = 3

    plans = scheduler._build_decode_plans(
        force_request_ids={"req-1"},
        request_ids=["req-1"],
    )

    assert len(plans) == 1
    plan = plans[0]
    assert plan.start == 3
    assert plan.end == 5
    assert plan.trim == 4
    assert tuple(plan.codes.shape) == (2, 4)
    assert torch.equal(plan.codes[0], torch.tensor([2, 3, 4, 5]))
    assert torch.equal(plan.codes[1], torch.tensor([11, 12, 13, 14]))
    assert scheduler._code_buffers["req-1"].length == 5
    assert scheduler._code_chunks["req-1"] == []


def test_qwen_code2wav_tensor_buffer_grows_without_losing_chunks() -> None:
    scheduler = Code2WavScheduler(
        FakeCode2WavModel(total_upsample=2),
        device="cpu",
        stream_chunk_size=64,
        left_context_size=0,
        sample_rate=24000,
    )
    _seed_payloads(scheduler, ["req-1"])

    for idx in range(20):
        scheduler._append_stream_chunk("req-1", _chunk(idx, idx + 1, idx + 100))

    buffer = scheduler._code_buffers["req-1"]
    assert buffer.length == 20
    assert buffer.chunks is not None
    assert int(buffer.chunks.shape[0]) >= 20
    window = scheduler._code_window("req-1", 0, 20).transpose(0, 1)
    assert torch.equal(window[0], torch.arange(1, 21))
    assert torch.equal(window[1], torch.arange(100, 120))


def test_qwen_code2wav_tensor_buffer_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_OMNI_QWEN3_CODE2WAV_TENSOR_BUFFER", "0")
    scheduler = Code2WavScheduler(
        FakeCode2WavModel(total_upsample=2),
        device="cpu",
        stream_chunk_size=64,
        left_context_size=0,
        sample_rate=24000,
    )
    _seed_payloads(scheduler, ["req-1"])

    scheduler._append_stream_chunk("req-1", _chunk(0, 1, 10))

    assert scheduler._code_buffers == {}
    assert len(scheduler._code_chunks["req-1"]) == 1
    assert torch.equal(
        scheduler._code_window("req-1", 0, 1).squeeze(0),
        torch.tensor([1, 10]),
    )
