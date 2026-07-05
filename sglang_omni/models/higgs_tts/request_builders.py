# SPDX-License-Identifier: Apache-2.0
"""Per-request data + StagePayload <-> scheduler adapters for Higgs TTS (V1)."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.models.higgs_tts.rollout_trace import build_omni_rollout_trace
from sglang_omni.models.higgs_tts.vocoder_scheduler import (
    DEFAULT_HIGGS_INITIAL_CHUNK_FRAMES,
    DEFAULT_HIGGS_STREAM_FOLLOWUP_STRIDE,
    DEFAULT_HIGGS_STREAM_STRIDE,
    HIGGS_STREAM_FOLLOWUP_STRIDE_METADATA,
    HIGGS_STREAM_STRIDE_METADATA,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData
from sglang_omni.scheduling.streaming_vocoder import (
    INITIAL_CODEC_CHUNK_FRAMES_PARAM,
    resolve_initial_codec_chunk_frames,
)


@dataclass
class HiggsSGLangRequestData(SGLangARRequestData):
    """Per-request state for the Higgs TTS scheduler."""

    reference_codes_delayed: list[list[int]] | None = None
    num_codebooks: int = 8
    codebook_size: int = 1026
    output_codes: list[torch.Tensor] = field(default_factory=list)
    output_code_buffer: torch.Tensor | None = None
    output_code_count: int = 0
    output_logprobs: list[torch.Tensor] = field(default_factory=list)
    return_omni_rollout: bool = False
    generation_done: bool = False
    engine_start_s: float = 0.0
    stream_metadata: dict[str, Any] | None = None
    stream_code_buffer: list[torch.Tensor] = field(default_factory=list)
    stream_code_first_flush_done: bool = False
    stream_code_seen_rows: int = 0
    stream_code_next_flush_rows: int = 0

    # Voice fusion. ``fusion_group_id`` is shared by all siblings of one fused
    # request (None for ordinary requests); ``fusion_weight`` is this sibling's
    # blend weight; ``fusion_is_leader`` marks the one sibling whose decoded
    # codes are emitted as audio. ``fusion_siblings`` is a builder->scheduler
    # side-channel: the leader's req_data carries the follower req_datas so the
    # scheduler can enqueue the whole group atomically (see omni_scheduler
    # ``_enqueue_built_request``). Followers carry ``fusion_siblings=None``.
    fusion_group_id: str | None = None
    fusion_weight: float = 1.0
    fusion_is_leader: bool = True
    fusion_siblings: list["HiggsSGLangRequestData"] | None = None


class _ResettableHiggsModel(Protocol):
    def reset_request(self, req_id: str) -> None: ...


_HiggsRequestBuilder = Callable[[StagePayload], HiggsSGLangRequestData]
_HiggsResultAdapter = Callable[[HiggsSGLangRequestData], StagePayload]


def _perf_counter() -> float:
    return time.perf_counter()


def _ref_audio_fingerprint(codes: list[list[int]] | None) -> str | None:
    """Stable hash of the full N-codebook ref-audio sequence.

    Returned as a short hex string used as ``Req.extra_key``. ``None`` for
    zero-shot (no ref audio) so all zero-shot requests share the radix subtree.
    Each codec value packs into 2 bytes (range 0..1025) so the hash is
    sensitive to every codebook, not just cb0.
    """
    if not codes:
        return None
    buf = bytearray(2 * sum(len(row) for row in codes))
    i = 0
    for row in codes:
        for c in row:
            buf[i] = c & 0xFF
            buf[i + 1] = (c >> 8) & 0xFF
            i += 2
    return hashlib.blake2b(bytes(buf), digest_size=16).hexdigest()


def _build_one_higgs_request(
    *,
    prompt_token_ids: list[int],
    reference_codes_delayed: list[list[int]] | None,
    num_codebooks: int,
    codebook_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    seed: int | None,
    request_id: str,
    return_logprob: bool = False,
    return_omni_rollout: bool = False,
    fusion_group_id: str | None = None,
    fusion_weight: float = 1.0,
    fusion_is_leader: bool = True,
) -> HiggsSGLangRequestData:
    """Build one ``HiggsSGLangRequestData`` (one batch row / sampler slot).

    Shared by the ordinary single-voice path and each fan-out sibling of a
    voice-fusion request. ``seed`` MUST be a concrete int for fusion siblings
    (the caller assigns a shared seed) so they draw the same frame each step.
    """
    input_ids_list = list(prompt_token_ids)
    input_ids = torch.tensor(input_ids_list, dtype=torch.long)

    sp_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
    }
    if top_p is not None:
        sp_kwargs["top_p"] = float(top_p)
    if top_k is not None:
        sp_kwargs["top_k"] = int(top_k)
    if seed is not None:
        sp_kwargs["sampling_seed"] = int(seed)
    sampling_params = SamplingParams(**sp_kwargs)
    # tokenizer_manager.normalize() is bypassed in our custom pipeline;
    # without it stop_strs / stop_regex_strs stay None and the upstream
    # scheduler's update_finish_state trips on ``len(None)``.
    sampling_params.normalize(tokenizer=None)

    # vocab_size = backbone text vocab so cb0 rides sglang's standard sampler path.
    # extra_key namespaces the radix cache per ref-audio fingerprint so prompts
    # sharing the -100 placeholder prefix can never cross-contaminate KV. Each
    # fusion sibling has its own ref audio → its own fingerprint → its own KV.
    req = Req(
        rid=request_id,
        origin_input_text="",
        origin_input_ids=input_ids_list,
        sampling_params=sampling_params,
        vocab_size=151_936,
        extra_key=_ref_audio_fingerprint(reference_codes_delayed),
    )
    # V1's prefill manager probes these attrs; absence triggers AttributeError.
    req._codec_suppress_tokens = None
    req._input_embeds_are_projected = False

    return HiggsSGLangRequestData(
        input_ids=input_ids,
        req=req,
        reference_codes_delayed=reference_codes_delayed,
        num_codebooks=int(num_codebooks),
        codebook_size=int(codebook_size),
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p) if top_p is not None else 1.0,
        top_k=int(top_k) if top_k is not None else -1,
        return_logprob=bool(return_logprob),
        return_omni_rollout=bool(return_omni_rollout),
        fusion_group_id=fusion_group_id,
        fusion_weight=float(fusion_weight),
        fusion_is_leader=fusion_is_leader,
    )


def build_sglang_higgs_request(
    state: HiggsTtsState, *, request_id: str = ""
) -> HiggsSGLangRequestData:
    return _build_one_higgs_request(
        prompt_token_ids=state.prompt_token_ids,
        reference_codes_delayed=state.reference_codes_delayed,
        num_codebooks=int(state.num_codebooks),
        codebook_size=int(state.codebook_size),
        max_new_tokens=int(state.max_new_tokens),
        temperature=float(state.temperature),
        top_p=state.top_p,
        top_k=state.top_k,
        seed=state.seed,
        request_id=request_id,
        return_logprob=bool(state.return_logprob),
        return_omni_rollout=bool(state.return_omni_rollout),
    )


def _coerce_fusion_weights(refs: list[dict[str, Any]]) -> list[float]:
    """Per-ref blend weights, normalized to sum 1. Defaults to uniform when a
    ref omits ``weight`` or all weights are non-positive."""
    raw = [float(r.get("weight", 1.0)) for r in refs]
    if any(w < 0 for w in raw):
        raise ValueError(f"fusion weights must be >= 0, got {raw}")
    total = sum(raw)
    if total <= 0:
        return [1.0 / len(refs)] * len(refs)
    return [w / total for w in raw]


def build_fusion_sibling_requests(
    state: HiggsTtsState,
    *,
    request_id: str,
) -> HiggsSGLangRequestData:
    """Fan a multi-reference voice-fusion request out into N sibling rows.

    Each entry of ``state.fusion_refs`` (``{"codes_delayed", "weight",
    "prompt_token_ids"}`` — the per-sibling prompt is pre-built in the
    preprocessing stage, which owns the tokenizer) becomes one sibling
    ``HiggsSGLangRequestData`` that prefills its own reference voice into its
    own KV context. All siblings:

    - share one ``fusion_group_id`` (``request_id``) so the model blends their
      per-codebook output distributions before sampling (see ``fusion.py``);
    - share ONE concrete ``sampling_seed`` so, given the shared fused
      distribution, every sibling draws the identical frame each AR step and
      their delay/EOC state machines advance in lock-step (this is what makes
      the "any sibling done -> all done" barrier safe — see design doc);
    - sibling 0 is the leader (its decoded codes are emitted as audio);
      followers are deduped at collection time.

    Returns the leader req_data with the followers attached as
    ``fusion_siblings`` (a builder->scheduler side-channel; the scheduler
    enqueues the whole group atomically).
    """
    refs = state.fusion_refs or []
    if len(refs) < 2:
        raise ValueError(f"voice fusion needs >= 2 references, got {len(refs)}")
    weights = _coerce_fusion_weights(refs)

    # One shared concrete seed for the whole group. If the user pinned a seed we
    # honour it; otherwise derive a stable one from the request id so a retry of
    # the same request is reproducible. Per-sibling seeding is NOT allowed — it
    # would desync the draws and break the done-barrier invariant.
    if state.seed is not None:
        shared_seed = int(state.seed)
    else:
        shared_seed = (
            int.from_bytes(
                hashlib.blake2b(request_id.encode(), digest_size=8).digest(),
                "little",
                signed=False,
            )
            & 0x7FFF_FFFF_FFFF_FFFF
        )

    siblings: list[HiggsSGLangRequestData] = []
    for i, (ref, weight) in enumerate(zip(refs, weights)):
        codes_delayed = ref.get("codes_delayed")
        if not codes_delayed:
            raise ValueError(f"fusion ref {i} has no codes_delayed")
        prompt_ids = ref.get("prompt_token_ids")
        if not prompt_ids:
            raise ValueError(
                f"fusion ref {i} has no prompt_token_ids "
                f"(preprocessing must pre-build each sibling prompt)"
            )
        is_leader = i == 0
        sib = _build_one_higgs_request(
            prompt_token_ids=prompt_ids,
            reference_codes_delayed=codes_delayed,
            num_codebooks=int(state.num_codebooks),
            codebook_size=int(state.codebook_size),
            max_new_tokens=int(state.max_new_tokens),
            temperature=float(state.temperature),
            top_p=state.top_p,
            top_k=state.top_k,
            seed=shared_seed,
            request_id=request_id if is_leader else f"{request_id}#fuse{i}",
            # Rollout logprob/trace capture is meaningful only for the leader —
            # its codes are the request's real output; followers' codes are
            # duplicates that are discarded (see is_fusion_follower), so
            # capturing logprobs for them would be wasted compute.
            return_logprob=bool(state.return_logprob) if is_leader else False,
            return_omni_rollout=bool(state.return_omni_rollout) if is_leader else False,
            fusion_group_id=request_id,
            fusion_weight=float(weight),
            fusion_is_leader=is_leader,
        )
        siblings.append(sib)

    leader = siblings[0]
    leader.fusion_siblings = siblings[1:]
    return leader


def build_higgs_stream_metadata(
    payload: StagePayload,
    data: HiggsSGLangRequestData,
    *,
    stream_stride: int = DEFAULT_HIGGS_STREAM_STRIDE,
    stream_followup_stride: int = DEFAULT_HIGGS_STREAM_FOLLOWUP_STRIDE,
    initial_chunk_frames: int = DEFAULT_HIGGS_INITIAL_CHUNK_FRAMES,
) -> dict[str, Any] | None:
    params = payload.request.params
    if not isinstance(params, dict):
        raise TypeError(
            f"Higgs request params must be a dict, got {type(params).__name__}"
        )
    if not bool(params.get("stream", False)):
        return None

    num_codebooks = int(data.num_codebooks)
    codebook_size = int(data.codebook_size)
    if num_codebooks <= 0 or codebook_size <= 2:
        raise ValueError(
            f"Invalid Higgs stream codec contract: "
            f"num_codebooks={num_codebooks}, codebook_size={codebook_size}"
        )
    metadata: dict[str, Any] = {
        "modality": "audio_codes",
        "stream": True,
        "num_codebooks": num_codebooks,
        "codebook_size": codebook_size,
        HIGGS_STREAM_STRIDE_METADATA: stream_stride,
        HIGGS_STREAM_FOLLOWUP_STRIDE_METADATA: stream_followup_stride,
        INITIAL_CODEC_CHUNK_FRAMES_PARAM: resolve_initial_codec_chunk_frames(
            params,
            steady_chunk_frames=max(1, stream_stride - num_codebooks + 1),
            default_frames=initial_chunk_frames,
        ),
    }
    return metadata


def apply_higgs_result(state: HiggsTtsState, data: HiggsSGLangRequestData) -> None:
    num_codebooks = int(data.num_codebooks)
    if data.output_code_buffer is not None and data.output_code_count > 0:
        codes = data.output_code_buffer[: data.output_code_count].to(torch.long)
        state.output_codes_delayed = codes.tolist()
        state.completion_tokens = int(codes.shape[0])
    elif data.output_codes:
        codes = torch.stack(data.output_codes, dim=0).to(torch.long)
        state.output_codes_delayed = codes.tolist()
        state.completion_tokens = int(codes.shape[0])
    else:
        codes = torch.empty((0, num_codebooks), dtype=torch.long)
        state.output_codes_delayed = None

    if data.return_omni_rollout:
        logprobs = (
            torch.stack(data.output_logprobs, dim=0).to(torch.float32)
            if (data.return_logprob and data.output_logprobs)
            else None
        )
        state.omni_rollout = build_omni_rollout_trace(
            codes,
            num_codebooks=num_codebooks,
            codebook_vocab_size=int(data.codebook_size),
            delayed_logprobs=logprobs,
        )
    state.prompt_tokens = len(data.input_ids)


def make_higgs_scheduler_adapters(
    model: _ResettableHiggsModel,
    *,
    max_new_tokens_cap: int | None = None,
    stream_stride: int = DEFAULT_HIGGS_STREAM_STRIDE,
    stream_followup_stride: int = DEFAULT_HIGGS_STREAM_FOLLOWUP_STRIDE,
    initial_chunk_frames: int = DEFAULT_HIGGS_INITIAL_CHUNK_FRAMES,
) -> tuple[_HiggsRequestBuilder, _HiggsResultAdapter]:
    """Build scheduler request/result adapters for :class:`HiggsTTSModel`."""

    def _register_fusion_group(group: list[HiggsSGLangRequestData]) -> None:
        """Tell the model which req_ids share a fusion group + their weights, so
        the decode step blends their distributions and dedups followers."""
        register = getattr(model, "set_fusion_group", None)
        if register is None:
            raise RuntimeError(
                "voice fusion requested but model has no set_fusion_group(); "
                "the loaded model build does not support fusion."
            )
        for sib in group:
            register(
                sib.req.rid,
                sib.fusion_group_id,
                sib.fusion_weight,
                is_leader=sib.fusion_is_leader,
            )

    def _finalize(
        data: HiggsSGLangRequestData, payload: StagePayload
    ) -> HiggsSGLangRequestData:
        data.engine_start_s = _perf_counter()
        data.stage_payload = payload
        data.stream_metadata = build_higgs_stream_metadata(
            payload,
            data,
            stream_stride=stream_stride,
            stream_followup_stride=stream_followup_stride,
            initial_chunk_frames=initial_chunk_frames,
        )
        return data

    def request_builder(payload: StagePayload) -> HiggsSGLangRequestData:
        state = HiggsTtsState.from_dict(payload.data)
        if max_new_tokens_cap is not None:
            state.max_new_tokens = min(
                int(state.max_new_tokens),
                int(max_new_tokens_cap),
            )

        # Voice fusion: >= 2 references → fan out into N sibling rows that share
        # a fusion group id + one concrete seed. The leader carries the followers
        # as ``fusion_siblings``; the scheduler enqueues the group atomically.
        if state.fusion_refs and len(state.fusion_refs) >= 2:
            leader = build_fusion_sibling_requests(state, request_id=payload.request_id)
            followers = leader.fusion_siblings or []
            _register_fusion_group([leader, *followers])
            # Only the leader is finalized with the stage payload + stream
            # metadata (it is the row whose codes become audio and whose
            # result_adapter emits the StagePayload). Followers still need
            # engine_start_s set for limit checks but carry no stream metadata
            # and no stage payload — they never produce a result.
            _finalize(leader, payload)
            for sib in followers:
                sib.engine_start_s = _perf_counter()
            return leader

        data = build_sglang_higgs_request(state, request_id=payload.request_id)
        return _finalize(data, payload)

    def result_adapter(data: HiggsSGLangRequestData) -> StagePayload:
        payload = data.stage_payload
        state = HiggsTtsState.from_dict(payload.data)
        apply_higgs_result(state, data)
        if data.engine_start_s:
            state.engine_time_s = _perf_counter() - data.engine_start_s
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
        )

    return request_builder, result_adapter


__all__ = [
    "HiggsSGLangRequestData",
    "INITIAL_CODEC_CHUNK_FRAMES_PARAM",
    "apply_higgs_result",
    "build_higgs_stream_metadata",
    "build_sglang_higgs_request",
    "make_higgs_scheduler_adapters",
]
