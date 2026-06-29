# SPDX-License-Identifier: Apache-2.0
"""sglang-native Higgs Multimodal Qwen3 TTS model."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from torch import nn

from sglang_omni.models.higgs_tts.fusion import (
    fuse_group_generation_done,
    fuse_group_logits,
)
from sglang_omni.models.higgs_tts.hf_config import HiggsMultimodalQwen3Config
from sglang_omni.models.higgs_tts.modeling import (
    HiggsFusedMultiTextEmbedding,
    HiggsFusedMultiTextHead,
)
from sglang_omni.models.higgs_tts.sampler import (
    K_MAX,
    NO_SEED,
    HiggsBatchedSamplerState,
    batched_step,
    batched_step_direct,
)
from sglang_omni.models.higgs_tts.weight_loader import DiscreteWeightMapper
from sglang_omni.sampling.seed import resolve_row_seed

logger = logging.getLogger(__name__)

# Higgs ckpt prefixes → sglang Qwen3ForCausalLM parameter tree (under ``backbone.``).
_BACKBONE_PREFIX_MAP: dict[str, str] = {
    "tied.embedding.text_embedding.": "backbone.model.embed_tokens.",
    "body.layers.": "backbone.model.layers.",
    "body.norm.": "backbone.model.norm.",
    "tied.head.text_head.": "backbone.lm_head.",
}


@dataclass
class HiggsGenParams:
    """Per-request decoding parameters consumed by :func:`sampler.step`."""

    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    # Voice-fusion sibling grouping. ``fusion_group_id`` is ``None`` for a normal
    # (single-reference) request; for a fusion request all N sibling rows share
    # the same id and their output-distribution logits are weighted-averaged
    # before sampling (see :meth:`HiggsTTSModel.decode_codebooks_batch`).
    # ``fusion_weight`` is this row's blend weight (group weights are normalized
    # to sum to 1 at fuse time).
    fusion_group_id: str | None = None
    fusion_weight: float = 1.0


def _resolve_max_running_requests() -> int:
    try:
        from sglang.srt.server_args import get_global_server_args

        return int(get_global_server_args().max_running_requests)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        fallback = 64
        logger.warning(
            f"Falling back to Higgs max_running_requests={fallback} because "
            f"SGLang global server args are unavailable: {exc}"
        )
        return fallback


def _flat_sampling_attr(sampling_info, attr: str) -> list | None:
    """Return ``sampling_info.<attr>`` as a flat Python list, or ``None``.

    One D2H per attribute (not per row).
    """
    val = getattr(sampling_info, attr, None)
    if val is None:
        return None
    if hasattr(val, "cpu"):
        return val.detach().cpu().flatten().tolist()
    return list(val)


class _HiggsMultimodalEmbedding(nn.Module):
    """Container matching the Higgs checkpoint layout for straight prefix subst."""

    def __init__(self, num_codebooks: int, vocab_size: int, hidden_size: int):
        super().__init__()
        self.modality_embedding_0 = HiggsFusedMultiTextEmbedding(
            num_codebooks=num_codebooks,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )


class HiggsTTSModel(nn.Module):
    """Higgs Multimodal Qwen3 model (discrete TTS path) adapted for sglang.

    Composition over :class:`sglang.srt.models.qwen3.Qwen3ForCausalLM` —
    the backbone handles paged attention, KV cache, logits processing and
    standard text weight loading. This wrapper adds:

    - ``multimodal_embedding.modality_embedding_0``: the fused
      :class:`HiggsFusedMultiTextEmbedding` (shape ``[N*V, D]``).
    - ``modality_head``: the fused :class:`HiggsFusedMultiTextHead`, tied
      to the embedding weight when ``audio_encoder_config.tie_word_embeddings``.
    - :meth:`load_weights` that remaps Higgs checkpoint names and splits
      the stream between the backbone and the multimodal modules.

    Multi-codebook input embedding overlay (the ``-100`` placeholder paste
    from the reference audio) is performed by the engine model_runner; this
    model just consumes the prepared ``input_embeds`` in its forward.
    """

    def __init__(
        self,
        config: HiggsMultimodalQwen3Config,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        text_config = config.get_text_config()
        self.backbone = Qwen3ForCausalLM(
            text_config,
            quant_config=quant_config,
            prefix=prefix + "backbone" if prefix else "backbone",
        )

        enc_cfg = config.audio_encoder_config or {}
        encoder_type = enc_cfg.get("encoder_type", "discrete")
        if encoder_type != "discrete":
            raise NotImplementedError(
                f"HiggsTTSModel currently supports only the discrete "
                f"TTS path; got encoder_type={encoder_type!r}. Whisper/Qwen3-AUT "
                f"(ASR) encoders are planned for a future PR."
            )

        num_codebooks: int = int(enc_cfg["num_codebooks"])
        vocab_size: int = int(enc_cfg["vocab_size"])
        hidden_size: int = int(enc_cfg.get("out_dim", text_config.hidden_size))
        self._num_codebooks = num_codebooks
        self._codebook_vocab_size = vocab_size
        self._tie_modality = bool(enc_cfg.get("tie_word_embeddings", True))

        self.multimodal_embedding = _HiggsMultimodalEmbedding(
            num_codebooks=num_codebooks,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )
        self.modality_head = HiggsFusedMultiTextHead(
            num_codebooks=num_codebooks,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )
        # Match backbone bf16 dtype; fp32 fused embed accumulates ~1 ULP per AR step.
        backbone_dtype = self.backbone.model.embed_tokens.weight.dtype
        self.multimodal_embedding.to(dtype=backbone_dtype)
        self.modality_head.to(dtype=backbone_dtype)
        if self._tie_modality:
            self.modality_head.weight = (
                self.multimodal_embedding.modality_embedding_0.weight
            )

        self._sampler_pool_max_running_requests = _resolve_max_running_requests()
        pool_size = self._sampler_pool_max_running_requests + 1
        self._sampler_pool = HiggsBatchedSamplerState(
            max_batch_size=pool_size,
            num_codebooks=num_codebooks,
            device=self.backbone.model.embed_tokens.weight.device,
        )
        self._padding_row = self._sampler_pool_max_running_requests
        self._rid_to_row: dict[str, int] = {}
        self._free_rows: list[int] = list(
            range(self._sampler_pool_max_running_requests)
        )
        self._output_codes: dict[str, list[torch.Tensor]] = {}
        cg_device = self.backbone.model.embed_tokens.weight.device
        self._cg_row_indices = torch.zeros(
            pool_size, dtype=torch.long, device=cg_device
        )
        self._cg_temperature = torch.ones(
            pool_size, dtype=torch.float32, device=cg_device
        )
        self._cg_top_p = torch.ones(pool_size, dtype=torch.float32, device=cg_device)
        self._cg_top_k_buf = torch.full(
            (pool_size,),
            K_MAX,
            dtype=torch.long,
            device=cg_device,
        )
        self._cg_codes_BN = torch.zeros(
            pool_size, num_codebooks, dtype=torch.long, device=cg_device
        )
        # Note(Jiaxin): Packs codes_BN | was_done | active_generation_done into one buffer.
        self._cg_collect_staging = torch.zeros(
            pool_size, num_codebooks + 2, dtype=torch.long, device=cg_device
        )
        self._cg_was_done = torch.zeros(pool_size, dtype=torch.bool, device=cg_device)

        self._cg_active_delay_count = torch.zeros(
            pool_size, dtype=torch.int32, device=cg_device
        )
        self._cg_active_eoc_countdown = torch.full(
            (pool_size,), -1, dtype=torch.int32, device=cg_device
        )
        self._cg_active_generation_done = torch.zeros(
            pool_size, dtype=torch.bool, device=cg_device
        )
        self._cg_active_last_codes = torch.zeros(
            pool_size, num_codebooks, dtype=torch.long, device=cg_device
        )
        self._cg_active_seeds = torch.full(
            (pool_size,), NO_SEED, dtype=torch.long, device=cg_device
        )
        self._cg_active_step_count = torch.zeros(
            pool_size, dtype=torch.long, device=cg_device
        )

        # Voice-fusion shadow buffers. ``_cg_fusion_group`` holds, per batch
        # slot, the *batch-local* group id used to blend sibling rows' output
        # distributions in ``decode_codebooks_batch_cg``; ``_cg_fusion_weight``
        # is the per-row blend weight. Default (filled each step by the runner):
        # group id = own slot index, weight = 1.0 → the blend is a no-op, so
        # non-fusion decoding is numerically unchanged.
        self._cg_fusion_group = torch.arange(
            pool_size, dtype=torch.long, device=cg_device
        )
        self._cg_fusion_weight = torch.ones(
            pool_size, dtype=torch.float32, device=cg_device
        )
        # Engine-side fusion bookkeeping. ``_fusion_group_of[req_id]`` maps a
        # sibling request id to its shared fusion group id; ``_fusion_weight_of``
        # to its blend weight; ``_fusion_leader`` records which member emits
        # audio. Populated by :meth:`set_fusion_group`.
        self._fusion_group_of: dict[str, str] = {}
        self._fusion_weight_of: dict[str, float] = {}
        self._fusion_leader: dict[str, bool] = {}
        # Expected member count per fusion group id, set at registration. The
        # decode reduction checks the per-step batch contains all members; a
        # short count means the upstream scheduler split the group (e.g. KV
        # retract) and we fail loud rather than emit silently un-fused audio.
        self._fusion_group_size: dict[str, int] = {}

    @property
    def language_model(self) -> Qwen3ForCausalLM:
        """Decoder handle for SGLang prefill-graph discovery; a property
        keeps the parameter tree free of a duplicate alias."""
        return self.backbone

    def set_fusion_group(
        self, req_id: str, group_id: str | None, weight: float, *, is_leader: bool
    ) -> None:
        """Register ``req_id`` as a member of voice-fusion group ``group_id``.

        ``group_id is None`` clears any fusion membership (normal request).
        Sibling rows sharing a ``group_id`` get their per-codebook output
        distributions weighted-averaged before sampling; only the leader's
        decoded codes are emitted as audio. Idempotent.
        """
        if group_id is None:
            old = self._fusion_group_of.pop(req_id, None)
            self._fusion_weight_of.pop(req_id, None)
            self._fusion_leader.pop(req_id, None)
            if old is not None:
                # Drop the size entry once every member of the group has cleared.
                remaining = sum(1 for g in self._fusion_group_of.values() if g == old)
                if remaining == 0:
                    self._fusion_group_size.pop(old, None)
            return
        self._fusion_group_of[req_id] = group_id
        self._fusion_weight_of[req_id] = float(weight)
        self._fusion_leader[req_id] = bool(is_leader)
        # Track the expected member count so the decode-time reduction can
        # fail loud if the upstream scheduler ever splits a group across steps
        # (see fuse-group barrier; a partial group would silently un-fuse).
        self._fusion_group_size[group_id] = (
            self._fusion_group_size.get(group_id, 0) + 1
        )

    def is_fusion_leader(self, req_id: str) -> bool:
        """True iff ``req_id`` is a fusion member and the group's output leader."""
        return self._fusion_leader.get(req_id, True)

    def is_fusion_follower(self, req_id: str) -> bool:
        """True iff ``req_id`` is a fusion member that is NOT the leader (its
        decoded codes duplicate the leader's and must not be emitted)."""
        return req_id in self._fusion_group_of and not self._fusion_leader.get(
            req_id, True
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.backbone.get_input_embeddings()

    def get_multimodal_embedding(self) -> HiggsFusedMultiTextEmbedding:
        return self.multimodal_embedding.modality_embedding_0

    def get_modality_head(self) -> HiggsFusedMultiTextHead:
        return self.modality_head

    @property
    def num_codebooks(self) -> int:
        return self._num_codebooks

    @property
    def codebook_vocab_size(self) -> int:
        return self._codebook_vocab_size

    @property
    def sampler_pool_max_running_requests(self) -> int:
        return self._sampler_pool_max_running_requests

    def acquire_row(self, req_id: str) -> int:
        """Allocate or look up the sampler-pool row for ``req_id``. Idempotent."""
        row = self._rid_to_row.get(req_id)
        if row is not None:
            return row
        if not self._free_rows:
            max_running_requests = self._sampler_pool_max_running_requests
            raise RuntimeError(
                f"HiggsTTSModel sampler pool exhausted "
                f"(max_running_requests={max_running_requests}); raise "
                f"``max_running_requests`` or limit concurrent requests."
            )
        row = self._free_rows.pop()
        self._rid_to_row[req_id] = row
        self._sampler_pool.reset_row(row)
        return row

    def set_request_seed(self, req_id: str, seed: int | None) -> None:
        """Pin req_id's sampler seed (None -> unseeded torch.multinomial).

        Constant across the request's AR steps; consumed by
        multinomial_with_seed for seeded rows.
        """
        row = self.acquire_row(req_id)
        self._sampler_pool.seeds[row] = (
            NO_SEED if seed is None else resolve_row_seed(seed)
        )

    def release_row(self, req_id: str) -> None:
        """Return ``req_id``'s row to the free pool and drop its output codes."""
        row = self._rid_to_row.pop(req_id, None)
        if row is not None:
            self._free_rows.append(row)
        self._output_codes.pop(req_id, None)
        self.set_fusion_group(req_id, None, 1.0, is_leader=True)

    def reset_request(self, req_id: str) -> None:
        self.release_row(req_id)

    def get_output_codes(self, req_id: str) -> torch.Tensor:
        codes = self._output_codes.get(req_id)
        if not codes:
            return torch.empty(
                (0, self._num_codebooks),
                dtype=torch.long,
                device=self.multimodal_embedding.modality_embedding_0.weight.device,
            )
        return torch.stack(codes, dim=0).to(torch.long)

    def _batch_local_fusion(
        self, gen_params: list[HiggsGenParams], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Map per-row ``fusion_group_id`` to batch-local group indices.

        Returns ``(group_B, weight_B, is_fused)``:
        - ``group_B`` ``[B]`` long: rows sharing a fusion id get the same index;
          unfused rows are their own singleton group (index = own row). The
          indices are batch-local (in ``[0, B)``) so they can drive the
          ``scatter_add_`` blend directly.
        - ``weight_B`` ``[B]`` float: per-row blend weight (1.0 when unfused).
        - ``is_fused``: ``False`` when no row carries a fusion id, letting the
          caller skip the blend entirely.
        """
        B = len(gen_params)
        group = list(range(B))
        weight = [1.0] * B
        seen: dict[str, int] = {}
        present: dict[str, int] = {}
        is_fused = False
        for b, p in enumerate(gen_params):
            gid = p.fusion_group_id
            if gid is None:
                continue
            is_fused = True
            if gid not in seen:
                seen[gid] = b  # first member's row index anchors the group
            group[b] = seen[gid]
            weight[b] = p.fusion_weight
            present[gid] = present.get(gid, 0) + 1

        # Fail loud if the upstream scheduler split a group across decode steps
        # (e.g. KV-pressure retract). A partial group would blend an incomplete
        # set of sibling distributions and silently emit wrong (un-fused) audio;
        # better to fail the request — observable and retryable — than ship that.
        for gid, n_present in present.items():
            expected = self._fusion_group_size.get(gid)
            if expected is not None and n_present != expected:
                raise RuntimeError(
                    f"voice-fusion group {gid!r} split across a decode step: "
                    f"{n_present}/{expected} sibling rows present in the batch. "
                    f"The serving engine scheduled the group's rows apart "
                    f"(likely a KV-pressure retract); raise max_running_requests "
                    f"or reduce concurrency so fusion siblings stay co-batched."
                )

        group_B = torch.tensor(group, dtype=torch.long, device=device)
        weight_B = torch.tensor(weight, dtype=torch.float32, device=device)
        return group_B, weight_B, is_fused

    @torch.no_grad()
    def decode_codebooks_batch(
        self,
        hidden_states_BD: torch.Tensor,
        req_ids: list[str],
        gen_params: list[HiggsGenParams],
    ) -> torch.Tensor:
        """Sample multi-codebook tokens for one forward step."""
        batch_size = hidden_states_BD.shape[0]
        if len(req_ids) != batch_size or len(gen_params) != batch_size:
            raise ValueError(
                f"batch size mismatch: hidden={batch_size}, "
                f"req_ids={len(req_ids)}, gen_params={len(gen_params)}"
            )

        # fp32 for softmax numerical stability.
        logits_BNV = self.modality_head.generate(hidden_states_BD).to(torch.float32)
        device = logits_BNV.device

        row_indices = torch.tensor(
            [self.acquire_row(rid) for rid in req_ids],
            dtype=torch.long,
            device=device,
        )

        temperature = torch.tensor(
            [p.temperature for p in gen_params],
            dtype=torch.float32,
            device=device,
        )
        has_top_p = any(p.top_p is not None for p in gen_params)
        top_p = (
            torch.tensor(
                [p.top_p if p.top_p is not None else 1.0 for p in gen_params],
                dtype=torch.float32,
                device=device,
            )
            if has_top_p
            else None
        )
        top_k_buf = torch.tensor(
            [
                (p.top_k if (p.top_k is not None and p.top_k > 0) else K_MAX)
                for p in gen_params
            ],
            dtype=torch.long,
            device=device,
        )

        # Voice fusion (eager path, mirrors decode_codebooks_batch_cg). Map each
        # row's ``fusion_group_id`` to a batch-local group index; rows with no
        # fusion id form singleton groups (identity blend). The blended log-probs
        # then replace the raw logits and temperature is folded in, so the
        # sampler runs at temperature 1.
        #
        # ``_gen_params_for_batch`` cannot know fusion membership (it only sees
        # ``sampling_info``, not req_ids), so backfill each row's group id +
        # weight here from the registry keyed by req_id. Non-fusion req_ids are
        # absent from the maps → fields stay None/1.0 → singleton/identity.
        for rid, p in zip(req_ids, gen_params):
            p.fusion_group_id = self._fusion_group_of.get(rid)
            if p.fusion_group_id is not None:
                p.fusion_weight = self._fusion_weight_of.get(rid, 1.0)

        group_B, weight_B, is_fused = self._batch_local_fusion(gen_params, device)
        if is_fused:
            logits_BNV = fuse_group_logits(
                logits_BNV, group_B, weight_B, temperature_B=temperature
            )
            temperature = torch.ones_like(temperature)

        was_done = self._sampler_pool.generation_done[row_indices].clone()

        codes_BN = batched_step(
            logits_BNV,
            self._sampler_pool,
            row_indices,
            temperature=temperature,
            top_p=top_p,
            top_k_buf=top_k_buf,
        )

        if is_fused:
            # Group barrier: sync generation_done across siblings, then persist
            # it back into the pool so the scheduler ends the group together.
            synced_done = fuse_group_generation_done(
                self._sampler_pool.generation_done[row_indices], group_B
            )
            self._sampler_pool.generation_done[row_indices] = synced_done

        # Note(yichi): One D2H per step to skip STOP-sentinel rows in the Python append loop.
        was_done_cpu = was_done.cpu().tolist()
        codes_BN = codes_BN.detach().to(torch.long)
        for b in range(batch_size):
            if was_done_cpu[b]:
                continue
            # Fusion followers' codes duplicate the leader's; don't accumulate
            # them (only the leader is decoded to audio).
            if self.is_fusion_follower(req_ids[b]):
                continue
            self._output_codes.setdefault(req_ids[b], []).append(codes_BN[b])

        text_vocab_size = self.backbone.config.vocab_size
        return torch.zeros(
            (batch_size, text_vocab_size),
            device=device,
            dtype=torch.float32,
        )

    @torch.no_grad()
    def decode_codebooks_batch_cg(self, hidden_states_BD: torch.Tensor) -> torch.Tensor:
        """CG-friendly variant of :meth:`decode_codebooks_batch`: reads/writes
        only preallocated ``_cg_*`` buffers, no Python control flow on
        tensor values, no D2H syncs.
        """
        batch_size = hidden_states_BD.shape[0]
        device = hidden_states_BD.device

        logits_BNV = self.modality_head.generate(hidden_states_BD).to(torch.float32)

        temperature = self._cg_temperature[:batch_size]
        top_p = self._cg_top_p[:batch_size]
        top_k_buf = self._cg_top_k_buf[:batch_size]

        # Voice fusion: blend sibling rows' output distributions before sampling.
        # ``_cg_fusion_group``/``_cg_fusion_weight`` default to singleton groups
        # (group = own slot, weight = 1), making this a numerical no-op for
        # non-fusion batches. Returns log-probs that feed the sampler as logits;
        # the shared seed (set equal across siblings by the runner) then draws
        # the same frame for every group member.
        fusion_group_B = self._cg_fusion_group[:batch_size]
        fusion_weight_B = self._cg_fusion_weight[:batch_size]
        logits_BNV = fuse_group_logits(
            logits_BNV,
            fusion_group_B,
            fusion_weight_B,
            temperature_B=temperature,
        )

        delay_count_B = self._cg_active_delay_count[:batch_size].to(torch.long)
        eoc_countdown_B = self._cg_active_eoc_countdown[:batch_size].to(torch.long)
        generation_done_B = self._cg_active_generation_done[:batch_size]
        last_codes_BN_in = self._cg_active_last_codes[:batch_size]
        seeds_B = self._cg_active_seeds[:batch_size]
        step_count_B = self._cg_active_step_count[:batch_size]

        self._cg_was_done[:batch_size] = generation_done_B

        # ``fuse_group_logits`` already applied temperature; pass temperature=1
        # so the sampler doesn't divide the blended log-probs a second time.
        sampler_temperature = torch.ones_like(temperature)

        (
            codes_BN,
            new_delay_count_B,
            new_eoc_countdown_B,
            new_generation_done_B,
            new_last_codes_BN,
            new_step_count_B,
        ) = batched_step_direct(
            logits_BNV,
            delay_count_B,
            eoc_countdown_B,
            generation_done_B,
            last_codes_BN_in,
            temperature=sampler_temperature,
            top_p=top_p,
            top_k_buf=top_k_buf,
            seeds=seeds_B,
            step_count=step_count_B,
        )
        # Group barrier: any sibling reaching EOC ends the whole group on the
        # same step, so the N KV contexts never desynchronize. No-op for
        # singleton groups.
        new_generation_done_B = fuse_group_generation_done(
            new_generation_done_B, fusion_group_B
        )
        self._cg_active_step_count[:batch_size] = new_step_count_B
        self._cg_active_delay_count[:batch_size] = new_delay_count_B.to(
            self._cg_active_delay_count.dtype
        )
        self._cg_active_eoc_countdown[:batch_size] = new_eoc_countdown_B.to(
            self._cg_active_eoc_countdown.dtype
        )
        self._cg_active_generation_done[:batch_size] = new_generation_done_B
        self._cg_active_last_codes[:batch_size] = new_last_codes_BN
        self._cg_codes_BN[:batch_size] = codes_BN

        text_vocab_size = self.backbone.config.vocab_size
        return torch.zeros(
            (batch_size, text_vocab_size),
            device=device,
            dtype=torch.float32,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        input_embeds: torch.Tensor | None = None,
        omni_prefill_rids: list[str] | None = None,
        **kwargs,
    ):
        """Run the backbone then sample multi-codebook codes per request.

        Prefill takes runner-supplied ``input_embeds`` (ref-audio pasted
        at ``-100``); decode reads embeds and sampling state from
        ``_cg_active_*`` shadow buffers populated by the runner.
        """
        is_decode = self._is_decode_step(forward_batch)

        if is_decode:
            input_embeds = self._decode_step_embeds_cg(
                input_ids, batch_size=input_ids.shape[0]
            )
        else:
            if input_embeds is None:
                raise RuntimeError(
                    "Higgs prefill requires runner-composed input_embeds"
                )
            if omni_prefill_rids is None:
                raise RuntimeError(
                    "Higgs prefill requires omni_prefill_rids from ForwardBatch.rids"
                )
            req_ids, gen_params = self._extract_batch_metadata(
                forward_batch, omni_prefill_rids
            )

        hidden_states = self.backbone.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
        )

        if (
            not is_decode
            and hasattr(forward_batch, "forward_mode")
            and forward_batch.forward_mode.is_extend()
            and hasattr(forward_batch, "extend_seq_lens")
        ):
            last_index = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            hidden_states_last = hidden_states[last_index]
        else:
            hidden_states_last = hidden_states
            if hidden_states_last.ndim == 3:
                hidden_states_last = hidden_states_last[:, -1, :]

        if is_decode:
            text_logits_BV = self.decode_codebooks_batch_cg(hidden_states_last)
        else:
            text_logits_BV = self.decode_codebooks_batch(
                hidden_states_last, req_ids, gen_params
            )

        # Rows are per-request, not per-token; the graph runner's replay trim
        # is a no-op on these shapes only because bs <= extend tokens.
        return LogitsProcessorOutput(
            next_token_logits=text_logits_BV,
            hidden_states=hidden_states_last,
        )

    def _decode_step_embeds_cg(
        self, input_ids: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Graph-capture-friendly decode-step embedding lookup; reads from
        shadow `_cg_active_*[:bs]` populated by ``before_decode``.
        """
        delay_counts = self._cg_active_delay_count[:batch_size].to(torch.long)
        has_codes = (delay_counts > 0).unsqueeze(-1)

        last_codes_BN = self._cg_active_last_codes[:batch_size].to(torch.long)
        fused_embeds = self.multimodal_embedding.modality_embedding_0(last_codes_BN)

        text_embeds = self.backbone.model.embed_tokens(input_ids)
        if text_embeds.ndim == 3:
            text_embeds = text_embeds[:, -1, :]

        return torch.where(has_codes, fused_embeds.to(text_embeds.dtype), text_embeds)

    @staticmethod
    def _is_decode_step(forward_batch) -> bool:
        mode = getattr(forward_batch, "forward_mode", None)
        if mode is None:
            return False
        is_decode = getattr(mode, "is_decode", None)
        return bool(is_decode()) if callable(is_decode) else False

    def _extract_batch_metadata(
        self,
        forward_batch,
        req_ids: list[str],
    ) -> tuple[list[str], list[HiggsGenParams]]:
        batch_size = self._infer_batch_size(forward_batch)
        gen_params = self._gen_params_for_batch(forward_batch.sampling_info, batch_size)
        return req_ids, gen_params

    @staticmethod
    def _gen_params_for_batch(sampling_info, batch_size: int) -> list[HiggsGenParams]:
        """Pull per-row sampling params off ``sampling_info``."""
        if sampling_info is None:
            return [HiggsGenParams() for _ in range(batch_size)]

        temps = _flat_sampling_attr(sampling_info, "temperatures")
        top_ps = _flat_sampling_attr(sampling_info, "top_ps")
        top_ks = _flat_sampling_attr(sampling_info, "top_ks")

        params: list[HiggsGenParams] = []
        for b in range(batch_size):
            temp = float(temps[b]) if temps is not None else 1.0
            tp = float(top_ps[b]) if top_ps is not None else None
            tk_raw = int(top_ks[b]) if top_ks is not None else 0
            params.append(
                HiggsGenParams(
                    temperature=temp,
                    top_p=tp,
                    top_k=tk_raw or None,
                )
            )
        return params

    @staticmethod
    def _infer_batch_size(forward_batch) -> int:
        seq_lens = getattr(forward_batch, "seq_lens", None)
        if seq_lens is not None and hasattr(seq_lens, "shape"):
            return int(seq_lens.shape[0])
        return int(getattr(forward_batch, "batch_size", 1))

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """Remap Higgs ckpt names then split between backbone and own modules.

        Returns the set of *own* parameter names loaded (multimodal embedding +
        optionally the untied modality head). Text-backbone loading delegates
        to :meth:`Qwen3ForCausalLM.load_weights`, which does qkv / gate_up
        stacking and lm_head tying internally.
        """
        mapper = DiscreteWeightMapper(
            text_prefix_map=_BACKBONE_PREFIX_MAP,
            tie_modality=self._tie_modality,
        )

        backbone_weights: list[Tuple[str, torch.Tensor]] = []
        self_weights: list[Tuple[str, torch.Tensor]] = []
        loaded: set[str] = set()
        own_names = self._own_param_names()

        for name, tensor in weights:
            mapped = mapper.map(name)
            if mapped is None:
                continue
            if mapped.startswith("backbone."):
                backbone_weights.append((mapped[len("backbone.") :], tensor))
            elif mapped in own_names:
                self_weights.append((mapped, tensor))

        self.backbone.load_weights(iter(backbone_weights))

        own_params = dict(self.named_parameters(remove_duplicate=False))
        for name, tensor in self_weights:
            param = own_params.get(name)
            if param is None:
                continue
            if param.shape != tensor.shape:
                raise ValueError(
                    f"Shape mismatch for {name}: expected {tuple(param.shape)}, "
                    f"got {tuple(tensor.shape)}"
                )
            param.data.copy_(tensor.to(param.dtype))
            loaded.add(name)

        return loaded

    def _own_param_names(self) -> set[str]:
        names: set[str] = set()
        for name, _ in self.named_parameters(remove_duplicate=False):
            if not name.startswith("backbone."):
                names.add(name)
        return names


__all__ = ["HiggsGenParams", "HiggsTTSModel"]
