# SPDX-License-Identifier: Apache-2.0
"""Voice timbre fusion — output-distribution blending for Higgs TTS.

A *fusion request* conditions one synthesis on ``N`` reference voices at once.
Each reference is prefilled into its own KV context as a separate *sibling* row,
and at every AR decode step the sibling rows' per-codebook output distributions
are blended by weight **before** sampling. All siblings then sample the *same*
multi-codebook frame (shared seed), so their ``N`` KV contexts evolve in
lock-step and decode the same audio; only the group *leader* row is emitted.

This module holds the two pure, ``sgl_kernel``-free tensor ops that implement
the blend, so they are unit-testable without the full sglang engine:

- :func:`fuse_group_logits` — weighted probability average across group members,
  returned as log-probs ready to feed the standard sampler.
- :func:`fuse_group_generation_done` — "any sibling done ⇒ all done" barrier so
  group members terminate on the same step.

Both are CUDA-Graph friendly: fixed-shape ``scatter_add_`` / advanced-index ops,
no host-side control flow. They are identity no-ops for the default case where
every row is its own singleton group (``group_id[i] == i``, ``weight == 1``),
so non-fusion decoding is numerically unchanged (argmax/multinomial-invariant).
"""

from __future__ import annotations

import torch

# Floor added before ``log`` so a zeroed fused-prob row can't produce ``-inf``
# that poisons the downstream softmax. ~1e-30 is well below any real codec prob.
_LOG_FLOOR = 1e-30


def fuse_group_logits(
    logits_BNV: torch.Tensor,
    group_id_B: torch.Tensor,
    weight_B: torch.Tensor,
    *,
    temperature_B: torch.Tensor | None = None,
) -> torch.Tensor:
    """Blend per-codebook output distributions within each fusion group.

    Args:
        logits_BNV: raw head logits, shape ``[B, N, V]`` (B rows, N codebooks,
            V codec vocab).
        group_id_B: ``[B]`` int. Rows sharing a value are one fusion group.
            For a normal request each row is its own group (``group_id[i] == i``),
            making the blend an identity (up to a constant log shift that does
            not affect argmax / multinomial sampling).
        weight_B: ``[B]`` float blend weight per row. Weights need not be
            normalized; only their within-group ratio matters.
        temperature_B: optional ``[B]`` float. When given, probabilities are
            taken at ``softmax(logits / temperature)`` before blending so the
            blend happens in the same temperature-scaled space the sampler will
            use. ``None`` blends raw-logit softmax (temperature applied later).

    Returns:
        ``[B, V]``-per-codebook log-probabilities, shape ``[B, N, V]``, where
        every row carries its group's blended distribution. Feed these to the
        existing sampler exactly as if they were logits.

    The blend is ``log(Σ_g w_i · softmax(logits_i))`` over members ``i`` of each
    group ``g``, with group weights renormalized to sum to 1. Because group
    membership is expressed purely through ``scatter_add_`` + advanced indexing,
    the op is shape-static and safe inside a captured CUDA graph.

    Singleton-group rows (the entire non-fusion batch) are returned as
    ``logits / temperature`` **unchanged** — bit-identical to what the sampler
    would have received without fusion — rather than routed through
    ``log(softmax(...))``, whose ``exp``/``log`` round-trip and ``_LOG_FLOOR``
    would perturb the distribution tail. The singleton-vs-blended choice is a
    per-row ``torch.where`` (tensor op, no host branch), so it stays
    CUDA-Graph-safe in a mixed fusion/non-fusion batch.
    """
    if logits_BNV.ndim != 3:
        raise ValueError(f"logits_BNV must be [B, N, V], got {tuple(logits_BNV.shape)}")
    B, N, V = logits_BNV.shape
    device = logits_BNV.device

    logits = logits_BNV.float()
    if temperature_B is not None:
        safe_temp = temperature_B.to(device).clamp_min(1e-5).view(B, 1, 1)
        logits = logits / safe_temp
    probs_BNV = logits.softmax(dim=-1)

    gid = group_id_B.to(device=device, dtype=torch.long)
    w = weight_B.to(device=device, dtype=torch.float32)

    # Per-group member count + weight sum (both via scatter_add_, CG-safe).
    ones = torch.ones(B, dtype=torch.float32, device=device)
    group_count = torch.zeros(B, dtype=torch.float32, device=device)
    group_count.scatter_add_(0, gid, ones)
    group_weight_sum = torch.zeros(B, dtype=torch.float32, device=device)
    group_weight_sum.scatter_add_(0, gid, w)

    # Per-group weight normalization: divide each row's weight by its group's
    # total, so blended probabilities stay a valid distribution.
    norm_w = w / group_weight_sum[gid].clamp_min(_LOG_FLOOR)  # [B]

    weighted = probs_BNV * norm_w.view(B, 1, 1)  # [B, N, V]
    idx = gid.view(B, 1, 1).expand(B, N, V)
    fused = torch.zeros_like(probs_BNV)
    fused.scatter_add_(0, idx, weighted)  # group g accumulates its members

    # Broadcast each group's fused distribution back onto all its member rows,
    # then take log so the result feeds the sampler as logits.
    fused_BNV = fused.index_select(0, gid)
    blended_logits = (fused_BNV + _LOG_FLOOR).log()

    # Rows in a real (size > 1) group get the blended log-probs; singleton rows
    # keep their exact (temperature-scaled) logits so non-fusion decoding is
    # bit-identical to baseline. Per-row select — no host branch.
    is_grouped = (group_count.index_select(0, gid) > 1.5).view(B, 1, 1)
    return torch.where(is_grouped, blended_logits, logits)


def fuse_group_generation_done(
    generation_done_B: torch.Tensor,
    group_id_B: torch.Tensor,
) -> torch.Tensor:
    """"Any sibling done ⇒ all done" group barrier.

    Returns a ``[B]`` bool where a row is done iff *any* member of its fusion
    group is done. For singleton groups this is an identity. Keeping group
    members' ``generation_done`` synchronized makes them terminate on the same
    AR step, so their sibling KV contexts never desynchronize.
    """
    device = generation_done_B.device
    gid = group_id_B.to(device=device, dtype=torch.long)
    done_f = generation_done_B.to(torch.float32)
    group_any = torch.zeros(
        generation_done_B.shape[0], dtype=torch.float32, device=device
    )
    group_any.scatter_add_(0, gid, done_f)
    return group_any.index_select(0, gid) > 0


__all__ = ["fuse_group_logits", "fuse_group_generation_done"]
