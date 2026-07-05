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
  returned as log-probs ready to feed the standard sampler, plus a per-row
  ``is_grouped`` mask the caller MUST use to keep singleton rows sampling at
  their real (unfolded) temperature — see the "greedy" warning below.
- :func:`fuse_group_generation_done` — "any sibling done ⇒ all done" barrier so
  group members terminate on the same step.

Both are CUDA-Graph friendly: fixed-shape ``scatter_add_`` / advanced-index ops,
no host-side control flow. They are identity no-ops for the default case where
every row is its own singleton group (``group_id[i] == i``, ``weight == 1``).

Caller contract — do not fold temperature into the sampler call unconditionally:
``fuse_group_logits`` pre-applies ``temperature_B`` for grouped rows so the
blend happens in the same temperature-scaled space the sampler will use, and
the caller then samples the returned logits at ``temperature=1`` for those
rows. But the sampler's greedy short-circuit (:func:`sampler._sample_independent`
and its batched counterpart) is keyed on the *temperature it receives*, not on
the logits — it decides ``temperature <= _GREEDY_TEMP_THRESHOLD`` before ever
looking at the logits. If the caller passes ``temperature=1`` for EVERY row
(grouped or not, as a blanket simplification), a plain non-fusion request with
``temperature=0`` silently loses its argmax short-circuit: it becomes a
``multinomial`` draw over a near-one-hot distribution instead of a
deterministic ``argmax``, which both breaks determinism and burns global RNG
state that a truly-greedy row was never supposed to touch. The returned
``is_grouped`` mask is exactly what the caller needs to avoid this:
singleton rows must keep sampling at their real ``temperature_B``, and only
grouped rows get temperature folded away.
"""

from __future__ import annotations

import threading

import torch

# Floor added before ``log`` so a zeroed fused-prob row can't produce ``-inf``
# that poisons the downstream softmax. ~1e-30 is well below any real codec prob.
_LOG_FLOOR = 1e-30


class FusionRegistry:
    """Thread-safe registry of which in-flight requests belong to which
    voice-fusion group, at what blend weight, and whether they are the
    group's audio-emitting leader.

    Written by the scheduler's request-build thread (:meth:`set`) and read
    every decode step by the GPU-worker thread (:meth:`is_follower`,
    :meth:`expected_size`, ...). ``_lock`` guards every access so a decode
    step can never observe a half-registered group (which would otherwise
    spuriously trip a group-completeness check). The lock is held only for
    cheap dict ops, never across a GPU forward.

    Pure Python (no torch/sglang dependency) so it is unit-testable standalone
    — see ``test_voice_fusion.py``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._group_of: dict[str, str] = {}
        self._weight_of: dict[str, float] = {}
        self._leader: dict[str, bool] = {}
        # Cheap, lock-free "is any fusion request live right now" signal for
        # the hot (non-fusion) path: a server with zero fusion traffic must
        # not pay a per-decode-step lock acquisition + dict-comprehension tax
        # just to learn that, yes, it's still zero. Maintained under
        # ``_lock`` (write side, cheap int increment/decrement) but read
        # without it — a reader can observe a value that's one register/clear
        # stale, which only means an all-singleton step occasionally still
        # takes the (harmless, correct) fusion-aware path, never the reverse.
        self._active_count = 0

    def set(
        self, req_id: str, group_id: str | None, weight: float, *, is_leader: bool
    ) -> None:
        """Register ``req_id`` as a member of voice-fusion group ``group_id``.

        ``group_id is None`` clears any fusion membership (normal request).
        Idempotent: re-registering the same ``req_id`` overwrites in place (no
        double-counting), so a retry that reuses a request id can't inflate
        the group.
        """
        with self._lock:
            if group_id is None:
                if self._group_of.pop(req_id, None) is not None:
                    self._active_count -= 1
                self._weight_of.pop(req_id, None)
                self._leader.pop(req_id, None)
                return
            if req_id not in self._group_of:
                self._active_count += 1
            self._group_of[req_id] = group_id
            self._weight_of[req_id] = float(weight)
            self._leader[req_id] = bool(is_leader)

    def has_any(self) -> bool:
        """Lock-free, best-effort "is any fusion request registered right now".

        For the overwhelmingly common non-fusion server, this lets the decode
        hot path skip the fusion bookkeeping (buffer population, per-row
        follower checks) entirely without ever taking the lock. See
        ``_active_count``'s docstring for the staleness tradeoff this makes.
        """
        return self._active_count > 0

    def expected_size(self, group_id: str) -> int:
        """Number of currently-registered members of ``group_id`` (0 if none).

        Derived live from membership (not a separate counter) so it can never
        drift out of sync with the actual registry on retries or partial
        cleanup: a reused request id overwrites its own entry rather than
        incrementing a count.
        """
        with self._lock:
            return sum(1 for g in self._group_of.values() if g == group_id)

    def snapshot(self, req_ids: list[str]) -> tuple[dict[str, str], dict[str, float]]:
        """Atomic snapshot of (group_id, weight) for the given req_ids.

        Taken under the lock so a decode step sees a consistent view of every
        row's membership even if a concurrent register/clear is in flight.
        Returns ``(group_of, weight_of)`` restricted to req_ids that are
        fusion members; non-members are absent from both dicts.
        """
        with self._lock:
            group_of = {r: self._group_of[r] for r in req_ids if r in self._group_of}
            weight_of = {r: self._weight_of.get(r, 1.0) for r in group_of}
        return group_of, weight_of

    def is_leader(self, req_id: str) -> bool:
        """True iff ``req_id`` is a fusion member and the group's output leader."""
        with self._lock:
            return self._leader.get(req_id, True)

    def is_follower(self, req_id: str) -> bool:
        """True iff ``req_id`` is a fusion member that is NOT the leader (its
        decoded codes duplicate the leader's and must not be emitted)."""
        with self._lock:
            return req_id in self._group_of and not self._leader.get(req_id, True)


def fuse_group_logits(
    logits_BNV: torch.Tensor,
    group_id_B: torch.Tensor,
    weight_B: torch.Tensor,
    *,
    temperature_B: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        temperature_B: optional ``[B]`` float. When given, the *blend* is
            computed at ``softmax(logits / temperature)`` so grouped rows fuse
            in the same temperature-scaled space the sampler will use.
            ``None`` blends raw-logit softmax (temperature applied later by
            the caller for every row, grouped or not).

    Returns:
        ``(logits_out, is_grouped_B)``:

        - ``logits_out``: ``[B, N, V]``. Grouped rows carry the blended
          log-distribution (temperature already folded in — sample these at
          ``temperature=1``). Singleton rows carry their **raw, untouched**
          ``logits_BNV`` — the caller must apply their real ``temperature_B``
          when sampling them, exactly as it would with fusion disabled
          entirely. Do NOT apply temperature to singleton rows a second time:
          this function does not pre-divide them, precisely so the caller's
          one real division is the only one that happens.
        - ``is_grouped_B``: ``[B]`` bool, true for rows in a real (size > 1)
          group. The caller MUST sample grouped rows at ``temperature=1`` (the
          blend already applied ``temperature_B``) but sample singleton rows at
          their **real** ``temperature_B`` — folding every row to 1
          unconditionally defeats the sampler's greedy short-circuit for
          ordinary (non-fusion) requests. See the module docstring.

    The blend is ``log(Σ_g w_i · softmax(logits_i / T))`` over members ``i`` of
    each group ``g``, with group weights renormalized to sum to 1. Because
    group membership is expressed purely through ``scatter_add_`` + advanced
    indexing, the op is shape-static and safe inside a captured CUDA graph.

    Singleton-group rows (the entire non-fusion batch) are returned as
    ``logits_BNV`` **unchanged** — bit-identical to what the sampler would
    have received without fusion — rather than routed through
    ``log(softmax(...))``, whose ``exp``/``log`` round-trip and ``_LOG_FLOOR``
    would perturb the distribution tail, and rather than pre-divided by
    temperature, which the caller already does downstream (dividing twice
    would silently sharpen/dull every ordinary request's sampling — this was
    a real bug caught in review, not a hypothetical). The singleton-vs-blended
    choice is a per-row ``torch.where`` (tensor op, no host branch), so it
    stays CUDA-Graph-safe in a mixed fusion/non-fusion batch.
    """
    if logits_BNV.ndim != 3:
        raise ValueError(f"logits_BNV must be [B, N, V], got {tuple(logits_BNV.shape)}")
    B, N, V = logits_BNV.shape
    device = logits_BNV.device

    raw_logits = logits_BNV.float()
    logits = raw_logits
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

    # Rows in a real (size > 1) group get the blended log-probs (temperature
    # already folded in); singleton rows get their exact RAW logits back —
    # the caller applies the real per-row temperature exactly once, downstream
    # — so non-fusion decoding is bit-identical to baseline. Per-row select —
    # no host branch.
    is_grouped_B = group_count.index_select(0, gid) > 1.5
    logits_out = torch.where(is_grouped_B.view(B, 1, 1), blended_logits, raw_logits)
    return logits_out, is_grouped_B


def fuse_group_generation_done(
    generation_done_B: torch.Tensor,
    group_id_B: torch.Tensor,
) -> torch.Tensor:
    """ "Any sibling done ⇒ all done" group barrier.

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
