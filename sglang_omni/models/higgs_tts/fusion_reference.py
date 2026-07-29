# SPDX-License-Identifier: Apache-2.0
"""Reference-space voice fusion (zero training), built INSIDE the engine stage.

Instead of blending per-step output distributions across sibling rows (see
``fusion.py``, kept as the ``logits`` research mode), this module builds ONE
hybrid-timbre reference from N weighted reference voices and then serves the
request as a completely ordinary single-reference clone:

1. **Calibration**: each reference voice clones the same fixed calibration
   sentence (engine-internal requests, fixed seed + F0 quality gate), giving
   N same-content / different-timbre readings.
2. **WORLD morph**: DTW-align the readings on WORLD spectral-envelope
   features, then weight-interpolate log-F0 / log spectral envelope /
   aperiodicity and resynthesize a hybrid reference waveform.
3. **Serve**: codec-encode the hybrid waveform into an ordinary reference and
   enqueue the real request as a standard single-reference clone. The hybrid
   reference is cached per (reference codes, weights, algo version) so only
   the first request of a combination pays the build cost.

Why reference-space: per-step distribution pooling was live-verified to be
seed-bimodal (an intermediate-timbre frame is a low-probability tail of BOTH
experts, and AR hysteresis locks the register within a few frames — see
``docs/voice_fusion_design.md`` 第五阶段). Moving fusion into the reference
makes the intermediate timbre the MODE of the model's speaker posterior, and
single-reference cloning has no bimodality mechanism at all. Live E1 run:
8/8 mixed clones landed inside [0.35, 0.65] on the log-F0 axis (old
mechanism: ~15/16 locked to an endpoint) with strictly weight-monotonic
output across alpha ∈ {0, 0.25, 0.5, 0.75, 1}.

Everything here runs in the tts_engine stage process. Heavy work (CPU codec
decode/encode, pyworld analysis/synthesis) happens on a single worker thread;
scheduler-thread callbacks only collect codes and advance the state machine.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from sglang_omni.models.higgs_tts.utils import (
    apply_delay_pattern,
    get_or_load_codec,
)
from sglang_omni.utils.codec_delay import reverse_delay_pattern

logger = logging.getLogger(__name__)

# --- Configuration -----------------------------------------------------------

FUSION_MODE_ENV = "HIGGS_FUSION_MODE"  # "reference" (default) | "logits"
CAL_TEXT_ENV = "HIGGS_FUSION_CAL_TEXT"

# Calibration sentence: natural register, reasonably phoneme-rich, ~8 s read.
# Must stay stable across releases: it is part of the cache key and of the
# hybrid reference's transcript.
DEFAULT_CAL_TEXT = "今天天气不错，我们在花园里散步，聊起了旅行、音乐和美食，每个人都很开心。"

ALGO_VERSION = "ref-fusion-v1"

# Calibration sampling: fixed, user-independent (part of determinism + cache
# identity). Matches the live-validated E1 protocol.
CAL_SEEDS = (1234, 5678, 424242)
CAL_TEMPERATURE = 0.8
CAL_TOP_P = 0.8
CAL_TOP_K = 30
CAL_MAX_NEW_TOKENS = 1600  # codec is 75 Hz → ~21 s cap; calibration reads are ~8 s

# F0 quality gate: a calibration clone must sit within x1.35 of its own
# reference voice's F0 median, else retry with the next seed.
_GATE_LOG_RATIO = math.log(1.35)

# A build that hasn't finished within this window is failed + swept (covers
# rows silently removed from waiting_queue by abort, which never reach
# ``stream_output``).
BUILD_DEADLINE_S = 300.0

_CACHE_MAX_ENTRIES = 64

_WORLD_FRAME_PERIOD_MS = 5.0
_SAMPLE_RATE = 24_000


def fusion_mode() -> str:
    mode = os.environ.get(FUSION_MODE_ENV, "reference").strip().lower()
    if mode not in ("reference", "logits"):
        raise ValueError(
            f"{FUSION_MODE_ENV} must be 'reference' or 'logits', got {mode!r}"
        )
    return mode


def calibration_text() -> str:
    return os.environ.get(CAL_TEXT_ENV) or DEFAULT_CAL_TEXT


# --- WORLD-domain morph (numpy only; pyworld imported lazily) ----------------


def _pyworld():
    try:
        import pyworld
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "reference-space voice fusion requires the 'pyworld' package "
            "(WORLD vocoder analysis/synthesis) in the tts_engine environment"
        ) from exc
    return pyworld


def world_extract(
    wav: np.ndarray, fs: int = _SAMPLE_RATE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """mono float waveform → (f0, spectral envelope, aperiodicity)."""
    pw = _pyworld()
    x = np.ascontiguousarray(wav, dtype=np.float64)
    f0, t = pw.harvest(x, fs, frame_period=_WORLD_FRAME_PERIOD_MS)
    f0 = pw.stonemask(x, f0, t, fs)
    sp = pw.cheaptrick(x, f0, t, fs)
    ap = pw.d4c(x, f0, t, fs)
    return f0, sp, ap


def median_f0(wav: np.ndarray, fs: int = _SAMPLE_RATE) -> float | None:
    """Median voiced F0 via WORLD harvest; None when fully unvoiced."""
    pw = _pyworld()
    x = np.ascontiguousarray(wav, dtype=np.float64)
    f0, t = pw.harvest(x, fs, frame_period=_WORLD_FRAME_PERIOD_MS)
    f0 = pw.stonemask(x, f0, t, fs)
    voiced = f0[f0 > 0]
    if voiced.size == 0:
        return None
    return float(np.median(voiced))


def dtw_features(sp: np.ndarray, num_bands: int = 32) -> np.ndarray:
    """Per-frame alignment features: band-pooled, mean-centered log envelope."""
    lsp = np.log(sp + 1e-12)
    frames, bins = lsp.shape
    edges = np.linspace(0, bins, num_bands + 1).astype(int)
    feats = np.stack(
        [lsp[:, a:b].mean(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1
    )
    return feats - feats.mean(axis=0, keepdims=True)


def dtw_map(feats_a: np.ndarray, feats_b: np.ndarray) -> np.ndarray:
    """DTW-align B onto A's frame axis: returns ``map_ab[T_a] -> B index``."""
    cost = np.sqrt(
        np.maximum(
            (feats_a**2).sum(1)[:, None]
            + (feats_b**2).sum(1)[None, :]
            - 2.0 * feats_a @ feats_b.T,
            0.0,
        )
    )
    ta, tb = cost.shape
    acc = np.full((ta + 1, tb + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, ta + 1):
        cost_row = cost[i - 1]
        prev_row = acc[i - 1]
        cur_row = acc[i]
        for j in range(1, tb + 1):
            cur_row[j] = cost_row[j - 1] + min(
                prev_row[j], cur_row[j - 1], prev_row[j - 1]
            )
    i, j = ta, tb
    path: list[tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        _, i, j = min(
            (acc[i - 1, j - 1], i - 1, j - 1),
            (acc[i - 1, j], i - 1, j),
            (acc[i, j - 1], i, j - 1),
        )
    path.reverse()
    buckets: dict[int, list[int]] = {}
    for a, b in path:
        buckets.setdefault(a, []).append(b)
    out = np.zeros(ta, dtype=np.int64)
    last = 0
    for a in range(ta):
        if a in buckets:
            last = int(np.median(buckets[a]))
        out[a] = last
    return out


def _morph_pair(
    world_a: tuple[np.ndarray, np.ndarray, np.ndarray],
    world_b: tuple[np.ndarray, np.ndarray, np.ndarray],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blend B into A's time axis with share ``alpha`` for B (E1-validated).

    log-F0 weighted mean where both are voiced; a lone-voiced frame keeps its
    contour shifted by the global (median) pitch ratio so the register still
    lands at the blended point; log-envelope weighted mean; linear
    aperiodicity.
    """
    f0_a, sp_a, ap_a = world_a
    f0_b, sp_b, ap_b = world_b
    feats_a, feats_b = dtw_features(sp_a), dtw_features(sp_b)
    map_ab = dtw_map(feats_a, feats_b)
    f0_bw, sp_bw, ap_bw = f0_b[map_ab], sp_b[map_ab], ap_b[map_ab]

    voiced_a, voiced_b = f0_a > 0, f0_bw > 0
    gm_a = float(np.median(f0_a[f0_a > 0]))
    gm_b = float(np.median(f0_b[f0_b > 0]))

    f0_m = np.zeros_like(f0_a)
    both = voiced_a & voiced_b
    f0_m[both] = np.exp(
        (1 - alpha) * np.log(f0_a[both]) + alpha * np.log(f0_bw[both])
    )
    only_a = voiced_a & ~voiced_b
    f0_m[only_a] = f0_a[only_a] * (gm_b / gm_a) ** alpha
    only_b = (~voiced_a) & voiced_b
    f0_m[only_b] = f0_bw[only_b] * (gm_a / gm_b) ** (1 - alpha)

    sp_m = np.exp((1 - alpha) * np.log(sp_a + 1e-16) + alpha * np.log(sp_bw + 1e-16))
    ap_m = np.clip((1 - alpha) * ap_a + alpha * ap_bw, 0.001, 0.999)
    return f0_m, sp_m, ap_m


def build_fused_reference(
    cal_wavs: list[np.ndarray],
    weights: list[float],
    fs: int = _SAMPLE_RATE,
) -> np.ndarray:
    """N same-content calibration waveforms + weights → hybrid waveform.

    N == 2 is exactly the live-validated E1 morph. N > 2 reduces pairwise
    (always merging the two smallest current weights, Huffman-style) so every
    step goes through the validated binary blend; the time axis converges to
    the largest-weight member's.
    """
    if len(cal_wavs) != len(weights) or len(cal_wavs) < 2:
        raise ValueError(
            f"need >= 2 calibration waveforms with matching weights, got "
            f"{len(cal_wavs)} / {len(weights)}"
        )
    pw = _pyworld()
    entries = [
        {"world": world_extract(w, fs), "weight": float(wt)}
        for w, wt in zip(cal_wavs, weights)
    ]
    while len(entries) > 1:
        entries.sort(key=lambda e: e["weight"])
        low, high = entries.pop(0), entries.pop(0)
        # Base (time axis) = the heavier of the pair; alpha = lighter's share.
        alpha = low["weight"] / (low["weight"] + high["weight"])
        merged = _morph_pair(high["world"], low["world"], alpha)
        entries.append({"world": merged, "weight": low["weight"] + high["weight"]})

    f0_m, sp_m, ap_m = entries[0]["world"]
    wav = pw.synthesize(
        np.ascontiguousarray(f0_m),
        np.ascontiguousarray(sp_m),
        np.ascontiguousarray(ap_m),
        fs,
        _WORLD_FRAME_PERIOD_MS,
    )
    peak = max(float(np.abs(wav).max()), 1e-8)
    return (wav / peak * 0.9).astype(np.float32)


# --- Cache -------------------------------------------------------------------


def fused_reference_cache_key(
    refs: list[dict[str, Any]], weights: list[float], cal_text: str
) -> str:
    """Content key over the N reference code sequences + weights + algo id."""
    h = hashlib.blake2b(digest_size=16)
    h.update(ALGO_VERSION.encode())
    h.update(cal_text.encode())
    for ref, weight in zip(refs, weights):
        h.update(b"|ref|")
        h.update(f"{weight:.6f}".encode())
        for row in ref["codes_delayed"]:
            for c in row:
                h.update(int(c).to_bytes(2, "little"))
    return h.hexdigest()


# --- Engine-side orchestrator ------------------------------------------------


@dataclass
class _CalRow:
    ref_idx: int
    seed_idx: int
    rid: str


@dataclass
class _BuildGroup:
    request_id: str
    payload: Any
    cache_key: str
    refs: list[dict[str, Any]]  # {"codes_delayed", "weight", "cal_prompt_token_ids"}
    weights: list[float]
    cal_text: str
    make_real_request: Callable[[list[list[int]]], Any]
    deadline: float
    pending: dict[str, _CalRow] = field(default_factory=dict)
    collected: dict[int, torch.Tensor] = field(default_factory=dict)  # raw [T, N]
    seed_used: dict[int, int] = field(default_factory=dict)
    failed: bool = False


class FusionReferenceOrchestrator:
    """Per-engine-process build coordinator for reference-space fusion.

    Lives on the Higgs model object (see ``get_orchestrator``); bound to the
    OmniScheduler in ``HiggsTtsEngineBuilder.post_scheduler_setup``. Scheduler
    thread: ``on_internal_done`` (collect + advance). Worker thread: codec
    decode/encode + WORLD morph + real-request enqueue.
    """

    def __init__(self) -> None:
        self._scheduler: Any | None = None
        self._checkpoint_dir: str | None = None
        self._lock = threading.Lock()
        self._groups: dict[str, _BuildGroup] = {}
        self._cache: dict[str, list[list[int]]] = {}
        self._cache_order: list[str] = []
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="higgs-ref-fusion"
        )

    # -- wiring --

    def bind(self, scheduler: Any, checkpoint_dir: str) -> None:
        self._scheduler = scheduler
        self._checkpoint_dir = checkpoint_dir

    @property
    def is_bound(self) -> bool:
        return self._scheduler is not None

    # -- cache --

    def cache_get(self, key: str) -> list[list[int]] | None:
        with self._lock:
            return self._cache.get(key)

    def _cache_put(self, key: str, delayed_rows: list[list[int]]) -> None:
        with self._lock:
            if key not in self._cache:
                self._cache_order.append(key)
                while len(self._cache_order) > _CACHE_MAX_ENTRIES:
                    evicted = self._cache_order.pop(0)
                    self._cache.pop(evicted, None)
            self._cache[key] = delayed_rows

    # -- build lifecycle --

    def register_group(
        self,
        *,
        request_id: str,
        payload: Any,
        cache_key: str,
        refs: list[dict[str, Any]],
        weights: list[float],
        cal_text: str,
        make_real_request: Callable[[list[list[int]]], Any],
        cal_rows: list[_CalRow],
    ) -> None:
        group = _BuildGroup(
            request_id=request_id,
            payload=payload,
            cache_key=cache_key,
            refs=refs,
            weights=weights,
            cal_text=cal_text,
            make_real_request=make_real_request,
            deadline=time.monotonic() + BUILD_DEADLINE_S,
        )
        self._sweep_expired()
        for row in cal_rows:
            group.pending[row.rid] = row
            group.seed_used[row.ref_idx] = row.seed_idx
        with self._lock:
            self._groups[request_id] = group
        # Abort entry point: a client abort of ``request_id`` must cascade
        # into the in-flight calibration rows. The set deliberately does NOT
        # contain ``request_id`` itself — the atomic-admission gate withholds
        # any group whose members aren't all present in the waiting queue, and
        # ``request_id`` never has a queue row of its own during the build.
        # (The calibration rows' own member entries come from the
        # fusion_siblings enqueue channel and likewise only contain cal rids.)
        self._scheduler._fusion_group_members[request_id] = {
            row.rid for row in cal_rows
        }

    def make_done_callback(self, request_id: str, ref_idx: int) -> Callable[[Any], None]:
        def _done(req_data: Any) -> None:
            self.on_internal_done(request_id, ref_idx, req_data)

        return _done

    def on_internal_done(self, request_id: str, ref_idx: int, req_data: Any) -> None:
        """Scheduler-thread callback for a finished calibration row."""
        self._sweep_expired()
        with self._lock:
            group = self._groups.get(request_id)
        if group is None or group.failed:
            return
        group.pending.pop(req_data.req.rid, None)

        if (req_data.finish_reason or "").lower() == "abort":
            self._fail(group, RuntimeError("calibration row aborted"), emit=False)
            return
        if not req_data.output_codes:
            self._fail(
                group,
                RuntimeError(
                    f"calibration synthesis for reference {ref_idx} produced no codes"
                ),
            )
            return

        delayed = torch.stack(req_data.output_codes, dim=0).to(torch.long).cpu()
        group.collected[ref_idx] = delayed
        if group.pending or len(group.collected) < len(group.refs):
            return
        self._executor.submit(self._finalize_build, group)

    # -- heavy path (worker thread) --

    def _codec(self):
        assert self._checkpoint_dir is not None
        # fp32 CPU codec: the documented-stable decode path; loaded once per
        # engine process, only ever exercised on cold fusion builds.
        return get_or_load_codec(self._checkpoint_dir, "cpu", "float32")

    def _delayed_to_wav(self, delayed_LN: torch.Tensor) -> np.ndarray:
        raw = reverse_delay_pattern(delayed_LN, allow_short=True)
        raw = raw[(raw < 1024).all(dim=1)]
        if raw.shape[0] < 75:  # < 1 s of frames (codec runs at 75 Hz)
            raise RuntimeError(
                f"decoded reference is too short ({raw.shape[0]} frames)"
            )
        return self._codec().decode(raw).numpy().astype(np.float64)

    def _finalize_build(self, group: _BuildGroup) -> None:
        try:
            self._finalize_build_inner(group)
        except Exception as exc:  # noqa: BLE001 - single failure funnel
            logger.exception(
                "reference-fusion build failed for %s", group.request_id
            )
            self._fail(group, exc)

    def _finalize_build_inner(self, group: _BuildGroup) -> None:
        scheduler = self._scheduler
        if group.request_id in scheduler._aborted_request_ids:
            self._drop(group)
            return

        cal_wavs: list[np.ndarray] = []
        retry_rows: list[tuple[int, int]] = []  # (ref_idx, next_seed_idx)
        for idx, ref in enumerate(group.refs):
            cal_wav = self._delayed_to_wav(group.collected[idx])
            anchor_wav = self._delayed_to_wav(
                torch.tensor(ref["codes_delayed"], dtype=torch.long)
            )
            cal_f0 = median_f0(cal_wav)
            anchor_f0 = median_f0(anchor_wav)
            if cal_f0 is None or anchor_f0 is None:
                deviation = None
            else:
                deviation = abs(math.log(cal_f0 / anchor_f0))
            if deviation is None or deviation > _GATE_LOG_RATIO:
                next_seed = group.seed_used[idx] + 1
                if next_seed < len(CAL_SEEDS):
                    retry_rows.append((idx, next_seed))
                    logger.warning(
                        "reference-fusion %s: calibration for ref %d failed the "
                        "F0 gate (cal=%s anchor=%s), retrying with seed #%d",
                        group.request_id,
                        idx,
                        cal_f0,
                        anchor_f0,
                        next_seed,
                    )
                    continue
                raise RuntimeError(
                    f"calibration for reference {idx} failed the F0 quality "
                    f"gate on all {len(CAL_SEEDS)} seeds "
                    f"(last: cal_f0={cal_f0}, anchor_f0={anchor_f0})"
                )
            cal_wavs.append(cal_wav)

        if retry_rows:
            self._enqueue_retries(group, retry_rows)
            return

        hybrid = build_fused_reference(cal_wavs, group.weights)
        codes_TN = self._codec().encode_reference(
            torch.from_numpy(hybrid), sample_rate=_SAMPLE_RATE
        )
        delayed_rows = apply_delay_pattern(codes_TN).tolist()
        self._cache_put(group.cache_key, delayed_rows)

        if group.request_id in scheduler._aborted_request_ids:
            self._drop(group)
            return
        real_req_data = group.make_real_request(delayed_rows)
        self._drop(group)
        scheduler._enqueue_built_request(group.payload, False, real_req_data)
        logger.info(
            "reference-fusion %s: hybrid reference built (%d frames), real "
            "request enqueued",
            group.request_id,
            len(delayed_rows),
        )

    def _enqueue_retries(
        self, group: _BuildGroup, retry_rows: list[tuple[int, int]]
    ) -> None:
        # Import here: request_builders imports this module at load time.
        from sglang_omni.models.higgs_tts.request_builders import (
            build_calibration_request,
        )

        # Register all retries as pending before enqueueing any, so a
        # lightning-fast completion can't observe a half-updated group. Retry
        # rows get no member entry of their own (a single row is trivially
        # "complete" for the admission gate); they are only added to the
        # client-facing abort entry.
        pending_requests = []
        for ref_idx, seed_idx in retry_rows:
            group.collected.pop(ref_idx, None)
            group.seed_used[ref_idx] = seed_idx
            rid = f"{group.request_id}#cal{ref_idx}r{seed_idx}"
            row = _CalRow(ref_idx=ref_idx, seed_idx=seed_idx, rid=rid)
            group.pending[rid] = row
            pending_requests.append(
                build_calibration_request(
                    ref=group.refs[ref_idx],
                    rid=rid,
                    seed=CAL_SEEDS[seed_idx],
                    done_callback=self.make_done_callback(group.request_id, ref_idx),
                )
            )
            members = self._scheduler._fusion_group_members.get(group.request_id)
            if members is not None:
                members.add(rid)
        for req_data in pending_requests:
            self._scheduler._enqueue_built_request(group.payload, False, req_data)

    # -- failure / cleanup --

    def _drop(self, group: _BuildGroup) -> None:
        with self._lock:
            self._groups.pop(group.request_id, None)
        members = self._scheduler._fusion_group_members.pop(group.request_id, None)
        if members is not None:
            for rid in members:
                self._scheduler._fusion_group_members.pop(rid, None)

    def _fail(self, group: _BuildGroup, exc: Exception, *, emit: bool = True) -> None:
        with self._lock:
            if group.failed:
                return
            group.failed = True
        if emit:
            self._scheduler._emit_request_error(group.request_id, exc)
        # Cascade-kill any still-pending calibration rows for this group.
        for rid in list(group.pending):
            try:
                self._scheduler.abort(rid)
            except Exception:  # noqa: BLE001 - best-effort cleanup
                logger.exception("failed to abort calibration row %s", rid)
        self._drop(group)

    def _sweep_expired(self) -> None:
        now = time.monotonic()
        with self._lock:
            expired = [g for g in self._groups.values() if now > g.deadline]
        for group in expired:
            self._fail(
                group,
                RuntimeError(
                    f"reference-fusion build timed out after {BUILD_DEADLINE_S:.0f}s"
                ),
            )


_ORCHESTRATOR_ATTR = "_higgs_fusion_reference_orchestrator"


def get_orchestrator(model: Any) -> FusionReferenceOrchestrator:
    """Engine-process singleton, keyed on the Higgs model object."""
    orch = getattr(model, _ORCHESTRATOR_ATTR, None)
    if orch is None:
        orch = FusionReferenceOrchestrator()
        setattr(model, _ORCHESTRATOR_ATTR, orch)
    return orch


__all__ = [
    "ALGO_VERSION",
    "CAL_MAX_NEW_TOKENS",
    "CAL_SEEDS",
    "CAL_TEMPERATURE",
    "CAL_TOP_K",
    "CAL_TOP_P",
    "DEFAULT_CAL_TEXT",
    "FusionReferenceOrchestrator",
    "build_fused_reference",
    "calibration_text",
    "dtw_features",
    "dtw_map",
    "fused_reference_cache_key",
    "fusion_mode",
    "get_orchestrator",
    "median_f0",
    "world_extract",
]
