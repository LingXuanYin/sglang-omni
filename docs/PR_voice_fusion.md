# Higgs TTS: voice timbre fusion (multi-reference output-distribution blending)

## What

Adds **voice timbre fusion** to the Higgs TTS (`higgs_multimodal_qwen3`) path: a single
synthesis can be conditioned on **N reference voices at once**, blended by weight, producing
one new voice that interpolates the references — not a switch between them, not a concatenation.

API (OpenAI-compatible `/v1/audio/speech`), backward compatible:

```jsonc
{
  "input": "Text to speak in the blended voice.",
  "references": [
    {"audio_path": "a.wav", "text": "...", "weight": 0.6},
    {"audio_path": "b.wav", "text": "...", "weight": 0.4}
  ]
}
```

`>= 2` references each carrying a `weight` ⇒ fusion request. Anything else keeps the exact
legacy single-voice behavior. Weights are ratios (need not sum to 1). `N > 2` supported.

## How it works

Higgs has **no speaker/style embedding** — timbre is conditioned entirely by reference audio
codes in-context. So fusion is done at the **output-distribution layer**, the one place a
clean, continuous blend is possible:

1. **Fan-out** (`request_builders.py`): a fusion request is split into `N` *sibling* rows,
   each prefilling one reference voice into its own KV context. All siblings share one
   `fusion_group_id` and **one concrete sampling seed**.
2. **Blend** (`fusion.py`, `model.py`): at every AR decode step, the `N` siblings'
   per-codebook output distributions are weighted-averaged **before** sampling
   (`log(Σ wᵢ·softmax(logitsᵢ/T))`). All siblings then sample the **same** frame (shared
   seed), so their `N` KV contexts stay in lock-step and decode identical codes; only the
   group **leader** is emitted as audio.
3. **Co-retire** (group barrier): "any sibling done ⇒ all done", so the group's rows finish
   on the same step.

The blend op is CUDA-graph friendly (fixed-shape `scatter_add_`/advanced-index, no host
control flow) and is a **byte-identical no-op** for non-fusion rows (each is its own singleton
group), so the production decode path is unchanged for ordinary requests.

## Co-batching: two layers

The siblings must be co-batched every step for the shared-seed lock-step to hold. Continuous
batching schedules rows independently, so this is enforced in two layers:

- **Layer 1 — group-atomic prefill admission** (`OmniScheduler.get_next_batch_to_run`
  override): if a prefill batch contains only *some* members of a fusion group, the present
  members are deferred back to the waiting queue so the whole group enters together. Normal
  path; keeps Layer 2 from ever firing.
- **Layer 2 — decode-time group-completeness guard**: if a group is ever split mid-decode
  (e.g. a KV-pressure retract — the only place this is reachable, since retraction only acts
  on the running/decode batch), the *present* rows of that group are isolated — degraded to
  independent singletons and aborted (`FINISH_ABORT`) — without affecting the rest of the
  batch (other fusion groups, ordinary requests). The prefill-side counterpart
  (`model.py::_batch_local_fusion`) keeps a hard `RuntimeError` instead: Layer 1 should make a
  split prefill batch unreachable there, so tripping it means an invariant broke, not a normal
  runtime condition to route around.

## Files

| File | Change |
|---|---|
| `models/higgs_tts/fusion.py` | **new** — `fuse_group_logits` / `fuse_group_generation_done` (pure torch) + `FusionRegistry` (pure Python, thread-safe group/weight/leader bookkeeping), all unit-tested with no engine dependency |
| `models/higgs_tts/model.py` | delegates to `FusionRegistry`, blends logits in both decode paths, drives the group barrier |
| `models/higgs_tts/model_runner.py` | CG fusion-buffer population (with an early-out for zero-fusion traffic), follower output dedup via `_finish_fusion_follower` |
| `models/higgs_tts/request_builders.py` | sibling fan-out, shared seed, leader/follower |
| `models/higgs_tts/stages.py` | preprocessing/audio-encoder multi-reference handling |
| `models/higgs_tts/payload_types.py` | `HiggsTtsState.fusion_refs` |
| `scheduling/omni_scheduler.py` | group-atomic admission, abort cascade, follower lifecycle |
| `serve/protocol.py` | `SpeechReference.weight` / `reference_codes` |
| `tests/unit_test/higgs_tts/` | blend numerics (15 cases) + pipeline fan-out tests |

## Testing

- **`test_voice_fusion.py`** (no GPU, no engine): 21 cases — blend ops (singleton
  byte-identity, weighted average, weight ratio, shared-seed identical draw, mixed batch,
  group barrier, temperature-before-blend, the greedy-sampling regression guard) plus
  `FusionRegistry` bookkeeping (register/clear/reuse/snapshot). **All pass locally.**
- **`test_voice_fusion_pipeline.py`**: fan-out shape/seed/leader/weight (skips cleanly without
  the engine installed).

## Known items to verify on Linux + GPU (could not run here)

These are flagged honestly — the author's dev box is Windows (no `sgl_kernel`/engine), so the
blend math + serialization are verified but the full engine path is not:

1. **KV reclaim**: `get_next_batch_to_run` now releases each deferred sibling's KV via the
   same path `abort` uses before requeuing it, so a repeatedly-deferred group can't leak pool
   slots by construction — but this hasn't been load-tested against a real engine's
   `PrefillAdder` under repeated partial-group defers.
2. **Deadlock guard**: group-atomic admission assumes KV capacity ≥ one group's prefill.
   Deployments must budget `max_running_requests`/KV as **1 fusion request = N rows**.
3. **prefill→decode transition**: confirm a co-prefilled group enters the first decode step
   together (no single row decoding one step ahead).
4. **CUDA-graph replay**: confirm per-step `_cg_fusion_group`/`_cg_fusion_weight` repopulation
   composes with graph capture/replay.

Feedback welcome on whether group-atomic co-batching should instead ride a first-class
parallel-sampling (`n>1`) mechanism if the engine grows one.
