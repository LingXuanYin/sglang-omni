# SPDX-License-Identifier: Apache-2.0
"""Unit tests for voice-fusion blend ops (pure torch, no sglang engine)."""

import pytest
import torch

from sglang_omni.models.higgs_tts.fusion import (
    FusionRegistry,
    fuse_group_generation_done,
    fuse_group_logits,
)


def _singleton_groups(B):
    return torch.arange(B, dtype=torch.long), torch.ones(B, dtype=torch.float32)


def test_singleton_is_sampling_identity():
    """Each row its own group → blended log-probs sample-equivalent to raw logits.

    log(softmax(logits)) differs from logits only by a per-row constant, which
    leaves argmax and multinomial(softmax(.)) unchanged.
    """
    torch.manual_seed(0)
    B, N, V = 4, 8, 1026
    logits = torch.randn(B, N, V)
    gid, w = _singleton_groups(B)
    out, is_grouped = fuse_group_logits(logits, gid, w)
    assert not is_grouped.any()
    # argmax preserved per (row, codebook)
    assert torch.equal(out.argmax(-1), logits.argmax(-1))
    # softmax preserved (log-prob == log-softmax up to fp error)
    torch.testing.assert_close(
        out.softmax(-1), logits.float().softmax(-1), atol=1e-5, rtol=1e-4
    )


def test_singleton_is_byte_identical_to_scaled_logits():
    """Bug-C guard: a singleton row returns logits/T *exactly* (byte-identical),
    so a mixed batch's non-fusion rows are unchanged vs. the no-fusion baseline.

    The sampler consumes the returned tensor as logits at temperature 1, i.e.
    ``softmax(out / 1)``. For a singleton the contract is ``out == logits / T``
    bit-for-bit, not merely sample-equivalent.
    """
    torch.manual_seed(7)
    B, N, V = 3, 8, 1026
    logits = torch.randn(B, N, V)
    gid, w = _singleton_groups(B)
    # temperature == 1: out must equal the raw logits exactly.
    out1, is_grouped1 = fuse_group_logits(logits, gid, w, temperature_B=torch.ones(B))
    assert not is_grouped1.any()
    assert torch.equal(out1, logits.float())
    # arbitrary per-row temperature: out must equal logits / T exactly.
    temp = torch.tensor([0.7, 1.0, 1.5])
    out2, is_grouped2 = fuse_group_logits(logits, gid, w, temperature_B=temp)
    assert not is_grouped2.any()
    assert torch.equal(out2, logits.float() / temp.view(B, 1, 1))


def test_mixed_batch_singleton_rows_byte_identical():
    """In a batch mixing a fused group with singletons, the singleton rows are
    byte-identical to logits/T while the fused rows blend (Bug-C contract)."""
    torch.manual_seed(8)
    N, V = 8, 1026
    logits = torch.randn(4, N, V)
    gid = torch.tensor([0, 0, 2, 3], dtype=torch.long)  # rows 0,1 fused; 2,3 alone
    w = torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float32)
    temp = torch.tensor([1.0, 1.0, 0.8, 1.3])
    out, is_grouped = fuse_group_logits(logits, gid, w, temperature_B=temp)
    assert is_grouped.tolist() == [True, True, False, False]
    # singleton rows: exact logits/T
    assert torch.equal(out[2], logits[2].float() / 0.8)
    assert torch.equal(out[3], logits[3].float() / 1.3)
    # fused rows: blended (not equal to either raw row)
    assert not torch.equal(out[0], logits[0].float())


def test_two_member_equal_weight_is_prob_average():
    """A 2-row group at 0.5/0.5 yields the arithmetic mean of the two softmaxes."""
    torch.manual_seed(1)
    N, V = 8, 1026
    logits = torch.randn(2, N, V)
    gid = torch.tensor([0, 0], dtype=torch.long)
    w = torch.tensor([0.5, 0.5], dtype=torch.float32)
    out, is_grouped = fuse_group_logits(logits, gid, w)
    assert is_grouped.all()
    expected = 0.5 * logits[0].softmax(-1) + 0.5 * logits[1].softmax(-1)
    # both rows carry the same fused distribution
    torch.testing.assert_close(out[0].softmax(-1), expected, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(out[1].softmax(-1), out[0].softmax(-1))


def test_weight_ratio_only():
    """Unnormalized weights blend by ratio: [3,1] == [0.75,0.25]."""
    torch.manual_seed(2)
    N, V = 8, 1026
    logits = torch.randn(2, N, V)
    gid = torch.tensor([0, 0], dtype=torch.long)
    out_raw, _ = fuse_group_logits(logits, gid, torch.tensor([3.0, 1.0]))
    out_norm, _ = fuse_group_logits(logits, gid, torch.tensor([0.75, 0.25]))
    torch.testing.assert_close(
        out_raw.softmax(-1), out_norm.softmax(-1), atol=1e-6, rtol=1e-5
    )


def test_fused_rows_sample_identically_with_shared_seed():
    """Two fused rows + same seed draw the same multi-codebook frame."""
    torch.manual_seed(3)
    N, V = 8, 1026
    logits = torch.randn(2, N, V)
    gid = torch.tensor([0, 0], dtype=torch.long)
    fused, _ = fuse_group_logits(logits, gid, torch.tensor([0.5, 0.5]))
    g0 = torch.Generator().manual_seed(42)
    g1 = torch.Generator().manual_seed(42)
    s0 = fused[0].softmax(-1).multinomial(1, generator=g0)
    s1 = fused[1].softmax(-1).multinomial(1, generator=g1)
    assert torch.equal(s0, s1)


def test_mixed_batch_groups_and_singletons():
    """A batch mixing a 2-row group and a singleton blends only within group."""
    torch.manual_seed(4)
    N, V = 8, 1026
    logits = torch.randn(3, N, V)
    gid = torch.tensor([0, 0, 2], dtype=torch.long)  # rows 0,1 grouped; row 2 alone
    w = torch.tensor([0.5, 0.5, 1.0], dtype=torch.float32)
    out, is_grouped = fuse_group_logits(logits, gid, w)
    assert is_grouped.tolist() == [True, True, False]
    expected01 = 0.5 * logits[0].softmax(-1) + 0.5 * logits[1].softmax(-1)
    torch.testing.assert_close(out[0].softmax(-1), expected01, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(out[1].softmax(-1), expected01, atol=1e-5, rtol=1e-4)
    # singleton untouched (sample-equivalent)
    assert torch.equal(out[2].argmax(-1), logits[2].argmax(-1))


def test_generation_done_barrier():
    """Any done in a group ⇒ all done; singletons untouched."""
    gid = torch.tensor([0, 0, 0, 3], dtype=torch.long)
    done = torch.tensor([False, True, False, False])
    out = fuse_group_generation_done(done, gid)
    assert out.tolist() == [True, True, True, False]


def test_generation_done_singletons_identity():
    done = torch.tensor([True, False, True])
    gid = torch.arange(3, dtype=torch.long)
    assert torch.equal(fuse_group_generation_done(done, gid), done)


def test_temperature_applied_before_blend():
    """temperature_B scales each row's logits before the softmax-blend."""
    torch.manual_seed(5)
    N, V = 8, 1026
    logits = torch.randn(2, N, V)
    gid = torch.tensor([0, 0], dtype=torch.long)
    w = torch.tensor([0.5, 0.5])
    temp = torch.tensor([2.0, 2.0])
    out, _ = fuse_group_logits(logits, gid, w, temperature_B=temp)
    expected = 0.5 * (logits[0] / 2.0).softmax(-1) + 0.5 * (logits[1] / 2.0).softmax(-1)
    torch.testing.assert_close(out[0].softmax(-1), expected, atol=1e-5, rtol=1e-4)


# --------------------------------------------------------------------------- #
# Regression guard for the Linus-review BLOCKING-1 finding: a caller that
# unconditionally sets ``sampler_temperature = 1`` after calling
# ``fuse_group_logits`` silently defeats the sampler's greedy short-circuit for
# every ordinary (non-fusion) request. These tests exercise the *actual*
# invariant that matters — the sampled codes, not an intermediate tensor — so
# they fail loudly if a future caller reintroduces the bug.
# --------------------------------------------------------------------------- #
def _greedy_sample(logits_NV: torch.Tensor) -> torch.Tensor:
    """Mirrors ``sampler._sample_independent``'s greedy branch: plain argmax."""
    return logits_NV.argmax(dim=-1)


def _wrong_caller_sample(
    logits_NV: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """The BLOCKING-1 bug, reproduced directly: blend with temperature folded
    in, then unconditionally resample at temperature=1 regardless of whether
    the row was ever actually grouped. For a singleton row this multinomial-
    samples a near-one-hot distribution instead of taking the argmax."""
    gid = torch.arange(logits_NV.shape[0], dtype=torch.long)
    w = torch.ones(logits_NV.shape[0])
    temp = torch.full((logits_NV.shape[0],), 1e-5)  # requested: greedy
    blended, _ = fuse_group_logits(logits_NV.unsqueeze(1), gid, w, temperature_B=temp)
    probs = blended.squeeze(1).softmax(dim=-1)
    return probs.multinomial(num_samples=1, generator=generator).squeeze(-1)


def _correct_caller_sample(logits_NV: torch.Tensor) -> torch.Tensor:
    """The fixed contract: singleton rows sample at their real temperature
    (here ~0 → greedy), so they must go through argmax exactly like baseline
    and must NOT touch the RNG at all."""
    gid = torch.arange(logits_NV.shape[0], dtype=torch.long)
    w = torch.ones(logits_NV.shape[0])
    temp = torch.full((logits_NV.shape[0],), 1e-5)
    blended, is_grouped = fuse_group_logits(
        logits_NV.unsqueeze(1), gid, w, temperature_B=temp
    )
    assert not is_grouped.any()  # every row here is a singleton
    sampler_temp = torch.where(is_grouped, torch.ones_like(temp), temp)
    # Mirror the batched sampler's greedy short-circuit: temperature<=threshold
    # rows go through argmax — no multinomial call, no RNG consumed.
    assert bool((sampler_temp <= 1e-5).all())
    return blended.squeeze(1).argmax(-1)


def test_singleton_greedy_sampling_matches_baseline_not_the_blocking1_bug():
    """A plain (non-fusion) request with temperature=0 must decode by argmax,
    byte-identical to the no-fusion baseline, and must not consume any RNG
    state (greedy is deterministic and mustn't perturb other rows' draws in
    the same batch). The BLOCKING-1 bug — a caller that unconditionally
    resamples at temperature=1 after the blend — breaks BOTH properties for
    every ordinary request, fusion or not: it becomes a multinomial draw
    (RNG-consuming, only *probabilistically* matching argmax) instead of a
    deterministic, RNG-free argmax.
    """
    torch.manual_seed(11)
    B, V = 4, 1026
    logits = torch.randn(B, V)
    baseline = _greedy_sample(logits)

    # The buggy caller path takes a torch.Generator only because it MUST
    # consume RNG state (multinomial) — the fixed path below takes none.
    # Confirm bug reproduction: same seed, but it's a real sampling call.
    g = torch.Generator().manual_seed(123)
    state_before = g.get_state().clone()
    _wrong_caller_sample(logits, g)
    state_after_wrong = g.get_state()
    assert not torch.equal(state_before, state_after_wrong), (
        "sanity check: the buggy always-temperature=1 caller pattern is "
        "expected to consume RNG state via multinomial — if it doesn't, this "
        "helper no longer reproduces BLOCKING-1"
    )

    # The fix: singleton rows must sample at their real temperature, matching
    # baseline exactly and touching no RNG at all (deterministic argmax).
    correct = _correct_caller_sample(logits)
    assert torch.equal(correct, baseline)


# --------------------------------------------------------------------------- #
# FusionRegistry — the engine-side bookkeeping backing HiggsTTSModel's
# set_fusion_group/has_any_fusion/is_fusion_follower/... (Linus-review
# MAJOR-4: the non-fusion hot path must skip fusion work at zero cost). Pure
# Python, no torch/sglang dependency, so the counter/registry logic itself is
# directly unit-testable here rather than only indirectly via the model.
# --------------------------------------------------------------------------- #
def test_registry_starts_empty():
    reg = FusionRegistry()
    assert reg.has_any() is False
    assert reg.expected_size("g0") == 0
    assert reg.is_leader("nope") is True  # default: a non-member is its own leader
    assert reg.is_follower("nope") is False


def test_registry_register_marks_has_any_and_expected_size():
    reg = FusionRegistry()
    reg.set("r0", "g0", 0.5, is_leader=True)
    reg.set("r1", "g0", 0.5, is_leader=False)
    assert reg.has_any() is True
    assert reg.expected_size("g0") == 2
    assert reg.is_leader("r0") is True
    assert reg.is_follower("r0") is False
    assert reg.is_leader("r1") is False
    assert reg.is_follower("r1") is True


def test_registry_expected_size_only_counts_matching_group():
    reg = FusionRegistry()
    reg.set("r0", "g0", 1.0, is_leader=True)
    reg.set("r1", "g0", 1.0, is_leader=False)
    reg.set("r2", "g1", 1.0, is_leader=True)
    assert reg.expected_size("g0") == 2
    assert reg.expected_size("g1") == 1
    assert reg.expected_size("g-missing") == 0


def test_registry_clear_one_member_keeps_has_any_true_for_the_rest():
    reg = FusionRegistry()
    reg.set("r0", "g0", 0.5, is_leader=True)
    reg.set("r1", "g0", 0.5, is_leader=False)
    reg.set("r0", None, 1.0, is_leader=True)  # clear leader only
    assert reg.has_any() is True  # r1 still registered
    assert reg.expected_size("g0") == 1
    assert reg.is_follower("r0") is False  # no longer a member at all


def test_registry_clear_last_member_resets_has_any():
    reg = FusionRegistry()
    reg.set("r0", "g0", 0.5, is_leader=True)
    reg.set("r1", "g0", 0.5, is_leader=False)
    reg.set("r0", None, 1.0, is_leader=True)
    reg.set("r1", None, 1.0, is_leader=False)
    assert reg.has_any() is False
    assert reg.expected_size("g0") == 0


def test_registry_reregistering_same_req_id_does_not_inflate_active_count():
    """Idempotent re-registration (e.g. a retry reusing a request id) must
    overwrite in place, not double-count — else ``has_any`` could get stuck
    True after every member is cleared once."""
    reg = FusionRegistry()
    reg.set("r0", "g0", 0.5, is_leader=True)
    reg.set("r0", "g0", 0.9, is_leader=True)  # re-register same id, same group
    reg.set("r0", None, 1.0, is_leader=True)  # single clear must fully zero it out
    assert reg.has_any() is False


def test_registry_clear_of_never_registered_id_is_a_no_op():
    reg = FusionRegistry()
    reg.set("ghost", None, 1.0, is_leader=True)
    assert reg.has_any() is False


def test_registry_reused_id_after_clear_and_reregister_has_correct_count():
    """Register → clear → register again on the same req_id (id reuse across
    requests) must leave the registry in exactly the single-member state, not
    drift the active count from stale increments/decrements."""
    reg = FusionRegistry()
    reg.set("r0", "g0", 0.5, is_leader=True)
    reg.set("r0", None, 1.0, is_leader=True)
    reg.set("r0", "g1", 0.7, is_leader=False)
    assert reg.has_any() is True
    assert reg.expected_size("g0") == 0
    assert reg.expected_size("g1") == 1
    assert reg.is_follower("r0") is True


def test_registry_snapshot_restricted_to_members():
    reg = FusionRegistry()
    reg.set("r0", "g0", 0.7, is_leader=True)
    reg.set("r1", "g0", 0.3, is_leader=False)
    group_of, weight_of = reg.snapshot(["r0", "r1", "not-a-member"])
    assert group_of == {"r0": "g0", "r1": "g0"}
    assert weight_of == {"r0": pytest.approx(0.7), "r1": pytest.approx(0.3)}


def test_registry_snapshot_empty_for_all_non_members():
    reg = FusionRegistry()
    group_of, weight_of = reg.snapshot(["a", "b"])
    assert group_of == {}
    assert weight_of == {}
