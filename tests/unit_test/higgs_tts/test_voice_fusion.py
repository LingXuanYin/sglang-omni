# SPDX-License-Identifier: Apache-2.0
"""Unit tests for voice-fusion blend ops (pure torch, no sglang engine)."""

import torch

from sglang_omni.models.higgs_tts.fusion import (
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
    out = fuse_group_logits(logits, gid, w)
    # argmax preserved per (row, codebook)
    assert torch.equal(out.argmax(-1), logits.argmax(-1))
    # softmax preserved (log-prob == log-softmax up to fp error)
    torch.testing.assert_close(out.softmax(-1), logits.float().softmax(-1), atol=1e-5, rtol=1e-4)


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
    out1 = fuse_group_logits(logits, gid, w, temperature_B=torch.ones(B))
    assert torch.equal(out1, logits.float())
    # arbitrary per-row temperature: out must equal logits / T exactly.
    temp = torch.tensor([0.7, 1.0, 1.5])
    out2 = fuse_group_logits(logits, gid, w, temperature_B=temp)
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
    out = fuse_group_logits(logits, gid, w, temperature_B=temp)
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
    out = fuse_group_logits(logits, gid, w)
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
    out_raw = fuse_group_logits(logits, gid, torch.tensor([3.0, 1.0]))
    out_norm = fuse_group_logits(logits, gid, torch.tensor([0.75, 0.25]))
    torch.testing.assert_close(out_raw.softmax(-1), out_norm.softmax(-1), atol=1e-6, rtol=1e-5)


def test_fused_rows_sample_identically_with_shared_seed():
    """Two fused rows + same seed draw the same multi-codebook frame."""
    torch.manual_seed(3)
    N, V = 8, 1026
    logits = torch.randn(2, N, V)
    gid = torch.tensor([0, 0], dtype=torch.long)
    fused = fuse_group_logits(logits, gid, torch.tensor([0.5, 0.5]))
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
    out = fuse_group_logits(logits, gid, w)
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
    out = fuse_group_logits(logits, gid, w, temperature_B=temp)
    expected = 0.5 * (logits[0] / 2.0).softmax(-1) + 0.5 * (logits[1] / 2.0).softmax(-1)
    torch.testing.assert_close(out[0].softmax(-1), expected, atol=1e-5, rtol=1e-4)
