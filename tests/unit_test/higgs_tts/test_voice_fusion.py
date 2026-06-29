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
