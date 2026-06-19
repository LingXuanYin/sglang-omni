# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared sampling-seed helpers."""

from __future__ import annotations

from sglang_omni.sampling.seed import (
    SAMPLING_SEED_MASK,
    new_random_sampling_seed,
    resolve_row_seed,
)


def test_new_random_sampling_seed_in_int32_range():
    assert 0 <= new_random_sampling_seed() <= SAMPLING_SEED_MASK


def test_resolve_row_seed_modes():
    # in-range seed is used as-is
    assert resolve_row_seed(42) == 42
    # out-of-range seed is masked to positive int32
    assert resolve_row_seed(0xFFFFFFFF) == (0xFFFFFFFF & SAMPLING_SEED_MASK)
    # None -> a fresh in-range seed
    assert 0 <= resolve_row_seed(None) <= SAMPLING_SEED_MASK
