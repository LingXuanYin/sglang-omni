# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared sampling-seed helpers."""

from __future__ import annotations

from sglang_omni.sampling.seed import (
    SAMPLING_SEED_MASK,
    derive_sampling_seed,
    resolve_row_seed,
)


def test_derive_is_deterministic_namespaced_and_in_int32_range():
    assert derive_sampling_seed(1234, "higgs") == derive_sampling_seed(1234, "higgs")
    assert derive_sampling_seed(1234, "a") != derive_sampling_seed(1234, "b")
    assert 0 <= derive_sampling_seed(1234, "higgs") <= SAMPLING_SEED_MASK


def test_resolve_row_seed_modes():
    # no namespace -> raw seed masked to positive int32
    assert resolve_row_seed(42) == 42
    assert resolve_row_seed(0xFFFFFFFF) == (0xFFFFFFFF & SAMPLING_SEED_MASK)
    # namespace -> derived
    assert resolve_row_seed(42, namespace="higgs") == derive_sampling_seed(42, "higgs")
    # None -> a fresh in-range seed
    assert 0 <= resolve_row_seed(None) <= SAMPLING_SEED_MASK
