# SPDX-License-Identifier: Apache-2.0
"""Base-runner _install_sampling_seeds: wire a request seed to the sampler."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.model_runner.base import ModelRunner


def _req(seed):
    sp = SimpleNamespace(sampling_seed=seed)
    return SimpleNamespace(
        data=SimpleNamespace(req=SimpleNamespace(sampling_params=sp))
    )


def _fb(sampling_seed=None):
    return SimpleNamespace(
        sampling_info=SimpleNamespace(device="cpu", sampling_seed=sampling_seed)
    )


def test_installs_per_row_seeds_and_noops_without_one():
    runner = object.__new__(ModelRunner)
    # seeded rows -> per-row int64 seed tensor; unseeded row stays in range
    fb = _fb()
    runner._install_sampling_seeds(fb, [_req(42), _req(None)])
    ss = fb.sampling_info.sampling_seed
    assert isinstance(ss, torch.Tensor) and ss.dtype == torch.long
    assert int(ss[0]) == 42 and ss.shape == (2,)
    # no seed anywhere -> left unseeded (preserves random sampling)
    fb2 = _fb()
    runner._install_sampling_seeds(fb2, [_req(None), _req(None)])
    assert fb2.sampling_info.sampling_seed is None


def test_does_not_clobber_subclass_installed_seed():
    runner = object.__new__(ModelRunner)
    preset = torch.tensor([1, 2, 3])
    fb = _fb(sampling_seed=preset)
    runner._install_sampling_seeds(fb, [_req(42), _req(42), _req(42)])
    assert fb.sampling_info.sampling_seed is preset
