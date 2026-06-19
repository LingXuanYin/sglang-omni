# SPDX-License-Identifier: Apache-2.0
"""Base-runner _install_sampling_seeds: wire a request seed to the sampler."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner.base import ModelRunner


def _req(seed):
    sp = SimpleNamespace(sampling_seed=seed)
    return SimpleNamespace(
        data=SimpleNamespace(req=SimpleNamespace(sampling_params=sp))
    )


def _fb(sampling_seed=None, *, top_p=False, top_k=False, min_p=False):
    return SimpleNamespace(
        sampling_info=SimpleNamespace(
            device="cpu",
            sampling_seed=sampling_seed,
            need_top_p_sampling=top_p,
            need_top_k_sampling=top_k,
            need_min_p_sampling=min_p,
        )
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


def test_unseeded_row_seed_is_resolved_once_and_cached():
    """An unseeded row in a seeded batch mints ONE random seed (cached on
    sampling_params), reused across decode steps — no per-step os.urandom."""
    runner = object.__new__(ModelRunner)
    requests = [_req(42), _req(None)]
    runner._install_sampling_seeds(_fb(), requests)
    cached = requests[1].data.req.sampling_params.sampling_seed
    assert cached is not None  # minted + cached back
    runner._install_sampling_seeds(_fb(), requests)  # next step
    assert requests[1].data.req.sampling_params.sampling_seed == cached  # reused


def test_rejects_seeded_min_p_before_upstream_sampler():
    runner = object.__new__(ModelRunner)
    with pytest.raises(ValueError, match="min_p"):
        runner._install_sampling_seeds(_fb(min_p=True), [_req(42)])


def test_rejects_seeded_flashinfer_top_p_before_upstream_sampler(monkeypatch):
    monkeypatch.setattr(
        "sglang_omni.model_runner.base._current_sglang_sampling_backend",
        lambda: "flashinfer",
    )
    runner = object.__new__(ModelRunner)
    with pytest.raises(ValueError, match="flashinfer"):
        runner._install_sampling_seeds(_fb(top_p=True), [_req(42)])


def test_allows_seeded_pytorch_top_p(monkeypatch):
    monkeypatch.setattr(
        "sglang_omni.model_runner.base._current_sglang_sampling_backend",
        lambda: "pytorch",
    )
    runner = object.__new__(ModelRunner)
    fb = _fb(top_p=True)
    runner._install_sampling_seeds(fb, [_req(42)])
    assert int(fb.sampling_info.sampling_seed[0]) == 42
