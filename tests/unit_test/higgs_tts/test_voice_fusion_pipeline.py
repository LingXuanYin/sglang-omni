# SPDX-License-Identifier: Apache-2.0
"""Pipeline-level tests for Higgs voice fusion: preprocessing detection,
builder fan-out, and group lifecycle.

These import the request/scheduler layer, which pulls in ``sglang`` — so they
run only in a full sglang-omni environment (Linux + sgl_kernel), not in a bare
torch venv. The pure-tensor blend math is covered separately by
``test_voice_fusion.py`` (no sglang import), which runs anywhere.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sglang")

from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.models.higgs_tts.request_builders import (
    build_fusion_sibling_requests,
)
from sglang_omni.models.higgs_tts.stages import _fusion_ref_entries


# --------------------------------------------------------------------------- #
# Fusion-request detection (_fusion_ref_entries)
# --------------------------------------------------------------------------- #
def test_detect_requires_two_weighted_refs():
    # < 2 refs → not fusion
    assert _fusion_ref_entries({"references": [{"audio_path": "a.wav", "weight": 1.0}]}) is None
    # 2 refs but no weight → legacy single-ref path, not fusion
    assert _fusion_ref_entries(
        {"references": [{"audio_path": "a.wav"}, {"audio_path": "b.wav"}]}
    ) is None


def test_detect_two_weighted_refs():
    specs = _fusion_ref_entries(
        {
            "references": [
                {"audio_path": "a.wav", "weight": 0.7, "text": "hi"},
                {"audio_path": "b.wav", "weight": 0.3},
            ]
        }
    )
    assert specs is not None
    assert len(specs) == 2
    assert specs[0]["weight"] == 0.7
    assert specs[0]["audio"] == "a.wav"
    assert specs[0]["reference_text"] == "hi"
    assert specs[1]["weight"] == 0.3


def test_detect_rejects_negative_weight():
    with pytest.raises(ValueError, match="weight"):
        _fusion_ref_entries(
            {
                "references": [
                    {"audio_path": "a.wav", "weight": -1.0},
                    {"audio_path": "b.wav", "weight": 0.5},
                ]
            }
        )


def test_detect_pre_encoded_codes():
    specs = _fusion_ref_entries(
        {
            "references": [
                {"reference_codes": [[1, 2, 3, 4, 5, 6, 7, 8]], "weight": 0.5},
                {"reference_codes": [[8, 7, 6, 5, 4, 3, 2, 1]], "weight": 0.5},
            ]
        }
    )
    assert specs is not None
    assert specs[0]["codes"] == [[1, 2, 3, 4, 5, 6, 7, 8]]
    assert specs[0]["audio"] is None


# --------------------------------------------------------------------------- #
# Builder fan-out (build_fusion_sibling_requests)
# --------------------------------------------------------------------------- #
def _fusion_state(n: int) -> HiggsTtsState:
    """A fusion state with ``n`` pre-built sibling refs (prompt + delayed codes)."""
    refs = []
    for i in range(n):
        refs.append(
            {
                "codes_delayed": [[i % 1024] * 8 for _ in range(3 + i)],
                "weight": 1.0,
                "prompt_token_ids": [10, 20, -100, -100, -100, 30, 40],
                "reference_text": None,
            }
        )
    return HiggsTtsState(
        prompt_token_ids=[],
        fusion_refs=refs,
        target_text="hello",
        num_codebooks=8,
        codebook_size=1026,
        max_new_tokens=256,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
    )


def test_fanout_produces_n_siblings():
    leader = build_fusion_sibling_requests(_fusion_state(3), request_id="rid-x")
    followers = leader.fusion_siblings
    assert followers is not None
    assert len(followers) == 2  # leader + 2 followers = 3
    group = [leader, *followers]
    assert all(s.fusion_group_id == "rid-x" for s in group)


def test_fanout_leader_and_followers():
    leader = build_fusion_sibling_requests(_fusion_state(3), request_id="rid-y")
    group = [leader, *(leader.fusion_siblings or [])]
    assert leader.fusion_is_leader is True
    assert sum(1 for s in group if s.fusion_is_leader) == 1
    assert [s.fusion_is_leader for s in group[1:]] == [False, False]


def test_fanout_shares_one_seed():
    leader = build_fusion_sibling_requests(_fusion_state(3), request_id="rid-z")
    group = [leader, *(leader.fusion_siblings or [])]
    seeds = {s.req.sampling_params.sampling_seed for s in group}
    assert len(seeds) == 1  # all siblings share one concrete seed
    assert next(iter(seeds)) is not None


def test_fanout_distinct_rids():
    leader = build_fusion_sibling_requests(_fusion_state(4), request_id="rid-w")
    group = [leader, *(leader.fusion_siblings or [])]
    rids = [s.req.rid for s in group]
    assert rids[0] == "rid-w"
    assert len(set(rids)) == 4  # all distinct


def test_fanout_weights_preserved():
    state = _fusion_state(2)
    state.fusion_refs[0]["weight"] = 0.7
    state.fusion_refs[1]["weight"] = 0.3
    leader = build_fusion_sibling_requests(state, request_id="rid-v")
    group = [leader, *(leader.fusion_siblings or [])]
    assert group[0].fusion_weight == pytest.approx(0.7)
    assert group[1].fusion_weight == pytest.approx(0.3)


def test_fanout_rejects_single_ref():
    with pytest.raises(ValueError, match=">= 2"):
        build_fusion_sibling_requests(_fusion_state(1), request_id="rid-u")
