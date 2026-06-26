# SPDX-License-Identifier: Apache-2.0
"""TypedTensor: exact bytes+dtype+shape round-trip for pipeline-state tensors.

This is the primary escape hatch for :class:`PipelineStateBase` subclasses
whose tensor fields must survive serialization byte-for-byte across the relay
side-channel without a lossy ``tolist()``. The integer code tensor is packed as
``{key}_bytes`` / ``{key}_shape`` / ``{key}_dtype`` (narrowest of uint16/int32
that holds the values), and decoded back to an int64 tensor. ``legacy_key``
keeps payloads written before this encoding readable.

Reference implementation: Voxtral TTS ``audio_codes`` (``models/voxtral_tts``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def encode_typed_tensor(value: Any, *, key: str) -> dict[str, Any]:
    """Pack an integer code tensor/array as ``{key}_bytes/_shape/_dtype``.

    Picks the narrowest of uint16/int32 that holds the values so the payload
    stays compact. Returns a dict the caller merges into ``StagePayload.data``.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size == 0:
        array = array.astype(np.uint16, copy=False)
    elif int(array.min()) >= 0 and int(array.max()) <= np.iinfo(np.uint16).max:
        array = array.astype(np.uint16, copy=False)
    else:
        array = array.astype(np.int32, copy=False)
    contiguous = np.ascontiguousarray(array)
    return {
        f"{key}_bytes": contiguous.tobytes(),
        f"{key}_shape": list(contiguous.shape),
        f"{key}_dtype": str(contiguous.dtype),
    }


def decode_typed_tensor(
    data: dict[str, Any], *, key: str, legacy_key: str | None = None
) -> torch.Tensor | None:
    """Inverse of :func:`encode_typed_tensor`; returns an int64 tensor or None.

    ``legacy_key`` keeps backward compatibility with payloads that stored a
    plain list/tensor under a single key before the bytes encoding existed.
    """
    if legacy_key is not None:
        legacy = data.get(legacy_key)
        if legacy is not None:
            if isinstance(legacy, list):
                return torch.tensor(legacy)
            return legacy

    raw = data.get(f"{key}_bytes")
    shape = data.get(f"{key}_shape")
    if raw is None or shape is None:
        return None
    dtype = np.dtype(data.get(f"{key}_dtype", "uint16"))
    array = np.frombuffer(raw, dtype=dtype).reshape(shape).astype(np.int64)
    return torch.from_numpy(array)
