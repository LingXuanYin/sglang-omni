# SPDX-License-Identifier: Apache-2.0
"""Reusable per-request sampling-seed helpers shared across model runners.

SGLang's ``multinomial_with_seed`` combines a per-row seed with a per-position
value, so a row's draw depends only on its own seed -- reproducible at any batch
size. These helpers derive the int32 per-row seeds the base runner installs onto
``forward_batch.sampling_info`` so a request ``seed`` is honored uniformly.
"""

from __future__ import annotations

import hashlib
import os

# multinomial_with_seed requires a positive int32 seed.
SAMPLING_SEED_MASK = 0x7FFFFFFF


def new_random_sampling_seed() -> int:
    """A fresh random positive-int32 seed (for rows the caller left unseeded)."""
    return int.from_bytes(os.urandom(4), "little") & SAMPLING_SEED_MASK


def derive_sampling_seed(public_seed: int, namespace: str) -> int:
    """Stable positive-int32 seed from a public ``seed``, namespaced per use site
    so different stages of one request don't draw identically."""
    digest = hashlib.blake2b(
        f"{namespace}:{int(public_seed)}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little") & SAMPLING_SEED_MASK


def resolve_row_seed(public_seed: int | None, *, namespace: str | None = None) -> int:
    """Per-row seed: derive (namespaced) or mask the user seed when given, else a
    fresh random seed so unseeded rows stay random."""
    if public_seed is None:
        return new_random_sampling_seed()
    if namespace is None:
        return int(public_seed) & SAMPLING_SEED_MASK
    return derive_sampling_seed(public_seed, namespace)


__all__ = [
    "SAMPLING_SEED_MASK",
    "derive_sampling_seed",
    "new_random_sampling_seed",
    "resolve_row_seed",
]
