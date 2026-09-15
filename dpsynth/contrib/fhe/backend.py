# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Backend interface for the arithmetic needed by FHAIM.

The interface is deliberately small: it is the set of CKKS-style SIMD
operations the FHAIM protocols use, and nothing else. A ciphertext is an opaque
handle holding a vector of `slots` real numbers. Slots beyond the length of
the encrypted vector are zero at encryption time but may hold arbitrary values
after `sum_slots` or `inner_product`; only the documented slots of each result
are meaningful.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
import dataclasses
from typing import Any

import numpy as np

# Opaque handles. Concrete backends decide the real types.
Ciphertext = Any
Plaintext = Any
PublicKey = Any
SecretKey = Any


@dataclasses.dataclass(frozen=True)
class KeyPair:
  public: PublicKey
  secret: SecretKey


class FHEBackend(abc.ABC):
  """Arithmetic over encrypted vectors.

  Depth budget per operation (for CKKS backends): `add`, `sub` and `pack` cost
  no multiplicative levels; `mul`, `scale` and `inner_product` cost one.
  """

  @property
  @abc.abstractmethod
  def slots(self) -> int:
    """Number of vector slots in a single ciphertext."""

  @abc.abstractmethod
  def keygen(self) -> KeyPair:
    """Generates a fresh key pair (including any evaluation keys needed)."""

  @abc.abstractmethod
  def encrypt(self, x: np.ndarray, public_key: PublicKey) -> Ciphertext:
    """Encrypts a vector of length <= slots, zero-padded to `slots`."""

  @abc.abstractmethod
  def encode(self, x: np.ndarray) -> Plaintext:
    """Encodes a public vector of length <= slots as a plaintext operand."""

  @abc.abstractmethod
  def add(self, a: Ciphertext, b: Ciphertext | Plaintext) -> Ciphertext:
    """Slot-wise a + b."""

  @abc.abstractmethod
  def sub(self, a: Ciphertext, b: Ciphertext | Plaintext) -> Ciphertext:
    """Slot-wise a - b."""

  @abc.abstractmethod
  def mul(self, a: Ciphertext, b: Ciphertext | Plaintext) -> Ciphertext:
    """Slot-wise a * b."""

  @abc.abstractmethod
  def scale(self, a: Ciphertext, c: float) -> Ciphertext:
    """Slot-wise a * c for a public scalar c."""

  @abc.abstractmethod
  def sum_slots(self, a: Ciphertext, n: int) -> Ciphertext:
    """Sum of the first n slots of a, returned in slot 0.

    Other slots of the result are unspecified.
    """

  @abc.abstractmethod
  def inner_product(self, a: Ciphertext, b: Ciphertext, n: int) -> Ciphertext:
    """Inner product of the first n slots of a and b, returned in slot 0.

    Other slots of the result are unspecified.
    """

  @abc.abstractmethod
  def pack(self, scalars: Sequence[Ciphertext]) -> Ciphertext:
    """Packs slot 0 of each input into slot i of the result.

    This is the pi_COMB sub-protocol: mask slot 0, rotate to position i, add.
    Slots beyond len(scalars) are zero.
    """

  @abc.abstractmethod
  def decrypt(
      self, a: Ciphertext, secret_key: SecretKey, n: int
  ) -> np.ndarray:
    """Decrypts the first n slots of a."""


class PlaintextBackend(FHEBackend):
  """NumPy simulation of `FHEBackend` with exact arithmetic.

  A "ciphertext" is a float64 array of length `slots`. Keys are inert tokens
  that are checked on decrypt so that key-custody bugs (decrypting with the
  wrong party's key) still surface in tests. The unspecified-slot semantics of
  `sum_slots` and `inner_product` are simulated by filling those slots with
  NaN, so any code that accidentally relies on them fails loudly.
  """

  def __init__(self, slots: int = 4096):
    if slots <= 0:
      raise ValueError('slots must be positive.')
    self._slots = slots
    self._next_key_id = 0

  @property
  def slots(self) -> int:
    return self._slots

  def keygen(self) -> KeyPair:
    key_id = self._next_key_id
    self._next_key_id += 1
    return KeyPair(public=('pk', key_id), secret=('sk', key_id))

  def _check_length(self, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size > self._slots:
      raise ValueError(
          f'Vector of length {x.size} does not fit in {self._slots} slots.'
      )
    return x

  def _pad(self, x: np.ndarray) -> np.ndarray:
    out = np.zeros(self._slots, dtype=np.float64)
    out[: x.size] = x
    return out

  def encrypt(self, x: np.ndarray, public_key: PublicKey) -> np.ndarray:
    if not (isinstance(public_key, tuple) and public_key[0] == 'pk'):
      raise ValueError('encrypt requires a public key.')
    return self._pad(self._check_length(x))

  def encode(self, x: np.ndarray) -> np.ndarray:
    return self._pad(self._check_length(x))

  def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b

  def sub(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a - b

  def mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b

  def scale(self, a: np.ndarray, c: float) -> np.ndarray:
    return a * float(c)

  def _slot0(self, value: float) -> np.ndarray:
    out = np.full(self._slots, np.nan, dtype=np.float64)
    out[0] = value
    return out

  def sum_slots(self, a: np.ndarray, n: int) -> np.ndarray:
    return self._slot0(float(np.sum(a[:n])))

  def inner_product(self, a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    return self._slot0(float(np.dot(a[:n], b[:n])))

  def pack(self, scalars: Sequence[np.ndarray]) -> np.ndarray:
    if len(scalars) > self._slots:
      raise ValueError(
          f'Cannot pack {len(scalars)} scalars into {self._slots} slots.'
      )
    out = np.zeros(self._slots, dtype=np.float64)
    for i, ct in enumerate(scalars):
      out[i] = ct[0]
    return out

  def decrypt(
      self, a: np.ndarray, secret_key: SecretKey, n: int
  ) -> np.ndarray:
    if not (isinstance(secret_key, tuple) and secret_key[0] == 'sk'):
      raise ValueError('decrypt requires a secret key.')
    return np.array(a[:n], dtype=np.float64)
