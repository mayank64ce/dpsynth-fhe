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

"""The parties of the FHAIM protocol and the objects they exchange.

FHAIM has a data entity that encrypts the data and a pool of unit noise
samples, a computation entity that runs the encrypted protocols, and a
crypto-service entity that holds the secret key and decrypts only noisy
statistics. `DataOwner` and `KeyHolder` model the first and the last; the
mechanism itself plays the computation entity. All three run in-process here,
but nothing below depends on that: the objects that cross party boundaries
(`EncryptedColumns`, `NoisePool`, ciphertexts handed to `KeyHolder.decrypt`)
are the only interfaces between them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses

from dpsynth.contrib.fhe import backend as backend_lib
import mbi
import numpy as np


@dataclasses.dataclass(frozen=True)
class EncryptedColumns:
  """One-hot encoded dataset, encrypted column-wise and chunked row-wise.

  Attributes:
    domain: The discrete domain of the dataset.
    columns: For each attribute, a list over categories; each entry is a list
      over row chunks of ciphertexts. Slot r of chunk c holds the indicator that
      row (c * chunk + r) takes that category.
    chunk_rows: The number of rows in each chunk. All but the last chunk hold
      `backend.slots` rows.
  """

  domain: mbi.Domain
  columns: Mapping[str | int, Sequence[Sequence[backend_lib.Ciphertext]]]
  chunk_rows: tuple[int, ...]

  @property
  def num_records(self) -> int:
    return int(sum(self.chunk_rows))


@dataclasses.dataclass(frozen=True)
class NoisePlan:
  """How many unit noise vectors to pre-encrypt.

  Attributes:
    gaussian: Mapping from vector length to the number of unit Gaussian vectors
      of that length needed for pi_MEASURE.
    gumbel_length: Length of each unit Gumbel vector for pi_SELECT.
    gumbel_count: Number of unit Gumbel vectors.
  """

  gaussian: Mapping[int, int]
  gumbel_length: int
  gumbel_count: int

  @classmethod
  def for_aim(
      cls,
      domain: mbi.Domain,
      candidates: Sequence[mbi.Clique],
      one_way_cliques: Sequence[mbi.Clique],
      max_rounds: int,
  ) -> NoisePlan:
    """Upper-bounds the noise AIM can consume.

    Each round measures exactly one candidate, whose size is unknown in
    advance, so `max_rounds` vectors are reserved for every distinct candidate
    size. Each round also draws one Gumbel sample per candidate.
    """
    gaussian: dict[int, int] = {}
    for cl in one_way_cliques:
      n = domain.size(cl)
      gaussian[n] = gaussian.get(n, 0) + 1
    for n in {domain.size(cl) for cl in candidates}:
      gaussian[n] = gaussian.get(n, 0) + max_rounds
    return cls(
        gaussian=gaussian,
        gumbel_length=len(candidates),
        gumbel_count=max_rounds,
    )


class NoisePool:
  """Pre-encrypted unit noise vectors, consumed in order and never reused."""

  def __init__(
      self,
      gaussian: Mapping[int, Sequence[backend_lib.Ciphertext]],
      gumbel: Sequence[backend_lib.Ciphertext],
      gumbel_length: int,
  ):
    self._gaussian = {n: list(cts) for n, cts in gaussian.items()}
    self._gumbel = list(gumbel)
    self._gumbel_length = gumbel_length

  def next_gaussian(self, n: int) -> backend_lib.Ciphertext:
    """Returns a fresh ciphertext of n unit Gaussian samples."""
    pool = self._gaussian.get(n)
    if not pool:
      raise RuntimeError(
          f'Noise pool exhausted for Gaussian vectors of length {n}.'
      )
    return pool.pop()

  def next_gumbel(self, n: int) -> backend_lib.Ciphertext:
    """Returns a fresh ciphertext whose first n slots are unit Gumbel samples."""
    if n > self._gumbel_length:
      raise ValueError(
          f'Requested {n} Gumbel samples but vectors hold {self._gumbel_length}.'
      )
    if not self._gumbel:
      raise RuntimeError('Noise pool exhausted for Gumbel vectors.')
    return self._gumbel.pop()


class KeyHolder:
  """The crypto-service entity: owns the secret key, decrypts on request.

  The public key is exposed for the data owner to encrypt with. Everything
  passed to `decrypt` is visible to this party, so callers must only decrypt
  values that already carry their DP noise.
  """

  def __init__(self, backend: backend_lib.FHEBackend):
    self._backend = backend
    self._keys = backend.keygen()

  @property
  def public_key(self) -> backend_lib.PublicKey:
    return self._keys.public

  def decrypt(self, ct: backend_lib.Ciphertext, n: int) -> np.ndarray:
    return self._backend.decrypt(ct, self._keys.secret, n)


class DataOwner:
  """The data entity: encrypts the dataset and a pool of unit noise samples."""

  def __init__(
      self,
      backend: backend_lib.FHEBackend,
      public_key: backend_lib.PublicKey,
  ):
    self._backend = backend
    self._public_key = public_key

  def encrypt_dataset(self, data: mbi.Dataset) -> EncryptedColumns:
    """One-hot encodes each attribute and encrypts every indicator column."""
    if data.weights is not None and not np.all(data.weights == 1):
      raise ValueError('Weighted datasets are not supported.')
    slots = self._backend.slots
    n = data.records
    starts = list(range(0, n, slots))
    chunk_rows = tuple(min(slots, n - s) for s in starts)
    columns = {}
    for attr in data.domain:
      values = np.asarray(data.data[attr])
      per_category = []
      for category in range(data.domain[attr]):
        indicator = (values == category).astype(np.float64)
        per_category.append([
            self._backend.encrypt(indicator[s : s + slots], self._public_key)
            for s in starts
        ])
      columns[attr] = per_category
    return EncryptedColumns(
        domain=data.domain, columns=columns, chunk_rows=chunk_rows
    )

  def encrypt_noise(
      self, rng: np.random.Generator, plan: NoisePlan
  ) -> NoisePool:
    """Samples and encrypts the unit noise described by `plan`."""
    gaussian = {
        n: [
            self._backend.encrypt(rng.normal(size=n), self._public_key)
            for _ in range(count)
        ]
        for n, count in plan.gaussian.items()
    }
    gumbel = [
        self._backend.encrypt(
            rng.gumbel(size=plan.gumbel_length), self._public_key
        )
        for _ in range(plan.gumbel_count)
    ]
    return NoisePool(gaussian, gumbel, plan.gumbel_length)
