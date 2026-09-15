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

"""Encrypted marginal computation (pi_COMP)."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import functools
import itertools

from dpsynth.contrib.fhe import backend as backend_lib
from dpsynth.contrib.fhe import parties
import mbi


@dataclasses.dataclass(frozen=True)
class EncryptedMarginals:
  """All candidate marginals of a dataset, computed once under encryption.

  Attributes:
    domain: The discrete domain of the underlying dataset.
    cliques: The cliques whose marginals are held.
    packed: For each clique, one ciphertext whose slot i holds cell i of the
      marginal, in the same row-major order as `Dataset.project(cl).datavector()`.
  """

  domain: mbi.Domain
  cliques: tuple[mbi.Clique, ...]
  packed: dict[mbi.Clique, backend_lib.Ciphertext]

  def __contains__(self, clique: mbi.Clique) -> bool:
    return tuple(clique) in self.packed

  def __getitem__(self, clique: mbi.Clique) -> backend_lib.Ciphertext:
    return self.packed[tuple(clique)]

  @classmethod
  def compute(
      cls,
      backend: backend_lib.FHEBackend,
      columns: parties.EncryptedColumns,
      cliques: Sequence[mbi.Clique],
      max_degree: int = 2,
  ) -> EncryptedMarginals:
    """Computes the marginal of every clique from encrypted one-hot columns.

    A k-way marginal cell is the row-wise product of k indicator columns,
    summed over rows. Multiplicative depth is k - 1, independent of the number
    of rows: row chunks are combined by addition.

    Args:
      backend: The FHE backend.
      columns: The encrypted one-hot dataset.
      cliques: The cliques to compute. Duplicates are computed once.
      max_degree: Refuse cliques with more attributes than this. Higher degrees
        are correct but cost one multiplicative level each.

    Returns:
      The packed encrypted marginals.
    """
    domain = columns.domain
    unique = list(dict.fromkeys(tuple(cl) for cl in cliques))
    packed = {}
    for cl in unique:
      if len(cl) > max_degree:
        raise ValueError(
            f'Clique {cl} has degree {len(cl)} > max_degree={max_degree}.'
        )
      size = domain.size(cl)
      if size > backend.slots:
        raise ValueError(
            f'Marginal {cl} has {size} cells, more than {backend.slots} slots.'
        )
      cells = [
          _cell(backend, columns, cl, categories)
          for categories in itertools.product(*(range(domain[a]) for a in cl))
      ]
      packed[cl] = backend.pack(cells)
    return cls(domain=domain, cliques=tuple(unique), packed=packed)


def _cell(
    backend: backend_lib.FHEBackend,
    columns: parties.EncryptedColumns,
    clique: mbi.Clique,
    categories: Sequence[int],
) -> backend_lib.Ciphertext:
  """Counts rows where every attribute in `clique` takes the given category."""
  total = None
  for chunk, rows in enumerate(columns.chunk_rows):
    indicators = [
        columns.columns[attr][cat][chunk]
        for attr, cat in zip(clique, categories)
    ]
    product = functools.reduce(backend.mul, indicators)
    partial = backend.sum_slots(product, rows)
    total = partial if total is None else backend.add(total, partial)
  assert total is not None
  return total
