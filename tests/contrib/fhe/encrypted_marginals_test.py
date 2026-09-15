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

from absl.testing import absltest
from absl.testing import parameterized
from dpsynth.contrib.fhe import backend as backend_lib
from dpsynth.contrib.fhe import encrypted_marginals
from dpsynth.contrib.fhe import parties
import mbi
import numpy as np


def _dataset(n=50, seed=0):
  domain = mbi.Domain(['a', 'b', 'c'], [2, 3, 4])
  rng = np.random.default_rng(seed)
  data = {a: rng.integers(0, domain[a], size=n) for a in domain}
  return mbi.Dataset(data, domain)


class EncryptedMarginalsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('single_chunk', 64),
      ('many_chunks', 16),
  )
  def test_matches_plaintext_marginals(self, slots):
    data = _dataset()
    backend = backend_lib.PlaintextBackend(slots=slots)
    holder = parties.KeyHolder(backend)
    owner = parties.DataOwner(backend, holder.public_key)
    columns = owner.encrypt_dataset(data)
    self.assertEqual(columns.num_records, data.records)
    self.assertLen(columns.chunk_rows, -(-data.records // slots))

    cliques = [('a',), ('b',), ('c',), ('a', 'b'), ('c', 'a'), ('b', 'c')]
    marginals = encrypted_marginals.EncryptedMarginals.compute(
        backend, columns, cliques
    )
    for cl in cliques:
      expected = data.project(cl).datavector()
      actual = holder.decrypt(marginals[cl], data.domain.size(cl))
      np.testing.assert_allclose(actual, expected, err_msg=str(cl))

  def test_three_way_marginal_when_degree_allowed(self):
    data = _dataset()
    backend = backend_lib.PlaintextBackend(slots=32)
    holder = parties.KeyHolder(backend)
    columns = parties.DataOwner(backend, holder.public_key).encrypt_dataset(data)
    cl = ('a', 'b', 'c')
    with self.assertRaises(ValueError):
      encrypted_marginals.EncryptedMarginals.compute(backend, columns, [cl])
    marginals = encrypted_marginals.EncryptedMarginals.compute(
        backend, columns, [cl], max_degree=3
    )
    np.testing.assert_allclose(
        holder.decrypt(marginals[cl], 24), data.project(cl).datavector()
    )

  def test_rejects_marginal_larger_than_slots(self):
    data = _dataset()
    backend = backend_lib.PlaintextBackend(slots=8)
    holder = parties.KeyHolder(backend)
    columns = parties.DataOwner(backend, holder.public_key).encrypt_dataset(data)
    with self.assertRaises(ValueError):
      encrypted_marginals.EncryptedMarginals.compute(
          backend, columns, [('b', 'c')]
      )

  def test_duplicate_cliques_computed_once(self):
    data = _dataset()
    backend = backend_lib.PlaintextBackend(slots=64)
    holder = parties.KeyHolder(backend)
    columns = parties.DataOwner(backend, holder.public_key).encrypt_dataset(data)
    marginals = encrypted_marginals.EncryptedMarginals.compute(
        backend, columns, [('a',), ('a',), ('a', 'b')]
    )
    self.assertEqual(marginals.cliques, (('a',), ('a', 'b')))


class NoisePoolTest(absltest.TestCase):

  def test_plan_and_pool_consumption(self):
    domain = mbi.Domain(['a', 'b'], [2, 3])
    candidates = [('a',), ('b',), ('a', 'b')]
    plan = parties.NoisePlan.for_aim(
        domain, candidates, one_way_cliques=[('a',), ('b',)], max_rounds=2
    )
    # One extra vector per one-way clique, max_rounds per distinct size.
    self.assertEqual(plan.gaussian, {2: 3, 3: 3, 6: 2})
    self.assertEqual((plan.gumbel_length, plan.gumbel_count), (3, 2))

    backend = backend_lib.PlaintextBackend(slots=8)
    holder = parties.KeyHolder(backend)
    owner = parties.DataOwner(backend, holder.public_key)
    pool = owner.encrypt_noise(np.random.default_rng(0), plan)
    for _ in range(2):
      self.assertEqual(holder.decrypt(pool.next_gaussian(6), 6).shape, (6,))
    with self.assertRaises(RuntimeError):
      pool.next_gaussian(6)
    with self.assertRaises(ValueError):
      pool.next_gumbel(4)
    pool.next_gumbel(3)
    pool.next_gumbel(3)
    with self.assertRaises(RuntimeError):
      pool.next_gumbel(3)


if __name__ == '__main__':
  absltest.main()
