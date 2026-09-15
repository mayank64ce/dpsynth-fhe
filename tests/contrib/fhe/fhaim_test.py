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

import dataclasses

from absl.testing import absltest
from dpsynth import api
from dpsynth.contrib.fhe import backend as backend_lib
from dpsynth.contrib.fhe import fhaim
from dpsynth.contrib.fhe import parties
from dpsynth.discrete_mechanisms import accounting
from dpsynth.discrete_mechanisms import aim
from dpsynth.discrete_mechanisms import common
from dpsynth.discrete_mechanisms import discrete
import dp_accounting
import mbi
import numpy as np


def _correlated_dataset(rng, n=500):
  domain = mbi.Domain(['a', 'b', 'c'], [3, 3, 3])
  a = rng.integers(0, 3, size=n)
  b = np.where(rng.random(n) < 0.75, a, rng.integers(0, 3, size=n))
  c = (a + b + rng.integers(0, 2, size=n)) % 3
  return mbi.Dataset({'a': a, 'b': b, 'c': c}, domain)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _ReferenceL2AIM(aim.AIM):
  """Plaintext AIM with squared-L2 selection, fed an explicit noise pool.

  Mirrors what FHAIM computes, written directly against the dataset with
  NumPy, so that FHAIM on `PlaintextBackend` can be checked round by round.
  """

  noise: parties.NoisePool
  num_records: int

  def _measure(self, rng, data, clique, sigma):
    del rng
    n = data.domain.size(clique)
    stddev = self.max_records_per_user * sigma
    x = data.project(clique).datavector()
    y = x + stddev * self.noise.next_gaussian(n)[:n]
    return mbi.LinearMeasurement(y, clique, stddev)

  def _select(self, rng, candidates, data, estimates, epsilon, sigma):
    del rng
    m = self.max_records_per_user
    stddev = m * sigma
    cliques = list(candidates)
    scores = []
    for cl in cliques:
      n = data.domain.size(cl)
      x = data.project(cl).datavector()
      est = np.clip(np.asarray(estimates[cl].datavector()), 0, self.num_records)
      scores.append(candidates[cl] * (np.sum((x - est) ** 2) - n * stddev**2))
    sensitivity = m * max(candidates.values()) * (2 * self.num_records + 1)
    gumbel = self.noise.next_gumbel(len(cliques))[: len(cliques)]
    noisy = np.array(scores) + 2 * sensitivity / epsilon * gumbel
    return cliques[int(np.argmax(noisy))]


class FHAIMTest(absltest.TestCase):

  def test_matches_plaintext_reference_round_by_round(self):
    data = _correlated_dataset(np.random.default_rng(1))
    workload = [('a', 'b'), ('b', 'c'), ('a', 'c')]
    rho = 2.0
    config = fhaim.FHAIMConfig(workload=workload, max_rounds=6, pgm_iters=300)
    result = config.configure(zcdp_rho=rho)(np.random.default_rng(0), data)

    # Rebuild the exact noise FHAIM drew: same plan, same seed, same order.
    candidates = common.compiled_workload(data.domain, workload)
    one_way = [(a,) for a in data.domain]
    plan = parties.NoisePlan.for_aim(data.domain, list(candidates), one_way, 6)
    backend = backend_lib.PlaintextBackend()
    holder = parties.KeyHolder(backend)
    pool = parties.DataOwner(backend, holder.public_key).encrypt_noise(
        np.random.default_rng(0), plan
    )
    one_way_rho = rho * config.one_way_budget_fraction
    reference = _ReferenceL2AIM(
        config=aim.AIMConfig(workload=workload, max_rounds=6, pgm_iters=300),
        zcdp_rho=rho - one_way_rho,
        noise=pool,
        num_records=data.records,
    )
    sigma = accounting.zcdp_gaussian_sigma(one_way_rho / len(one_way))
    initial = [reference._measure(None, data, cl, sigma) for cl in one_way]
    expected = reference(
        np.random.default_rng(0), data, initial_measurements=initial
    )

    self.assertLen(result.measurements, len(expected.measurements))
    for got, want in zip(result.measurements, expected.measurements):
      self.assertEqual(got.clique, want.clique)
      self.assertAlmostEqual(got.stddev, want.stddev)
      np.testing.assert_allclose(got.noisy_measurement, want.noisy_measurement)

  def test_fits_marginals_with_large_budget(self):
    data = _correlated_dataset(np.random.default_rng(2))
    config = fhaim.FHAIMConfig(
        workload=[('a', 'b'), ('b', 'c')], max_rounds=4, pgm_iters=500
    )
    result = config.configure(zcdp_rho=10000)(np.random.default_rng(0), data)
    self.assertIsInstance(result, common.DiscreteMechanismResult)
    for cl in [('a',), ('b',), ('c',), ('a', 'b'), ('b', 'c')]:
      np.testing.assert_allclose(
          result.model.project(cl).datavector(),
          data.project(cl).datavector(),
          atol=1,
          err_msg=str(cl),
      )

  def test_calibrate_and_dp_event(self):
    config = fhaim.FHAIMConfig(workload=[('a', 'b')], max_rounds=2)
    mechanism = config.configure(zcdp_rho=1.5)
    self.assertEqual(mechanism.dp_event, dp_accounting.ZCDpEvent(1.5))
    calibrated = config.calibrate(epsilon=1.0, delta=1e-6)
    self.assertIsInstance(calibrated, fhaim.FHAIM)
    self.assertGreater(calibrated.zcdp_rho, 0)

  def test_rejects_precomputed_marginals_from_discrete_config(self):
    data = _correlated_dataset(np.random.default_rng(3))
    wrapped = discrete.DiscreteConfig(
        mechanism=fhaim.FHAIMConfig(workload=[('a', 'b')], max_rounds=2)
    )
    with self.assertRaisesRegex(TypeError, 'mbi.Dataset'):
      wrapped.configure(zcdp_rho=1.0)(np.random.default_rng(0), data)

  def test_rejects_unknown_backend_and_bad_fraction(self):
    with self.assertRaises(ValueError):
      fhaim.FHAIMConfig(backend='nope')
    with self.assertRaises(ValueError):
      fhaim.FHAIMConfig(one_way_budget_fraction=1.0)

  def test_rejects_three_way_workload_by_default(self):
    data = _correlated_dataset(np.random.default_rng(4))
    config = fhaim.FHAIMConfig(workload=[('a', 'b', 'c')], max_rounds=2)
    with self.assertRaisesRegex(ValueError, 'max_degree'):
      config.configure(zcdp_rho=1.0)(np.random.default_rng(0), data)

  def test_is_a_mechanism_config(self):
    self.assertTrue(issubclass(fhaim.FHAIMConfig, api.MechanismConfig))
    self.assertIs(api.MechanismConfig.get_subclass('FHAIMConfig'), fhaim.FHAIMConfig)


if __name__ == '__main__':
  absltest.main()
