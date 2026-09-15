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

"""FHAIM: the AIM mechanism with selection and measurement under encryption.

See https://arxiv.org/abs/2602.05838. The AIM loop, budget schedule and
annealing are inherited from `dpsynth.discrete_mechanisms.aim.AIM`; only the
two hooks that touch sensitive data (`_select`, `_measure`) are replaced.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import dataclasses

from absl import logging
from dpsynth import api
from dpsynth.contrib.fhe import backend as backend_lib
from dpsynth.contrib.fhe import encrypted_marginals
from dpsynth.contrib.fhe import parties
from dpsynth.discrete_mechanisms import accounting
from dpsynth.discrete_mechanisms import aim
from dpsynth.discrete_mechanisms import common
import mbi
import numpy as np


BackendFactory = Callable[[], backend_lib.FHEBackend]

_BACKENDS: dict[str, BackendFactory] = {
    'plaintext': backend_lib.PlaintextBackend,
}


def register_backend(name: str, factory: BackendFactory) -> None:
  """Registers a backend factory under `name` for use in `FHAIMConfig`."""
  _BACKENDS[name] = factory


@dataclasses.dataclass(frozen=True)
class EncryptedSession:
  """Everything the computation entity holds during one FHAIM run.

  This object stands in for the dataset inside the inherited AIM loop, which
  only reads `.domain` from it; the overridden hooks read the rest.
  """

  backend: backend_lib.FHEBackend
  key_holder: parties.KeyHolder
  marginals: encrypted_marginals.EncryptedMarginals
  noise: parties.NoisePool
  num_records: int

  @property
  def domain(self) -> mbi.Domain:
    return self.marginals.domain


@dataclasses.dataclass(frozen=True)
class FHAIMConfig(aim.AIMConfig):
  """Configuration for FHAIM.

  Accepts every `AIMConfig` parameter. Selection uses the squared L2 error
  (rather than AIM's L1 error) because it needs one multiplicative level under
  CKKS instead of a polynomial approximation of |x|.

  FHAIM must be run directly on an `mbi.Dataset`. Do not wrap it in
  `DiscreteConfig`: that layer measures one-way marginals and precomputes
  workload marginals in plaintext before calling the inner mechanism. FHAIM
  measures its own one-way marginals under encryption using
  `one_way_budget_fraction` of its budget.

  Attributes:
    backend: Name of a registered backend. 'plaintext' is a NumPy simulation.
    one_way_budget_fraction: Fraction of zCDP budget spent on the initial
      one-way marginal measurements.
    num_records: Public upper bound on the number of records, used in the
      sensitivity of the squared L2 selection score. If None, the true record
      count is used, which treats the dataset size as public information.
    max_degree: Refuse workloads with marginals of more than this many
      attributes (each extra attribute costs one multiplicative level).
  """

  backend: str = 'plaintext'
  one_way_budget_fraction: float = 0.1
  num_records: int | None = None
  max_degree: int = 2

  def __post_init__(self):
    if not 0 <= self.one_way_budget_fraction < 1:
      raise ValueError('one_way_budget_fraction must be in [0, 1).')
    if self.backend not in _BACKENDS:
      raise ValueError(
          f'Unknown backend {self.backend!r}; known: {sorted(_BACKENDS)}.'
      )

  def configure(self, _=None, *, zcdp_rho, delta=0, max_records_per_user=1):
    api.validate_max_records_per_user(max_records_per_user)
    return FHAIM(
        config=self,
        zcdp_rho=zcdp_rho,
        max_records_per_user=max_records_per_user,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class FHAIM(aim.AIM):
  """Calibrated FHAIM instance."""

  config: FHAIMConfig

  def __call__(
      self,
      rng: np.random.Generator,
      data: mbi.Dataset | mbi.CliqueVector,
      *,
      initial_measurements: Sequence[mbi.LinearMeasurement] | None = None,
      constraints: Sequence[mbi.Constraint] = (),
  ) -> common.DiscreteMechanismResult:
    if not isinstance(data, mbi.Dataset):
      raise TypeError(
          'FHAIM encrypts the dataset itself and must be given an'
          f' mbi.Dataset, not {type(data).__name__}. Run it directly rather'
          ' than inside DiscreteConfig.'
      )

    cfg = self.config
    candidates = common.compiled_workload(
        data.domain, cfg.workload, cfg.max_marginal_size
    )
    measure_one_way = (
        not initial_measurements and cfg.one_way_budget_fraction > 0
    )
    one_way = [(a,) for a in data.domain] if measure_one_way else []
    max_rounds = cfg.max_rounds or 16 * len(data.domain)

    #########################################################################
    # Setup: parties, encrypted data, pre-encrypted noise, pi_COMP.         #
    #########################################################################
    backend = _BACKENDS[cfg.backend]()
    key_holder = parties.KeyHolder(backend)
    owner = parties.DataOwner(backend, key_holder.public_key)
    columns = owner.encrypt_dataset(data)
    plan = parties.NoisePlan.for_aim(
        data.domain, list(candidates), one_way, max_rounds
    )
    noise = owner.encrypt_noise(rng, plan)
    logging.info(
        '[FHAIM] Computing %d encrypted marginals with backend %s.',
        len(set(candidates) | set(one_way)),
        cfg.backend,
    )
    marginals = encrypted_marginals.EncryptedMarginals.compute(
        backend, columns, [*one_way, *candidates], max_degree=cfg.max_degree
    )
    session = EncryptedSession(
        backend=backend,
        key_holder=key_holder,
        marginals=marginals,
        noise=noise,
        num_records=cfg.num_records or columns.num_records,
    )

    #########################################################################
    # One-way marginals under encryption, then the inherited AIM loop.      #
    #########################################################################
    common.validate_initial_measurements(initial_measurements)
    measurements = list(initial_measurements) if initial_measurements else []
    loop_rho = self.zcdp_rho
    if one_way:
      one_way_rho = self.zcdp_rho * cfg.one_way_budget_fraction
      loop_rho -= one_way_rho
      sigma = accounting.zcdp_gaussian_sigma(one_way_rho / len(one_way))
      measurements = [self._measure(rng, session, cl, sigma) for cl in one_way]

    loop = dataclasses.replace(self, zcdp_rho=loop_rho)
    return aim.AIM.__call__(
        loop,
        rng,
        session,  # pyrefly: ignore[bad-argument-type]
        initial_measurements=measurements,
        constraints=constraints,
    )

  def _measure(
      self,
      rng: np.random.Generator,
      data: EncryptedSession,  # pyrefly: ignore[bad-override]
      clique: mbi.Clique,
      sigma: float,
  ) -> mbi.LinearMeasurement:
    """pi_MEASURE: adds scaled pre-encrypted Gaussian noise, then decrypts."""
    del rng  # Noise was pre-sampled by the data owner.
    backend = data.backend
    n = data.domain.size(clique)
    stddev = self.max_records_per_user * sigma
    noisy = backend.add(
        data.marginals[clique],
        backend.scale(data.noise.next_gaussian(n), stddev),
    )
    y = data.key_holder.decrypt(noisy, n)
    return mbi.LinearMeasurement(y, clique, stddev)

  def _select(
      self,
      rng: np.random.Generator,
      candidates: Mapping[mbi.Clique, float],
      data: EncryptedSession,  # pyrefly: ignore[bad-override]
      estimates: mbi.CliqueVector,
      epsilon: float,
      sigma: float,
  ) -> mbi.Clique:
    """pi_SELECT: encrypted squared-L2 scores + Gumbel noise, argmax in clear.

    The score of clique c is w_c * (||x_c - est_c||^2 - n_c * stddev^2), the
    bias term being the expected squared norm of the measurement noise. Under
    add/remove neighbouring datasets, removing one record changes
    ||x - est||^2 by at most 2N + 1 when 0 <= est <= N, so the estimates are
    clipped to [0, N] before encoding (a post-processing of released
    statistics) and the sensitivity is w_max * m * (2N + 1) for m records per
    user. Adding Gumbel(2 * sensitivity / epsilon) noise and taking the argmax
    is the epsilon-DP exponential mechanism, matching
    `common.exponential_mechanism` with monotonic=False.
    """
    del rng  # Noise was pre-sampled by the data owner.
    backend = data.backend
    m = self.max_records_per_user
    stddev = m * sigma
    big_n = data.num_records
    cliques = list(candidates)
    scores = []
    for cl in cliques:
      n = data.domain.size(cl)
      est = np.clip(np.asarray(estimates[cl].datavector()), 0.0, big_n)
      diff = backend.sub(data.marginals[cl], backend.encode(est))
      sq_err = backend.inner_product(diff, diff, n)
      centered = backend.add(sq_err, backend.encode([-n * stddev**2]))
      scores.append(backend.scale(centered, candidates[cl]))
    sensitivity = m * max(candidates.values()) * (2 * big_n + 1)
    noisy = backend.add(
        backend.pack(scores),
        backend.scale(
            data.noise.next_gumbel(len(cliques)), 2 * sensitivity / epsilon
        ),
    )
    noisy_scores = data.key_holder.decrypt(noisy, len(cliques))
    return cliques[int(np.argmax(noisy_scores))]
