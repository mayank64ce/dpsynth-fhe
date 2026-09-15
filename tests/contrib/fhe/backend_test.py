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
from dpsynth.contrib.fhe import backend as backend_lib
import numpy as np


class PlaintextBackendTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.backend = backend_lib.PlaintextBackend(slots=8)
    self.keys = self.backend.keygen()

  def test_encrypt_decrypt_roundtrip_pads_to_slots(self):
    ct = self.backend.encrypt(np.array([1.0, 2.0, 3.0]), self.keys.public)
    self.assertEqual(ct.shape, (8,))
    np.testing.assert_array_equal(
        self.backend.decrypt(ct, self.keys.secret, 3), [1.0, 2.0, 3.0]
    )

  def test_encrypt_rejects_oversized_vector(self):
    with self.assertRaises(ValueError):
      self.backend.encrypt(np.ones(9), self.keys.public)

  def test_decrypt_requires_secret_key(self):
    ct = self.backend.encrypt(np.ones(2), self.keys.public)
    with self.assertRaises(ValueError):
      self.backend.decrypt(ct, self.keys.public, 2)

  def test_encrypt_requires_public_key(self):
    with self.assertRaises(ValueError):
      self.backend.encrypt(np.ones(2), self.keys.secret)

  def test_arithmetic(self):
    a = self.backend.encrypt(np.array([1.0, 2.0, 3.0]), self.keys.public)
    b = self.backend.encode(np.array([10.0, 20.0, 30.0]))
    dec = lambda ct: self.backend.decrypt(ct, self.keys.secret, 3)
    np.testing.assert_array_equal(dec(self.backend.add(a, b)), [11, 22, 33])
    np.testing.assert_array_equal(dec(self.backend.sub(a, b)), [-9, -18, -27])
    np.testing.assert_array_equal(dec(self.backend.mul(a, b)), [10, 40, 90])
    np.testing.assert_array_equal(dec(self.backend.scale(a, 2.0)), [2, 4, 6])

  def test_sum_slots_and_inner_product_only_define_slot_zero(self):
    a = self.backend.encrypt(np.array([1.0, 2.0, 3.0, 100.0]), self.keys.public)
    total = self.backend.sum_slots(a, 3)
    self.assertEqual(self.backend.decrypt(total, self.keys.secret, 1)[0], 6.0)
    self.assertTrue(np.isnan(total[1]))
    ip = self.backend.inner_product(a, a, 3)
    self.assertEqual(self.backend.decrypt(ip, self.keys.secret, 1)[0], 14.0)
    self.assertTrue(np.isnan(ip[1]))

  def test_pack_takes_slot_zero_of_each_input(self):
    scalars = [
        self.backend.sum_slots(
            self.backend.encrypt(np.array([float(i)]), self.keys.public), 1
        )
        for i in range(3)
    ]
    packed = self.backend.pack(scalars)
    np.testing.assert_array_equal(
        self.backend.decrypt(packed, self.keys.secret, 4), [0, 1, 2, 0]
    )
    with self.assertRaises(ValueError):
      self.backend.pack(scalars * 3)


if __name__ == '__main__':
  absltest.main()
