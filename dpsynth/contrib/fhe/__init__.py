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

"""Fully homomorphic AIM (FHAIM): AIM over encrypted tabular data.

Implements the protocols from "FHAIM: Fully Homomorphic AIM For Private
Synthetic Data Generation" (https://arxiv.org/abs/2602.05838) on top of the
core AIM loop in `dpsynth.discrete_mechanisms.aim`.

The sensitive dataset is one-hot encoded and encrypted column-wise by a
`DataOwner`. All candidate marginals are computed once under encryption
(`EncryptedMarginals`). Each AIM round then scores candidates under encryption,
adds pre-encrypted Gumbel noise, decrypts the noisy score vector and takes the
argmax in the clear; the selected marginal is released after pre-encrypted
Gaussian noise is added under encryption. Model fitting and sampling operate on
released (already privatized) statistics and are inherited from AIM unchanged.

Scheme-specific arithmetic lives behind `FHEBackend`. `PlaintextBackend` is a
NumPy simulation with identical semantics used for testing and for validating
protocol changes without an FHE library installed.
"""

from dpsynth.contrib.fhe.backend import FHEBackend
from dpsynth.contrib.fhe.backend import KeyPair
from dpsynth.contrib.fhe.backend import PlaintextBackend
from dpsynth.contrib.fhe.encrypted_marginals import EncryptedMarginals
from dpsynth.contrib.fhe.fhaim import FHAIM
from dpsynth.contrib.fhe.fhaim import FHAIMConfig
from dpsynth.contrib.fhe.parties import DataOwner
from dpsynth.contrib.fhe.parties import EncryptedColumns
from dpsynth.contrib.fhe.parties import KeyHolder
from dpsynth.contrib.fhe.parties import NoisePlan
from dpsynth.contrib.fhe.parties import NoisePool

__all__ = [
    'DataOwner',
    'EncryptedColumns',
    'EncryptedMarginals',
    'FHAIM',
    'FHAIMConfig',
    'FHEBackend',
    'KeyHolder',
    'KeyPair',
    'NoisePlan',
    'NoisePool',
    'PlaintextBackend',
]
