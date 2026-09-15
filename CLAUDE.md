# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

DPSynth: differentially private (DP) synthetic tabular data generation. A fork of
`google/dpsynth`. Python 3.12/3.13, Google style (2-space indent, absl, Apache
license headers on every file). Heavy lifting is done by the external `mbi`
library (Private-PGM / Markov Random Field inference) and `dp_accounting`.

## Commands

Environment is managed with `uv` and a PEP 751 lockfile (`pylock.toml`):

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r pylock.toml
uv pip install -e .            # CI uses --no-deps -e .
```

Any change to `pyproject.toml` must be accompanied by regenerating the lockfile,
or CI fails:

```bash
uv export --format pylock.toml --all-extras --default-index https://pypi.org/simple -o pylock.toml --python 3.12
go run github.com/google/addlicense@v1.2.0 -c "Google LLC" -y "2026" -l apache pylock.toml
```

Tests (pytest with `--import-mode=importlib` and `pythonpath=["."]` set in
`pyproject.toml`; tests are `absltest.TestCase` classes under `tests/`, which
mirrors the `dpsynth/` layout):

```bash
pytest -n auto                                   # full suite, as in CI
pytest tests/discrete_mechanisms/aim_test.py     # one file
pytest tests/discrete_mechanisms/aim_test.py -k test_name   # one test
```

`conftest.py` pre-parses absl flags so `create_tempfile()` works under pytest.

Type checking (runs in CI, preset "basic"):

```bash
pyrefly check
```

CLI entry points live in `bin/` (see `bin/README.md`), e.g.
`python3 bin/main.py --dataset=... --domain=domain.yaml --epsilon=1.0 --delta=1e-8 --mechanism=mst --output_path=...`.

## Architecture

### Three independent code paths

1. **In-memory mode** (the main one): `dpsynth.TabularConfig` in
   `dpsynth/data_generation_v3.py`, operating on Pandas/NumPy/JAX. Uses
   `discrete_mechanisms/` for the DP algorithms and `local_mode/` for DP
   quantiles and partition selection. CLI: `bin/main.py`.
2. **Scalable pipeline mode**: `dpsynth.data_generation.generate()` on top of
   `pipeline_dp.PipelineBackend` (Apache Beam or `LocalBackend`). Uses
   `dataset_descriptors/` (schema + format converters for CSV/TFRecord) and
   `pipeline_transformations/`. Requires the `pipeline` extra. CLI:
   `bin/run_data_generation.py`.
3. **Post-processing mode**: `dpsynth/postprocessing.py` fits a model from
   externally computed noisy marginals, no measurement step.

The two main paths were developed independently; budget splitting and feature
availability differ slightly. Shared by both: `domain.py` (public attribute
types), `constraints.py`, `transformations.py`.

### Mechanism API (`dpsynth/api.py`)

Every DP algorithm is split into two frozen dataclasses:

- `MechanismConfig`: serializable hyperparameter recipe (YAML via
  `dpsynth.to_yaml` / `from_yaml`). Subclasses must implement
  `configure(domain, *, zcdp_rho, delta)`, which maps a zCDP rho budget to
  concrete noise parameters. Subclasses auto-register by name via
  `__init_subclass__` (`MechanismConfig.get_subclass`).
- `CalibratedMechanism`: runnable instance with concrete parameters. Exposes
  `dp_event` (exact `dp_accounting.DpEvent`) and `__call__(rng, data)`.

`MechanismConfig.calibrate(domain, *, epsilon, delta)` is the user-facing entry:
it numerically searches for the largest rho whose `dp_event` satisfies
(epsilon, delta), taking the max over PLD and RDP accountants. Composition is
hierarchical: `TabularConfig` wraps a `DiscreteConfig`
(`discrete_mechanisms/discrete.py`), which wraps the per-mechanism configs, and
each layer subdivides its rho among children (e.g. `init_budget_fraction`,
`select_budget_fraction`). `MultiTableConfig` in `relational/` wraps
`TabularConfig` again. Full design rationale: `docs/mechanism_api.md`.

### Discrete mechanisms (`dpsynth/discrete_mechanisms/`)

All operate on integer-coded data (`mbi.Dataset`) and must contain no
distributed-framework references. `discrete.py` owns the shared
select/measure/estimate lifecycle (the `README.md` there still calls it
`base.py`); `common.py` holds shared utilities (noisy marginals, domain
compression, `DiscreteMechanismResult`). Mechanisms: `independent`, `direct`
(caller-specified workload), `mst`, `aim`, `swift` (with `clique_tree.py`,
`swift_utils.py`). Keep selection policy in the mechanism file, generic logic in
`common.py`. `accounting.py` has legacy zCDP/GDP conversions slated to move to
`dp_accounting`.

### Processing lifecycle (both paths)

Initialization (DP quantiles for numericals, DP partition selection for
open-set categoricals) -> integer encoding -> domain compression (rare values
into an "Other" bucket) -> Private-PGM inference -> sampling, uncompression,
decoding. See `docs/processing_lifecycle.md`.

### Pipeline code rules (`pipeline_transformations/`, `eval/`)

Functions take a `backend: PipelineBackend` and must use only its methods
(`map`, `filter`, `group_by_key`, ...) with a descriptive stage name. Never use
`beam.Map`, the `|` pipe, or list/PCollection-specific methods; the same
function must run under both `LocalBackend` and `BeamBackend`. DP aggregations
go through `pipeline_dp.DPEngine`.

### Other packages

- `relational/`: experimental multi-table synthesis (cascading down foreign
  keys, PrivPetal-style permutation modeling). APIs unstable.
- `text/`: DP text generation / fine-tuning (Flax, Gemma, jax_privacy); needs
  the `text` extra.
- `adapters/`: Pydantic, protobuf, and Beam adapters over the core API.
- `experimental/`, `contrib/`: staging areas. New standalone mechanisms go in
  `contrib/` first.
- `eval/`: tabular evaluation engine (TV distance, Cramer's V), CLI
  `bin/run_tabular_eval.py`.

## Contribution constraints (from CONTRIBUTING.md)

- No PyTorch, TensorFlow, or other heavy dependencies. Use JAX for numerics and
  Flax for neural layers. Discuss any new dependency first.
- New mechanisms: aim for a single file of about 500 lines or fewer, conforming
  to the `MechanismConfig` / `CalibratedMechanism` contract.
- APIs are not yet backwards-compatibility stable; `CHANGELOG.md` is maintained.
