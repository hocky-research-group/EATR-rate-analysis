# Plan: Genuine Parallelism for `eatr-analysis`

## Problem Summary

`eatr-analysis --threads N --bootstrap` currently does nothing useful:

- **Bootstrap path**: `threaded_bootstrap()` uses `ThreadPoolExecutor`, but every worker
  calls `scipy.optimize.minimize_scalar` or `curve_fit` with Python-level callbacks.
  The GIL is held throughout — threads serialize, CPU stays at ~100% regardless of `--threads`.
- **Non-bootstrap path**: `cores = args.threads` is computed and passed through to every
  `RM.*` function, but `_map_with_cores` (defined in `rate_methods_library.py`) is never
  actually called. The `cores` parameter is accepted and silently ignored everywhere.

The fix is `ProcessPoolExecutor` for bootstrap. Processes bypass the GIL.
The constraint is that `ProcessPoolExecutor` (which uses `spawn` on macOS) requires
picklable callables — the current code passes lambdas everywhere.

`eatr-flooding-analysis` is **not affected**: its bootstrap workers do large numpy array
operations (`_scan_barrier`, `_prepare_flat`) that release the GIL, so `ThreadPoolExecutor`
already provides real parallelism there. Leave it unchanged.

---

## Phase 1 — Foundation: create `_bootstrap_workers.py`

**New file:** `eatr_rates/_bootstrap_workers.py`

Contains one frozen dataclass per method (holding the parameters that would have been
captured by the lambda) plus a single top-level dispatch function `_run_worker(task)`.
Using one dispatch function keeps the pickling surface minimal — only one name needs
to be importable from a fresh worker process.

```python
# eatr_rates/_bootstrap_workers.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import rate_methods_library as RM


@dataclass(frozen=True)
class ObsLogRateConfig:
    pass

@dataclass(frozen=True)
class IMetaDMLEConfig:
    beta: float
    bias_shift: float

@dataclass(frozen=True)
class IMetaDCDFConfig:
    beta: float
    bias_shift: float
    k_bounds: tuple
    k_guess: float | None

@dataclass(frozen=True)
class KTRMLEConfig:
    beta: float
    gamma_bounds: tuple
    log_trick: bool
    do_bopt: bool
    bias_shift: float

@dataclass(frozen=True)
class KTRCDFConfig:
    beta: float
    k_bounds: tuple
    gamma_bounds: tuple
    log_trick: bool
    init_guess: tuple
    do_bopt: bool
    bias_shift: float

@dataclass(frozen=True)
class EATRMLEConfig:
    beta: float
    gamma_bounds: tuple
    log_trick: bool
    do_bopt: bool
    bias_shift: float

@dataclass(frozen=True)
class EATRCDFConfig:
    beta: float
    k_bounds: tuple
    gamma_bounds: tuple
    log_trick: bool
    init_guess: tuple
    do_bopt: bool
    bias_shift: float


def _run_worker(task):
    """Picklable dispatch entry point for ProcessPoolExecutor.

    task: (indices, sample, event, config)
      indices : np.ndarray of int resample indices
      sample  : list[np.ndarray]
      event   : np.ndarray | None
      config  : one of the *Config dataclasses above
    """
    indices, sample, event, config = task
    resample = [sample[int(i)] for i in indices]
    eve = None if event is None else np.array([event[int(i)] for i in indices])

    if isinstance(config, ObsLogRateConfig):
        total_time = sum(float(traj[-1, 0]) for traj in resample)
        n_events = float(eve.sum()) if eve is not None else float(len(resample))
        return float(n_events / total_time)

    if isinstance(config, IMetaDMLEConfig):
        return RM.iMetaD_invMRT(resample, config.beta, event=eve,
                                 bias_shift=config.bias_shift)

    if isinstance(config, IMetaDCDFConfig):
        return RM.iMetaD_FitCDF(resample, config.beta, event=eve,
                                 bias_shift=config.bias_shift,
                                 k_bounds=config.k_bounds, k_guess=config.k_guess)

    if isinstance(config, KTRMLEConfig):
        return RM.KTR_MLE_rate(resample, config.beta, event=eve,
                                gamma_bounds=config.gamma_bounds,
                                logTrick=config.log_trick,
                                do_bopt=config.do_bopt,
                                bias_shift=config.bias_shift)

    if isinstance(config, KTRCDFConfig):
        return RM.KTR_CDF_rate(resample, config.beta, event=eve,
                                k_bounds=config.k_bounds,
                                gamma_bounds=config.gamma_bounds,
                                logTrick=config.log_trick,
                                init_guess=list(config.init_guess),
                                do_bopt=config.do_bopt,
                                bias_shift=config.bias_shift)

    if isinstance(config, EATRMLEConfig):
        return RM.EATR_MLE_rate(resample, config.beta, event=eve,
                                  gamma_bounds=config.gamma_bounds,
                                  logTrick=config.log_trick,
                                  do_bopt=config.do_bopt,
                                  bias_shift=config.bias_shift)

    if isinstance(config, EATRCDFConfig):
        return RM.EATR_CDF_rate(resample, config.beta, event=eve,
                                  k_bounds=config.k_bounds,
                                  gamma_bounds=config.gamma_bounds,
                                  logTrick=config.log_trick,
                                  init_guess=list(config.init_guess),
                                  do_bopt=config.do_bopt,
                                  bias_shift=config.bias_shift)

    raise TypeError(f"Unknown config type: {type(config)}")
```

**Spawn-safety checklist:**
- No top-level `mp.Pool` or `ThreadPoolExecutor` construction anywhere in this file.
- `rate_methods_library` is already guarded: `bayes_opt` import in try/except,
  `warnings.filterwarnings` call is benign.
- Bootstrap index arrays are generated in the main process and passed as plain
  `np.ndarray` to workers → reproducibility is preserved across process boundary.

---

## Phase 2 — Rewrite `threaded_bootstrap` and call sites in `rates_cmd.py`

### 2a. Imports to add at top of `rates_cmd.py`

```python
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from eatr_rates._bootstrap_workers import (
    _run_worker,
    ObsLogRateConfig, IMetaDMLEConfig, IMetaDCDFConfig,
    KTRMLEConfig, KTRCDFConfig, EATRMLEConfig, EATRCDFConfig,
)
```

Remove the existing `from concurrent.futures import ThreadPoolExecutor` line.

### 2b. Rewrite `threaded_bootstrap` (currently lines 208–228)

Keep the external signature but replace the body:

```python
def threaded_bootstrap(
    sample: list[np.ndarray],
    config,               # one of the *Config dataclasses
    nresamples: int,
    *,
    event: np.ndarray | None = None,
    double: bool = False,
    seed: int | None = None,
    threads: int = 1,
):
    rng = np.random.default_rng(seed=seed)
    sample_size = len(sample)
    index_sets = rng.integers(0, sample_size, size=(nresamples, sample_size))
    tasks = [(indices, sample, event, config) for indices in index_sets]

    if threads <= 1:
        results = [_run_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=threads) as executor:
            results = list(executor.map(_run_worker, tasks))

    return np.array(results, dtype=float)
```

`double=False` retained for API compatibility; shape of result array is naturally
determined by what the worker returns (scalar or 2-element array).

**Data serialization note:** the full `sample` list (all numpy arrays) is serialized
into each worker task. For typical datasets (tens–hundreds of trajectories, ~thousands
of rows each) this is fast. If it becomes a bottleneck, `multiprocessing.shared_memory`
can be added later without changing the external API.

### 2c. Remove `thread_map` from `rates_cmd.py`

`thread_map` (lines 55–59) is no longer called after this change. Remove it.
`rates_eatr_opes.py` defines its own copy — unaffected.

### 2d. Remove dead `cores` variables (lines 308–309)

```python
cores = 1 if use_threaded_bootstrap else args.threads   # DELETE
bootstrap_cores = 1                                      # DELETE
```

Remove all `cores=cores` and `cores=bootstrap_cores` keyword arguments from the
non-bootstrap and bootstrap method calls respectively.

### 2e. Replace every lambda call site in `analyze()`

All `use_threaded_bootstrap` branches (7 call sites) replace
`lambda subset, eve: RM.SomeMethod(...)` with a config object:

| Approx line | Method | Config to construct |
|-------------|--------|---------------------|
| ~350 | `observed_log_rate` bootstrap | `ObsLogRateConfig()` |
| ~381 | `iMetaD_invMRT` | `IMetaDMLEConfig(beta=beta, bias_shift=args.barrier)` |
| ~414 | `iMetaD_FitCDF` | `IMetaDCDFConfig(beta, args.barrier, k_bounds, init_guess[0])` |
| ~456 | `KTR_MLE_rate` | `KTRMLEConfig(beta, gamma_bounds, args.logtrick, args.bayesopt, args.barrier)` |
| ~494 | `KTR_CDF_rate` | `KTRCDFConfig(beta, k_bounds, gamma_bounds, args.logtrick, tuple(init_guess), args.bayesopt, args.barrier)` |
| ~532 | `EATR_MLE_rate` | `EATRMLEConfig(beta, gamma_bounds, args.logtrick, args.bayesopt, args.barrier)` |
| ~572 | `EATR_CDF_rate` | `EATRCDFConfig(beta, k_bounds, gamma_bounds, args.logtrick, tuple(init_guess), args.bayesopt, args.barrier)` |

Example (EATR MLE):
```python
# Before
sample = threaded_bootstrap(
    data,
    lambda subset, eve: RM.EATR_MLE_rate(subset, beta, event=eve,
        gamma_bounds=gamma_bounds, cores=bootstrap_cores,
        logTrick=args.logtrick, do_bopt=args.bayesopt, bias_shift=args.barrier),
    args.numboots, event=event, double=True, seed=seed, threads=args.threads)

# After
config = EATRMLEConfig(
    beta=beta, gamma_bounds=gamma_bounds,
    log_trick=args.logtrick, do_bopt=args.bayesopt, bias_shift=args.barrier)
sample = threaded_bootstrap(
    data, config, args.numboots,
    event=event, double=True, seed=seed, threads=args.threads)
```

### 2f. Update `--threads` help string (line ~111)

```python
# Before
"number of parallel workers: thread count for bootstrap resamples, or "
"multiprocessing core count when not bootstrapping (DEFAULT: 1)"

# After
"number of parallel worker processes for bootstrap resampling (DEFAULT: 1)"
```

---

## Phase 3 — Tests

### 3a. New file `tests/test_bootstrap_workers.py`

```python
import pickle
import numpy as np
import unittest
from eatr_rates._bootstrap_workers import (
    _run_worker,
    ObsLogRateConfig, IMetaDMLEConfig, IMetaDCDFConfig,
    KTRMLEConfig, KTRCDFConfig, EATRMLEConfig, EATRCDFConfig,
)

class BootstrapWorkerTests(unittest.TestCase):
    def test_all_configs_picklable(self):
        configs = [
            ObsLogRateConfig(),
            IMetaDMLEConfig(beta=1.0, bias_shift=0.0),
            IMetaDCDFConfig(beta=1.0, bias_shift=0.0, k_bounds=(0.0, float("inf")), k_guess=None),
            KTRMLEConfig(beta=1.0, gamma_bounds=(0.0, 1.0), log_trick=False, do_bopt=False, bias_shift=0.0),
            KTRCDFConfig(beta=1.0, k_bounds=(0.0, float("inf")), gamma_bounds=(0.0, 1.0),
                         log_trick=False, init_guess=(1.0, 0.9), do_bopt=False, bias_shift=0.0),
            EATRMLEConfig(beta=1.0, gamma_bounds=(0.0, 1.0), log_trick=False, do_bopt=False, bias_shift=0.0),
            EATRCDFConfig(beta=1.0, k_bounds=(0.0, float("inf")), gamma_bounds=(0.0, 1.0),
                          log_trick=False, init_guess=(1.0, 0.9), do_bopt=False, bias_shift=0.0),
        ]
        for cfg in configs:
            assert pickle.loads(pickle.dumps(cfg)) == cfg

    def test_run_worker_eatr_mle_returns_finite(self):
        traj = np.column_stack([np.arange(4, dtype=float), np.zeros(4), np.linspace(0.1, 0.4, 4)])
        sample = [traj, traj * 1.1, traj * 0.9]
        event = np.array([True, True, False])
        config = EATRMLEConfig(beta=1.0, gamma_bounds=(0.0, 1.0),
                               log_trick=False, do_bopt=False, bias_shift=0.0)
        result = _run_worker((np.array([0, 1, 2]), sample, event, config))
        assert len(result) == 2
        assert all(np.isfinite(result))
```

### 3b. Existing test coverage

`test_rates_cli_bootstrap_threads_writes_ci_output` (test_cli.py line 77) already exercises
`--threads 2 --bootstrap`. It will continue to be the key regression test after the switch
to `ProcessPoolExecutor`.

---

## Phase 4 — Ordering (keep tests green throughout)

1. Create `eatr_rates/_bootstrap_workers.py` — no behavior change, all tests pass.
2. Add `tests/test_bootstrap_workers.py` with picklability + correctness tests.
3. Run full test suite — must be green.
4. Rewrite `threaded_bootstrap` in `rates_cmd.py` to use `ProcessPoolExecutor`.
5. Replace all 7 lambda call sites in `analyze()` with config objects.
6. Remove `cores` / `bootstrap_cores` variables and all dead `cores=` kwargs.
7. Update `--threads` help string.
8. Run full test suite — must be green.
9. Commit.

---

## Out of scope / future work

- `eatr-flooding-analysis`: leave on `ThreadPoolExecutor`; workers are numpy-dominated
  and the GIL-holding scipy call (`minimize_scalar` for gamma fit) is cheap.
- Option B (parallelize multiple methods concurrently in non-bootstrap mode): deferred.
- Shared-memory optimization for large datasets: deferred.
- Removing `cores` parameter from `rate_methods_library.py` function signatures:
  deferred (separate refactor, potential external callers).
