from __future__ import annotations

import pickle
import unittest
from importlib.util import find_spec

import numpy as np


def _make_traj(scale: float = 1.0) -> np.ndarray:
    """3-column COLVAR: time, zero, bias."""
    return np.column_stack([
        np.arange(4, dtype=float),
        np.zeros(4),
        np.linspace(0.1, 0.4, 4) * scale,
    ])


@unittest.skipUnless(find_spec("numpy") and find_spec("scipy"), "numpy and scipy required")
class BootstrapWorkerTests(unittest.TestCase):
    def setUp(self):
        from eatr_rates._bootstrap_workers import (
            ObsLogRateConfig,
            IMetaDMLEConfig,
            IMetaDCDFConfig,
            KTRMLEConfig,
            KTRCDFConfig,
            EATRMLEConfig,
            EATRCDFConfig,
            _run_worker,
        )
        self.ObsLogRateConfig = ObsLogRateConfig
        self.IMetaDMLEConfig = IMetaDMLEConfig
        self.IMetaDCDFConfig = IMetaDCDFConfig
        self.KTRMLEConfig = KTRMLEConfig
        self.KTRCDFConfig = KTRCDFConfig
        self.EATRMLEConfig = EATRMLEConfig
        self.EATRCDFConfig = EATRCDFConfig
        self._run_worker = _run_worker

    def _all_configs(self):
        return [
            self.ObsLogRateConfig(),
            self.IMetaDMLEConfig(beta=1.0, bias_shift=0.0),
            self.IMetaDCDFConfig(beta=1.0, bias_shift=0.0,
                                  k_bounds=(0.0, float("inf")), k_guess=None),
            self.KTRMLEConfig(beta=1.0, gamma_bounds=(0.0, 1.0),
                               log_trick=False, do_bopt=False, bias_shift=0.0),
            self.KTRCDFConfig(beta=1.0, k_bounds=(0.0, float("inf")),
                               gamma_bounds=(0.0, 1.0), log_trick=False,
                               init_guess=(1.0, 0.9), do_bopt=False, bias_shift=0.0),
            self.EATRMLEConfig(beta=1.0, gamma_bounds=(0.0, 1.0),
                                log_trick=False, do_bopt=False, bias_shift=0.0),
            self.EATRCDFConfig(beta=1.0, k_bounds=(0.0, float("inf")),
                                gamma_bounds=(0.0, 1.0), log_trick=False,
                                init_guess=(1.0, 0.9), do_bopt=False, bias_shift=0.0),
        ]

    def test_all_configs_are_picklable(self):
        for cfg in self._all_configs():
            self.assertEqual(pickle.loads(pickle.dumps(cfg)), cfg)

    def test_run_worker_eatr_mle_returns_finite_pair(self):
        sample = [_make_traj(s) for s in (1.0, 1.1, 0.9)]
        event = np.array([True, True, False])
        config = self.EATRMLEConfig(beta=1.0, gamma_bounds=(0.0, 1.0),
                                     log_trick=False, do_bopt=False, bias_shift=0.0)
        result = self._run_worker((np.array([0, 1, 2]), sample, event, config))
        self.assertEqual(len(result), 2)
        self.assertTrue(all(np.isfinite(result)))

    def test_run_worker_imetad_mle_returns_finite_scalar(self):
        sample = [_make_traj(s) for s in (1.0, 1.1, 0.9)]
        event = np.array([True, True, False])
        config = self.IMetaDMLEConfig(beta=1.0, bias_shift=0.0)
        result = self._run_worker((np.array([0, 1, 2]), sample, event, config))
        self.assertTrue(np.isfinite(float(result)))

    def test_run_worker_obs_log_rate_returns_positive(self):
        sample = [_make_traj(s) for s in (1.0, 1.1, 0.9)]
        event = np.array([True, True, False])
        config = self.ObsLogRateConfig()
        result = self._run_worker((np.array([0, 1, 2]), sample, event, config))
        self.assertGreater(float(result), 0.0)
