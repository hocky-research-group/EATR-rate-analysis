"""Picklable worker functions for ProcessPoolExecutor bootstrap in eatr-analysis.

All public names here must be importable without side effects and must not
capture closures or lambdas — required for multiprocessing 'spawn' on macOS.
"""
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
    """Single dispatch entry point called by ProcessPoolExecutor.

    Parameters
    ----------
    task : tuple
        (indices, sample, event, config) where
        - indices : np.ndarray of int resample indices into sample
        - sample  : list[np.ndarray], full dataset
        - event   : np.ndarray | None, full event array
        - config  : one of the *Config dataclasses above

    Returns
    -------
    float or np.ndarray of float
        Scalar for single-valued methods; 2-element array for methods that
        return (k0, gamma).
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
                                 k_bounds=config.k_bounds,
                                 k_guess=config.k_guess)

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
