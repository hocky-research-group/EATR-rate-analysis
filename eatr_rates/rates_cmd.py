from __future__ import annotations

import argparse
import glob
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import gamma as gamma_func
from scipy.stats import ks_1samp, ks_2samp

import rate_methods_library as RM
from eatr_rates._bootstrap_workers import (
    _run_worker,
    ObsLogRateConfig,
    IMetaDMLEConfig,
    IMetaDCDFConfig,
    KTRMLEConfig,
    KTRCDFConfig,
    EATRMLEConfig,
    EATRCDFConfig,
)
from eatr_rates.time_units import TIME_UNIT_CHOICES, resolve_time_unit

bopt_avail = False
try:
    from bayes_opt import BayesianOptimization as bopt
    from bayes_opt import acquisition

    bopt_avail = True
except Exception:
    bopt_avail = False

boots_avail = False
try:
    from scipy.stats import bootstrap as bootstr

    boots_avail = True
except Exception:
    boots_avail = False


@dataclass
class AnalysisResult:
    beta: float
    data: list[np.ndarray]
    event: np.ndarray
    results: dict[str, object] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)


def _expand_globs(patterns: list[str]) -> list[str]:
    """Expand a list of glob patterns to a sorted list of file paths (Python-side expansion)."""
    result = []
    for pattern in patterns:
        matches = sorted(p for p in glob.glob(pattern, recursive=True) if not os.path.basename(p).startswith("bck"))
        if not matches:
            raise SystemExit(f"No files matched glob pattern: {pattern!r}")
        result.extend(matches)
    return result




def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    temperature = parser.add_mutually_exclusive_group()
    event_find = parser.add_mutually_exclusive_group()
    parser.add_argument("-i", "--input", type=str, help="the simulation COLVAR files to analyze", nargs="+")
    parser.add_argument("-g", "--input-glob", type=str, dest="input_glob", nargs="+", help="glob patterns for input COLVAR files, expanded by Python (quote to prevent shell expansion, e.g. -g 'path/to/*/run_*.colvar')")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="rates.json",
        help="the name of the output JSON file (DEFAULT: rates.json)",
    )
    temperature.add_argument(
        "--temp",
        type=np.float64,
        default=298,
        help="the temperature (in Kelvin) that the simulation was run at (make sure that ENERGYUNIT is correct) (DEFAULT: 298K)",
    )
    temperature.add_argument("--kt", type=np.float64, default=None, help="the temperature (in kBT) that the simulation was run at")
    temperature.add_argument("--beta", type=np.float64, default=None, help="the inverse temperature 1/kBT that the simulation was run at")
    parser.add_argument("--tcol", type=int, default=0, help="the time column index in the COLVAR file. (DEFAULT: 0)")
    parser.add_argument("--vcol", type=int, default=2, help="the bias column index in the COLVAR file. (DEFAULT: 2)")
    parser.add_argument("--acol", type=int, default=None, help="the acceleration factor column index in the COLVAR file, if present. (DEFAULT: None)")
    parser.add_argument("--mcol", type=int, default=None, help="the max bias column index in the COLVAR file, if present. (DEFAULT: None)")
    parser.add_argument("--stride", type=int, default=1, help="keep only every Nth row of each COLVAR (the final row is always kept). Use to thin finely-printed COLVAR files for speed. (DEFAULT: 1)")
    parser.add_argument("--cdf-weights", choices=["none","binomial"], default="none", dest="cdf_weights", help="weighting for CDF least-squares fits (iMetaD/KTR/EATR). 'none' (default) is the historical unweighted fit; 'binomial' weights each empirical-CDF point by its sampling std sqrt(F(1-F)), which is typically more accurate at fast bias-deposition. (DEFAULT: none)")
    parser.add_argument("--cdf-qmin", type=float, default=0.0, dest="cdf_qmin", help="exclude empirical-CDF points below this quantile from CDF fits. Use to ignore anomalously fast runs (e.g. trajectories that commit in the first frame because of a bad starting structure). The runs still count towards N, so the observed rate and CDF asymptote are unchanged. (DEFAULT: 0.0)")
    parser.add_argument("--cdf-qmax", type=float, default=1.0, dest="cdf_qmax", help="exclude empirical-CDF points above this quantile from CDF fits. Keep at 1.0 unless the long-time tail is known to be contaminated: it carries the censoring information. (DEFAULT: 1.0)")
    parser.add_argument("--subsample-min-points", type=int, default=0, dest="subsample_min_points", help="when using --stride, reduce the stride so even the shortest trajectory keeps at least this many rows (one uniform stride is used for all trajectories). Protects short, fast-transitioning runs from being over-thinned. (DEFAULT: 0, no floor)")
    parser.add_argument("--subsample-runs", type=str, default=None, dest="subsample_runs", help="ALSO fit using only this many of the input trajectories, e.g. '20' or a comma-separated sweep '10,20,30'. The full-set fit is still reported at the top level; subset fits go in the 'subsample' block. Use to measure how much the answer depends on how many runs you did. (DEFAULT: None)")
    parser.add_argument("--subsample-reps", type=int, default=1, dest="subsample_reps", help="number of independent random subsets to draw at each --subsample-runs size. The SPREAD across these is the honest uncertainty at that size. (DEFAULT: 1)")
    parser.add_argument("--subsample-bootstrap", action="store_true", dest="subsample_bootstrap", help="also bootstrap inside each subset. Off by default: with several replicates the spread ACROSS subsets already measures the uncertainty, and nesting the bootstrap multiplies the cost by --numboots. (DEFAULT: off)")
    parser.add_argument("--subsample-replace", action="store_true", dest="subsample_replace", help="draw subsets WITH replacement. Off by default; with-replacement resampling at full size is what --bootstrap already does, and conflating the two is rarely what you want. (DEFAULT: off, i.e. without replacement)")
    parser.add_argument(
        "--timeunit",
        type=np.float64,
        default=1e-12,
        help="the conversion factor from the time unit used in PLUMED to seconds (DEFAULT: 1e-12, for picoseconds)",
    )
    parser.add_argument(
        "--energyunit",
        type=np.float64,
        default=1,
        help="the conversion factor from the energy unit used in PLUMED to kJ/mol (only needed if temperature was given in Kelvin) (DEFAULT: 1, for kJ/mol)",
    )
    parser.add_argument(
        "--barrier",
        type=np.float64,
        default=0,
        help="the BARRIER parameter in PLUMED for OPES (it is not a good idea to use this script to run KTR and EATR on OPES simulations btw) (DEFAULT: 0)",
    )
    parser.add_argument("--gammamin", type=np.float64, default=0, help="the minimum value of gamma to be checked in KTR and EATR (DEFAULT: 0)")
    parser.add_argument("--gammamax", type=np.float64, default=1, help="the maximum value of gamma to be checked in KTR and EATR (DEFAULT: 1)")
    parser.add_argument("--lnkmin", type=np.float64, default=-np.inf, help="the minimum value of lnk0 to be checked in CDF fitting (DEFAULT: -inf)")
    parser.add_argument("--lnkmax", type=np.float64, default=np.inf, help="the maximum value of lnk0 to be checked in CDF fitting (DEFAULT: inf)")
    parser.add_argument("--initguess", type=np.float64, default=[None, None], help="the initial guess for lnk0 and gamma, respectively, in CDF fitting (DEFAULT: use iMetaD MLE estimate and γ = 0.9)", nargs=2)
    parser.add_argument("--seed", type=int, default=None, help="the random number generator seed to use (for repeatability) (DEFAULT: None)")
    parser.add_argument("--threads", type=int, default=1, help="number of parallel worker processes for bootstrap resampling (DEFAULT: 1)")
    event_find.add_argument(
        "--maxlen",
        type=int,
        default=None,
        help="the maximum number of rows in each COLVAR file before the simulation runs out of time (DEFAULT: Do not use this to determine which simulations transitioned.)",
    )
    event_find.add_argument(
        "--maxtime",
        type=np.float64,
        default=None,
        help="the maximum time that can appear in each COLVAR file (try to make it slightly less for floating point reasons) (DEFAULT: Do not use this to determine which simulations transitioned.)",
    )
    event_find.add_argument(
        "--numevents",
        type=int,
        default=None,
        help="the number of simulations that transitioned (DEFAULT: Do not use this to determine which simulations transitioned.)",
    )
    event_find.add_argument(
        "--logfiles",
        type=str,
        default=None,
        help="the files that contains the PLUMED logs. Use check_order.py to make sure that the correct COLVAR files are paired with the correct log files (DEFAULT: Do not use this to determine which simulations transitioned.)",
        nargs="+",
    )
    parser.add_argument(
        "--logfiles-glob",
        type=str,
        dest="logfiles_glob",
        default=None,
        nargs="+",
        help="glob patterns for PLUMED log files, expanded by Python (alternative to --logfiles; quote to prevent shell expansion)",
    )
    parser.add_argument("-m", "--imetadmle", action="store_true", help="run the Tiwary rate estimator for infrequent metadynamics")
    parser.add_argument("-M", "--imetadcdf", action="store_true", help="run the Salvalaglio rate estimator for infrequent metadynamics")
    parser.add_argument("-k", "--ktrmle", action="store_true", help="run original KTR method")
    parser.add_argument("-K", "--ktrcdf", action="store_true", help="run KTR method estimating gamma and k0 with CDF")
    parser.add_argument("-e", "--eatrmle", action="store_true", help="run EATR method estimating gamma and k0 with likelihood")
    parser.add_argument("-E", "--eatrcdf", action="store_true", help="run EATR method estimating gamma and k0 with CDF")
    parser.add_argument("-b", "--bootstrap", action="store_true", help="calculate errorbars with bootstrap analysis")
    parser.add_argument("--std", action="store_true", help="use standard deviations in bootstrap analysis even if SciPy has the bootstrap method")
    parser.add_argument("--numboots", type=int, default=100, help="the number of bootstrap samples to use in bootsrapping if enabled (DEFAULT: 100)")
    parser.add_argument("-B", "--bayesopt", action="store_true", help="use Bayseian Optimization algorithm for optimizing if available")
    parser.add_argument("-l", "--logtrick", action="store_true", help="use log-sum-exp trick to potentially increase precision (generally unneeded)")
    parser.add_argument("--require-convergence", action="store_true", dest="require_convergence", help="raise an error if any CDF fit does not converge (default: save best fit and record convergence status in output)")
    parser.add_argument("-q", "--quiet", action="store_true", help="do not print the results to the terminal as they are calculated")
    parser.add_argument("--plot-time-unit", choices=TIME_UNIT_CHOICES, default="seconds", help="preferred time unit for downstream plotting metadata (DEFAULT: seconds)")
    return parser


def parse_beta(args: argparse.Namespace) -> float:
    if args.beta is not None:
        return args.beta
    if args.kt is not None:
        return 1 / args.kt
    return args.energyunit / (0.008314 * args.temp)


def validate_args(args: argparse.Namespace) -> None:
    if not (args.imetadmle or args.imetadcdf or args.ktrmle or args.ktrcdf or args.eatrmle or args.eatrcdf):
        raise SystemExit("Specify at least one rate method to perform from -m -M -k -K -e -E (M=iMetaD, K=KTR, E=EATR; lowercase is MLE and uppercase is CDF).")
    if getattr(args, "subsample_runs", None):
        for size in parse_subsample_sizes(args.subsample_runs):
            if size < 2:
                raise SystemExit(f"--subsample-runs must be at least 2, got {size}.")
        if args.subsample_reps < 1:
            raise SystemExit("--subsample-reps must be at least 1.")


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "low") and hasattr(value, "high"):
        return {"low": json_ready(value.low), "high": json_ready(value.high)}
    return value


def add_message(run: AnalysisResult, message: str) -> None:
    run.messages.append(message)


def write_results(path: str, results: dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_ready(results), handle)


def percentile_interval(values: np.ndarray, alpha: float = 0.05) -> list[float]:
    lower = float(np.quantile(values, alpha / 2.0))
    upper = float(np.quantile(values, 1.0 - alpha / 2.0))
    return [lower, upper]


def observed_log_rate(data: list[np.ndarray], event: np.ndarray) -> float:
    final_times = np.array([traj[-1, 0] for traj in data], dtype=float)
    return float(np.log(event.sum() / np.sum(final_times)))


def threaded_bootstrap(
    sample: list[np.ndarray],
    config,
    nresamples: int,
    *,
    event: np.ndarray | None = None,
    double: bool = False,
    seed: int | None = None,
    threads: int = 1,
):
    """Run bootstrap resampling using ProcessPoolExecutor (bypasses the GIL).

    config must be one of the picklable *Config dataclasses from
    eatr_rates._bootstrap_workers.  Each worker process receives a copy of the
    resampled data and returns a scalar or 2-element array.
    """
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


def report_beta(run: AnalysisResult) -> None:
    add_message(run, f"Using β = {run.beta}")


def report_beta_from_temp(run: AnalysisResult, energyunit: float) -> None:
    add_message(run, f"Using β = 1/kBT = {run.beta}, with PLUMED energy unit equivalent to {energyunit} kJ/mol")


def report_bootstrap_mode(run: AnalysisResult, bootstrap_enabled: bool, ci_bootstrap: bool, threaded_bootstrap_enabled: bool) -> None:
    if not bootstrap_enabled:
        return
    if threaded_bootstrap_enabled and ci_bootstrap:
        add_message(run, "Bootstrapping is activated. Will use parallel process bootstrap (errors are 95% percentile confidence intervals).")
    elif ci_bootstrap:
        add_message(run, "Bootstrapping is activated. Will use SciPy bootstrap method (errors are 95% confidence intervals).")
    else:
        add_message(run, "SciPy bootstrap method is not available. Will use internal bootstrap method (errors are standard deviations).")


def report_method_result(run: AnalysisResult, method_label: str, log_key: str, ks_stat: float, p_value: float, bootstrap_enabled: bool, ci_bootstrap: bool, gamma_key: str | None = None) -> None:
    if not bootstrap_enabled:
        if gamma_key is None:
            add_message(run, f"{method_label}: lnk0 = {run.results[log_key]} (s^-1); KS: {ks_stat}, p = {p_value}")
        else:
            add_message(run, f"{method_label}: lnk0 = {run.results[log_key]} (s^-1), γ = {run.results[gamma_key]}; KS: {ks_stat}, p = {p_value}")
        return
    if ci_bootstrap:
        if gamma_key is None:
            ci = run.results[f"{log_key} CI"]
            add_message(run, f"{method_label}: lnk0 = {run.results[log_key]} (s^-1), 95% CI: {ci[0]} to {ci[1]}; KS: {ks_stat}, p = {p_value}")
        else:
            log_ci = run.results[f"{log_key} CI"]
            gamma_ci = run.results[f"{gamma_key} CI"]
            add_message(run, f"{method_label}: lnk0 = {run.results[log_key]} (s^-1), 95% CI: {log_ci[0]} to {log_ci[1]}, γ = {run.results[gamma_key]}, 95% CI: {gamma_ci[0]} to {gamma_ci[1]}; KS: {ks_stat}, p = {p_value}")
        return
    if gamma_key is None:
        add_message(run, f"{method_label}: lnk0 = {run.results[log_key]} +/- {run.results[f'{log_key} std']} (s^-1); KS: {ks_stat}, p = {p_value}")
    else:
        add_message(run, f"{method_label}: lnk0 = {run.results[log_key]} +/- {run.results[f'{log_key} std']} (s^-1), γ = {run.results[gamma_key]} +/- {run.results[f'{gamma_key} std']}; KS: {ks_stat}, p = {p_value}")


def emit_messages(run: AnalysisResult, quiet: bool) -> None:
    if quiet:
        return
    for message in run.messages:
        print(message)


def build_eatr_cdf_plot_payload(data, event, final_time_indices, log_average_exp, log_k: float, log_trick: bool, gamma: float | None = None):
    event = np.asarray(event, dtype=bool)
    if event.sum() == 0:
        return {"time": [], "ecdf": [], "fit": [], "n_total": len(data), "n_events": 0, "ln_k": log_k}
    final_times = np.array([traj[-1, 0] for traj in data], dtype=float)
    transitioned_times = final_times[event]
    transitioned_indices = np.asarray(final_time_indices, dtype=int)[event]
    order = np.argsort(transitioned_times)
    sorted_times = transitioned_times[order]
    sorted_indices = transitioned_indices[order]
    ecdf = np.arange(1, len(sorted_times) + 1, dtype=float) / len(data)
    fit = RM.EATR_CDF(sorted_indices, np.exp(log_k), log_average_exp, logTrick=log_trick)
    payload = {
        "time": sorted_times.tolist(),
        "ecdf": ecdf.tolist(),
        "fit": np.asarray(fit, dtype=float).tolist(),
        "n_total": len(data),
        "n_events": int(event.sum()),
        "ln_k": float(log_k),
    }
    if gamma is not None:
        payload["gamma"] = float(gamma)
    return payload


def build_exp_cdf_plot_payload(times, event, log_k: float):
    event = np.asarray(event, dtype=bool)
    times = np.asarray(times, dtype=float)
    if event.sum() == 0:
        return {"time": [], "ecdf": [], "fit": [], "n_total": len(times), "n_events": 0, "ln_k": log_k}
    transitioned_times = times[event]
    sorted_times = np.sort(transitioned_times)
    ecdf = np.arange(1, len(sorted_times) + 1, dtype=float) / len(times)
    fit = 1.0 - np.exp(-np.exp(log_k) * sorted_times)
    return {
        "time": sorted_times.tolist(),
        "ecdf": ecdf.tolist(),
        "fit": np.asarray(fit, dtype=float).tolist(),
        "n_total": len(times),
        "n_events": int(event.sum()),
        "ln_k": float(log_k),
    }


def parse_subsample_sizes(spec: str | None) -> list[int]:
    """Parse "10,20,30" (or "20") into [10, 20, 30]."""
    if not spec:
        return []
    sizes = []
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            sizes.append(int(token))
        except ValueError:
            raise SystemExit(f"--subsample-runs expects integers, got {token!r}.")
    return sizes


def summarise_subsamples(records: list[dict]) -> dict:
    """mean/std across replicates, per subset size, for every numeric scalar recorded.

    The std across independent subsets is the quantity of interest: it says how much the
    answer depends on WHICH runs you happened to do. It is NOT the bootstrap CI, and for
    subsets drawn from a common parent it is a LOWER bound, because the subsets overlap
    (e.g. 30 drawn from 40 share 75% of their trajectories). `parent_n` is recorded so a
    finite-population correction can be applied downstream.
    """
    summary: dict[str, dict] = {}
    by_size: dict[int, list[dict]] = {}
    for rec in records:
        by_size.setdefault(rec["n"], []).append(rec)
    for size, recs in sorted(by_size.items()):
        entry: dict[str, object] = {"n_fits": len(recs)}
        keys = {k for r in recs for k, v in r.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("n", "rep")}
        for key in sorted(keys):
            vals = [float(r[key]) for r in recs if isinstance(r.get(key), (int, float))
                    and not isinstance(r.get(key), bool)]
            if not vals:
                continue
            entry[f"{key} mean"] = float(np.mean(vals))
            entry[f"{key} std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        summary[str(size)] = entry
    return summary


def _run_estimators(
    run: AnalysisResult,
    data: list,
    event: np.ndarray,
    args: argparse.Namespace,
    *,
    beta,
    seed,
    k_bounds,
    gamma_bounds,
    init_guess,
    use_scipy_bootstrap: bool,
    use_threaded_bootstrap: bool,
    bootstrap_ci: bool,
) -> AnalysisResult:
    """Run every requested estimator on (data, event), writing into run.results.

    Split out of analyze() so the same estimator stack can be applied both to the full
    dataset and to subsamples of it (--subsample-runs) without re-reading the COLVARs,
    which is by far the expensive part of a fit.

    `data` and `event` MUST be index-aligned; a mismatch silently produces a wrong
    answer rather than an error, so callers that slice them do so together.
    """
    run.results["ln(k_obs)"] = observed_log_rate(data, event)

    if args.bootstrap:
        run.results["numboots"] = args.numboots
        observed_log_rate_sample = threaded_bootstrap(
            data,
            ObsLogRateConfig(),
            args.numboots,
            event=event,
            seed=seed,
            threads=args.threads,
        )
        run.results["ln(k_obs) avg"] = float(np.mean(np.log(observed_log_rate_sample)))
        run.results["ln(k_obs) std"] = float(np.std(np.log(observed_log_rate_sample)))
        seed = seed if seed is None else seed + 1

    if args.imetadmle or args.imetadcdf:
        rescaled_times = RM.iMetaD_rescaled_times(data, beta, bias_shift=args.barrier)

    if args.imetadmle:
        if not args.bootstrap:
            run.results["iMetaD MLE ln k"] = np.log(RM.iMetaD_invMRT_times(rescaled_times, event=event))
        else:
            if use_scipy_bootstrap:
                indices = list(range(len(data)))
                run.results["iMetaD MLE ln k"] = np.log(RM.iMetaD_invMRT_times(rescaled_times, event=event))
                res = bootstr(
                    (indices,),
                    lambda idxs: np.log(RM.iMetaD_invMRT([data[idx] for idx in idxs], beta, event=np.array([event[idx] for idx in idxs]), bias_shift=args.barrier)),
                    random_state=seed,
                    vectorized=False,
                    n_resamples=args.numboots,
                )
                run.results["iMetaD MLE ln k CI"] = res.confidence_interval
            elif use_threaded_bootstrap:
                sample = threaded_bootstrap(data, IMetaDMLEConfig(beta=beta, bias_shift=args.barrier), args.numboots, event=event, seed=seed, threads=args.threads)
                log_sample = np.log(sample)
                run.results["iMetaD MLE ln k"] = float(np.mean(log_sample))
                run.results["iMetaD MLE ln k CI"] = percentile_interval(log_sample)
            else:
                sample = RM.bootstrap(data, lambda subset, eve: RM.iMetaD_invMRT(subset, beta, event=eve, bias_shift=args.barrier), args.numboots, event=event, return_stat=True, seed=seed)
                run.results["iMetaD MLE ln k"] = np.mean(np.log(sample))
                run.results["iMetaD MLE ln k std"] = np.std(np.log(sample))
            seed = seed if seed is None else seed + 1
        size = np.int64(len(data) * 5e4)
        rvs1 = gamma_func.rvs(1, scale=np.exp(-run.results["iMetaD MLE ln k"]), size=size, random_state=seed)
        seed = seed if seed is None else seed + 1
        ks_stat, p = ks_2samp(rvs1, rescaled_times[event])
        run.results["iMetaD MLE KS stat"] = ks_stat
        run.results["iMetaD MLE p value"] = p
        report_method_result(run, "iMetaD MLE", "iMetaD MLE ln k", ks_stat, p, args.bootstrap, bootstrap_ci)

    if args.imetadcdf:
        if not args.bootstrap:
            k, converged = RM.iMetaD_FitCDF_times(rescaled_times, event=event, k_bounds=k_bounds, k_guess=init_guess[0], require_convergence=args.require_convergence, cdf_weights=args.cdf_weights, cdf_qmin=args.cdf_qmin, cdf_qmax=args.cdf_qmax)
            run.results["iMetaD CDF ln k"] = np.log(k)
            run.results["iMetaD CDF converged"] = converged
        else:
            if use_scipy_bootstrap:
                indices = list(range(len(data)))
                k, converged = RM.iMetaD_FitCDF_times(rescaled_times, event=event, k_bounds=k_bounds, k_guess=init_guess[0], require_convergence=args.require_convergence, cdf_weights=args.cdf_weights, cdf_qmin=args.cdf_qmin, cdf_qmax=args.cdf_qmax)
                run.results["iMetaD CDF ln k"] = np.log(k)
                run.results["iMetaD CDF converged"] = converged
                _imetad_cdf_boot_convergence: list[bool] = []
                def _imetad_cdf_scipy_boot(idxs):
                    k_b, conv_b = RM.iMetaD_FitCDF([data[idx] for idx in idxs], beta, event=np.array([event[idx] for idx in idxs]), bias_shift=args.barrier, k_bounds=k_bounds, k_guess=init_guess[0], require_convergence=args.require_convergence, cdf_weights=args.cdf_weights, cdf_qmin=args.cdf_qmin, cdf_qmax=args.cdf_qmax)
                    _imetad_cdf_boot_convergence.append(conv_b)
                    return np.log(k_b)
                res = bootstr(
                    (indices,),
                    _imetad_cdf_scipy_boot,
                    random_state=seed,
                    vectorized=False,
                    n_resamples=args.numboots,
                )
                run.results["iMetaD CDF ln k CI"] = res.confidence_interval
                run.results["iMetaD CDF n_unconverged_boots"] = _imetad_cdf_boot_convergence.count(False)
            elif use_threaded_bootstrap:
                raw = threaded_bootstrap(data, IMetaDCDFConfig(beta=beta, bias_shift=args.barrier, k_bounds=k_bounds, k_guess=init_guess[0], require_convergence=args.require_convergence, cdf_weights=args.cdf_weights, cdf_qmin=args.cdf_qmin, cdf_qmax=args.cdf_qmax), args.numboots, event=event, seed=seed, threads=args.threads)
                k_sample = raw[:, 0]
                log_sample = np.log(k_sample)
                run.results["iMetaD CDF ln k"] = float(np.mean(log_sample))
                run.results["iMetaD CDF ln k CI"] = percentile_interval(log_sample)
                run.results["iMetaD CDF converged"] = bool(np.all(raw[:, 1] > 0.5))
                run.results["iMetaD CDF n_unconverged_boots"] = int(np.sum(raw[:, 1] < 0.5))
            else:
                _imetad_cdf_conv: list[bool] = []
                def _imetad_cdf_boot(subset, eve):
                    k_b, conv_b = RM.iMetaD_FitCDF(subset, beta, event=eve, bias_shift=args.barrier, k_guess=init_guess[0], require_convergence=args.require_convergence, cdf_weights=args.cdf_weights, cdf_qmin=args.cdf_qmin, cdf_qmax=args.cdf_qmax)
                    _imetad_cdf_conv.append(conv_b)
                    return k_b
                sample = RM.bootstrap(data, _imetad_cdf_boot, args.numboots, event=event, return_stat=True, seed=seed)
                run.results["iMetaD CDF ln k"] = np.mean(np.log(sample))
                run.results["iMetaD CDF ln k std"] = np.std(np.log(sample))
                run.results["iMetaD CDF n_unconverged_boots"] = _imetad_cdf_conv.count(False)
            seed = seed if seed is None else seed + 1
        run.results["iMetaD CDF plot"] = build_exp_cdf_plot_payload(rescaled_times, event, run.results["iMetaD CDF ln k"])
        size = np.int64(len(data) * 5e4)
        rvs1 = gamma_func.rvs(1, scale=np.exp(-run.results["iMetaD CDF ln k"]), size=size, random_state=seed)
        seed = seed if seed is None else seed + 1
        ks_stat, p = ks_2samp(rvs1, rescaled_times[event])
        run.results["iMetaD CDF KS stat"] = ks_stat
        run.results["iMetaD CDF p value"] = p
        report_method_result(run, "iMetaD CDF", "iMetaD CDF ln k", ks_stat, p, args.bootstrap, bootstrap_ci)

    final_time_indices = np.array([int(len(traj) - 1) for traj in data])
    if args.ktrmle or args.ktrcdf:
        vmb_average = RM.avg_max_bias(data, beta, bias_shift=args.barrier)

    if args.ktrmle:
        if not args.bootstrap:
            result = RM.KTR_MLE_rate_VMB(vmb_average, final_time_indices, event=event, gamma_bounds=gamma_bounds, logTrick=args.logtrick, do_bopt=args.bayesopt)
            run.results["KTR MLE ln k"] = np.log(result[0])
            run.results["KTR MLE gamma"] = result[1]
        else:
            if use_scipy_bootstrap:
                indices = list(range(len(data)))
                result = RM.KTR_MLE_rate_VMB(vmb_average, final_time_indices, event=event, gamma_bounds=gamma_bounds, logTrick=args.logtrick, do_bopt=args.bayesopt)
                res = bootstr(
                    (indices,),
                    lambda idxs: RM.KTR_MLE_rate([data[idx] for idx in idxs], beta, event=np.array([event[idx] for idx in idxs]), gamma_bounds=gamma_bounds, logTrick=args.logtrick, do_bopt=args.bayesopt, bias_shift=args.barrier),
                    random_state=seed,
                    vectorized=False,
                    n_resamples=args.numboots,
                )
                run.results["KTR MLE ln k"] = np.log(result[0])
                run.results["KTR MLE gamma"] = result[1]
                run.results["KTR MLE ln k CI"] = [np.log(res.confidence_interval.low[0]), np.log(res.confidence_interval.high[0])]
                run.results["KTR MLE gamma CI"] = [res.confidence_interval.low[1], res.confidence_interval.high[1]]
            elif use_threaded_bootstrap:
                sample = threaded_bootstrap(data, KTRMLEConfig(beta=beta, gamma_bounds=gamma_bounds, log_trick=args.logtrick, do_bopt=args.bayesopt, bias_shift=args.barrier), args.numboots, event=event, double=True, seed=seed, threads=args.threads)
                run.results["KTR MLE ln k"] = float(np.mean(np.log(sample[:, 0])))
                run.results["KTR MLE gamma"] = float(np.mean(sample[:, 1]))
                run.results["KTR MLE ln k CI"] = percentile_interval(np.log(sample[:, 0]))
                run.results["KTR MLE gamma CI"] = percentile_interval(sample[:, 1])
            else:
                sample = RM.bootstrap(data, lambda subset, eve: RM.KTR_MLE_rate(subset, beta, event=eve, gamma_bounds=gamma_bounds, logTrick=args.logtrick, do_bopt=args.bayesopt, bias_shift=args.barrier), args.numboots, double=True, event=event, return_stat=True, seed=seed)
                run.results["KTR MLE ln k"] = np.mean(np.log(sample[:, 0]))
                run.results["KTR MLE gamma"] = np.mean(sample[:, 1])
                run.results["KTR MLE ln k std"] = np.std(np.log(sample[:, 0]))
                run.results["KTR MLE gamma std"] = np.std(sample[:, 1])
            seed = seed if seed is None else seed + 1
        ks_stat, p = ks_1samp(final_time_indices[event], lambda idx: RM.KTR_CDF(idx, np.exp(run.results["KTR MLE ln k"]), run.results["KTR MLE gamma"], vmb_average, logTrick=args.logtrick))
        run.results["KTR MLE KS stat"] = ks_stat
        run.results["KTR MLE p value"] = p
        report_method_result(run, "KTR MLE", "KTR MLE ln k", ks_stat, p, args.bootstrap, bootstrap_ci, gamma_key="KTR MLE gamma")

    if args.ktrcdf:
        if not args.bootstrap:
            result, converged = RM.KTR_CDF_rate_VMB(vmb_average, final_time_indices, event=event, k_bounds=k_bounds, gamma_bounds=gamma_bounds, logTrick=args.logtrick, init_guess=init_guess, do_bopt=args.bayesopt, require_convergence=args.require_convergence, cdf_weights=args.cdf_weights)
            run.results["KTR CDF ln k"] = np.log(result[0])
            run.results["KTR CDF gamma"] = result[1]
            run.results["KTR CDF converged"] = converged
        else:
            if use_scipy_bootstrap:
                indices = list(range(len(data)))
                result, converged = RM.KTR_CDF_rate_VMB(vmb_average, final_time_indices, event=event, k_bounds=k_bounds, gamma_bounds=gamma_bounds, logTrick=args.logtrick, init_guess=init_guess, do_bopt=args.bayesopt, require_convergence=args.require_convergence, cdf_weights=args.cdf_weights)
                run.results["KTR CDF ln k"] = np.log(result[0])
                run.results["KTR CDF gamma"] = result[1]
                run.results["KTR CDF converged"] = converged
                _ktr_cdf_conv: list[bool] = []
                def _ktr_cdf_scipy_boot(idxs):
                    r, conv_b = RM.KTR_CDF_rate([data[idx] for idx in idxs], beta, event=np.array([event[idx] for idx in idxs]), k_bounds=k_bounds, gamma_bounds=gamma_bounds, logTrick=args.logtrick, init_guess=init_guess, do_bopt=args.bayesopt, bias_shift=args.barrier, require_convergence=args.require_convergence, cdf_weights=args.cdf_weights)
                    _ktr_cdf_conv.append(conv_b)
                    return r
                res = bootstr(
                    (indices,),
                    _ktr_cdf_scipy_boot,
                    random_state=seed,
                    vectorized=False,
                    n_resamples=args.numboots,
                )
                run.results["KTR CDF ln k CI"] = [np.log(res.confidence_interval.low[0]), np.log(res.confidence_interval.high[0])]
                run.results["KTR CDF gamma CI"] = [res.confidence_interval.low[1], res.confidence_interval.high[1]]
                run.results["KTR CDF n_unconverged_boots"] = _ktr_cdf_conv.count(False)
            elif use_threaded_bootstrap:
                raw = threaded_bootstrap(data, KTRCDFConfig(beta=beta, k_bounds=k_bounds, gamma_bounds=gamma_bounds, log_trick=args.logtrick, init_guess=tuple(init_guess), do_bopt=args.bayesopt, bias_shift=args.barrier, require_convergence=args.require_convergence, cdf_weights=args.cdf_weights), args.numboots, event=event, double=True, seed=seed, threads=args.threads)
                run.results["KTR CDF ln k"] = float(np.mean(np.log(raw[:, 0])))
                run.results["KTR CDF gamma"] = float(np.mean(raw[:, 1]))
                run.results["KTR CDF ln k CI"] = percentile_interval(np.log(raw[:, 0]))
                run.results["KTR CDF gamma CI"] = percentile_interval(raw[:, 1])
                run.results["KTR CDF converged"] = bool(np.all(raw[:, 2] > 0.5))
                run.results["KTR CDF n_unconverged_boots"] = int(np.sum(raw[:, 2] < 0.5))
            else:
                _ktr_cdf_conv2: list[bool] = []
                def _ktr_cdf_boot(subset, eve):
                    r, conv_b = RM.KTR_CDF_rate(subset, beta, event=eve, k_bounds=k_bounds, gamma_bounds=gamma_bounds, logTrick=args.logtrick, init_guess=init_guess, do_bopt=args.bayesopt, bias_shift=args.barrier, require_convergence=args.require_convergence, cdf_weights=args.cdf_weights)
                    _ktr_cdf_conv2.append(conv_b)
                    return r
                sample = RM.bootstrap(data, _ktr_cdf_boot, args.numboots, double=True, event=event, return_stat=True, seed=seed)
                run.results["KTR CDF ln k"] = np.mean(np.log(sample[:, 0]))
                run.results["KTR CDF gamma"] = np.mean(sample[:, 1])
                run.results["KTR CDF ln k std"] = np.std(np.log(sample[:, 0]))
                run.results["KTR CDF gamma std"] = np.std(sample[:, 1])
                run.results["KTR CDF n_unconverged_boots"] = _ktr_cdf_conv2.count(False)
            seed = seed if seed is None else seed + 1
        ks_stat, p = ks_1samp(final_time_indices[event], lambda idx: RM.KTR_CDF(idx, np.exp(run.results["KTR CDF ln k"]), run.results["KTR CDF gamma"], vmb_average, logTrick=args.logtrick))
        run.results["KTR CDF KS stat"] = ks_stat
        run.results["KTR CDF p value"] = p
        report_method_result(run, "KTR CDF", "KTR CDF ln k", ks_stat, p, args.bootstrap, bootstrap_ci, gamma_key="KTR CDF gamma")

    if args.eatrmle:
        if not args.bootstrap:
            result = RM.EATR_MLE_rate(data, beta, event=event, gamma_bounds=gamma_bounds, logTrick=args.logtrick, do_bopt=args.bayesopt, bias_shift=args.barrier)
            run.results["EATR MLE ln k"] = np.log(result[0])
            run.results["EATR MLE gamma"] = result[1]
        else:
            if use_scipy_bootstrap:
                indices = list(range(len(data)))
                result = RM.EATR_MLE_rate(data, beta, event=event, gamma_bounds=gamma_bounds, logTrick=args.logtrick, do_bopt=args.bayesopt, bias_shift=args.barrier)
                res = bootstr(
                    (indices,),
                    lambda idxs: RM.EATR_MLE_rate([data[idx] for idx in idxs], beta, event=np.array([event[idx] for idx in idxs]), gamma_bounds=gamma_bounds, logTrick=args.logtrick, do_bopt=args.bayesopt, bias_shift=args.barrier),
                    random_state=seed,
                    vectorized=False,
                    n_resamples=args.numboots,
                )
                run.results["EATR MLE ln k"] = np.log(result[0])
                run.results["EATR MLE gamma"] = result[1]
                run.results["EATR MLE ln k CI"] = [np.log(res.confidence_interval.low[0]), np.log(res.confidence_interval.high[0])]
                run.results["EATR MLE gamma CI"] = [res.confidence_interval.low[1], res.confidence_interval.high[1]]
            elif use_threaded_bootstrap:
                sample = threaded_bootstrap(data, EATRMLEConfig(beta=beta, gamma_bounds=gamma_bounds, log_trick=args.logtrick, do_bopt=args.bayesopt, bias_shift=args.barrier), args.numboots, event=event, double=True, seed=seed, threads=args.threads)
                run.results["EATR MLE ln k"] = float(np.mean(np.log(sample[:, 0])))
                run.results["EATR MLE gamma"] = float(np.mean(sample[:, 1]))
                run.results["EATR MLE ln k CI"] = percentile_interval(np.log(sample[:, 0]))
                run.results["EATR MLE gamma CI"] = percentile_interval(sample[:, 1])
            else:
                sample = RM.bootstrap(data, lambda subset, eve: RM.EATR_MLE_rate(subset, beta, event=eve, gamma_bounds=gamma_bounds, logTrick=args.logtrick, do_bopt=args.bayesopt, bias_shift=args.barrier), args.numboots, double=True, event=event, return_stat=True, seed=seed)
                run.results["EATR MLE ln k"] = np.mean(np.log(sample[:, 0]))
                run.results["EATR MLE gamma"] = np.mean(sample[:, 1])
                run.results["EATR MLE ln k std"] = np.std(np.log(sample[:, 0]))
                run.results["EATR MLE gamma std"] = np.std(sample[:, 1])
            seed = seed if seed is None else seed + 1
        log_average_exp = RM.avg_exponential(data, beta, run.results["EATR MLE gamma"], bias_shift=args.barrier)
        run.results["EATR MLE CDF plot"] = build_eatr_cdf_plot_payload(data, event, final_time_indices, log_average_exp, run.results["EATR MLE ln k"], args.logtrick, gamma=run.results["EATR MLE gamma"])
        ks_stat, p = ks_1samp(final_time_indices[event], lambda idx: RM.EATR_CDF(idx, np.exp(run.results["EATR MLE ln k"]), log_average_exp, logTrick=args.logtrick))
        run.results["EATR MLE KS stat"] = ks_stat
        run.results["EATR MLE p value"] = p
        report_method_result(run, "EATR MLE", "EATR MLE ln k", ks_stat, p, args.bootstrap, bootstrap_ci, gamma_key="EATR MLE gamma")

    if args.eatrcdf:
        if not args.bootstrap:
            result, converged = RM.EATR_CDF_rate(data, beta, event=event, k_bounds=k_bounds, gamma_bounds=gamma_bounds, init_guess=init_guess, logTrick=args.logtrick, do_bopt=args.bayesopt, bias_shift=args.barrier, require_convergence=args.require_convergence, cdf_weights=args.cdf_weights, cdf_qmin=args.cdf_qmin, cdf_qmax=args.cdf_qmax)
            run.results["EATR CDF ln k"] = np.log(result[0])
            run.results["EATR CDF gamma"] = result[1]
            run.results["EATR CDF converged"] = converged
        else:
            if use_scipy_bootstrap:
                indices = list(range(len(data)))
                result, converged = RM.EATR_CDF_rate(data, beta, event=event, k_bounds=k_bounds, gamma_bounds=gamma_bounds, init_guess=init_guess, logTrick=args.logtrick, do_bopt=args.bayesopt, bias_shift=args.barrier, require_convergence=args.require_convergence, cdf_weights=args.cdf_weights, cdf_qmin=args.cdf_qmin, cdf_qmax=args.cdf_qmax)
                run.results["EATR CDF ln k"] = np.log(result[0])
                run.results["EATR CDF gamma"] = result[1]
                run.results["EATR CDF converged"] = converged
                _eatr_cdf_conv: list[bool] = []
                def _eatr_cdf_scipy_boot(idxs):
                    r, conv_b = RM.EATR_CDF_rate([data[idx] for idx in idxs], beta, event=np.array([event[idx] for idx in idxs]), k_bounds=k_bounds, gamma_bounds=gamma_bounds, init_guess=init_guess, logTrick=args.logtrick, do_bopt=args.bayesopt, bias_shift=args.barrier, require_convergence=args.require_convergence, cdf_weights=args.cdf_weights, cdf_qmin=args.cdf_qmin, cdf_qmax=args.cdf_qmax)
                    _eatr_cdf_conv.append(conv_b)
                    return r
                res = bootstr(
                    (indices,),
                    _eatr_cdf_scipy_boot,
                    random_state=seed,
                    vectorized=False,
                    n_resamples=args.numboots,
                )
                run.results["EATR CDF ln k CI"] = [np.log(res.confidence_interval.low[0]), np.log(res.confidence_interval.high[0])]
                run.results["EATR CDF gamma CI"] = [res.confidence_interval.low[1], res.confidence_interval.high[1]]
                run.results["EATR CDF n_unconverged_boots"] = _eatr_cdf_conv.count(False)
            elif use_threaded_bootstrap:
                raw = threaded_bootstrap(data, EATRCDFConfig(beta=beta, k_bounds=k_bounds, gamma_bounds=gamma_bounds, log_trick=args.logtrick, init_guess=tuple(init_guess), do_bopt=args.bayesopt, bias_shift=args.barrier, require_convergence=args.require_convergence, cdf_weights=args.cdf_weights, cdf_qmin=args.cdf_qmin, cdf_qmax=args.cdf_qmax), args.numboots, event=event, double=True, seed=seed, threads=args.threads)
                run.results["EATR CDF ln k"] = float(np.mean(np.log(raw[:, 0])))
                run.results["EATR CDF gamma"] = float(np.mean(raw[:, 1]))
                run.results["EATR CDF ln k CI"] = percentile_interval(np.log(raw[:, 0]))
                run.results["EATR CDF gamma CI"] = percentile_interval(raw[:, 1])
                run.results["EATR CDF converged"] = bool(np.all(raw[:, 2] > 0.5))
                run.results["EATR CDF n_unconverged_boots"] = int(np.sum(raw[:, 2] < 0.5))
            else:
                _eatr_cdf_conv2: list[bool] = []
                def _eatr_cdf_boot(subset, eve):
                    r, conv_b = RM.EATR_CDF_rate(subset, beta, event=eve, k_bounds=k_bounds, gamma_bounds=gamma_bounds, init_guess=init_guess, logTrick=args.logtrick, do_bopt=args.bayesopt, bias_shift=args.barrier, require_convergence=args.require_convergence, cdf_weights=args.cdf_weights, cdf_qmin=args.cdf_qmin, cdf_qmax=args.cdf_qmax)
                    _eatr_cdf_conv2.append(conv_b)
                    return r
                sample = RM.bootstrap(data, _eatr_cdf_boot, args.numboots, double=True, event=event, return_stat=True, seed=seed)
                run.results["EATR CDF ln k"] = np.mean(np.log(sample[:, 0]))
                run.results["EATR CDF gamma"] = np.mean(sample[:, 1])
                run.results["EATR CDF ln k std"] = np.std(np.log(sample[:, 0]))
                run.results["EATR CDF gamma std"] = np.std(sample[:, 1])
                run.results["EATR CDF n_unconverged_boots"] = _eatr_cdf_conv2.count(False)
            seed = seed if seed is None else seed + 1
        log_average_exp = RM.avg_exponential(data, beta, run.results["EATR CDF gamma"], bias_shift=args.barrier)
        run.results["EATR CDF plot"] = build_eatr_cdf_plot_payload(data, event, final_time_indices, log_average_exp, run.results["EATR CDF ln k"], args.logtrick, gamma=run.results["EATR CDF gamma"])
        ks_stat, p = ks_1samp(final_time_indices[event], lambda idx: RM.EATR_CDF(idx, np.exp(run.results["EATR CDF ln k"]), log_average_exp, logTrick=args.logtrick))
        run.results["EATR CDF KS stat"] = ks_stat
        run.results["EATR CDF p value"] = p
        report_method_result(run, "EATR CDF", "EATR CDF ln k", ks_stat, p, args.bootstrap, bootstrap_ci, gamma_key="EATR CDF gamma")



def analyze(args: argparse.Namespace) -> AnalysisResult:
    # --threads controls the ProcessPoolExecutor worker count for bootstrap.
    use_threaded_bootstrap = args.bootstrap and args.threads > 1
    use_scipy_bootstrap = args.bootstrap and boots_avail and not args.std and not use_threaded_bootstrap
    bootstrap_ci = use_scipy_bootstrap or use_threaded_bootstrap

    beta = parse_beta(args)
    validate_args(args)
    plot_time_unit, _, _ = resolve_time_unit(args.plot_time_unit)
    run = AnalysisResult(beta=beta, data=[], event=np.array([]))
    run.results["plot_time_unit"] = plot_time_unit
    # Record the subsampling/fitting options so a JSON is self-describing:
    # results produced with different settings are not directly comparable.
    if args.stride != 1:
        run.results["stride"] = args.stride
    if args.subsample_min_points:
        run.results["subsample_min_points"] = args.subsample_min_points
    if args.cdf_weights != "none":
        run.results["cdf_weights"] = args.cdf_weights
    if args.cdf_qmin > 0.0 or args.cdf_qmax < 1.0:
        run.results["cdf_quantile_range"] = [args.cdf_qmin, args.cdf_qmax]

    gamma_bounds = (args.gammamin, args.gammamax)
    k_bounds = (np.exp(args.lnkmin), np.exp(args.lnkmax))
    init_guess = args.initguess if args.initguess[0] is None else (np.exp(args.initguess[0]), args.initguess[1])

    if args.bayesopt and bopt_avail:
        add_message(run, "Bayesian Optimization module activated.")
    elif args.bayesopt:
        add_message(run, "The Bayesian Optimization module was not able to be loaded. Defaulting to local optimizers.")
    if args.beta is not None or args.kt is not None:
        report_beta(run)
    else:
        report_beta_from_temp(run, args.energyunit)
    report_bootstrap_mode(run, args.bootstrap, bootstrap_ci, use_threaded_bootstrap)

    if args.barrier > 0 and (args.ktrmle or args.ktrcdf or args.eatrmle or args.eatrcdf):
        add_message(run, "WARNING: Running KTR and/or EATR on OPES simulations using this analysis script is not expected to work properly! You should instead use the EATR-OPES method (not published yet).")

    seed = args.seed
    random.seed(seed)
    seed = seed if seed is None else seed + 1

    input_files = list(args.input or []) + (_expand_globs(args.input_glob) if args.input_glob else [])
    if args.logfiles_glob:
        logfiles = list(args.logfiles or []) + _expand_globs(args.logfiles_glob)
    else:
        logfiles = list(args.logfiles) if args.logfiles else None
    data, skipped, subsample_info = RM.get_data(input_files, args.tcol, args.vcol, acc_col=args.acol, maxbias_col=args.mcol, time_scale_factor=args.timeunit, threads=args.threads, stride=args.stride, subsample_min_points=args.subsample_min_points, return_info=True)
    if args.stride != 1 or args.subsample_min_points:
        # effective_stride can differ from the requested stride when
        # --subsample-min-points lowers it, so record what was actually applied
        # together with how many rows survived per trajectory.
        run.results["subsampling"] = subsample_info
    if skipped:
        for idx in skipped:
            add_message(run, f"WARNING: skipping unreadable COLVAR file: {input_files[idx]}")
        if logfiles is not None:
            logfiles = [logfiles[i] for i in range(len(logfiles)) if i not in skipped]
    RM.check_acc_consistency(data, beta)
    event = RM.get_event(data, maxlen=args.maxlen, maxtime=args.maxtime, num_events=args.numevents, log_files=logfiles, quiet=args.quiet)
    run.data = data
    run.event = event
    _run_estimators(
        run,
        data,
        event,
        args,
        beta=beta,
        seed=seed,
        k_bounds=k_bounds,
        gamma_bounds=gamma_bounds,
        init_guess=init_guess,
        use_scipy_bootstrap=use_scipy_bootstrap,
        use_threaded_bootstrap=use_threaded_bootstrap,
        bootstrap_ci=bootstrap_ci,
    )

    sizes = parse_subsample_sizes(getattr(args, "subsample_runs", None))
    if sizes:
        parent_n = len(data)
        too_big = [n for n in sizes if n > parent_n and not args.subsample_replace]
        if too_big:
            # Fail loudly: silently shrinking the request would quietly answer a
            # different question than the one that was asked.
            raise SystemExit(
                f"--subsample-runs {','.join(str(n) for n in too_big)} exceeds the "
                f"{parent_n} trajectories available (use --subsample-replace to draw "
                f"with replacement)."
            )
        rng = np.random.default_rng(args.seed)
        # Bootstrapping inside every subset multiplies cost by --numboots for information
        # the across-replicate spread already carries, so it is opt-in.
        sub_args = argparse.Namespace(**{**vars(args), "bootstrap": args.subsample_bootstrap})
        sub_boot = args.subsample_bootstrap and args.threads > 1
        records = []
        for size in sizes:
            # A subset that is the whole parent is deterministic, so extra replicates
            # would be identical fits.
            reps = 1 if (size >= parent_n and not args.subsample_replace) else args.subsample_reps
            for rep in range(1, reps + 1):
                idx = rng.choice(parent_n, size=size, replace=args.subsample_replace)
                sub_data = [data[i] for i in idx]
                sub_event = np.asarray(event)[idx]
                # data and event MUST stay index-aligned; a mismatch here would produce a
                # wrong number rather than an error.
                if len(sub_data) != len(sub_event):
                    raise SystemExit("internal error: subsample data/event length mismatch")
                sub_run = AnalysisResult(beta=beta, data=[], event=np.array([]))
                _run_estimators(
                    sub_run,
                    sub_data,
                    sub_event,
                    sub_args,
                    beta=beta,
                    seed=seed,
                    k_bounds=k_bounds,
                    gamma_bounds=gamma_bounds,
                    init_guess=init_guess,
                    use_scipy_bootstrap=False,
                    use_threaded_bootstrap=sub_boot,
                    bootstrap_ci=sub_boot,
                )
                rec: dict[str, object] = {"n": int(size), "rep": rep,
                                          "indices": [int(i) for i in idx]}
                for key, value in sub_run.results.items():
                    # keep the scalars; the big per-fit CDF payloads are not useful here
                    if isinstance(value, (int, float, bool, str)) or value is None:
                        rec[key] = value
                records.append(rec)
        run.results["subsample"] = {
            "parent_n": parent_n,
            "sizes": sizes,
            "reps": args.subsample_reps,
            "replace": bool(args.subsample_replace),
            "bootstrap_per_subset": bool(args.subsample_bootstrap),
            "results": records,
            "summary": summarise_subsamples(records),
        }

    return run


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run = analyze(args)
    emit_messages(run, args.quiet)
    write_results(args.output, run.results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
