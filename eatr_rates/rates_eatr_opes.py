from __future__ import annotations

import argparse
import glob
import json
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import optimize
from scipy.stats import ks_1samp

import ks_censored as ksc
import rate_methods_library as RM
from eatr_rates.plot_results import plot_flooding_payload
from eatr_rates.time_units import TIME_UNIT_CHOICES, resolve_time_unit


def thread_map(func, values, threads: int):
    if threads <= 1:
        return [func(value) for value in values]
    with ThreadPoolExecutor(max_workers=threads) as executor:
        return list(executor.map(func, values))


def _prepare_flat(bv_2d: np.ndarray, axis_first: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a NaN-padded (N, T) beta*V matrix to valid-only 1-D arrays.

    Returns ``(bv_flat, w_flat)`` such that::

        float(np.dot(w_flat, np.exp(gamma * bv_flat)))

    is numerically identical to::

        np.mean(np.nanmean(np.exp(gamma * bv_2d), axis=axis_first))

    Skipping the ~55-65 % of padded NaN cells gives roughly an 8x speedup
    in the per-barrier gamma loop compared to operating on the full 2-D matrix.
    """
    valid = ~np.isnan(bv_2d)
    N, T = bv_2d.shape
    if axis_first == 0:
        # average over sims first, then over time steps
        counts = np.maximum(valid.sum(axis=0), 1)          # (T,)
        weights = valid / (counts[np.newaxis, :] * T)       # (N, T)
    else:
        # average over time first, then over sims
        traj_len = np.maximum(valid.sum(axis=1), 1)         # (N,)
        weights = valid / (traj_len[:, np.newaxis] * N)     # (N, T)
    return bv_2d[valid], weights[valid]


def _scan_barrier(bv_flat: np.ndarray, w_flat: np.ndarray, obs_rate: float,
                  gamma_start: float, dgamma: float, n_steps: int) -> list[float]:
    """Scan a uniform gamma grid for one barrier using the multiplicative step trick.

    Returns a list of ln(k0) estimates, one per grid point.
    """
    exp_flat = np.exp(gamma_start * bv_flat)
    step_flat = np.exp(dgamma * bv_flat) if dgamma != 0.0 else None
    log_obs = float(np.log(obs_rate))
    logk0s: list[float] = []
    for k in range(n_steps):
        if k > 0 and step_flat is not None:
            exp_flat *= step_flat
        logk0s.append(log_obs - float(np.log(np.dot(w_flat, exp_flat))))
    return logk0s


@dataclass
class FloodingSetReport:
    barrier: float
    transitioned: int
    total: int
    avg_max_bias: float
    tau_obs: float
    k_obs: float
    log_k_obs: float
    ks_stat: float
    p_value: float
    ln_exp_beta_v: float
    avg_work_kbt: float | None = None


@dataclass
class FloodingAnalysisResult:
    beta: float
    logk0: float
    gamma: float
    opes_logk0: float | None
    set_reports: list[FloodingSetReport] = field(default_factory=list)
    flooding_diagnostics: dict[str, object] | None = None
    bootstrap_logk0_std: float | None = None
    bootstrap_gamma_std: float | None = None
    bootstrap_opes_logk0_std: float | None = None
    bootstrap_per_set_log_k_obs_std: list[float] | None = None
    bootstrap_per_set_ln_exp_beta_v_std: list[float] | None = None
    bootstrap_iterations: list[int] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


def _expand_globs(patterns: list[str]) -> list[str]:
    """Expand a list of glob patterns to a sorted list of file paths (Python-side expansion)."""
    result = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if not matches:
            raise SystemExit(f"No files matched glob pattern: {pattern!r}")
        result.extend(matches)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    barr = parser.add_mutually_exclusive_group()
    temperature = parser.add_mutually_exclusive_group()
    event_find = parser.add_mutually_exclusive_group()
    parser.add_argument("-i", "--input", type=str, action="append", help="the files for a simulation set (call once for each set, i.e. -i path/to/1/*.colvar -i path/to/2/*.colvar etc.)", nargs="+")
    parser.add_argument("--input-glob", type=str, action="append", dest="input_glob", nargs="+", help="glob patterns for one simulation set, expanded by Python (call once per set; quote to prevent shell expansion, e.g. --input-glob 'path/to/set1/*/run_*.colvar')")
    parser.add_argument("-o", "--output", type=str, default="flooding_rates.json", help="the name of the output JSON file (DEFAULT: flooding_rates.json)")
    barr.add_argument("--barrier", type=np.float64, action="append", help="the BARRIER parameter in PLUMED for the simulation set (i.e. -i path/to/1/*.colvar --barrier 1 -i path/to/2/*.colvar --barrier 2 etc.)")
    barr.add_argument("--barriers", type=np.float64, help="the BARRIER parameter in PLUMED for each simulation set, defined all at once (i.e. -i path/to/1/*.colvar -i path/to/2/*.colvar etc. --barriers 1 2 etc.)", nargs="+")
    temperature.add_argument("--temp", type=np.float64, default=298, help="the temperature (in Kelvin) that the simulation was run at (make sure that ENERGYUNIT is correct)")
    temperature.add_argument("--kt", type=np.float64, default=None, help="the temperature (in kBT) that the simulation was run at")
    temperature.add_argument("--beta", type=np.float64, default=None, help="the inverse temperature 1/kBT that the simulation was run at")
    parser.add_argument("--tcol", type=int, default=0, help="the time column index in the COLVAR file")
    parser.add_argument("--vcol", type=int, default=2, help="the bias column index in the COLVAR file")
    parser.add_argument("--acol", type=int, default=None, help="the acceleration factor column index in the COLVAR file (only useful if OPESF is set)")
    parser.add_argument("--wcol", type=int, default=None, help="the cumulative work column index in the COLVAR file (e.g. metad.work); if given, produces a ln(k_obs) vs ln(W/kBT) plot")
    parser.add_argument("--timeunit", type=np.float64, default=1e-12, help="the conversion factor from the time unit used in PLUMED to seconds")
    parser.add_argument("--energyunit", type=np.float64, default=1, help="the conversion factor from the energy unit used in PLUMED to kJ/mol (only needed if temperature was given in Kelvin)")
    parser.add_argument("--gammamin", type=np.float64, default=0, help="the minimum value of gamma to be checked")
    parser.add_argument("--gammamax", type=np.float64, default=1, help="the maximum value of gamma to be checked")
    parser.add_argument("--avgover", type=np.float64, default=-1, help="only include Vi(t) from the first transition time to when this fraction of transitioned simulations have transitioned in the set (-1 to include all data, 1 to only exclude Vi(t) before first transition)")
    parser.add_argument("--seed", type=int, default=None, help="the random number generator seed to use (for repeatability)")
    parser.add_argument("--threads", type=int, default=1, help="the number of threads to use for independent set/bootstrap work (DEFAULT: 1)")
    event_find.add_argument("--maxlen", type=int, default=None, help="the maximum number of rows in each COLVAR file before the simulation runs out of time")
    event_find.add_argument("--maxtime", type=np.float64, default=None, help="the maximum time that can appear in each COLVAR file (try to make it slightly less for floating point reasons)")
    event_find.add_argument("--numevents", type=int, default=None, action="append", help="the number of simulations that transitioned for each simulation set (i.e. -i path/to/1/*.colvar --numevents 20 -i path/to/2/*.colvar --numevents 18 etc.)")
    event_find.add_argument("--logfiles", type=str, default=None, action="append", help="the name of the file that contains the PLUMED log for each simulation in each set (i.e. -i path/to/1/*.colvar --logfiles path/to/1/*.log -i path/to/2/*.colvar --logfiles path/to/2/*.log etc.). Use check_order.py to make sure that the correct COLVAR files are paired with the correct log files.", nargs="+")
    parser.add_argument("--logfiles-glob", type=str, default=None, action="append", dest="logfiles_glob", nargs="+", help="glob patterns for logfiles of one simulation set, expanded by Python (call once per set; alternative to --logfiles; quote to prevent shell expansion)")
    parser.add_argument("-b", "--bootstrap", action="store_true", help="calculate errorbars with bootstrap analysis")
    parser.add_argument("--numboots", type=int, default=100, help="the number of bootstrap samples to use in bootsrapping if enabled")
    parser.add_argument("-q", "--quiet", action="store_true", help="do not print the results to the terminal as they are calculated")
    parser.add_argument("--cdf", action="store_true", help="estimate the biased observed rates using CDF fitting (not recommended if you have arbitrarily right-censored data, such as simulations being killed before reaching max steps)")
    parser.add_argument("--timefirst", action="store_true", help="estimate ln<e^βγV> by averaging over time for each simulation, then over the simulations (default is over simulations first)")
    parser.add_argument("--nooffset", action="store_true", help="do not add the BARRIER parameter to the bias (OPES simulations in PLUMED offset the bias by -1*BARRIER, so do not use this for such simulations)")
    parser.add_argument("--opesf", action="store_true", help="also run the OPES flooding analysis on all of the data")
    parser.add_argument("--no-plots", action="store_true", help="do not write the flooding diagnostic plots alongside the JSON output")
    parser.add_argument("--plot-prefix", type=str, default=None, help="output prefix for generated flooding figures (default: output JSON path without .json)")
    parser.add_argument("--condition-label", type=str, default="Bias label", help="label for the per-set condition values in generated plots")
    parser.add_argument("--condition-unit", type=str, default="", help="unit suffix for the per-set condition values in generated plots")
    parser.add_argument("--title-prefix", type=str, default="Flooding analysis", help="title prefix for the generated diagnostic figure")
    parser.add_argument("--plot-time-unit", choices=TIME_UNIT_CHOICES, default="seconds", help="display time unit for generated plots and JSON metadata (DEFAULT: seconds)")
    parser.add_argument("--truerate", type=float, default=None, help="reference ln(k0) value in the display time unit; drawn as a dashed line on the acceleration and diagnostics plots")
    parser.add_argument("--nsets", type=int, default=None, help="number of simulation sets to use for the final fit, ordered from lowest to highest acceleration factor (default: use all sets)")
    parser.add_argument("--gamma-nsteps", type=int, default=401, help="number of gamma grid points for the diagnostic scan (DEFAULT: 401)")
    parser.add_argument("--adaptive-gamma", action="store_true", help="use an adaptive coarse+fine gamma grid: coarse scan over [gammamin,gammamax] (n//4 steps) then fine scan in a ±2-step window around the minimum (3n//4 steps)")
    return parser


def parse_beta(args: argparse.Namespace) -> float:
    if args.beta is not None:
        return args.beta
    if args.kt is not None:
        return 1 / args.kt
    return args.energyunit / (8.314462618e-3 * args.temp)


def emit_messages(result: FloodingAnalysisResult, quiet: bool) -> None:
    if quiet:
        return
    for message in result.messages:
        print(message)


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_results(path: str, payload: dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle)


def result_payload(result: FloodingAnalysisResult, plot_time_unit: str) -> dict[str, object]:
    plot_time_unit, _, _ = resolve_time_unit(plot_time_unit)
    return {
        "plot_time_unit": plot_time_unit,
        "beta": result.beta,
        "logk0": result.logk0,
        "gamma": result.gamma,
        "opes_logk0": result.opes_logk0,
        "flooding_diagnostics": result.flooding_diagnostics,
        "bootstrap_logk0_std": result.bootstrap_logk0_std,
        "bootstrap_gamma_std": result.bootstrap_gamma_std,
        "bootstrap_opes_logk0_std": result.bootstrap_opes_logk0_std,
        "bootstrap_per_set_log_k_obs_std": result.bootstrap_per_set_log_k_obs_std,
        "bootstrap_per_set_ln_exp_beta_v_std": result.bootstrap_per_set_ln_exp_beta_v_std,
        "bootstrap_iterations": result.bootstrap_iterations,
        "set_reports": [
            {
                "barrier": report.barrier,
                "transitioned": report.transitioned,
                "total": report.total,
                "avg_max_bias": report.avg_max_bias,
                "tau_obs": report.tau_obs,
                "k_obs": report.k_obs,
                "log_k_obs": report.log_k_obs,
                "ks_stat": report.ks_stat,
                "p_value": report.p_value,
                "ln_exp_beta_v": report.ln_exp_beta_v,
                "avg_work_kbt": report.avg_work_kbt,
            }
            for report in result.set_reports
        ],
    }


def format_flooding_result(result: FloodingAnalysisResult) -> list[str]:
    lines = [f"Using β = {result.beta}"]
    for report in result.set_reports:
        lines.append(f"Simulation Set: BARRIER = {result.beta * report.barrier} kBT")
        lines.append(f"{report.transitioned} out of {report.total} simulations transitioned.")
        lines.append(f"avg. max. bias: {report.avg_max_bias}")
        lines.append(f"tau_obs: {report.tau_obs}, k_obs: {report.k_obs}, log k_obs: {report.log_k_obs}")
        lines.append(f"KS stat: {report.ks_stat}; p = {report.p_value}")
        lines.append(rf"ln<e^βV>: {report.ln_exp_beta_v}")
        lines.append("")
    lines.extend(str(iteration) for iteration in result.bootstrap_iterations)
    if result.bootstrap_logk0_std is None:
        suffix = f", OPES logk0: {result.opes_logk0} s^-1" if result.opes_logk0 is not None else ""
        lines.append(f"\nk0: {np.exp(result.logk0)} s^-1, logk0: {result.logk0} s^-1, τ0: {np.exp(-result.logk0)} s, gamma: {result.gamma}{suffix}")
    else:
        suffix = ""
        if result.opes_logk0 is not None and result.bootstrap_opes_logk0_std is not None:
            suffix = f", OPES logk0: {result.opes_logk0} +/- σ {result.bootstrap_opes_logk0_std} s^-1"
        lines.append(f"logk0: {result.logk0} +/- σ {result.bootstrap_logk0_std} s^-1, τ0: {np.exp(-result.logk0)} s, gamma: {result.gamma} +/- σ {result.bootstrap_gamma_std}{suffix}")
    return lines


def analyze(args: argparse.Namespace) -> FloodingAnalysisResult:
    beta = parse_beta(args)
    random.seed(args.seed)

    if args.input_glob:
        args.input = (args.input or []) + [_expand_globs(pats) for pats in args.input_glob]
    if args.logfiles_glob:
        args.logfiles = (args.logfiles or []) + [_expand_globs(pats) for pats in args.logfiles_glob]

    barriers = args.barriers if args.barriers is not None else args.barrier
    if args.input is None or len(args.input) < 2:
        raise SystemExit("You need at least two sets of simulations, each with a different value for BARRIER in PLUMED.")
    if barriers is None or len(args.input) != len(barriers):
        raise SystemExit(f"You must specify the same number of BARRIER values as simulation sets in INPUT. There are {len(barriers) if barriers is not None else 0} BARRIER values and {len(args.input)} simulation sets.")

    num_eventss = args.numevents if args.numevents is not None else [None] * len(barriers)
    log_filess = args.logfiles if args.logfiles is not None else [None] * len(barriers)
    gamma_bounds = (args.gammamin, args.gammamax)
    axis_first = 1 if args.timefirst else 0

    datas = []
    events = []
    for i, colvars in enumerate(args.input):
        data = RM.get_data(colvars, args.tcol, args.vcol, acc_col=args.acol, time_scale_factor=args.timeunit, threads=args.threads, work_col=args.wcol)
        event = RM.get_event(data, maxlen=args.maxlen, maxtime=args.maxtime, num_events=num_eventss[i], log_files=log_filess[i], quiet=True)
        datas.append(data)
        events.append(event)

    def compute_diagnostics(flat_bv: dict[float, tuple], obs_rates: dict[float, float],
                            threads: int = 1) -> dict[str, object]:
        n_steps = args.gamma_nsteps
        g_lo, g_hi = args.gammamin, args.gammamax

        if args.adaptive_gamma and n_steps >= 10:
            # Phase 1: coarse scan across the full range.
            n_coarse = max(5, n_steps // 4)
            dg_c = (g_hi - g_lo) / (n_coarse - 1) if n_coarse > 1 else 0.0

            def _coarse(b):
                bv, w = flat_bv[b]
                return _scan_barrier(bv, w, obs_rates[b], g_lo, dg_c, n_coarse)

            coarse_results = thread_map(_coarse, barriers, threads)
            coarse_var = [float(np.var([coarse_results[bi][k] for bi in range(len(barriers))]))
                          for k in range(n_coarse)]
            best_c = int(np.argmin(coarse_var))
            g_best_c = g_lo + best_c * dg_c

            # Phase 2: fine scan in a ±2-coarse-step window around the minimum.
            n_fine = n_steps - n_coarse
            fine_lo = max(g_lo, g_best_c - 2 * dg_c)
            fine_hi = min(g_hi, g_best_c + 2 * dg_c)
            dg_f = (fine_hi - fine_lo) / (n_fine - 1) if n_fine > 1 else 0.0

            def _fine(b):
                bv, w = flat_bv[b]
                return _scan_barrier(bv, w, obs_rates[b], fine_lo, dg_f, n_fine)

            fine_results = thread_map(_fine, barriers, threads)

            # Merge coarse + fine, sort by gamma value.
            gamma_c = [g_lo + k * dg_c for k in range(n_coarse)]
            gamma_f = [fine_lo + k * dg_f for k in range(n_fine)]
            gamma_all = gamma_c + gamma_f
            per_set_all = [[coarse_results[bi][k] for bi in range(len(barriers))] for k in range(n_coarse)] + \
                          [[fine_results[bi][k] for bi in range(len(barriers))] for k in range(n_fine)]
            order = list(np.argsort(gamma_all))
            gamma_grid = [gamma_all[i] for i in order]
            per_set_logk0 = [per_set_all[i] for i in order]
        else:
            # Uniform grid across [gammamin, gammamax].
            dgamma = (g_hi - g_lo) / (n_steps - 1) if n_steps > 1 else 0.0

            def _uniform(b):
                bv, w = flat_bv[b]
                return _scan_barrier(bv, w, obs_rates[b], g_lo, dgamma, n_steps)

            results = thread_map(_uniform, barriers, threads)
            gamma_grid = [g_lo + k * dgamma for k in range(n_steps)]
            per_set_logk0 = [[results[bi][k] for bi in range(len(barriers))] for k in range(n_steps)]

        mean_logk0 = [float(np.mean(row)) for row in per_set_logk0]
        var_logk0  = [float(np.var(row))  for row in per_set_logk0]
        best_index = int(np.argmin(var_logk0))
        return {
            "gamma_grid": gamma_grid,
            "per_set_ln_k0": per_set_logk0,
            "mean_ln_k0": mean_logk0,
            "var_ln_k0": var_logk0,
            "gamma_best": float(gamma_grid[best_index]),
            "logk0_best": float(mean_logk0[best_index]),
            "ln_k0_per_set_best": per_set_logk0[best_index],
        }

    def analyze_indices(indicess: list[list[int]], nsets: int | None = None, threads: int = 1) -> tuple[float, float, float | None, list[FloodingSetReport], dict[str, object]]:
        logk0_opesf = None
        opesf_times: list[float] = []
        opesf_event: list[bool] = []
        v_datas: dict[float, np.ndarray] = {}
        obs_rates: dict[float, float] = {}
        set_reports: list[FloodingSetReport] = []

        def analyze_set(index_and_barrier: tuple[int, float]):
            i, barrier = index_and_barrier
            barrier_add = 0 if args.nooffset else barrier
            data = [datas[i][j] for j in indicess[i]]
            event = np.array([events[i][j] for j in indicess[i]])
            max_biases = [np.max(traj[:, 1]) + barrier for traj in data]

            colvar_row_counts = np.sort([len(traj[:, 0]) for traj in data])
            max_index = colvar_row_counts[-1] if args.avgover < 0 else colvar_row_counts[int(abs(args.avgover) * np.sum(event))]
            min_index = 0 if args.avgover < 0 else colvar_row_counts[0]
            v_data = np.full((len(data), max_index - min_index), np.nan)
            for traj_index, traj in enumerate(data):
                v_data[traj_index, : (min(len(traj), max_index) - min_index)] = traj[min_index:max_index, 1] + barrier_add
            final_times = np.array([traj[-1, 0] for traj in data])
            opesf_times_local: list[float] = []
            opesf_event_local: list[bool] = []
            if args.opesf:
                rescaled_times = RM.iMetaD_rescaled_times(data, beta, bias_shift=barrier_add)
                opesf_times_local.extend(list(rescaled_times))
                opesf_event_local.extend(list(event))

            ecdfxs = np.sort(final_times)
            ecdfys = np.linspace(1 / len(event), 1, len(event))
            emp_rate = event.sum() / final_times.sum()
            if args.cdf:
                obs_rate = optimize.curve_fit(lambda t, k: 1 - np.exp(-k * t), ecdfxs[: event.sum()], ecdfys[: event.sum()], p0=emp_rate)[0][0]
                ks_stat, p = ks_1samp(ecdfxs[: event.sum()], lambda t: 1 - np.exp(-obs_rate * t))
            else:
                obs_rate = emp_rate
                ks_stat, p = ksc.ks_1samp_censored(final_times, event, lambda t: np.exp(-emp_rate * t))

            avg = np.mean(np.nanmean(np.exp(beta * v_data), axis=0))
            avg_work_kbt: float | None = None
            if args.wcol is not None:
                # Column 4 is the cumulative work (kJ/mol); average the final value across all trajectories.
                work_finals = np.array([float(traj[-1, 4]) for traj in data])
                avg_work_kbt = float(beta * np.mean(work_finals))
            report = FloodingSetReport(
                barrier=barrier,
                transitioned=int(event.sum()),
                total=len(data),
                avg_max_bias=float(np.mean(max_biases)),
                tau_obs=float(1 / obs_rate),
                k_obs=float(obs_rate),
                log_k_obs=float(np.log(obs_rate)),
                ks_stat=float(ks_stat),
                p_value=float(p),
                ln_exp_beta_v=float(np.log(avg)),
                avg_work_kbt=avg_work_kbt,
            )
            return barrier, v_data, float(obs_rate), report, opesf_times_local, opesf_event_local

        set_results = thread_map(analyze_set, list(enumerate(barriers)), threads)
        for barrier, v_data, obs_rate, report, opesf_times_local, opesf_event_local in set_results:
            v_datas[barrier] = v_data
            obs_rates[barrier] = obs_rate
            set_reports.append(report)
            if args.opesf:
                opesf_times.extend(opesf_times_local)
                opesf_event.extend(opesf_event_local)

        if args.opesf:
            logk0_opesf = np.log(RM.iMetaD_FitCDF_times(np.array(opesf_times), event=np.array(opesf_event)))

        beta_v_datas = {barrier: beta * v_data for barrier, v_data in v_datas.items()}
        flat_bv = {b: _prepare_flat(bv, axis_first) for b, bv in beta_v_datas.items()}

        # Sort barriers by ascending ln_acceleration so convergence analysis runs
        # from least- to most-biased sets.
        ln_accels = [
            float(np.log(float(np.dot(flat_bv[b][1], np.exp(flat_bv[b][0])))))
            for b in barriers
        ]
        sort_order = list(np.argsort(ln_accels))
        sorted_barriers = [barriers[i] for i in sort_order]

        # Convergence analysis: fit using the n lowest-alpha subsets (n = 3 … N).
        n_total = len(sorted_barriers)
        min_n_conv = min(3, n_total)
        conv_n: list[int] = []
        conv_gamma: list[float] = []
        conv_logk0: list[float] = []
        for n in range(min_n_conv, n_total + 1):
            subset = sorted_barriers[:n]

            def var_subset(gamma: float, sb: list = subset) -> float:
                lk = [
                    np.log(obs_rates[b]) - np.log(float(np.dot(flat_bv[b][1], np.exp(gamma * flat_bv[b][0]))))
                    for b in sb
                ]
                return float(np.var(lk)) if len(lk) > 1 else 0.0

            gn = float(optimize.minimize_scalar(var_subset, bounds=gamma_bounds, method="bounded").x)
            lk0s_n = [
                np.log(obs_rates[b]) - np.log(float(np.dot(flat_bv[b][1], np.exp(gn * flat_bv[b][0]))))
                for b in subset
            ]
            conv_n.append(n)
            conv_gamma.append(gn)
            conv_logk0.append(float(np.mean(lk0s_n)))

        # Determine which barriers to use for the final reported fit.
        selected_nsets = nsets if (nsets is not None and 1 <= nsets < n_total) else n_total
        fit_barriers = sorted_barriers[:selected_nsets]

        def variance(gamma: float) -> float:
            logk0s = []
            for barrier in fit_barriers:
                avg = float(np.dot(flat_bv[barrier][1], np.exp(gamma * flat_bv[barrier][0])))
                logk0s.append(np.log(obs_rates[barrier]) - np.log(avg))
            return np.var(logk0s)

        diagnostics = compute_diagnostics(flat_bv, obs_rates, threads=threads)
        gamma_best = optimize.minimize_scalar(variance, bounds=gamma_bounds, method="bounded").x
        logk0s = []
        for barrier in fit_barriers:
            avg = float(np.dot(flat_bv[barrier][1], np.exp(gamma_best * flat_bv[barrier][0])))
            logk0s.append(np.log(obs_rates[barrier]) - np.log(avg))
        logk0_best = np.mean(logk0s)
        diagnostics["gamma_best"] = float(gamma_best)
        diagnostics["logk0_best"] = float(logk0_best)
        diagnostics["ln_k0_per_set_best"] = [float(value) for value in logk0s]
        diagnostics["convergence_analysis"] = {
            "n_sets": conv_n,
            "gamma": conv_gamma,
            "logk0": conv_logk0,
            "sorted_barrier_labels": [float(sorted_barriers[i]) for i in range(n_total)],
            "selected_nsets": selected_nsets,
        }
        return logk0_best, gamma_best, logk0_opesf, set_reports, diagnostics

    if not args.bootstrap:
        logk0_best, gamma_best, logk0_opes, set_reports, diagnostics = analyze_indices([list(range(len(data))) for data in datas], args.nsets, threads=args.threads)
        result = FloodingAnalysisResult(beta=beta, logk0=float(logk0_best), gamma=float(gamma_best), opes_logk0=None if logk0_opes is None else float(logk0_opes), set_reports=set_reports, flooding_diagnostics=diagnostics)
        result.messages = format_flooding_result(result)
        return result

    sample_logk0 = []
    sample_gamma = []
    sample_opesf = []
    sample_set_log_k_obs: list[list[float]] = [[] for _ in barriers]
    sample_set_ln_exp_beta_v: list[list[float]] = [[] for _ in barriers]
    set_reports: list[FloodingSetReport] = []
    diagnostics: dict[str, object] | None = None
    iterations: list[int] = []
    def bootstrap_worker(i: int):
        rng = random.Random(None if args.seed is None else args.seed + i + 1)
        indicess = [rng.choices(list(range(len(data))), k=len(data)) for data in datas]
        return i, analyze_indices(indicess, args.nsets, threads=1)

    bootstrap_results = thread_map(bootstrap_worker, list(range(args.numboots)), args.threads)
    for i, (logk0, gamma, logk0_opesf, current_reports, current_diagnostics) in bootstrap_results:
        sample_logk0.append(logk0)
        sample_gamma.append(gamma)
        sample_opesf.append(logk0_opesf)
        set_reports = current_reports
        diagnostics = current_diagnostics
        iterations.append(i)
        for j, report in enumerate(current_reports):
            sample_set_log_k_obs[j].append(report.log_k_obs)
            sample_set_ln_exp_beta_v[j].append(report.ln_exp_beta_v)
    result = FloodingAnalysisResult(
        beta=beta,
        logk0=float(np.mean(sample_logk0)),
        gamma=float(np.mean(sample_gamma)),
        opes_logk0=None if sample_opesf[0] is None else float(np.mean(sample_opesf)),
        set_reports=set_reports,
        flooding_diagnostics=diagnostics,
        bootstrap_logk0_std=float(np.std(sample_logk0)),
        bootstrap_gamma_std=float(np.std(sample_gamma)),
        bootstrap_opes_logk0_std=None if sample_opesf[0] is None else float(np.std(sample_opesf)),
        bootstrap_per_set_log_k_obs_std=[float(np.std(s)) for s in sample_set_log_k_obs],
        bootstrap_per_set_ln_exp_beta_v_std=[float(np.std(s)) for s in sample_set_ln_exp_beta_v],
        bootstrap_iterations=iterations,
    )
    result.messages = format_flooding_result(result)
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = analyze(args)
    emit_messages(result, args.quiet)
    payload = result_payload(result, args.plot_time_unit)
    if args.bootstrap:
        payload["numboots"] = args.numboots
    write_results(args.output, payload)
    if not args.no_plots:
        prefix = args.plot_prefix if args.plot_prefix is not None else str(Path(args.output).with_suffix(""))
        plot_flooding_payload(
            payload,
            output_prefix=prefix,
            condition_label=args.condition_label,
            condition_unit=args.condition_unit,
            title_prefix=args.title_prefix,
            time_unit=args.plot_time_unit,
            truerate=args.truerate,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
