from __future__ import annotations

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import optimize

import ks_censored as ksc
import rate_methods_library as RM
ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = ROOT / "example-data" / "Ree_Data"
OUTPUT_ROOT = ROOT / "example-data" / "test_results"
MPL_CONFIG_DIR = ROOT / ".matplotlib-cache"
XDG_CACHE_DIR = ROOT / ".cache"

TIMEUNIT_SECONDS = 1e-12
MICROSECONDS_PER_SECOND = 1e6
TIMEUNIT_MICROSECONDS = TIMEUNIT_SECONDS * MICROSECONDS_PER_SECOND
TEMPERATURE_K = 312.0
BOOTSTRAP_RESAMPLES = 50
BOOTSTRAP_SEED = 20260501
LN_US_PER_S = np.log(MICROSECONDS_PER_SECOND)
DEFAULT_THREADS = max(1, int(os.environ.get("EATR_THREADS", "1")))


def ensure_output_root() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
    os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))
    os.environ.setdefault("MPLBACKEND", "Agg")


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def sorted_run_files(base_dir: Path, filename: str) -> list[str]:
    return [str(path) for path in sorted(base_dir.glob(f"run_*/{filename}"))]


def pace_to_ps(pace_steps: float) -> float:
    return pace_steps * 10.0 * 1e-3


def rate_s_to_us(rate_s: float) -> float:
    return rate_s / MICROSECONDS_PER_SECOND


def rate_std_s_to_us(rate_std_s: float) -> float:
    return rate_std_s / MICROSECONDS_PER_SECOND


def ln_rate_s_to_ln_us(ln_rate_s: float) -> float:
    return ln_rate_s - LN_US_PER_S


def percentile_interval(samples: np.ndarray, level: float = 95.0) -> tuple[float, float]:
    alpha = (100.0 - level) / 2.0
    return float(np.percentile(samples, alpha)), float(np.percentile(samples, 100.0 - alpha))


def thread_map(func, values, threads: int):
    if threads <= 1:
        return [func(value) for value in values]
    with ThreadPoolExecutor(max_workers=threads) as executor:
        return list(executor.map(func, values))


def bootstrap_index_sets(size: int, n_resamples: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, size, size=(n_resamples, size))


def build_prepared_data(data: list[np.ndarray], event: np.ndarray, bias_shift: float = 0.0) -> dict[str, object]:
    row_count = max(len(traj[:, 0]) for traj in data)
    dt = float(data[0][1, 0] - data[0][0, 0])
    time_grid = np.linspace(0.0, row_count * dt, row_count)
    v_data = np.full((len(data), row_count), np.nan)
    final_time_indices = np.array([len(traj) - 1 for traj in data], dtype=int)
    final_times = np.array([traj[-1, 0] for traj in data], dtype=float)
    for traj_index, traj in enumerate(data):
        v_data[traj_index, : len(traj)] = traj[:, 1] + bias_shift
    return {
        "data": data,
        "event": np.array(event, dtype=bool),
        "time_grid": time_grid,
        "v_data": v_data,
        "final_time_indices": final_time_indices,
        "final_times": final_times,
    }


def resample_prepared_data(prepared: dict[str, object], indices: np.ndarray) -> dict[str, object]:
    return {
        "data": None,
        "event": np.asarray(prepared["event"])[indices],
        "time_grid": prepared["time_grid"],
        "v_data": np.asarray(prepared["v_data"])[indices],
        "final_time_indices": np.asarray(prepared["final_time_indices"])[indices],
        "final_times": np.asarray(prepared["final_times"])[indices],
    }


def cumulative_trapezoid_grid(time_grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    increments = 0.5 * (values[:-1] + values[1:]) * np.diff(time_grid)
    return np.concatenate(([0.0], np.cumsum(increments)))


def eatr_log_average_exp(prepared: dict[str, object], beta: float, gamma: float) -> np.ndarray:
    mean_exp = np.nanmean(np.exp(beta * gamma * np.asarray(prepared["v_data"], dtype=float)), axis=0)
    return np.log(mean_exp)


def eatr_mle_from_prepared(prepared: dict[str, object], beta: float, gamma_bounds: tuple[float, float] = (0.0, 1.0)) -> dict[str, float | np.ndarray]:
    event = np.asarray(prepared["event"], dtype=bool)
    final_time_indices = np.asarray(prepared["final_time_indices"], dtype=int)
    time_grid = np.asarray(prepared["time_grid"], dtype=float)

    def objective(gamma: float) -> float:
        log_average_exp = eatr_log_average_exp(prepared, beta, gamma)
        cum_hazard_grid = cumulative_trapezoid_grid(time_grid, np.exp(log_average_exp))
        cum_hazard = cum_hazard_grid[final_time_indices]
        log_hazard = log_average_exp[final_time_indices]
        mean_t = cum_hazard.sum() / event.sum()
        log_l = -event.sum() * np.log(mean_t) + log_hazard[event].sum() - (1.0 / mean_t) * cum_hazard.sum()
        return -float(log_l)

    optimum = optimize.minimize_scalar(objective, bounds=gamma_bounds, method="bounded")
    gamma = float(optimum.x)
    log_average_exp = eatr_log_average_exp(prepared, beta, gamma)
    cum_hazard_grid = cumulative_trapezoid_grid(time_grid, np.exp(log_average_exp))
    cum_hazard = cum_hazard_grid[final_time_indices]
    k0 = float(event.sum() / cum_hazard.sum())
    return {
        "k0": k0,
        "gamma": gamma,
        "log_average_exp": np.column_stack((time_grid, log_average_exp)),
    }


def bootstrap_regular_eatr(prepared: dict[str, object], beta: float, n_resamples: int, rng: np.random.Generator, threads: int = DEFAULT_THREADS) -> dict[str, float]:
    index_sets = bootstrap_index_sets(len(np.asarray(prepared["event"])), n_resamples, rng)

    def worker(indices: np.ndarray) -> tuple[float, float]:
        resampled = resample_prepared_data(prepared, indices)
        fit = eatr_mle_from_prepared(resampled, beta)
        return float(np.log(fit["k0"])), float(fit["gamma"])

    results = thread_map(worker, index_sets, threads)
    sample_ln_k = np.array([result[0] for result in results], dtype=float)
    sample_gamma = np.array([result[1] for result in results], dtype=float)
    ln_k_ci = percentile_interval(sample_ln_k)
    gamma_ci = percentile_interval(sample_gamma)
    return {
        "n_resamples": int(n_resamples),
        "ln_k_std": float(np.std(sample_ln_k)),
        "ln_k_ci95_low": ln_k_ci[0],
        "ln_k_ci95_high": ln_k_ci[1],
        "gamma_std": float(np.std(sample_gamma)),
        "gamma_ci95_low": gamma_ci[0],
        "gamma_ci95_high": gamma_ci[1],
    }


def run_regular_wt_eatr() -> dict[str, object]:
    wt_root = EXAMPLE_ROOT / "E_end_end_distance_wt"
    pace_dirs = sorted(wt_root.glob("eruns_pace*"), key=lambda path: float(path.name.split("pace")[1]))
    summaries: list[dict[str, float | str]] = []
    beta = beta_value()
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for pace_dir in pace_dirs:
        pace_steps = float(pace_dir.name.split("pace")[1])
        data = RM.get_data(sorted_run_files(pace_dir, "metad.colvar"), 0, 2, acc_col=4, time_scale_factor=TIMEUNIT_SECONDS)
        event = RM.get_event(data, log_files=sorted_run_files(pace_dir, "p.log"), quiet=True)
        prepared = build_prepared_data(data, event, bias_shift=0.0)
        mle_fit = eatr_mle_from_prepared(prepared, beta)
        mle_rate = float(mle_fit["k0"])
        mle_gamma = float(mle_fit["gamma"])
        log_average_exp_mle = mle_fit["log_average_exp"]
        final_time_indices = np.asarray(prepared["final_time_indices"], dtype=int)
        mle_ks_stat, mle_p_value = ks_censored_ks(final_time_indices[event], np.exp(np.log(mle_rate)), log_average_exp_mle)
        mle_bootstrap = bootstrap_regular_eatr(prepared, beta, BOOTSTRAP_RESAMPLES, rng, threads=DEFAULT_THREADS)

        cdf_ln_k = math.nan
        cdf_gamma = math.nan
        cdf_ks_stat = math.nan
        cdf_p_value = math.nan
        cdf_status = "ok"
        try:
            cdf_rate, cdf_gamma_value = RM.EATR_CDF_rate(data, beta, event=event, k_bounds=(1e-30, np.inf), gamma_bounds=(0.0, 1.0), cores=1, logTrick=False, do_bopt=False, bias_shift=0.0)
            cdf_ln_k = float(np.log(cdf_rate))
            cdf_gamma = float(cdf_gamma_value)
            log_average_exp_cdf = RM.avg_exponential(data, beta, cdf_gamma_value, bias_shift=0.0)
            cdf_ks_stat, cdf_p_value = ks_censored_ks(final_time_indices[event], cdf_rate, log_average_exp_cdf)
        except Exception as exc:
            cdf_status = f"failed: {type(exc).__name__}"

        summaries.append(
            {
                "set": pace_dir.name,
                "pace_steps": pace_steps,
                "pace_ps": pace_to_ps(pace_steps),
                "transitioned": int(event.sum()),
                "total": len(event),
                "eatr_mle_ln_k": float(ln_rate_s_to_ln_us(np.log(mle_rate))),
                "eatr_mle_gamma": float(mle_gamma),
                "eatr_cdf_ln_k": float(ln_rate_s_to_ln_us(cdf_ln_k)) if np.isfinite(cdf_ln_k) else cdf_ln_k,
                "eatr_cdf_gamma": cdf_gamma,
                "eatr_mle_ks_stat": float(mle_ks_stat),
                "eatr_mle_p_value": float(mle_p_value),
                "eatr_mle_bootstrap_n": int(mle_bootstrap["n_resamples"]),
                "eatr_mle_ln_k_std": float(mle_bootstrap["ln_k_std"]),
                "eatr_mle_ln_k_ci95_low": float(ln_rate_s_to_ln_us(mle_bootstrap["ln_k_ci95_low"])),
                "eatr_mle_ln_k_ci95_high": float(ln_rate_s_to_ln_us(mle_bootstrap["ln_k_ci95_high"])),
                "eatr_mle_gamma_std": float(mle_bootstrap["gamma_std"]),
                "eatr_mle_gamma_ci95_low": float(mle_bootstrap["gamma_ci95_low"]),
                "eatr_mle_gamma_ci95_high": float(mle_bootstrap["gamma_ci95_high"]),
                "eatr_cdf_ks_stat": cdf_ks_stat,
                "eatr_cdf_p_value": cdf_p_value,
                "eatr_cdf_status": cdf_status,
            }
        )

    output = {
        "temperature_K": TEMPERATURE_K,
        "timeunit_seconds": TIMEUNIT_SECONDS,
        "timeunit_microseconds": TIMEUNIT_MICROSECONDS,
        "rate_unit": "us^-1",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "sets": summaries,
    }
    with open(OUTPUT_ROOT / "wt_regular_eatr_summary.json", "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    plot_regular_eatr(summaries)
    return output


def plot_regular_eatr(summaries: list[dict[str, float | str]]) -> None:
    plt = pyplot()
    pace_ps = np.array([entry["pace_ps"] for entry in summaries], dtype=float)
    ln_k_mle = np.array([entry["eatr_mle_ln_k"] for entry in summaries], dtype=float)
    ln_k_cdf = np.array([entry["eatr_cdf_ln_k"] for entry in summaries], dtype=float)
    gamma_mle = np.array([entry["eatr_mle_gamma"] for entry in summaries], dtype=float)
    gamma_cdf = np.array([entry["eatr_cdf_gamma"] for entry in summaries], dtype=float)
    ln_k_mle_err = np.array([entry["eatr_mle_ln_k_std"] for entry in summaries], dtype=float)
    gamma_mle_err = np.array([entry["eatr_mle_gamma_std"] for entry in summaries], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].errorbar(pace_ps, ln_k_mle, yerr=ln_k_mle_err, marker="o", capsize=3, label="EATR MLE bootstrap σ")
    axes[0].plot(pace_ps, ln_k_cdf, marker="s", label="EATR CDF")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("MetaD hill deposition pace (ps)")
    axes[0].set_ylabel(r"Estimated ln($k_0$ / us$^{-1}$)")
    axes[0].set_title("WT-MetaD Regular EATR")
    axes[0].legend()

    axes[1].errorbar(pace_ps, gamma_mle, yerr=gamma_mle_err, marker="o", capsize=3, label="EATR MLE bootstrap σ")
    axes[1].plot(pace_ps, gamma_cdf, marker="s", label="EATR CDF")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("MetaD hill deposition pace (ps)")
    axes[1].set_ylabel("Estimated γ")
    axes[1].set_title("WT-MetaD Biasing Efficiency")
    axes[1].legend()

    fig.savefig(OUTPUT_ROOT / "wt_regular_eatr_vs_pace.png", dpi=200)
    plt.close(fig)


def ks_censored_ks(final_time_indices: np.ndarray, rate: float, log_average_exp: np.ndarray) -> tuple[float, float]:
    return ksc.ks_1samp_censored(
        final_time_indices,
        np.array([True for _ in final_time_indices]),
        lambda idx: np.exp(-rate * RM.EATR_calculate_cum_hazard(log_average_exp, False, idx)),
    )


def load_flooding_set(colvar_files: list[str], log_files: list[str], bias_col: int, acc_col: int | None, bias_shift: float, cdf_fit: bool) -> dict[str, object]:
    data = RM.get_data(colvar_files, 0, bias_col, acc_col=acc_col, time_scale_factor=TIMEUNIT_SECONDS)
    event = RM.get_event(data, log_files=log_files, quiet=True)
    prepared = build_prepared_data(data, event, bias_shift=bias_shift)
    final_times = np.asarray(prepared["final_times"], dtype=float)
    obs_rate = float(event.sum() / final_times.sum())
    if cdf_fit:
        ecdfxs = np.sort(final_times)
        ecdfys = np.linspace(1 / len(event), 1, len(event))
        obs_rate = float(optimize.curve_fit(lambda t, k: 1 - np.exp(-k * t), ecdfxs[: event.sum()], ecdfys[: event.sum()], p0=obs_rate)[0][0])
    ks_stat, p_value = ksc.ks_1samp_censored(final_times, event, lambda t: np.exp(-obs_rate * t))

    return {
        "data": data,
        "event": event,
        "obs_rate": obs_rate,
        "log_obs_rate": float(np.log(obs_rate)),
        "ks_stat": float(ks_stat),
        "p_value": float(p_value),
        "v_data": prepared["v_data"],
        "time_grid": prepared["time_grid"],
        "final_times": prepared["final_times"],
        "final_time_indices": prepared["final_time_indices"],
        "avg_bias_gamma1": float(np.log(np.mean(np.nanmean(np.exp(beta_value() * np.asarray(prepared["v_data"], dtype=float)), axis=0)))),
        "avg_acceleration_factor_gamma1": float(np.exp(np.log(np.mean(np.nanmean(np.exp(beta_value() * np.asarray(prepared["v_data"], dtype=float)), axis=0))))),
    }


def beta_value() -> float:
    return 1.0 / (8.314462618e-3 * TEMPERATURE_K)


def log_mean_exp(values: np.ndarray, axis: int) -> np.ndarray:
    max_values = np.nanmax(values, axis=axis, keepdims=True)
    shifted = np.exp(values - max_values)
    return np.squeeze(max_values, axis=axis) + np.log(np.nanmean(shifted, axis=axis))


def flooding_log_average(spec: dict[str, object], gamma: float, beta: float) -> float:
    scaled_bias = beta * gamma * np.asarray(spec["v_data"], dtype=float)
    log_mean_over_runs = log_mean_exp(scaled_bias, axis=0)
    return float(log_mean_exp(log_mean_over_runs, axis=0))


def flooding_ln_k0_by_set(set_specs: list[dict[str, object]], gamma: float, beta: float, axis_first: int = 0) -> list[float]:
    ln_k0s = []
    for spec in set_specs:
        ln_avg = flooding_log_average(spec, gamma, beta)
        ln_k0s.append(float(spec["log_obs_rate"] - ln_avg))
    return ln_k0s


def flooding_best_fit(set_specs: list[dict[str, object]], axis_first: int = 0) -> dict[str, object]:
    beta = beta_value()
    objective = lambda gamma: np.var(flooding_ln_k0_by_set(set_specs, float(gamma), beta, axis_first=axis_first))
    optimum = optimize.minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
    gamma_best = float(optimum.x)
    ln_k0_per_set_best = flooding_ln_k0_by_set(set_specs, gamma_best, beta, axis_first=axis_first)
    logk0_best = float(np.mean(ln_k0_per_set_best))
    return {
        "gamma_best": gamma_best,
        "logk0_best": logk0_best,
        "ln_k0_per_set_best": ln_k0_per_set_best,
    }


def flooding_diagnostics(set_specs: list[dict[str, object]], axis_first: int = 0, threads: int = DEFAULT_THREADS) -> dict[str, object]:
    beta = beta_value()
    gamma_grid = np.linspace(0.0, 1.0, 401)
    def gamma_worker(gamma: float) -> tuple[list[float], float, float]:
        ln_k0s = flooding_ln_k0_by_set(set_specs, float(gamma), beta, axis_first=axis_first)
        return ln_k0s, float(np.mean(ln_k0s)), float(np.var(ln_k0s))

    gamma_results = thread_map(gamma_worker, gamma_grid, threads)
    per_set_ln_k0 = [result[0] for result in gamma_results]
    mean_ln_k0 = [result[1] for result in gamma_results]
    var_ln_k0 = [result[2] for result in gamma_results]

    best_fit = flooding_best_fit(set_specs, axis_first=axis_first)

    return {
        "gamma_grid": gamma_grid.tolist(),
        "per_set_ln_k0": per_set_ln_k0,
        "mean_ln_k0": mean_ln_k0,
        "var_ln_k0": var_ln_k0,
        "gamma_best": best_fit["gamma_best"],
        "logk0_best": best_fit["logk0_best"],
        "ln_k0_per_set_best": best_fit["ln_k0_per_set_best"],
    }


def bootstrap_flooding(set_specs: list[dict[str, object]], n_resamples: int, rng: np.random.Generator, threads: int = DEFAULT_THREADS) -> dict[str, object]:
    obs_rate_samples = np.full((n_resamples, len(set_specs)), np.nan)
    index_sets = [bootstrap_index_sets(len(np.asarray(spec["event"])), n_resamples, rng) for spec in set_specs]

    def bootstrap_worker(bootstrap_index: int) -> tuple[float, float, np.ndarray]:
        resampled_specs = []
        obs_rates = np.full(len(set_specs), np.nan)
        for set_index, spec in enumerate(set_specs):
            indices = index_sets[set_index][bootstrap_index]
            event = np.asarray(spec["event"], dtype=bool)[indices]
            final_times = np.asarray(spec["final_times"], dtype=float)[indices]
            obs_rate = float(event.sum() / final_times.sum())
            obs_rates[set_index] = obs_rate
            resampled_specs.append(
                {
                    "obs_rate": obs_rate,
                    "log_obs_rate": float(np.log(obs_rate)),
                    "v_data": np.asarray(spec["v_data"], dtype=float)[indices],
                }
            )
        best_fit = flooding_best_fit(resampled_specs)
        return float(best_fit["gamma_best"]), float(best_fit["logk0_best"]), obs_rates

    bootstrap_results = thread_map(bootstrap_worker, range(n_resamples), threads)
    gamma_samples_array = np.array([result[0] for result in bootstrap_results], dtype=float)
    logk0_samples_array = np.array([result[1] for result in bootstrap_results], dtype=float)
    for bootstrap_index, result in enumerate(bootstrap_results):
        obs_rate_samples[bootstrap_index, :] = result[2]
    gamma_ci = percentile_interval(gamma_samples_array)
    logk0_ci = percentile_interval(logk0_samples_array)
    per_set = []
    for set_index, spec in enumerate(set_specs):
        obs_ci = percentile_interval(obs_rate_samples[:, set_index])
        per_set.append(
            {
                "set": spec["label"],
                "obs_rate_std": float(np.std(obs_rate_samples[:, set_index])),
                "obs_rate_ci95_low": obs_ci[0],
                "obs_rate_ci95_high": obs_ci[1],
            }
        )
    return {
        "n_resamples": int(n_resamples),
        "gamma_std": float(np.std(gamma_samples_array)),
        "gamma_ci95_low": gamma_ci[0],
        "gamma_ci95_high": gamma_ci[1],
        "logk0_std": float(np.std(logk0_samples_array)),
        "logk0_ci95_low": logk0_ci[0],
        "logk0_ci95_high": logk0_ci[1],
        "per_set": per_set,
    }


def save_flooding_plot(title: str, diagnostics: dict[str, object], set_labels: list[str], output_name: str, bootstrap_stats: dict[str, object] | None = None, reference_lines: dict[str, float] | None = None, truerate: float | None = None) -> None:
    plt = pyplot()
    gamma_grid = np.array(diagnostics["gamma_grid"], dtype=float)
    per_set = np.array(diagnostics["per_set_ln_k0"], dtype=float)
    mean_ln_k0 = np.array(diagnostics["mean_ln_k0"], dtype=float)
    var_ln_k0 = np.array(diagnostics["var_ln_k0"], dtype=float)
    gamma_best = float(diagnostics["gamma_best"])
    logk0_best = float(ln_rate_s_to_ln_us(diagnostics["logk0_best"]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    for idx, label in enumerate(set_labels):
        axes[0].plot(gamma_grid, per_set[:, idx], label=label)
    axes[0].set_xlabel("γ")
    axes[0].set_ylabel(r"Predicted ln($k_0$ / us$^{-1}$)")
    axes[0].set_title(f"{title}: ln k0 by set")
    axes[0].legend(fontsize=8, ncol=2)

    std_ln_k0 = np.sqrt(var_ln_k0)
    axes[1].plot(gamma_grid, mean_ln_k0, color="black")
    axes[1].fill_between(gamma_grid, mean_ln_k0 - std_ln_k0, mean_ln_k0 + std_ln_k0, color="tab:blue", alpha=0.2)
    axes[1].axvline(gamma_best, color="tab:red", linestyle="--", label=f"γ*={gamma_best:.3f}")
    axes[1].axhline(logk0_best, color="tab:green", linestyle="--", label=f"ln k0*={logk0_best:.3f}")
    if reference_lines:
        for name, value in reference_lines.items():
            axes[1].axhline(value, linestyle=":", label=f"{name}={value:.3f}")
    if truerate is not None:
        axes[1].axhline(truerate, linestyle=":", color="black", label=f"reference ln(k0)={truerate:.3f}")
    axes[1].set_xlabel("γ")
    axes[1].set_ylabel(r"Mean ln($k_0$ / us$^{-1}$)")
    axes[1].set_title(f"{title}: mean ± std")
    axes[1].legend(fontsize=8)

    axes[2].plot(gamma_grid, var_ln_k0, color="tab:purple")
    axes[2].axvline(gamma_best, color="tab:red", linestyle="--")
    axes[2].set_xlabel("γ")
    axes[2].set_ylabel("Var[ln k0]")
    axes[2].set_title(f"{title}: variance")

    if bootstrap_stats is not None:
        fig.suptitle(
            f"bootstrap σ(γ*)={float(bootstrap_stats['gamma_std']):.3f}, "
            f"σ(ln k0*)={float(bootstrap_stats['logk0_std']):.3f}",
            fontsize=10,
            y=1.02,
        )

    fig.savefig(OUTPUT_ROOT / output_name, dpi=220)
    plt.close(fig)


def plot_opes_observed_rate_vs_barrier(set_specs: list[dict[str, object]], bootstrap_stats: dict[str, object]) -> None:
    plt = pyplot()
    barriers = np.array([spec["barrier_kj_mol"] for spec in set_specs], dtype=float)
    obs_rate = np.array([rate_s_to_us(spec["obs_rate"]) for spec in set_specs], dtype=float)
    ln_obs_rate = np.log(obs_rate)
    ln_obs_rate_err = np.array([rate_std_s_to_us(entry["obs_rate_std"]) / rate_s_to_us(spec["obs_rate"]) for spec, entry in zip(set_specs, bootstrap_stats["per_set"])], dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    ax.errorbar(barriers, ln_obs_rate, yerr=ln_obs_rate_err, marker="o", capsize=3)
    ax.set_xlabel(r"OPES barrier (kJ mol$^{-1}$)")
    ax.set_ylabel(r"Observed ln($k_{\mathrm{obs}}$ / us$^{-1}$)")
    ax.set_title("OPES ln observed rate by barrier")
    fig.savefig(OUTPUT_ROOT / "opes_observed_rate_vs_barrier.png", dpi=220)
    plt.close(fig)


def plot_ln_kobs_vs_acceleration(
    set_specs: list[dict[str, object]],
    diagnostics: dict[str, object],
    bootstrap_stats: dict[str, object],
    point_labels: list[str],
    series_label: str,
    title: str,
    output_name: str,
) -> None:
    plt = pyplot()
    ln_acceleration = np.array([spec["avg_bias_gamma1"] for spec in set_specs], dtype=float)
    ln_kobs_us = np.log(np.array([rate_s_to_us(spec["obs_rate"]) for spec in set_specs], dtype=float))
    ln_kobs_err = np.array(
        [rate_std_s_to_us(entry["obs_rate_std"]) / rate_s_to_us(spec["obs_rate"]) for spec, entry in zip(set_specs, bootstrap_stats["per_set"])],
        dtype=float,
    )
    gamma_best = float(diagnostics["gamma_best"])
    logk0_best = float(ln_rate_s_to_ln_us(diagnostics["logk0_best"]))
    x_fit = np.linspace(float(np.min(ln_acceleration)) * 0.98, float(np.max(ln_acceleration)) * 1.02, 200)
    y_fit = logk0_best + gamma_best * x_fit

    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    ax.errorbar(ln_acceleration, ln_kobs_us, yerr=ln_kobs_err, marker="o", linestyle="none", capsize=3, label=series_label)
    ax.plot(x_fit, y_fit, color="tab:red", label=fr"fit: ln($k_{{obs}}$) = ln($k_0$) + $\gamma$ ln($\alpha$)")
    for point_label, x_value, y_value in zip(point_labels, ln_acceleration, ln_kobs_us):
        ax.annotate(point_label, (x_value, y_value), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel(r"ln acceleration factor, ln($\alpha$)")
    ax.set_ylabel(r"ln($k_{\mathrm{obs}}$ / us$^{-1}$)")
    ax.set_title(title)
    ax.text(
        0.03,
        0.97,
        f"slope (gamma) = {gamma_best:.3f}\nintercept ln(k0 / us^-1) = {logk0_best:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )
    ax.legend()
    fig.savefig(OUTPUT_ROOT / output_name, dpi=220)
    plt.close(fig)


def plot_opes_ln_kobs_vs_acceleration(set_specs: list[dict[str, object]], diagnostics: dict[str, object], bootstrap_stats: dict[str, object]) -> None:
    plot_ln_kobs_vs_acceleration(
        set_specs,
        diagnostics,
        bootstrap_stats,
        [str(spec["barrier_kj_mol"]) for spec in set_specs],
        "OPES sets",
        "OPES slope-style rate scaling",
        "opes_ln_kobs_vs_acceleration.png",
    )


def run_opes_flooding() -> dict[str, object]:
    opes_root = EXAMPLE_ROOT / "E_end_end_distance_opes"
    set_specs = []
    labels = []
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    for path in sorted(opes_root.glob("eruns_barr*"), key=lambda item: float(item.name.split("barr")[1])):
        barrier = float(path.name.split("barr")[1])
        set_info = load_flooding_set(sorted_run_files(path, "opes_short.colvar"), sorted_run_files(path, "p.log"), bias_col=4, acc_col=None, bias_shift=barrier, cdf_fit=False)
        set_info["label"] = path.name
        set_info["barrier_kj_mol"] = barrier
        set_specs.append(set_info)
        labels.append(path.name)

    diagnostics = flooding_diagnostics(set_specs, threads=DEFAULT_THREADS)
    bootstrap_stats = bootstrap_flooding(set_specs, BOOTSTRAP_RESAMPLES, rng, threads=DEFAULT_THREADS)
    per_set_bootstrap = {entry["set"]: entry for entry in bootstrap_stats["per_set"]}
    output = {
        "temperature_K": TEMPERATURE_K,
        "timeunit_seconds": TIMEUNIT_SECONDS,
        "timeunit_microseconds": TIMEUNIT_MICROSECONDS,
        "rate_unit": "us^-1",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "sets": [
            {
                "set": spec["label"],
                "barrier_kj_mol": spec["barrier_kj_mol"],
                "obs_rate": rate_s_to_us(spec["obs_rate"]),
                "obs_rate_std": rate_std_s_to_us(per_set_bootstrap[spec["label"]]["obs_rate_std"]),
                "obs_rate_ci95_low": rate_s_to_us(per_set_bootstrap[spec["label"]]["obs_rate_ci95_low"]),
                "obs_rate_ci95_high": rate_s_to_us(per_set_bootstrap[spec["label"]]["obs_rate_ci95_high"]),
                "ks_stat": spec["ks_stat"],
                "p_value": spec["p_value"],
                "ln_avg_exp_beta_v_gamma1": spec["avg_bias_gamma1"],
                "avg_acceleration_factor_gamma1": spec["avg_acceleration_factor_gamma1"],
            }
            for spec in set_specs
        ],
        "flooding_fit": {
            **diagnostics,
            "logk0_best": ln_rate_s_to_ln_us(float(diagnostics["logk0_best"])),
            "ln_k0_per_set_best": [ln_rate_s_to_ln_us(float(value)) for value in diagnostics["ln_k0_per_set_best"]],
            "mean_ln_k0": [ln_rate_s_to_ln_us(float(value)) for value in diagnostics["mean_ln_k0"]],
        },
        "flooding_fit_bootstrap": {
            **bootstrap_stats,
            "logk0_ci95_low": ln_rate_s_to_ln_us(float(bootstrap_stats["logk0_ci95_low"])),
            "logk0_ci95_high": ln_rate_s_to_ln_us(float(bootstrap_stats["logk0_ci95_high"])),
        },
    }
    with open(OUTPUT_ROOT / "opes_flooding_summary.json", "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    save_flooding_plot("OPES EATR-flooding", diagnostics, labels, "opes_flooding_diagnostics.png", bootstrap_stats=bootstrap_stats, truerate=0.35967608559103206)
    plot_opes_observed_rate_vs_barrier(set_specs, bootstrap_stats)
    plot_opes_ln_kobs_vs_acceleration(set_specs, diagnostics, bootstrap_stats)
    return output


def run_wt_flooding() -> dict[str, object]:
    wt_root = EXAMPLE_ROOT / "E_end_end_distance_wt"
    pace_dirs = sorted(wt_root.glob("eruns_pace*"), key=lambda path: float(path.name.split("pace")[1]))
    set_specs = []
    labels = []
    rng = np.random.default_rng(BOOTSTRAP_SEED + 2)
    for pace_dir in pace_dirs:
        pace_steps = float(pace_dir.name.split("pace")[1])
        set_info = load_flooding_set(sorted_run_files(pace_dir, "metad.colvar"), sorted_run_files(pace_dir, "p.log"), bias_col=2, acc_col=4, bias_shift=0.0, cdf_fit=False)
        set_info["label"] = pace_dir.name
        set_info["pace_steps"] = pace_steps
        set_info["pace_ps"] = pace_to_ps(pace_steps)
        set_specs.append(set_info)
        labels.append(pace_dir.name)

    diagnostics_all = flooding_diagnostics(set_specs, threads=DEFAULT_THREADS)
    filtered_specs = [spec for spec in set_specs if float(spec["pace_ps"]) >= 100.0]
    filtered_labels = [spec["label"] for spec in filtered_specs]
    diagnostics_filtered = flooding_diagnostics(filtered_specs, threads=DEFAULT_THREADS)
    bootstrap_all = bootstrap_flooding(set_specs, BOOTSTRAP_RESAMPLES, rng, threads=DEFAULT_THREADS)
    bootstrap_filtered = bootstrap_flooding(filtered_specs, BOOTSTRAP_RESAMPLES, rng, threads=DEFAULT_THREADS)
    per_set_bootstrap = {entry["set"]: entry for entry in bootstrap_all["per_set"]}

    output = {
        "temperature_K": TEMPERATURE_K,
        "timeunit_seconds": TIMEUNIT_SECONDS,
        "timeunit_microseconds": TIMEUNIT_MICROSECONDS,
        "rate_unit": "us^-1",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "sets": [
            {
                "set": spec["label"],
                "pace_steps": spec["pace_steps"],
                "pace_ps": spec["pace_ps"],
                "obs_rate": rate_s_to_us(spec["obs_rate"]),
                "obs_rate_std": rate_std_s_to_us(per_set_bootstrap[spec["label"]]["obs_rate_std"]),
                "obs_rate_ci95_low": rate_s_to_us(per_set_bootstrap[spec["label"]]["obs_rate_ci95_low"]),
                "obs_rate_ci95_high": rate_s_to_us(per_set_bootstrap[spec["label"]]["obs_rate_ci95_high"]),
                "ks_stat": spec["ks_stat"],
                "p_value": spec["p_value"],
                "ln_avg_exp_beta_v_gamma1": spec["avg_bias_gamma1"],
                "avg_acceleration_factor_gamma1": spec["avg_acceleration_factor_gamma1"],
            }
            for spec in set_specs
        ],
        "flooding_fit_all_sets": {
            **diagnostics_all,
            "logk0_best": ln_rate_s_to_ln_us(float(diagnostics_all["logk0_best"])),
            "ln_k0_per_set_best": [ln_rate_s_to_ln_us(float(value)) for value in diagnostics_all["ln_k0_per_set_best"]],
            "mean_ln_k0": [ln_rate_s_to_ln_us(float(value)) for value in diagnostics_all["mean_ln_k0"]],
        },
        "flooding_fit_filtered_pace_ge_100ps": {
            **diagnostics_filtered,
            "logk0_best": ln_rate_s_to_ln_us(float(diagnostics_filtered["logk0_best"])),
            "ln_k0_per_set_best": [ln_rate_s_to_ln_us(float(value)) for value in diagnostics_filtered["ln_k0_per_set_best"]],
            "mean_ln_k0": [ln_rate_s_to_ln_us(float(value)) for value in diagnostics_filtered["mean_ln_k0"]],
        },
        "flooding_fit_all_sets_bootstrap": {
            **bootstrap_all,
            "logk0_ci95_low": ln_rate_s_to_ln_us(float(bootstrap_all["logk0_ci95_low"])),
            "logk0_ci95_high": ln_rate_s_to_ln_us(float(bootstrap_all["logk0_ci95_high"])),
        },
        "flooding_fit_filtered_pace_ge_100ps_bootstrap": {
            **bootstrap_filtered,
            "logk0_ci95_low": ln_rate_s_to_ln_us(float(bootstrap_filtered["logk0_ci95_low"])),
            "logk0_ci95_high": ln_rate_s_to_ln_us(float(bootstrap_filtered["logk0_ci95_high"])),
        },
    }
    with open(OUTPUT_ROOT / "wt_flooding_summary.json", "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    save_flooding_plot("WT-MetaD EATR-flooding (all paces)", diagnostics_all, labels, "wt_flooding_all_paces.png", bootstrap_stats=bootstrap_all, truerate=0.35967608559103206)
    save_flooding_plot("WT-MetaD EATR-flooding (pace ≥ 100 ps)", diagnostics_filtered, filtered_labels, "wt_flooding_filtered_paces.png", bootstrap_stats=bootstrap_filtered, truerate=0.35967608559103206)
    plot_wt_observed_rate_vs_pace(set_specs, diagnostics_filtered, bootstrap_all)
    plot_wt_ln_kobs_vs_acceleration(set_specs, diagnostics_all, bootstrap_all)
    return output


def plot_wt_observed_rate_vs_pace(set_specs: list[dict[str, object]], diagnostics_filtered: dict[str, object], bootstrap_stats: dict[str, object]) -> None:
    plt = pyplot()
    pace_ps = np.array([spec["pace_ps"] for spec in set_specs], dtype=float)
    obs_rate = np.array([rate_s_to_us(spec["obs_rate"]) for spec in set_specs], dtype=float)
    ln_obs_rate = np.log(obs_rate)
    ln_obs_rate_err = np.array([rate_std_s_to_us(entry["obs_rate_std"]) / rate_s_to_us(spec["obs_rate"]) for spec, entry in zip(set_specs, bootstrap_stats["per_set"])], dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    ax.errorbar(pace_ps, ln_obs_rate, yerr=ln_obs_rate_err, marker="o", capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("MetaD hill deposition pace (ps)")
    ax.set_ylabel(r"Observed ln($k_{\mathrm{obs}}$ / us$^{-1}$)")
    ax.set_title("WT-MetaD ln observed rate by pace")
    ax.axhline(float(ln_rate_s_to_ln_us(diagnostics_filtered["logk0_best"])), color="tab:red", linestyle="--", label="Filtered flooding ln k0*")
    ax.legend()
    fig.savefig(OUTPUT_ROOT / "wt_observed_rate_vs_pace.png", dpi=220)
    plt.close(fig)


def plot_wt_ln_kobs_vs_acceleration(set_specs: list[dict[str, object]], diagnostics: dict[str, object], bootstrap_stats: dict[str, object]) -> None:
    plot_ln_kobs_vs_acceleration(
        set_specs,
        diagnostics,
        bootstrap_stats,
        [spec["label"].replace("eruns_pace", "") for spec in set_specs],
        "WT pace sets",
        "WT-MetaD slope-style rate scaling",
        "wt_ln_kobs_vs_acceleration.png",
    )


def main() -> None:
    ensure_output_root()
    regular = run_regular_wt_eatr()
    opes = run_opes_flooding()
    wt_flooding = run_wt_flooding()

    manifest = {
        "generated_files": sorted(path.name for path in OUTPUT_ROOT.iterdir()),
        "notes": [
            "Protein G Ree example trajectories are written in PLUMED picoseconds, so times were converted with 1e-12 s/ps.",
            "Reported rate constants and observed rates in these example outputs are converted to us^-1.",
            "WT flooding analysis is reported for all pace sets and for a manuscript-style filtered subset with pace >= 100 ps.",
            f"Bootstrap uncertainties use {BOOTSTRAP_RESAMPLES} trajectory-resampling replicas per analysis.",
            f"Optional multithreading is controlled with EATR_THREADS; current default is {DEFAULT_THREADS}.",
        ],
        "wt_regular_summary": "wt_regular_eatr_summary.json",
        "opes_flooding_summary": "opes_flooding_summary.json",
        "wt_flooding_summary": "wt_flooding_summary.json",
    }
    with open(OUTPUT_ROOT / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
