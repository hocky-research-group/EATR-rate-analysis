# Script adapted from Palacio-Rodriguez et al. at https://github.com/kpalaciorodr/KTR/tree/master from J. Phys. Chem. Lett. 2022, 13, 32, 7490-7496.

import numpy as np

if hasattr(np, 'trapezoid'):
    trapezoid = np.trapezoid
else:
    trapezoid = np.trapz

import sys
from scipy import optimize
from scipy.integrate import cumulative_trapezoid
import warnings
import multiprocessing as mp
from functools import partial

bopt_avail = False
try:
    from bayes_opt import BayesianOptimization as bopt
    from bayes_opt import acquisition
    bopt_avail = True
except:
    bopt_avail = False

boots_avail = False
try:
    from scipy.stats import bootstrap as bootstr
    boots_avail = True
except:
    boots_avail = False

warnings.filterwarnings('ignore')


def ecdf_sigma(ecdfy, weights="none"):
    """Per-point sigma for a CDF least-squares fit against an empirical CDF.

    ``"none"`` (default) returns None, i.e. an unweighted fit -- the historical
    behaviour. ``"binomial"`` returns sqrt(F(1-F)), the sampling standard
    deviation of an empirical CDF value, so the fit weights each quantile by how
    well determined it is instead of treating all quantiles equally. The
    variance is clipped away from zero so the extreme quantiles, where
    F(1-F) -> 0, cannot dominate.
    """
    if weights in (None, "none"):
        return None
    if weights != "binomial":
        raise ValueError(f"unknown cdf weights {weights!r} (expected 'none' or 'binomial')")
    y = np.asarray(ecdfy, dtype=float)
    return np.sqrt(np.clip(y * (1.0 - y), 1e-3, None))


def _curve_fit_lenient(func, x, y, p0, bounds, max_nfev=None, sigma=None):
    """Like curve_fit but returns (popt, converged: bool) and never raises.

    - OptimizeWarning (max iterations reached): returns params + converged=False.
    - RuntimeError (complete failure): falls back to least_squares for the best
      iterate, returns it + converged=False.

    ``sigma`` is passed through to curve_fit (per-point standard deviations, so
    residuals are divided by it); None keeps the fit unweighted.
    """
    from scipy.optimize import OptimizeWarning, least_squares as _ls
    # kwargs are shared by curve_fit and the least_squares fallback, so sigma
    # (curve_fit-only) is applied separately -- in the fallback by scaling the
    # residuals, which is exactly what curve_fit's sigma does.
    kwargs = {} if max_nfev is None else {"max_nfev": max_nfev}
    fit_kwargs = dict(kwargs)
    if sigma is not None:
        fit_kwargs["sigma"] = sigma
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            popt = optimize.curve_fit(func, x, y, p0=p0, bounds=bounds, **fit_kwargs)[0]
            converged = not any(issubclass(w.category, OptimizeWarning) for w in caught)
            return popt, converged
        except RuntimeError:
            p0_arr = np.atleast_1d(np.asarray(p0, dtype=float))
            scale = 1.0 if sigma is None else np.asarray(sigma, dtype=float)
            def residuals(params):
                return (func(x, *params) - y) / scale
            res = _ls(residuals, p0_arr, bounds=bounds, **kwargs)
            return res.x, False


def _default_event(values):
    return np.array([True for _ in values])


def _loadtxt_with_optional_header(colvar, usecols):
    try:
        return np.loadtxt(colvar, usecols=usecols)
    except Exception:
        return np.loadtxt(colvar, usecols=usecols, skiprows=1)


def _pad_trajectory_columns(traj, missing_columns):
    dummy = np.array([None for _ in traj])
    extra = [dummy for _ in range(missing_columns)]
    return np.vstack([traj.T, *extra]).T


def _map_with_cores(func, values, cores):
    if cores > 1:
        with mp.Pool(cores) as pool:
            return np.array(pool.map(func, values))
    return np.array([func(value) for value in values])


def _build_v_data(data, bias_shift=0.0):
    """Build the padded (n_trajs × T_max) bias matrix and time axis once."""
    colvar_maxrow_count = max(len(traj[:, 0]) for traj in data)
    time_list = np.linspace(0, colvar_maxrow_count * (data[0][1, 0] - data[0][0, 0]), colvar_maxrow_count)
    v_data = np.full((len(data), colvar_maxrow_count), np.nan)
    for i, traj in enumerate(data):
        v_data[i, :len(traj)] = traj[:, 1] + bias_shift
    return v_data, np.isnan(v_data), time_list


def _avg_exponential_from_v_data(v_data, mask, time_list, beta, gamma, logTrick=False):
    """Compute log<e^{βγV}>(t) from a precomputed v_data matrix."""
    if logTrick:
        simmax_v = np.nanmax(v_data, axis=0)
        masked_exp = np.ma.masked_array(np.exp(beta * gamma * (v_data - simmax_v)), mask)
        log_vals = beta * gamma * simmax_v + np.log(np.ma.average(masked_exp.T, axis=1))
    else:
        masked_exp = np.ma.masked_array(np.exp(beta * gamma * v_data), mask)
        log_vals = np.log(np.ma.average(masked_exp.T, axis=1))
    return np.vstack((time_list, np.asarray(log_vals))).T


def _cum_hazards_eatr(log_average_exp, final_time_indices, logTrick=False):
    """Compute ∫₀ᵗⁱ <e^{γβV}>(t) dt for all indices in one vectorised pass.

    Falls back to per-element EATR_calculate_cum_hazard only when logTrick=True.
    """
    if logTrick:
        func = partial(EATR_calculate_cum_hazard, log_average_exp, True)
        return np.array([func(i) for i in final_time_indices], dtype=float)
    exp_vals = np.exp(np.asarray(log_average_exp[:, 1], dtype=float))
    cum_vals = cumulative_trapezoid(exp_vals, log_average_exp[:, 0], initial=0.0)
    indices = np.asarray(final_time_indices, dtype=int)
    result = cum_vals[indices].copy()
    result[indices <= 1] = 0.0
    return result


def _cum_hazards_ktr(vmb_average, gamma, final_time_indices, logTrick=False):
    """Compute ∫₀ᵗⁱ e^{γβVmb}(t) dt for all indices in one vectorised pass.

    Falls back to per-element KTR_calculate_cum_hazard only when logTrick=True.
    """
    if logTrick:
        func = partial(KTR_calculate_cum_hazard, gamma, vmb_average, True)
        return np.array([func(i) for i in final_time_indices], dtype=float)
    exp_vals = np.exp(gamma * np.asarray(vmb_average[:, 1], dtype=float))
    cum_vals = cumulative_trapezoid(exp_vals, vmb_average[:, 0], initial=0.0)
    indices = np.asarray(final_time_indices, dtype=int)
    result = cum_vals[indices].copy()
    result[indices <= 1] = 0.0
    return result


# data fmt:
# [
# [t0 V0 acc0 Vm0],
# [t1 V1 acc1 Vm1],
# [t2 V2 acc2 Vm2],
# ...
# ]
def get_data(colvars, time_col, bias_col, acc_col=None, maxbias_col=None, time_scale_factor=1.0, threads=1, work_col=None, skip_errors=True, stride=1, subsample_min_points=0, return_info=False):  # Changed "file_format" to "colvars"
    """Load trajectory data from COLVAR files.

    Parameters
    ----------
    stride : int, optional
        Keep only every ``stride``-th row of each trajectory (default 1, i.e.
        keep everything). The final row is always retained so that the
        transition/censoring time of each trajectory is preserved exactly. Use
        this to thin very finely-printed COLVAR files (e.g. a bias written every
        step) down to a spacing that still resolves the bias evolution, which
        speeds up the exponential-average and bootstrap steps substantially.
    subsample_min_points : int, optional
        Floor on the number of rows retained per trajectory when subsampling
        (default 0, i.e. no floor). The stride actually used is reduced so that
        even the *shortest* trajectory keeps at least this many rows::

            effective_stride = min(stride, shortest_trajectory // min_points)

        and that one stride is applied to every trajectory.

        Why this matters: a single global ``stride`` is unsafe when trajectory
        lengths vary. Fast-deposition metadynamics runs transition quickly and
        so are short, and striding them by the same factor used for long runs
        can leave only a handful of rows, which turns the EATR cumulative-hazard
        integral into a coarse staircase and corrupts the fit.

        Why the stride must stay *uniform*: downstream, ``_build_v_data``
        aligns trajectories by row index and builds a single time axis from the
        spacing of the first trajectory, i.e. it assumes every trajectory shares
        one time step. Thinning trajectories by different amounts would break
        that alignment and silently corrupt the exponential average, so the
        floor is applied globally (via the shortest trajectory) rather than
        per trajectory.

    Returns
    -------
    data : list of np.ndarray
        Successfully loaded trajectories.
    skipped : list of int
        Indices (into the original *colvars* list) of files that could not be
        loaded.  Empty when all files loaded successfully.  When skip_errors is
        False a ValueError is raised instead of collecting skipped indices.
    """
    if len(colvars) == 0:
        sys.exit(f"ERROR: No COLVAR files provided.")
    if stride < 1:
        raise ValueError(f"stride must be a positive integer, got {stride}")
    if subsample_min_points < 0:
        raise ValueError(f"subsample_min_points must be non-negative, got {subsample_min_points}")

    def _subsample(traj, step):
        # Keep every step-th row, but always retain the final row so the
        # trajectory's true final (transition) time is preserved.
        if step <= 1 or len(traj) <= 1:
            return traj
        keep = list(range(0, len(traj), step))
        if keep[-1] != len(traj) - 1:
            keep.append(len(traj) - 1)
        return traj[keep]

    def _load_one(colvar):
        if acc_col is None and maxbias_col is None:
            traj = _loadtxt_with_optional_header(colvar, (time_col, bias_col))
            traj = _pad_trajectory_columns(traj, 2)
        elif maxbias_col is None:
            traj = _loadtxt_with_optional_header(colvar, [time_col, bias_col, acc_col])
            traj = _pad_trajectory_columns(traj, 1)
        elif acc_col is not None:
            traj = _loadtxt_with_optional_header(colvar, [time_col, bias_col, acc_col, maxbias_col])
        else:
            traj = _loadtxt_with_optional_header(colvar, [time_col, bias_col, maxbias_col])
            dummy = np.array([None for _ in traj])
            traj = np.vstack([traj[:,:-1].T, dummy, traj[:,-1].T]).T
        traj[:, 0] *= time_scale_factor
        if work_col is not None:
            work = _loadtxt_with_optional_header(colvar, (work_col,))
            traj = np.hstack([traj, work.reshape(-1, 1).astype(float)])
        return traj

    def _load_safe(colvar):
        try:
            return _load_one(colvar)
        except Exception:
            return None

    if threads > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=threads) as executor:
            if skip_errors:
                results = list(executor.map(_load_safe, colvars))
            else:
                results = list(executor.map(_load_one, colvars))
    else:
        if skip_errors:
            results = [_load_safe(c) for c in colvars]
        else:
            results = [_load_one(c) for c in colvars]

    data = []
    skipped = []
    for i, result in enumerate(results):
        if result is None or result.ndim < 2 or result.shape[0] == 0:
            if result is not None:
                print(f"Warning: skipping {colvars[i]}: file loaded but contains no data rows", file=sys.stderr)
            skipped.append(i)
        else:
            data.append(result)

    # Subsample after loading, with ONE stride shared by every trajectory (see
    # the subsample_min_points docstring: downstream code assumes a common time
    # step). The floor is set by the shortest trajectory so that even the
    # fastest-transitioning runs keep enough points to resolve the bias.
    effective_stride = 1
    if stride > 1 and data:
        step = stride
        if subsample_min_points > 0:
            shortest = min(len(traj) for traj in data)
            step = min(step, max(1, shortest // subsample_min_points))
        effective_stride = step
        if step > 1:
            data = [_subsample(traj, step) for traj in data]
    if return_info:
        rows = [len(traj) for traj in data]
        info = {
            "requested_stride": int(stride),
            "subsample_min_points": int(subsample_min_points),
            # With subsample_min_points the stride is data dependent, so the
            # value actually applied is the one needed to reproduce a result.
            "effective_stride": int(effective_stride),
            "rows_per_trajectory_min": int(min(rows)) if rows else 0,
            "rows_per_trajectory_median": int(np.median(rows)) if rows else 0,
            "rows_per_trajectory_max": int(max(rows)) if rows else 0,
        }
        return data, skipped, info
    return data, skipped

def check_acc_consistency(data, beta, rtol=2.0, n_sample=5):
    """Check that the acc column (col 2) agrees with integral(exp(beta*V)) / t_final.

    Compares PLUMED's precomputed acceleration factor against what the code would
    compute from the bias column using the given beta.  A large disagreement
    (ratio outside [1/rtol, rtol]) almost always means the energy unit is wrong.

    Parameters
    ----------
    data   : list of trajectories as returned by get_data (with acc_col provided)
    beta   : inverse temperature in units matching the bias column
    rtol   : allowed ratio between acc_file and acc_computed before warning (default 2.0)
    n_sample : number of trajectories to check (picks longest ones for better statistics)
    """
    if data[0][0, 2] is None:
        return  # no acc column loaded, nothing to check

    from scipy.integrate import trapezoid as _trapz

    lengths = [len(traj) for traj in data]
    indices = sorted(range(len(data)), key=lambda i: lengths[i], reverse=True)[:n_sample]

    ratios = []
    for i in indices:
        traj = data[i]
        t = traj[:, 0].astype(float)
        v = traj[:, 1].astype(float)
        acc_file = float(traj[-1, 2])
        if acc_file == 0 or t[-1] == 0:
            continue
        acc_computed = _trapz(np.exp(beta * v), t) / t[-1]
        ratios.append(acc_file / acc_computed)

    if not ratios:
        return

    median_ratio = float(np.median(ratios))
    if not (1.0 / rtol <= median_ratio <= rtol):
        import warnings
        warnings.warn(
            f"The acc column in your COLVAR files disagrees with integral(exp(beta*V))/t "
            f"by a factor of {median_ratio:.2f} (checked {len(ratios)} trajectories). "
            f"This usually means the energy unit is wrong. "
            f"If your bias is in kcal/mol, pass --energyunit 4.184. "
            f"If it is in kJ/mol, pass --energyunit 1 (the default).",
            stacklevel=2,
        )


def get_event(data, maxlen=None, maxtime=None, num_events=None, log_files=None, quiet=False, qquiet=False):
    # Determine which simulations transitioned.
    # log_files: The simulations where the corresponding PLUMED log file contains the line "#! SET COMMIT(T)ED TO BASIN X" have transitioned.
    # maxlen: Simulations whose COLVAR files have fewer data rows than the maximum file length have transitioned.
    # maxtime: Simulations whose final times in their COLVAR files is less than the maximum simulation time have transitioned.
    # num_events: The simulations with the [num_events] lowest final times in their COLVAR files have transitioned.
    event = None
    if maxlen is None and maxtime is None and num_events is None and log_files is None:
        event = _default_event(data)
        if not qquiet:
            print('WARNING: Assuming all simulations transitioned.')
    elif np.sum([maxlen is not None,maxtime is not None,num_events is not None,log_files is not None]) > 1:
        print('Multiple transition counting methods have somehow been selected. Priority: log_files > maxlen > maxtime > num_events')
    if log_files is not None:
        # log_files needs to be a list of str for PLUMED log files in the same order as the corresponding trajectories in data
        event = []
        try:
            for log_file in log_files:
                transitioned = False
                with open(log_file, 'r') as f:
                    for line in f:
                        if 'SET COMMIT' in line:
                            transitioned = True
                event.append(transitioned)
            event = np.array(event)
            if not quiet:
                print(f"{event.sum()} out of {len(event)} simulations transitioned.")
        except:
            print('Could not load the PLUMED log. Defaulting to assuming all simulations transitioned.')
            event = get_event(data, qquiet=True)
    elif maxlen is not None:
        event = []
        for traj in data:
            event.append(len(traj) < maxlen)
        event = np.array(event)
        if not quiet:
            print(f"{event.sum()} out of {len(event)} simulations transitioned.")
    elif maxtime is not None:
        event = []
        for traj in data:
            event.append(traj[-1,0] < maxtime)
        event = np.array(event)
        if not quiet:
            print(f"{event.sum()} out of {len(event)} simulations transitioned.")
    elif num_events is not None:
        N = len(data)
        event = np.full(N,False)
        lowest_indices = sorted(range(N), key=lambda i: len(data[i]))[:num_events]
        for i in lowest_indices:
            event[i] = True
        if not quiet:
            print(f"{event.sum()} out of {len(event)} simulations are specified to have transitioned.")
    return event

def bootstrap(sample,func,nresamples,event=None,double=False,return_stat=False,seed=None):
    stat = []
    stat2 = []
    rng = np.random.default_rng(seed=seed)
    sample_size = len(sample)
    index_sets = rng.integers(0, sample_size, size=(nresamples, sample_size))
    for indices in index_sets:
        resample = [sample[index] for index in indices]
        if event is not None:
            resampled_event = np.array([event[index] for index in indices])
        else:
            resampled_event = None
        if double:
            a, b = func(resample,resampled_event)
            stat.append(a)
            stat2.append(b)
        else:
            stat.append(func(resample,resampled_event))
    if double:
        if return_stat:
            return np.array([[stat[i],stat2[i]] for i in range(len(stat))])
        else:
            return np.std(stat), np.std(stat2)
    else:
        if return_stat:
            return np.array(stat)
        else:
            return np.std(stat)
    
## Infrequent Metadynamics

# Evaluating the rescaled times τ_accel = α*t = <e^βV>*t
def iMetaD_rescaled_times(data, beta, bias_shift=0.0): # Consider cutting traj data to maxlen to make foolproof
    # Create the acceleration factor from the bias column if not provided
    if data[0][0,2] is None:
        times = np.array([traj[-1,0]*np.mean(np.exp(np.float64(beta*(traj[:,1] + bias_shift)))) for traj in data])
    else:
        times = np.array([traj[-1,0]*traj[-1,2] for traj in data])
    return times

# Infrequent Metadynamics Tiwary Estimator (directly from trajectory data)
def iMetaD_invMRT(data, beta, event=None, bias_shift=0.0):
    if event is None:
        event = _default_event(data) # Assume all simulations transition unless told otherwise
    times = iMetaD_rescaled_times(data, beta, bias_shift=bias_shift)
    return event.sum() / np.sum(times) # Σ_N t / M is the maximum likelihood estimate for right-censored data

# Infrequent Metadynamics Tiwary Estimator (from precomputed rescaled times)
def iMetaD_invMRT_times(times, event=None):
    if event is None:
        event = _default_event(times) # Assume all simulations transition unless told otherwise
    return event.sum() / np.sum(times) # Σ_N t / M is the maximum likelihood estimate for right-censored data

# Infrequent Metadynamics CDF Fit Least Squares Objective
def iMetaD_leastsq_cost(k, t, ecdfy):
    f = 1-np.exp(-k*t)
    sse = np.square(ecdfy-f).sum() # Sum of Squared Errors
    return sse

def iMetaD_FitCDF(data, beta, event=None, bias_shift=0.0, k_bounds=(-np.inf,np.inf), k_guess=None, require_convergence=False, cdf_weights="none"):
    if event is None:
        event = _default_event(data) # Assume all simulations transition unless told otherwise
    times = iMetaD_rescaled_times(data, beta, bias_shift=bias_shift)

    # Construct Empirical CDF
    ecdfx = np.sort(times[event])
    ecdfy = np.arange(1, event.sum()+1) / len(data)

    if k_guess is None:
        k_guess = event.sum() / np.sum(times) # Use maximum likelihood estimate as initial guess if the guess is not provided

    popt, converged = _curve_fit_lenient(lambda k,t:1-np.exp(-k*t), ecdfx, ecdfy, p0=k_guess, bounds=k_bounds, sigma=ecdf_sigma(ecdfy, cdf_weights))
    if require_convergence and not converged:
        raise RuntimeError("iMetaD CDF fit did not converge to tolerance")
    return popt[0], converged

def iMetaD_FitCDF_times(times, event=None, k_bounds=(-np.inf,np.inf), k_guess=None, require_convergence=False, cdf_weights="none"):
    if event is None:
        event = _default_event(times) # Assume all simulations transition unless told otherwise

    # Construct Empirical CDF
    ecdfx = np.sort(times[event])
    ecdfy = np.arange(1, event.sum()+1) / len(times)

    if k_guess is None:
        k_guess = event.sum() / np.sum(times) # Use maximum likelihood estimate as initial guess if the guess is not provided

    popt, converged = _curve_fit_lenient(lambda k,t:1-np.exp(-k*t), ecdfx, ecdfy, p0=k_guess, bounds=k_bounds, sigma=ecdf_sigma(ecdfy, cdf_weights))
    if require_convergence and not converged:
        raise RuntimeError("iMetaD CDF fit did not converge to tolerance")
    return popt[0], converged


## Kramers' Time-dependent Rate (KTR)

# Populate the maxbias column in data
def set_max_bias(data, bias_shift=0.0):
    for traj in data:
        maximum = -np.inf
        for point in traj:
            maximum = maximum if maximum > point[1] else point[1]
            point[3] = maximum

# Evaluating the average max bias Vmb(t)
def avg_max_bias(data, beta, bias_shift=0.0):

    # Populate maxbias column if needed
    if data[0][0,3] is None:
        set_max_bias(data, bias_shift=bias_shift)

    # Prepare rectangular masked ndarray for averaging
    colvar_maxrow_count = max(len(traj[:,0]) for traj in data)
    vmb_data = np.full((len(data), colvar_maxrow_count), np.nan)
    for i, traj in enumerate(data):
            vmb_data[i,:len(traj)] = traj[:,3]

    # Average across simulations
    masked_vmb = np.ma.masked_array(vmb_data, np.isnan(vmb_data))
    vmb_average = np.ma.average(masked_vmb.T, axis=1)
    time_list = np.linspace(0,colvar_maxrow_count*(data[0][1,0]-data[0][0,0]),colvar_maxrow_count)
    vmb_average = np.vstack((time_list, vmb_average)).T
    vmb_average[:,1] = (vmb_average[:,1] + bias_shift) * beta

    return vmb_average # Final result is of the form [ [t0 βVmb0], [t1 βVmb1], ... ]

# KTR log likelihood function. (The logTrick uses the log-sum-exp trick to ideally increase precision for large exponents.)
def KTR_calculate_neg_log_l(gamma, final_time_indices, vmb_average, event=None, cores=1, logTrick=False, reg_lambda=0.0):

    if event is None:
        event = _default_event(final_time_indices)

    cum_hazard = _cum_hazards_ktr(vmb_average, gamma, final_time_indices, logTrick)
    log_hazard = KTR_calculate_log_hazard(gamma, vmb_average, final_time_indices)

    mean_t = cum_hazard.sum() / event.sum()
    log_l = -event.sum() * np.log(mean_t) + log_hazard[event].sum() - (1 / mean_t) * cum_hazard.sum()

    gdiff = 0.5-gamma # Regularization for gamma in case you have a situation where gamma is crashing to 0

    return -log_l + reg_lambda*gdiff*gdiff

# integral of e^γβVmb from 0 to simulation i's transition time
def KTR_calculate_cum_hazard(gamma, vmb_average, logTrick, final_time_index):
    if int(final_time_index) <= 1:
        return 0.0
    dt=vmb_average[1,0]-vmb_average[0,0]
    if logTrick: # log-sum-exp trick; e^A+Σe^Bi = exp(A + ln(1+Σe^(Bi-A))); int_0^ti f(t)dt ~ (dt/2)*( f(0) + f(ti) + 2Σ_j=1^(i-1)f(tj) )
        max_vmb = max(vmb_average[:,1])
        return 0.5*dt*(1 + np.exp(gamma*vmb_average[int(final_time_index),1]) + 2*np.exp(gamma*max_vmb + np.log(np.exp(gamma*vmb_average[1:int(final_time_index),1] - gamma*max_vmb).sum())))
    else:
        int_Veff = trapezoid(np.exp(gamma*vmb_average[:int(final_time_index),1]),vmb_average[:int(final_time_index),0])
        return int_Veff

# γβVmb at simulation i's transition time
def KTR_calculate_log_hazard(gamma, vmb_average, final_time_index):
    return gamma*vmb_average[final_time_index,1]

# Theory CDF for KTR: S(t) = exp(-int_0^t k(t') dt') = exp(-k0 int_0^t e^γβVmb(t') dt')
def KTR_CDF(time_indices, k0, gamma, vmb_average, cores=1, logTrick=False):
    cum_hazard = _cum_hazards_ktr(vmb_average, gamma, time_indices, logTrick)
    return 1 - np.exp(-k0 * cum_hazard)

# KTR CDF Fit Least Squares Objective
def KTR_leastsq_cost(params, ecdfx_indices, ecdfy, vmb_average, cores=1, logTrick=False, reg_lambda=0.0, kIMD=1.0):
    f = KTR_CDF(ecdfx_indices, params[0], params[1], vmb_average, cores=cores, logTrick=logTrick)
    sse = np.square(ecdfy-f).sum()
    gdiff = 0.5 - params[1]
    kdiff = 10*kIMD - params[0]
    return sse + reg_lambda*(kdiff*kdiff + gdiff*gdiff)

# KTR Get MLE rate estimate (directly from trajectory data)
def KTR_MLE_rate(data, beta, event=None, gamma_bounds=(0.,1.), cores=1, logTrick=False, reg_lambda=0.0, do_bopt=False, bias_shift=0.0):

    # Get Vmb(t) and final_time_indices
    vmb_average = avg_max_bias(data, beta, bias_shift=bias_shift)
    final_time_indices = np.array([int(len(traj)-1) for traj in data])
    if event is None:
        event = _default_event(final_time_indices)
    
    if not do_bopt: # No Bayesian Optimization method: instead use bounded Brent method
        # Find the value of gamma that maximizes the likelihood
        neg_log_l = lambda gamma : KTR_calculate_neg_log_l(gamma, final_time_indices, vmb_average, event=event, cores=cores, logTrick=logTrick, reg_lambda=reg_lambda)
        opt = optimize.minimize_scalar(neg_log_l, bounds=gamma_bounds, method='bounded')
        gamma = opt.x
    else: # Bayesian Optimization selected
        # Find the value of gamma that maximizes the likelihood
        acquisition_function = acquisition.ExpectedImprovement(xi=0.1,exploration_decay=0.97,exploration_decay_delay=50)
        optimizer = bopt(
                f = lambda gamma : -KTR_calculate_neg_log_l(gamma, final_time_indices, vmb_average, event=event, cores=cores, logTrick=logTrick, reg_lambda=reg_lambda),
                acquisition_function = acquisition_function,
                pbounds = {'gamma': gamma_bounds},
                verbose = 0,
                random_state = 1
        )
        optimizer.maximize(init_points=25, n_iter=100)
        gamma = optimizer.max['params']['gamma']

    # Calculate k0* = M / ( Σ_N int_0^ti e^γβVmb(t') dt' )
    cum_hazard = _cum_hazards_ktr(vmb_average, gamma, final_time_indices, logTrick)
    k0 = event.sum() / cum_hazard.sum()

    return np.array([k0, gamma])

# KTR Get MLE rate estimate (from precomputed Vmb(t) and ti indices)
def KTR_MLE_rate_VMB(vmb_average, final_time_indices, event=None, gamma_bounds=(0.,1.), cores=1, logTrick=False, reg_lambda=0.0, do_bopt=False):

    # Assume all simulations transitioned unless explicitly told otherwise
    if event is None:
        event = _default_event(final_time_indices)
    
    if not do_bopt: # No Bayesian Optimization method: instead use bounded Brent method
        # Find the value of gamma that maximizes the likelihood
        neg_log_l = lambda gamma : KTR_calculate_neg_log_l(gamma, final_time_indices, vmb_average, event=event, cores=cores, logTrick=logTrick, reg_lambda=reg_lambda)
        opt = optimize.minimize_scalar(neg_log_l, bounds=gamma_bounds, method='bounded')
        gamma = opt.x
    else: # Bayesian Optimization selected
        # Find the value of gamma that maximizes the likelihood
        acquisition_function = acquisition.ExpectedImprovement(xi=0.1,exploration_decay=0.97,exploration_decay_delay=50)
        optimizer = bopt(
                f = lambda gamma : -KTR_calculate_neg_log_l(gamma, final_time_indices, vmb_average, event=event, cores=cores, logTrick=logTrick, reg_lambda=reg_lambda),
                acquisition_function = acquisition_function,
                pbounds = {'gamma': gamma_bounds},
                verbose = 0,
                random_state = 1
        )
        optimizer.maximize(init_points=25, n_iter=100)
        gamma = optimizer.max['params']['gamma']

    # Calculate k0* = M / ( Σ_N int_0^ti e^γβVmb(t') dt' )
    cum_hazard = _cum_hazards_ktr(vmb_average, gamma, final_time_indices, logTrick)
    k0 = event.sum() / cum_hazard.sum()

    return np.array([k0, gamma])

# KTR Get CDF rate estimate (directly from trajectory data)
def KTR_CDF_rate(data, beta, event=None, k_bounds=(-np.inf,np.inf), gamma_bounds=(0.,1.), cores=1, logTrick=False, init_guess=[None,None], reg_lambda=0.0, kIMD=1.0, do_bopt=False, bias_shift=0.0, require_convergence=False, cdf_weights="none"):

    # Get Vmb(t) and final_time_indices
    vmb_average = avg_max_bias(data, beta, bias_shift=bias_shift)
    final_time_indices = np.array([int(len(traj)-1) for traj in data])
    if event is None:
        event = _default_event(final_time_indices)
    if init_guess[0] is None:
        init_guess = (iMetaD_invMRT(data, beta, event=event, bias_shift=bias_shift),0.9)

    # 2-parameter CDF fitting for gamma and k0
    ecdfx_indices = np.sort(final_time_indices[event])
    ecdfy = np.arange(1, event.sum()+1) / len(data)

    if not do_bopt: # No Bayesian Optimization method: instead use Bounded Brent (if λ > 0) or Levenberg-Marquardt (if λ = 0) method
        options = {
            "maxiter":1000000
        }
        if reg_lambda == 0:
            cdf = lambda time_indices, k0, gamma: KTR_CDF(time_indices, k0, gamma, vmb_average, cores=cores, logTrick=logTrick)
            cdf_result, converged = _curve_fit_lenient(cdf, ecdfx_indices, ecdfy, p0=init_guess, bounds=([k_bounds[0],gamma_bounds[0]],[k_bounds[1],gamma_bounds[1]]), max_nfev=100000*len(ecdfy), sigma=ecdf_sigma(ecdfy, cdf_weights))
        else:
            leastsq = lambda params: KTR_leastsq_cost(params, ecdfx_indices, ecdfy, vmb_average, cores=cores, logTrick=logTrick, reg_lambda=reg_lambda, kIMD=kIMD)
            cdf_result = optimize.minimize(leastsq,init_guess,options=options).x
            converged = True
    else:
        acquisition_function = acquisition.ExpectedImprovement(xi=0.1,exploration_decay=0.97,exploration_decay_delay=50)
        optimizer = bopt(
                f = lambda logk0, gamma : -KTR_leastsq_cost((np.exp(logk0),gamma), int(ecdfx_indices), ecdfy, vmb_average, cores=cores, logTrick=logTrick, reg_lambda=reg_lambda, kIMD=kIMD),
                acquisition_function = acquisition_function,
                pbounds = {'logk0': (np.log(init_guess[0])-35,np.log(1/np.mean(vmb_average[final_time_indices[event],0]))+5), 'gamma': gamma_bounds},
                verbose = 0,
                random_state = 1
        )
        optimizer.probe(params={'logk0': np.log(init_guess[0]), 'gamma': init_guess[1]})
        optimizer.maximize(init_points=25, n_iter=100)
        cdf_result = np.array([np.exp(optimizer.max['params']['logk0']), optimizer.max['params']['gamma']])
        converged = True
    if require_convergence and not converged:
        raise RuntimeError("KTR CDF fit did not converge to tolerance")
    return cdf_result, converged

# KTR Get CDF rate estimate (with precomputed Vmb(t) and ti indices)
def KTR_CDF_rate_VMB(vmb_average, final_time_indices, event=None, k_bounds=(-np.inf,np.inf), gamma_bounds=(0.,1.), cores=1, logTrick=False, init_guess=[None,None], reg_lambda=0.0, kIMD=None, do_bopt=False, require_convergence=False, cdf_weights="none"):

    if event is None:
        event = _default_event(final_time_indices)
    if init_guess[0] is None:
        init_guess = (1/np.mean(vmb_average[final_time_indices,0]),0.9)
    if kIMD is None:
        kIMD = init_guess[0]

    # 2-parameter CDF fitting for gamma and k0
    ecdfx_indices = np.sort(final_time_indices[event])
    ecdfy = np.arange(1, event.sum()+1) / len(final_time_indices)

    if not do_bopt: # No Bayesian Optimization method: instead use Bounded Brent (if λ > 0) or Levenberg-Marquardt (if λ = 0) method
        options = {
            "maxiter":1000000
        }
        if reg_lambda == 0:
            cdf = lambda time_indices, k0, gamma: KTR_CDF(time_indices, k0, gamma, vmb_average, cores=cores, logTrick=logTrick)
            cdf_result, converged = _curve_fit_lenient(cdf, ecdfx_indices, ecdfy, p0=init_guess, bounds=([k_bounds[0],gamma_bounds[0]],[k_bounds[1],gamma_bounds[1]]), max_nfev=100000*len(ecdfy), sigma=ecdf_sigma(ecdfy, cdf_weights))
        else:
            leastsq = lambda params: KTR_leastsq_cost(params, ecdfx_indices, ecdfy, vmb_average, cores=cores, logTrick=logTrick, reg_lambda=reg_lambda, kIMD=kIMD)
            cdf_result = optimize.minimize(leastsq,init_guess,options=options).x
            converged = True
    else:
        acquisition_function = acquisition.ExpectedImprovement(xi=0.1,exploration_decay=0.97,exploration_decay_delay=50)
        optimizer = bopt(
                f = lambda logk0, gamma : -KTR_leastsq_cost((np.exp(logk0),gamma), ecdfx_indices, ecdfy, vmb_average, cores=cores, logTrick=logTrick, reg_lambda=reg_lambda, kIMD=kIMD),
                acquisition_function = acquisition_function,
                pbounds = {'logk0': (np.log(init_guess[0])-35,np.log(1/np.mean(vmb_average[final_time_indices[event],0]))+5), 'gamma': gamma_bounds},
                verbose = 0,
                random_state = 1
        )
        optimizer.probe(params={'logk0': np.log(init_guess[0]), 'gamma': init_guess[1]})
        optimizer.maximize(init_points=25, n_iter=100)
        cdf_result = np.array([np.exp(optimizer.max['params']['logk0']), optimizer.max['params']['gamma']])
        converged = True
    if require_convergence and not converged:
        raise RuntimeError("KTR CDF fit did not converge to tolerance")
    return cdf_result, converged

## Exponential Average Time-dependent Rate (EATR)

# Evaluating the average exponential <e^γβV> = 1/n(t) Σ_n(t) e^γβV(t) (where n(t) is the number of untransitioned simulations at t)
def avg_exponential(data, beta, gamma, logTrick=False, bias_shift=0.0):
    v_data, mask, time_list = _build_v_data(data, bias_shift)
    return _avg_exponential_from_v_data(v_data, mask, time_list, beta, gamma, logTrick)
    # Final result is of the form [ [t0 ln<e^γβV>0], [t1 ln<e^γβV>1], ... ]

#  EATR log likelihood expression as a function of γ alone (dependence on γ comes from log_average_exp)
def EATR_calculate_neg_log_l(gamma, final_time_indices, log_average_exp, event=None, cores=1, logTrick=False, reg_lambda=0.0):

    if event is None:
        event = _default_event(final_time_indices)

    cum_hazard = _cum_hazards_eatr(log_average_exp, final_time_indices, logTrick)
    log_hazard = EATR_calculate_log_hazard(final_time_indices, log_average_exp)

    mean_t = cum_hazard.sum() / event.sum()
    log_l = -event.sum() * np.log(mean_t) + log_hazard[event].sum() - (1 / mean_t) * cum_hazard.sum()

    gdiff = 0.5-gamma

    return -log_l + reg_lambda*gdiff*gdiff

# EATR log likelihood expression as a function of k0 and γ (dependence on γ comes from log_average_exp)
def EATR_calculate_neg_log_l_k0(k0, gamma, final_time_indices, log_average_exp, event=None, cores=1, logTrick=False, reg_lambda=0.0):

    if event is None:
        event = _default_event(final_time_indices)

    cum_hazard = _cum_hazards_eatr(log_average_exp, final_time_indices, logTrick)
    log_hazard = EATR_calculate_log_hazard(final_time_indices, log_average_exp)

    log_l = event.sum() * np.log(k0) + log_hazard[event].sum() - k0 * cum_hazard.sum()

    gdiff = 0.5-gamma

    return -log_l + reg_lambda*gdiff*gdiff

# Integral of <e^γβV> from 0 to ti where i is the given time index
def EATR_calculate_cum_hazard(log_average_exp, logTrick, final_time_index):
    if int(final_time_index) <= 1:
        return 0.0
    if logTrick:
        dt=log_average_exp[1,0]-log_average_exp[0,0]
        max_lae = max(log_average_exp[:,1])
        return 0.5*dt*(1 + np.exp(log_average_exp[int(final_time_index),1]) + 2*np.exp(max_lae + np.log(np.exp(log_average_exp[1:int(final_time_index),1] - max_lae).sum())))
    else:
        int_Veff = trapezoid(np.exp(log_average_exp[:int(final_time_index),1]),log_average_exp[:int(final_time_index),0])
        return int_Veff

# ln <e^γβV>
def EATR_calculate_log_hazard(final_time_index, log_average_exp):

    Veff = log_average_exp[final_time_index,1]
    return Veff

# Theory CDF for EATR: S(t) = exp(-int_0^t k(t') dt') = exp(-k0 int_0^t <e^γβV>(t') dt')
def EATR_CDF(time_indices, k0, log_average_exp, cores=1, logTrick=False):
    cum_hazard = _cum_hazards_eatr(log_average_exp, time_indices, logTrick)
    return 1 - np.exp(-k0 * cum_hazard)
    
# EATR CDF Fit Least Squares Objective
def EATR_leastsq_cost(params, ecdfx_indices, ecdfy, log_average_exp, cores=1, logTrick=False, reg_lambda=0.0, kIMD=1.0):
    f = EATR_CDF(ecdfx_indices, params[0], log_average_exp, cores=cores, logTrick=logTrick)
    sse = np.square(ecdfy-f).sum()
    gdiff = 0.5 - params[1]
    kdiff = 10*kIMD - params[0]
    return sse + reg_lambda*(kdiff*kdiff + gdiff*gdiff)

# EATR Get MLE rate estimate (directly from trajectory data) (cannot precompute ln<e^γβV> because that depends on γ.)
def EATR_MLE_rate(data, beta, event=None, gamma_bounds=(0.,1.), cores=1, logTrick=False, reg_lambda=0.0, do_bopt=False, bias_shift=0.0):

    # Get final_time_indices
    final_time_indices = np.array([int(len(traj)-1) for traj in data])
    if event is None:
        event = _default_event(final_time_indices)

    # Precompute bias matrix once; only the gamma scalar changes between optimizer steps.
    v_data, v_mask, time_list = _build_v_data(data, bias_shift)

    def neg_log_l(gamma):
        log_average_exp = _avg_exponential_from_v_data(v_data, v_mask, time_list, beta, gamma, logTrick)
        return EATR_calculate_neg_log_l(gamma, final_time_indices, log_average_exp, event=event, logTrick=logTrick, reg_lambda=reg_lambda)

    # Find MLE for γ with Brent method or Bayesian Optimization
    if not do_bopt:
        opt = optimize.minimize_scalar(neg_log_l, bounds=gamma_bounds, method='bounded')
        gamma = opt.x
    else:
        acquisition_function = acquisition.ExpectedImprovement(xi=0.06,exploration_decay=0.97,exploration_decay_delay=50)
        optimizer = bopt(
                f = lambda gamma : -neg_log_l(gamma),
                acquisition_function = acquisition_function,
                pbounds = {'gamma': gamma_bounds},
                verbose = 0,
                random_state = 1
        )
        optimizer.maximize(init_points=25, n_iter=100)
        gamma = optimizer.max['params']['gamma']

    # Calculate k0*
    log_average_exp = _avg_exponential_from_v_data(v_data, v_mask, time_list, beta, gamma, logTrick)
    cum_hazard = _cum_hazards_eatr(log_average_exp, final_time_indices, logTrick)
    k0 = event.sum() / cum_hazard.sum()

    return np.array([k0, gamma])

# EATR Get CDF rate estimate (directly from trajectory data) (cannot precompute ln<e^γβV> because that depends on γ.)
def EATR_CDF_rate(data, beta, event=None, k_bounds=(0.,np.inf), gamma_bounds=(0.,1.), cores=1, init_guess=[None,None], logTrick=False, reg_lambda=0.0, kIMD=1.0, do_bopt=False, bias_shift=0.0, require_convergence=False, cdf_weights="none"):

    # Get final_time_indices
    final_time_indices = np.array([int(len(traj)-1) for traj in data])
    if event is None:
        event = _default_event(final_time_indices)

    # 2-parameter CDF fitting for gamma and k0
    ecdfx_indices = np.sort(final_time_indices[event])
    ecdfy = np.arange(1, event.sum()+1) / len(data)
    
    # Precompute bias matrix once; only gamma changes between optimizer steps.
    v_data, v_mask, time_list = _build_v_data(data, bias_shift)

    def cdf(time_indices, k0, gamma):
        log_average_exp = _avg_exponential_from_v_data(v_data, v_mask, time_list, beta, gamma, logTrick)
        return EATR_CDF(time_indices, k0, log_average_exp, logTrick=logTrick)
    def get_cost(params):
        log_average_exp = _avg_exponential_from_v_data(v_data, v_mask, time_list, beta, params[1], logTrick)
        return EATR_leastsq_cost(params, ecdfx_indices, ecdfy, log_average_exp, logTrick=logTrick, reg_lambda=reg_lambda, kIMD=kIMD)

    def guess_is_finite(params):
        try:
            values = cdf(ecdfx_indices, params[0], params[1])
        except Exception:
            return False
        return np.isfinite(values).all()

    # initial guess should be finite for the CDF model on the actual dataset
    if init_guess[0] is None:
        imetad_guess = iMetaD_invMRT(data, beta, event=event, bias_shift=bias_shift)
        guess_candidates = []
        try:
            mle_guess = EATR_MLE_rate(data, beta, event=event, gamma_bounds=gamma_bounds, cores=cores, logTrick=logTrick, reg_lambda=reg_lambda, do_bopt=do_bopt, bias_shift=bias_shift)
            guess_candidates.append((mle_guess[0], mle_guess[1]))
        except Exception:
            pass
        guess_candidates.extend((imetad_guess, gamma) for gamma in (0.5, 0.3, 0.1, 0.7, 0.9))
        init_guess = next((candidate for candidate in guess_candidates if guess_is_finite(candidate)), (imetad_guess, 0.5))
    elif not guess_is_finite(init_guess):
        fallback_candidates = [(init_guess[0], gamma) for gamma in (0.5, 0.3, 0.1, 0.7, 0.9)]
        init_guess = next((candidate for candidate in fallback_candidates if guess_is_finite(candidate)), init_guess)

    if not do_bopt: # No Bayesian Optimization method: instead use Bounded Brent (if λ > 0) or Levenberg-Marquardt (if λ = 0) method
        options = {
                "maxiter":100000*len(ecdfy)
        }
        if reg_lambda == 0.0:
            cdf_result, converged = _curve_fit_lenient(cdf, ecdfx_indices, ecdfy, p0=init_guess, bounds=([k_bounds[0],gamma_bounds[0]],[k_bounds[1],gamma_bounds[1]]), max_nfev=100000*len(ecdfy), sigma=ecdf_sigma(ecdfy, cdf_weights))
        else:
            cdf_result = optimize.minimize(get_cost,init_guess,options=options).x
            converged = True
    else:
        acquisition_function = acquisition.ExpectedImprovement(xi=0.1,exploration_decay=0.97,exploration_decay_delay=50)
        optimizer = bopt(
                f = lambda logk0, gamma : -get_cost((np.exp(logk0),gamma)),
                acquisition_function = acquisition_function,
                pbounds = {'logk0': (np.log(init_guess[0])-35,np.log(1/np.mean(vmb_average[final_time_indices[event],0]))+5), 'gamma': gamma_bounds},
                verbose = 0,
                random_state = 1
        )
        optimizer.probe(params={'logk0': np.log(init_guess[0]), 'gamma': init_guess[1]})
        optimizer.maximize(init_points=25, n_iter=100)
        cdf_result = np.array([np.exp(optimizer.max['params']['logk0']), optimizer.max['params']['gamma']])
        converged = True
    if require_convergence and not converged:
        raise RuntimeError("EATR CDF fit did not converge to tolerance")
    return cdf_result, converged
