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
def get_data(colvars,time_col,bias_col,acc_col=None,maxbias_col=None,time_scale_factor=1.0): # Changed "file_format" to "colvars"
    if len(colvars) == 0:
        sys.exit(f"ERROR: No COLVAR files provided.")
    data = []
    for colvar in colvars:
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
        traj[:,0] *= time_scale_factor
        data.append(traj)
    return data

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

def iMetaD_FitCDF(data, beta, event=None, bias_shift=0.0, k_bounds=(-np.inf,np.inf), k_guess=None):
    if event is None:
        event = _default_event(data) # Assume all simulations transition unless told otherwise
    times = iMetaD_rescaled_times(data, beta, bias_shift=bias_shift)
    
    # Construct Empirical CDF
    ecdfx = np.sort(times[event])
    ecdfy = np.arange(1, event.sum()+1) / len(data)

    if k_guess is None:
        k_guess = event.sum() / np.sum(times) # Use maximum likelihood estimate as initial guess if the guess is not provided

    # Fit Poisson distribution CDF to data using Levinberg-Marquardt Method
    return optimize.curve_fit(lambda k,t:1-np.exp(-k*t), ecdfx, ecdfy, p0=k_guess, bounds=k_bounds)[0][0]

def iMetaD_FitCDF_times(times, event=None, k_bounds=(-np.inf,np.inf), k_guess=None):
    if event is None:
        event = _default_event(times) # Assume all simulations transition unless told otherwise

    # Construct Empirical CDF
    ecdfx = np.sort(times[event])
    ecdfy = np.arange(1, event.sum()+1) / len(times)

    if k_guess is None:
        k_guess = event.sum() / np.sum(times) # Use maximum likelihood estimate as initial guess if the guess is not provided

    # Fit Poisson distribution CDF to data using Levinberg-Marquardt Method
    return optimize.curve_fit(lambda k,t:1-np.exp(-k*t), ecdfx, ecdfy, p0=k_guess, bounds=k_bounds)[0][0]


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
def KTR_CDF_rate(data, beta, event=None, k_bounds=(-np.inf,np.inf), gamma_bounds=(0.,1.), cores=1, logTrick=False, init_guess=[None,None], reg_lambda=0.0, kIMD=1.0, do_bopt=False, bias_shift=0.0):

    # Get Vmb(t) and final_time_indices
    vmb_average = avg_max_bias(data, beta, bias_shift=bias_shift)
    final_time_indices = np.array([int(len(traj)-1) for traj in data])
    if event is None:
        event = _default_event(final_time_indices)
    if init_guess[0] is None:
        init_guess = (iMetaD_invMRT(data, beta, event=event, bias_shift=bias_shift),0.9)
    
    # 2-parameter CDF fitting for gamma and k0
    ecdfx_indices = np.sort(final_time_indices)
    ecdfy = np.arange(1, event.sum()+1) / len(data)

    if not do_bopt: # No Bayesian Optimization method: instead use Bounded Brent (if λ > 0) or Levenberg-Marquardt (if λ = 0) method
        options = {
            "maxiter":1000000
        }
        if reg_lambda == 0:
            cdf = lambda time_indices, k0, gamma: KTR_CDF(time_indices, k0, gamma, vmb_average, cores=cores, logTrick=logTrick)
            cdf_result = optimize.curve_fit(cdf, ecdfx_indices, ecdfy, p0=init_guess, bounds=([k_bounds[0],gamma_bounds[0]],[k_bounds[1],gamma_bounds[1]]), max_nfev=100000*len(ecdfy))[0]
        else:
            leastsq = lambda params: KTR_leastsq_cost(params, ecdfx_indices, ecdfy, vmb_average, cores=cores, logTrick=logTrick, reg_lambda=reg_lambda, kIMD=kIMD)
            cdf_result = optimize.minimize(leastsq,init_guess,options=options).x
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
        cdf_result = (np.exp(optimizer.max['params']['logk0']),optimizer.max['params']['gamma'])
    return cdf_result

# KTR Get CDF rate estimate (with precomputed Vmb(t) and ti indices)
def KTR_CDF_rate_VMB(vmb_average, final_time_indices, event=None, k_bounds=(-np.inf,np.inf), gamma_bounds=(0.,1.), cores=1, logTrick=False, init_guess=[None,None], reg_lambda=0.0, kIMD=None, do_bopt=False):
    
    if event is None:
        event = _default_event(final_time_indices)
    if init_guess[0] is None:
        init_guess = (1/np.mean(vmb_average[final_time_indices,0]),0.9)
    if kIMD is None:
        kIMD = init_guess[0]
    
    # 2-parameter CDF fitting for gamma and k0
    ecdfx_indices = np.sort(final_time_indices)
    ecdfy = np.arange(1, event.sum()+1) / len(final_time_indices)

    if not do_bopt: # No Bayesian Optimization method: instead use Bounded Brent (if λ > 0) or Levenberg-Marquardt (if λ = 0) method
        options = {
            "maxiter":1000000
        }
        if reg_lambda == 0:
            cdf = lambda time_indices, k0, gamma: KTR_CDF(time_indices, k0, gamma, vmb_average, cores=cores, logTrick=logTrick)
            cdf_result = optimize.curve_fit(cdf, ecdfx_indices, ecdfy, p0=init_guess, bounds=([k_bounds[0],gamma_bounds[0]],[k_bounds[1],gamma_bounds[1]]), max_nfev=100000*len(ecdfy))[0]
        else:
            leastsq = lambda params: KTR_leastsq_cost(params, ecdfx_indices, ecdfy, vmb_average, cores=cores, logTrick=logTrick, reg_lambda=reg_lambda, kIMD=kIMD)
            cdf_result = optimize.minimize(leastsq,init_guess,options=options).x
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
        cdf_result = (np.exp(optimizer.max['params']['logk0']),optimizer.max['params']['gamma'])
    return cdf_result

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
def EATR_CDF_rate(data, beta, event=None, k_bounds=(0.,np.inf), gamma_bounds=(0.,1.), cores=1, init_guess=[None,None], logTrick=False, reg_lambda=0.0, kIMD=1.0, do_bopt=False, bias_shift=0.0):

    # Get final_time_indices
    final_time_indices = np.array([int(len(traj)-1) for traj in data])
    if event is None:
        event = _default_event(final_time_indices)

    # 2-parameter CDF fitting for gamma and k0
    ecdfx_indices = np.sort(final_time_indices)
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
            cdf_result = optimize.curve_fit(cdf, ecdfx_indices, ecdfy, p0=init_guess, bounds=([k_bounds[0],gamma_bounds[0]],[k_bounds[1],gamma_bounds[1]]), max_nfev=100000*len(ecdfy))[0]
        else:
            cdf_result = optimize.minimize(get_cost,init_guess,options=options).x
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
        cdf_result = (np.exp(optimizer.max['params']['logk0']),optimizer.max['params']['gamma'])
    return cdf_result
