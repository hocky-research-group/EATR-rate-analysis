# EATR Rate Analysis

This repository provides command-line tools and Python modules for estimating unbiased rate constants from biased molecular dynamics simulations.

It supports:

- infrequent metadynamics / WT-MetaD style analyses
- OPES flooding analyses
- KTR and EATR estimators
- EATR-flooding across multiple sets of simulations with different bias strengths

The packaged commands are:

- `eatr-analysis`
- `eatr-flooding-analysis`
- `eatr-check-order`
- `eatr-analysis-plot`

The repository also includes config-driven analysis scripts for local dataset trees:

- `scripts/analyze_opes_dataset.py`
- `scripts/analyze_imetad_dataset.py`

Those scripts are intended for batch analysis of directory-structured datasets and can read `analysis.toml` files stored directly inside the relevant data folders.

## Theory

Rare-event kinetics are often estimated from biased simulations by relating the observed transition times under bias to an underlying unbiased rate constant `k0`.

This repository includes several related estimators:

- `iMetaD`
  Uses the standard infrequent metadynamics rescaling idea, where the observed time is accelerated by the bias.
- `KTR`
  Introduces a fitted efficiency parameter `γ` to account for the fact that the biased collective variable may not be an ideal reaction coordinate.
- `EATR`
  Uses an exponential average of the time-dependent bias to estimate both `k0` and `γ`. This is the main estimator introduced in Mazzaferro et al., JCTC 2024.
- `EATR-flooding`
  Extends the same idea to quasi-static or flooding-style biasing, especially OPES flooding, by comparing multiple sets of simulations performed with different bias strengths.

The relevant papers included in [papers](papers) are:

- [52_2024_Mazzaferro_EATR_JCTC.pdf](papers/52_2024_Mazzaferro_EATR_JCTC.pdf)
- [52_2024_Mazzaferro_EATR_JCTC_SI.pdf](papers/52_2024_Mazzaferro_EATR_JCTC_SI.pdf)
- [eatr-flooding-plusSI-arxiv.pdf](papers/eatr-flooding-plusSI-arxiv.pdf)

Practical guidance:

- Use `eatr-analysis` for time-dependent MetaD-style biasing.
- Use `eatr-flooding-analysis` for OPES flooding or any workflow where you intentionally vary the amount of bias across multiple simulation sets.
- You can also apply `eatr-flooding-analysis` to MetaD if you have multiple sets with different hill-deposition paces or other systematically varied biasing conditions.

## Installation

Install as a package:

```bash
pip install .
```

For development:

```bash
pip install -e ".[dev]"
```


## Command Overview

### `eatr-analysis`

This command analyzes one collection of trajectories from a single biasing protocol. It can compute:

- `iMetaD MLE`
- `iMetaD CDF`
- `KTR MLE`
- `KTR CDF`
- `EATR MLE`
- `EATR CDF`

Typical usage:

```bash
eatr-analysis -i run_*/*.colvar --temp 310 -E
```

Important arguments:

- `-i`, `--input`
  Input COLVAR files.
- `-o`, `--output`
  Output JSON file. Default: `rates.json`.
- `--temp`, `--kt`, `--beta`
  Mutually exclusive ways to specify temperature.
- `--timeunit`
  Conversion factor from the time unit in the COLVAR file to seconds.
- `--energyunit`
  Conversion factor from the energy unit in the COLVAR file to kJ/mol.
- `--tcol`, `--vcol`
  Time and bias column indices.
- `--acol`
  Acceleration-factor column index if present.
- `--mcol`
  Max-bias column index if present.
- `--logfiles`, `--maxlen`, `--maxtime`, `--numevents`
  Ways to determine which runs actually transitioned. Use exactly one when not all trajectories transition.
- `--threads`
  Number of parallel workers. When `--bootstrap` is set, runs bootstrap resamples concurrently using threads. Otherwise, used as the multiprocessing core count for inner CDF computations.
- `-m`/`-M`, `-k`/`-K`, `-e`/`-E`
  Select estimator(s). Lowercase is MLE, uppercase is CDF fitting:
  `-m`/`-M` iMetaD, `-k`/`-K` KTR, `-e`/`-E` EATR.
  Multiple flags can be combined (e.g. `-eE` runs both EATR MLE and CDF).
- `-b`, `--bootstrap`
  Enable bootstrap uncertainty analysis. Use `--numboots` to set the number
  of resamples (default: 100).
- `--plot-time-unit`
  Time unit for rate values in plots and JSON metadata (e.g. `microseconds`,
  `milliseconds`, `seconds`). Does not affect the numerical computation.
  Default: `seconds`.
- `-q`, `--quiet`
  Suppress terminal printing and only write JSON output.

Notes:

- `KTR` and regular `EATR` are not intended for OPES flooding trajectories. Use `eatr-flooding-analysis` for those.
- If your COLVAR file includes an acceleration column, passing `--acol` is preferable.
- If your COLVAR files were written in femtoseconds and you want SI rates, use `--timeunit 1e-15`.

### `eatr-flooding-analysis`

This command analyzes multiple sets of trajectories collected under different bias strengths and estimates a single unbiased `k0` plus a single `γ`.

Typical usage:

```bash
eatr-flooding-analysis \
  -i barrier5/*.colvar --barrier 5 \
  -i barrier10/*.colvar --barrier 10 \
  -i barrier15/*.colvar --barrier 15 \
  --temp 310
```

Equivalent form:

```bash
eatr-flooding-analysis \
  -i barrier5/*.colvar \
  -i barrier10/*.colvar \
  -i barrier15/*.colvar \
  --barriers 5 10 15 \
  --temp 310
```

Important arguments:

- `-i`, `--input`
  Supply one group of trajectory files per simulation set.
- `--barrier` or `--barriers`
  Bias-strength labels for each set. For OPES, this should usually be the PLUMED `BARRIER` value.
- `--timeunit`, `--energyunit`, `--temp`, `--kt`, `--beta`
  Unit and temperature handling, as in `eatr-analysis`.
- `--tcol`, `--vcol`, `--acol`
  Time, bias, and optional acceleration columns.
- `--threads`
  Run independent set/bootstrap work in parallel.
- `--logfiles`, `--maxlen`, `--maxtime`, `--numevents`
  Set-wise transition detection.
- `-b`, `--bootstrap`, `--numboots`
  Enable bootstrap uncertainty analysis. `--numboots` sets the number of
  resamples (default: 100).
- `--plot-time-unit`
  Time unit for rate values in generated plots and JSON metadata. Default: `seconds`.
- `--cdf`
  Fit the observed rate for each set using the CDF instead of the MLE.
- `--timefirst`
  Compute the exponential average `<e^{βγV}>` by first averaging over time
  within each trajectory and then over trajectories. Default is to average
  over trajectories first.
- `--nooffset`
  Disable automatic addition of the OPES barrier offset to the reported bias.
- `--opesf`
  Also report the standard OPES-flooding estimate alongside EATR-flooding.
- `--plot-prefix`
  Prefix for the generated flooding figures. By default this is derived from `--output`, so `-o opes_flooding.json` writes `opes_flooding_observed_rate.png`, `opes_flooding_ln_kobs_vs_acceleration.png`, and `opes_flooding_diagnostics.png`.
- `--condition-label`, `--condition-unit`, `--title-prefix`
  Labels used in the generated flooding figures.
- `--no-plots`
  Disable the automatic figure generation if you only want the JSON output.

Notes:

- For OPES data produced with `OPES_METAD ... BARRIER=...`, you usually want to pass the same `BARRIER` values here and leave `--nooffset` unset.
- The method is also useful for MetaD if you have several sets with systematically varied deposition pace.
- The command writes the flooding JSON and the diagnostic figures in one pass. Those plots are intended to be inspected together with the numerical fit.

### `eatr-check-order`

This helper writes the expanded order of COLVAR files, optionally paired with log files, so you can verify shell glob expansion and pairing.

Example:

```bash
eatr-check-order -i run_*/metad.colvar -l run_*/p.log -o order.dat
```

### `eatr-analysis-plot`

This plotting helper consumes JSON outputs written by `eatr-analysis` or `eatr-flooding-analysis` and generates figures without rerunning the numerical analysis. It is useful if you want to replot with different labels, a different output prefix, or a different time unit, since `eatr-flooding-analysis` already generates flooding plots by default.

Regular-series example (default method is `eatr-cdf`; pass multiple methods to overlay them):

```bash
eatr-analysis-plot regular-series \
  -i pace_1ps.json pace_10ps.json pace_100ps.json \
  --xvalues 1 10 100 \
  --xlabel "MetaD hill deposition pace (ps)" \
  --method eatr-cdf imetad-cdf \
  -o wt_regular_series.png
```

Flooding example:

```bash
eatr-analysis-plot flooding \
  -i opes_flooding.json \
  --condition-label "OPES barrier" \
  --condition-unit "kJ mol^-1" \
  --title-prefix "OPES flooding" \
  -o opes_figures
```

## Config-Driven Dataset Scripts

The packaged CLI tools are best when you want to specify inputs explicitly on the command line. For repeated analysis of a filesystem dataset with fixed conventions, use the local scripts plus TOML config files.

Example config files:

- [example-data/Ree_Data/E_end_end_distance_opes/analysis.toml](example-data/Ree_Data/E_end_end_distance_opes/analysis.toml)
- [example-data/Ree_Data/E_end_end_distance_wt/analysis.toml](example-data/Ree_Data/E_end_end_distance_wt/analysis.toml)

These configs control:

- input and output roots
- time and energy unit conversions
- temperature
- bias and acceleration column indices
- directory naming conventions
- bootstrap count
- OPES barrier filtering

### OPES dataset script

Run the Ree OPES example from the `analysis.toml` stored in that data folder:

```bash
EATR_THREADS=4 .venv/bin/python scripts/analyze_opes_dataset.py
```

That config points at [example-data/Ree_Data/E_end_end_distance_opes](example-data/Ree_Data/E_end_end_distance_opes) and sets:

- `timeunit_seconds = 1e-12`
- `temperature_k = 312.0`
- `bias_col = 4`
- directory prefixes `eruns_barr*` and `run_*`

To adapt this to a different OPES dataset, copy the TOML and change the roots, column indices, and directory/file naming conventions.

Restrict to one configured CV:

```bash
EATR_THREADS=4 .venv/bin/python scripts/analyze_opes_dataset.py \
  --cv E_end_end_distance_opes
```

Outputs per CV:

- flooding summary JSON
- flooding diagnostics plot
- `ln(k_obs)` vs barrier plot
- slope-style `ln(k_obs)` vs acceleration plot

### iMetaD dataset script

Run the Ree MetaD example from the `analysis.toml` stored in that data folder:

```bash
EATR_THREADS=4 .venv/bin/python scripts/analyze_imetad_dataset.py
```

That config points at [example-data/Ree_Data/E_end_end_distance_wt](example-data/Ree_Data/E_end_end_distance_wt) and sets:

- `timeunit_seconds = 1e-12`
- `timestep_ps = 0.01`
- `temperature_k = 312.0`
- `bias_col = 2`
- `acc_col = 4`
- `use_height_dirs = false` because the Ree MetaD example has `eruns_pace*` directly under the dataset root

Restrict to one configured CV:

```bash
EATR_THREADS=4 .venv/bin/python scripts/analyze_imetad_dataset.py \
  --cv E_end_end_distance_wt
```

Outputs per CV/height series:

- regular-EATR summary JSON
- `ln(k0)` and `gamma` vs pace plot
- `ln(k_obs)` vs pace plot

## Example Data

The repository includes two example collections under [example-data/Ree_Data](example-data/Ree_Data):

- [E_end_end_distance_opes](example-data/Ree_Data/E_end_end_distance_opes)
  OPES flooding simulations with sets `eruns_barr5`, `7`, `9`, `11`, `13`
- [E_end_end_distance_wt](example-data/Ree_Data/E_end_end_distance_wt)
  WT-MetaD simulations with sets `eruns_pace1e2`, `1e3`, `1e4`, `2e4`, `5e4`, `1e5`, `5e5`, `1e6`

For these protein G examples, PLUMED was configured to use the default unit of `ps`, so the correct timeunit parameter is:

```bash
--timeunit 1e-12
```

The temperature used in the examples is:

```bash
--temp 312
```

## Worked Commands

### 1. OPES flooding example

To analyze the OPES datasets with EATR-flooding:

```bash
eatr-flooding-analysis \
  -i example-data/Ree_Data/E_end_end_distance_opes/eruns_barr5/run_*/opes_short.colvar --barrier 5 \
  -i example-data/Ree_Data/E_end_end_distance_opes/eruns_barr7/run_*/opes_short.colvar --barrier 7 \
  -i example-data/Ree_Data/E_end_end_distance_opes/eruns_barr9/run_*/opes_short.colvar --barrier 9 \
  -i example-data/Ree_Data/E_end_end_distance_opes/eruns_barr11/run_*/opes_short.colvar --barrier 11 \
  -i example-data/Ree_Data/E_end_end_distance_opes/eruns_barr13/run_*/opes_short.colvar --barrier 13 \
  --logfiles example-data/Ree_Data/E_end_end_distance_opes/eruns_barr5/run_*/p.log \
  --logfiles example-data/Ree_Data/E_end_end_distance_opes/eruns_barr7/run_*/p.log \
  --logfiles example-data/Ree_Data/E_end_end_distance_opes/eruns_barr9/run_*/p.log \
  --logfiles example-data/Ree_Data/E_end_end_distance_opes/eruns_barr11/run_*/p.log \
  --logfiles example-data/Ree_Data/E_end_end_distance_opes/eruns_barr13/run_*/p.log \
  --temp 312 \
  --timeunit 1e-12 \
  --tcol 0 \
  --vcol 4 \
  --opesf
```

Why these columns:

- in `opes_short.colvar`, column 0 is time
- column 4 is `opes.bias`

### 2. Regular EATR on one WT-MetaD set

For a single MetaD set such as `eruns_pace1e4`:

```bash
eatr-analysis \
  -i example-data/Ree_Data/E_end_end_distance_wt/eruns_pace1e4/run_*/metad.colvar \
  --logfiles example-data/Ree_Data/E_end_end_distance_wt/eruns_pace1e4/run_*/p.log \
  --temp 312 \
  --timeunit 1e-12 \
  --tcol 0 \
  --vcol 2 \
  --acol 4 \
  -eE \
  -o example-data/test_results/pace1e4_rates.json
```

Why these columns:

- in `metad.colvar`, column 0 is time
- column 2 is `metad.bias`
- column 4 is `metad.acc`

### 3. Flooding-style analysis across WT-MetaD pace sets

The flooding paper shows that EATR-flooding can also be applied to MetaD by comparing sets with different deposition pace. In that interpretation, the pace is the stepped biasing condition.

The repository includes a CLI-only example workflow that runs the packaged commands and then plots from their JSON outputs:

```bash
bash scripts/run_example_cli.sh
```

That script writes JSON and figure outputs under [example-data/test_results_cli](example-data/test_results_cli).
For the WT pace ladder it uses EATR CDF (`-E`), and for the flooding workflows it enables bootstrap uncertainty analysis. By default it uses 50 bootstrap replicas where bootstrap is enabled; for a faster smoke run you can lower that with `EATR_NUMBOOTS`, for example `EATR_NUMBOOTS=5 bash scripts/run_example_cli.sh`.

For comparison, the repository also includes the Python example runner:

```bash
.venv/bin/python scripts/run_example_analyses.py
```

That script writes bootstrap-backed summaries and plots with reported rates converted to `us^-1` and pace units in `ps`. It uses 50 trajectory-resampling bootstrap replicas per analysis.

If you want to speed up the example workflow, you can enable threaded execution over independent gamma-grid and bootstrap tasks:

```bash
EATR_THREADS=4 .venv/bin/python scripts/run_example_analyses.py
```

The generated files are:

- [wt_regular_eatr_summary.json](example-data/test_results/wt_regular_eatr_summary.json)
- [wt_regular_eatr_vs_pace.png](example-data/test_results/wt_regular_eatr_vs_pace.png)
- [wt_flooding_summary.json](example-data/test_results/wt_flooding_summary.json)
- [wt_flooding_all_paces.png](example-data/test_results/wt_flooding_all_paces.png)
- [wt_flooding_filtered_paces.png](example-data/test_results/wt_flooding_filtered_paces.png)
- [wt_observed_rate_vs_pace.png](example-data/test_results/wt_observed_rate_vs_pace.png)
- [wt_ln_kobs_vs_acceleration.png](example-data/test_results/wt_ln_kobs_vs_acceleration.png)
- [opes_flooding_summary.json](example-data/test_results/opes_flooding_summary.json)
- [opes_flooding_diagnostics.png](example-data/test_results/opes_flooding_diagnostics.png)
- [opes_observed_rate_vs_barrier.png](example-data/test_results/opes_observed_rate_vs_barrier.png)
- [opes_ln_kobs_vs_acceleration.png](example-data/test_results/opes_ln_kobs_vs_acceleration.png)

## Python Usage

The library functions remain available from Python. The packaged CLI modules separate the numerical analysis from output formatting:

- [eatr_rates/rates_cmd.py](eatr_rates/rates_cmd.py)
- [eatr_rates/rates_eatr_opes.py](eatr_rates/rates_eatr_opes.py)
- [rate_methods_library.py](rate_methods_library.py)

If you want to build automated regression tests, the easiest target is the example runner in [scripts/run_example_analyses.py](scripts/run_example_analyses.py) and the JSON outputs it writes.

## Tests

Run the unit tests with:

```bash
pytest
```

Or with the standard library test runner:

```bash
python3 -m unittest discover -s tests -v
```
