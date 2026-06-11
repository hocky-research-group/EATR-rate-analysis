#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_BIN="${ROOT_DIR}/.venv/bin"
PYTHON_BIN="${VENV_BIN}/python"
OUT_DIR="${ROOT_DIR}/example-data/test_results_cli"
THREADS="${EATR_THREADS:-4}"
NUMBOOTS="${EATR_NUMBOOTS:-50}"
PLOT_TIME_UNIT="${EATR_PLOT_TIME_UNIT:-microseconds}"
MPL_CACHE_DIR="${ROOT_DIR}/.matplotlib-cache"
XDG_CACHE_DIR="${ROOT_DIR}/.cache"

mkdir -p "${MPL_CACHE_DIR}" "${XDG_CACHE_DIR}"
export MPLCONFIGDIR="${MPL_CACHE_DIR}"
export XDG_CACHE_HOME="${XDG_CACHE_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Expected Python interpreter under ${PYTHON_BIN}. Create the project .venv first." >&2
  exit 1
fi

set -x

mkdir -p "${OUT_DIR}/wt_regular"

PACE_STEPS=(1e2 1e3 1e4 2e4 5e4 1e5 5e5 1e6)
PACE_PS=(1 10 100 200 500 1000 5000 10000)

for idx in "${!PACE_STEPS[@]}"; do
  pace_step="${PACE_STEPS[$idx]}"
  pace_ps="${PACE_PS[$idx]}"
  pace_dir="${ROOT_DIR}/example-data/Ree_Data/E_end_end_distance_wt/eruns_pace${pace_step}"
  "${PYTHON_BIN}" -m eatr_rates \
    -i "${pace_dir}"/run_*/metad.colvar \
    --logfiles "${pace_dir}"/run_*/p.log \
    --temp 312 \
    --timeunit 1e-12 \
    --plot-time-unit "${PLOT_TIME_UNIT}" \
    --tcol 0 \
    --vcol 2 \
    --acol 4 \
    -eEmM \
    --bootstrap --numboots "${NUMBOOTS}" \
    --threads "${THREADS}" \
    -q \
    -o "${OUT_DIR}/wt_regular/pace_${pace_ps}ps.json"
done

"${PYTHON_BIN}" -m eatr_rates.plot_results regular-series \
  -i \
  "${OUT_DIR}/wt_regular/pace_1ps.json" \
  "${OUT_DIR}/wt_regular/pace_10ps.json" \
  "${OUT_DIR}/wt_regular/pace_100ps.json" \
  "${OUT_DIR}/wt_regular/pace_200ps.json" \
  "${OUT_DIR}/wt_regular/pace_500ps.json" \
  "${OUT_DIR}/wt_regular/pace_1000ps.json" \
  "${OUT_DIR}/wt_regular/pace_5000ps.json" \
  "${OUT_DIR}/wt_regular/pace_10000ps.json" \
  --xvalues 1 10 100 200 500 1000 5000 10000 \
  --labels 1e2 1e3 1e4 2e4 5e4 1e5 5e5 1e6 \
  --xlabel "MetaD hill deposition pace (ps)" \
  --method eatr-cdf imetad-cdf \
  --truerate 0.35967608559103206 \
  --time-unit "${PLOT_TIME_UNIT}" \
  -o "${OUT_DIR}/wt_regular_series.pdf"


"${PYTHON_BIN}" -m eatr_rates.rates_eatr_opes \
  -i "${ROOT_DIR}/example-data/Ree_Data/E_end_end_distance_opes/eruns_barr5"/run_*/opes_short.colvar --barrier 5 \
  -i "${ROOT_DIR}/example-data/Ree_Data/E_end_end_distance_opes/eruns_barr7"/run_*/opes_short.colvar --barrier 7 \
  -i "${ROOT_DIR}/example-data/Ree_Data/E_end_end_distance_opes/eruns_barr9"/run_*/opes_short.colvar --barrier 9 \
  -i "${ROOT_DIR}/example-data/Ree_Data/E_end_end_distance_opes/eruns_barr11"/run_*/opes_short.colvar --barrier 11 \
  -i "${ROOT_DIR}/example-data/Ree_Data/E_end_end_distance_opes/eruns_barr13"/run_*/opes_short.colvar --barrier 13 \
  --logfiles "${ROOT_DIR}/example-data/Ree_Data/E_end_end_distance_opes/eruns_barr5"/run_*/p.log \
  --logfiles "${ROOT_DIR}/example-data/Ree_Data/E_end_end_distance_opes/eruns_barr7"/run_*/p.log \
  --logfiles "${ROOT_DIR}/example-data/Ree_Data/E_end_end_distance_opes/eruns_barr9"/run_*/p.log \
  --logfiles "${ROOT_DIR}/example-data/Ree_Data/E_end_end_distance_opes/eruns_barr11"/run_*/p.log \
  --logfiles "${ROOT_DIR}/example-data/Ree_Data/E_end_end_distance_opes/eruns_barr13"/run_*/p.log \
  --temp 312 \
  --timeunit 1e-12 \
  --plot-time-unit "${PLOT_TIME_UNIT}" \
  --tcol 0 \
  --vcol 4 \
  --bootstrap --numboots "${NUMBOOTS}" \
  --threads "${THREADS}" \
  --condition-label "OPES barrier" \
  --condition-unit "kJ mol^-1" \
  --title-prefix "OPES flooding CLI" \
  --plot-prefix "${OUT_DIR}/opes_cli" \
  --truerate 0.35967608559103206 \
  -q \
  -o "${OUT_DIR}/opes_flooding.json"

# Alternatively, generate plots separately from an existing JSON (e.g. to
# change labels or time units without rerunning the analysis):
# "${PYTHON_BIN}" -m eatr_rates.plot_results flooding \
#   -i "${OUT_DIR}/opes_flooding.json" \
#   --condition-label "OPES barrier" \
#   --condition-unit "kJ mol^-1" \
#   --title-prefix "OPES flooding CLI" \
#   --time-unit "${PLOT_TIME_UNIT}" \
#   --truerate 0.35967608559103206 \
#   -o "${OUT_DIR}/opes_cli"

"${PYTHON_BIN}" -m eatr_rates.rates_eatr_opes \
  -i "${ROOT_DIR}/example-data/Ree_Data/E_end_end_distance_wt/eruns_pace1e2"/run_*/metad.colvar --barrier 1 \
  -i "${ROOT_DIR}/example-data/Ree_Data/E_end_end_distance_wt/eruns_pace1e3"/run_*/metad.colvar --barrier 10 \
  -i "${ROOT_DIR}/example-data/Ree_Data/E_end_end_distance_wt/eruns_pace1e4"/run_*/metad.colvar --barrier 100 \
  -i "${ROOT_DIR}/example-data/Ree_Data/E_end_end_distance_wt/eruns_pace2e4"/run_*/metad.colvar --barrier 200 \
  -i "${ROOT_DIR}/example-data/Ree_Data/E_end_end_distance_wt/eruns_pace5e4"/run_*/metad.colvar --barrier 500 \
  -i "${ROOT_DIR}/example-data/Ree_Data/E_end_end_distance_wt/eruns_pace1e5"/run_*/metad.colvar --barrier 1000 \
  -i "${ROOT_DIR}/example-data/Ree_Data/E_end_end_distance_wt/eruns_pace5e5"/run_*/metad.colvar --barrier 5000 \
  -i "${ROOT_DIR}/example-data/Ree_Data/E_end_end_distance_wt/eruns_pace1e6"/run_*/metad.colvar --barrier 10000 \
  --logfiles "${ROOT_DIR}/example-data/Ree_Data/E_end_end_distance_wt/eruns_pace1e2"/run_*/p.log \
  --logfiles "${ROOT_DIR}/example-data/Ree_Data/E_end_end_distance_wt/eruns_pace1e3"/run_*/p.log \
  --logfiles "${ROOT_DIR}/example-data/Ree_Data/E_end_end_distance_wt/eruns_pace1e4"/run_*/p.log \
  --logfiles "${ROOT_DIR}/example-data/Ree_Data/E_end_end_distance_wt/eruns_pace2e4"/run_*/p.log \
  --logfiles "${ROOT_DIR}/example-data/Ree_Data/E_end_end_distance_wt/eruns_pace5e4"/run_*/p.log \
  --logfiles "${ROOT_DIR}/example-data/Ree_Data/E_end_end_distance_wt/eruns_pace1e5"/run_*/p.log \
  --logfiles "${ROOT_DIR}/example-data/Ree_Data/E_end_end_distance_wt/eruns_pace5e5"/run_*/p.log \
  --logfiles "${ROOT_DIR}/example-data/Ree_Data/E_end_end_distance_wt/eruns_pace1e6"/run_*/p.log \
  --temp 312 \
  --timeunit 1e-12 \
  --plot-time-unit "${PLOT_TIME_UNIT}" \
  --tcol 0 \
  --vcol 2 \
  --acol 4 \
  --nooffset \
  --bootstrap --numboots "${NUMBOOTS}" \
  --threads "${THREADS}" \
  --condition-label "MetaD pace" \
  --condition-unit "ps" \
  --title-prefix "WT flooding CLI" \
  --plot-prefix "${OUT_DIR}/wt_cli" \
  --truerate 0.35967608559103206 \
  -q \
  -o "${OUT_DIR}/wt_flooding.json"

# Alternatively, generate plots separately from an existing JSON:
# "${PYTHON_BIN}" -m eatr_rates.plot_results flooding \
#   -i "${OUT_DIR}/wt_flooding.json" \
#   --condition-label "MetaD pace" \
#   --condition-unit "ps" \
#   --title-prefix "WT flooding CLI" \
#   --time-unit "${PLOT_TIME_UNIT}" \
#   --truerate 0.35967608559103206 \
#   -o "${OUT_DIR}/wt_cli"
