#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_BIN="${ROOT_DIR}/.venv/bin"

DATA_ROOT="${1:-${ROOT_DIR}/nnp-rate-data/imetad}"
OUT_ROOT="${2:-${ROOT_DIR}/nnp-rate-data/analysis/imetad}"

THREADS="${EATR_THREADS:-4}"
NUMBOOTS="${EATR_NUMBOOTS:-100}"
TEMP_K="${EATR_TEMP_K:-298.15}"
ENERGY_UNIT="${EATR_ENERGY_UNIT:-4.184}"
TIMEUNIT_SECONDS="${EATR_TIMEUNIT_SECONDS:-1e-12}"
PLOT_TIME_UNIT="${EATR_PLOT_TIME_UNIT:-microseconds}"
TCOL="${EATR_TCOL:-0}"
VCOL="${EATR_VCOL:-11}"
ACOL="${EATR_ACOL:-14}"
MPL_CACHE_DIR="${ROOT_DIR}/.matplotlib-cache"
XDG_CACHE_DIR="${ROOT_DIR}/.cache"

mkdir -p "${MPL_CACHE_DIR}" "${XDG_CACHE_DIR}"
export MPLCONFIGDIR="${MPL_CACHE_DIR}"
export XDG_CACHE_HOME="${XDG_CACHE_DIR}"

if [[ ! -x "${VENV_BIN}/eatr-analysis" || ! -x "${VENV_BIN}/eatr-analysis-plot" || ! -x "${VENV_BIN}/eatr-flooding-analysis" ]]; then
  echo "Expected CLI tools under ${VENV_BIN}. Install the package into .venv first." >&2
  exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Data root does not exist: ${DATA_ROOT}" >&2
  exit 1
fi

shopt -s nullglob

mkdir -p "${OUT_ROOT}"

collect_cv_dirs() {
  local root="$1"
  local cv_dirs=("${root}"/bias_*)
  if [[ ${#cv_dirs[@]} -gt 0 && -d "${cv_dirs[0]}" ]]; then
    printf '%s\n' "${cv_dirs[@]}"
    return
  fi
  printf '%s\n' "${root}"
}

run_series() {
  local cv_name="$1"
  local bf_name="$2"
  local height_dir="$3"
  local height_name
  local series_out
  local json_out
  local flooding_json
  local flooding_prefix
  local pace_dir
  local pace_name
  local pace_value
  local pace_ps
  local output_json
  local sorted_pace_dirs=()
  local insert_index
  local existing_pace_value
  local input_globs=()
  local logfile_globs=()
  local barrier_args=()
  local plot_inputs=()
  local plot_xvalues=()
  local plot_labels=()

  height_name="$(basename "${height_dir}")"
  series_out="${OUT_ROOT}/${cv_name}/${bf_name}/${height_name}"
  json_out="${series_out}/json"
  flooding_json="${series_out}/${cv_name}_${bf_name}_${height_name}_flooding.json"
  flooding_prefix="${series_out}/${cv_name}_${bf_name}_${height_name}_flooding"
  mkdir -p "${json_out}"

  for pace_dir in "${height_dir}"/pace*; do
    [[ -d "${pace_dir}" ]] || continue
    pace_name="$(basename "${pace_dir}")"
    pace_value="${pace_name#pace}"
    insert_index=${#sorted_pace_dirs[@]}
    while [[ ${insert_index} -gt 0 ]]; do
      existing_pace_value="${sorted_pace_dirs[$((insert_index - 1))]}"
      existing_pace_value="$(basename "${existing_pace_value}")"
      existing_pace_value="${existing_pace_value#pace}"
      if ! awk -v lhs="${pace_value}" -v rhs="${existing_pace_value}" 'BEGIN { exit !(lhs < rhs) }'; then
        break
      fi
      insert_index=$((insert_index - 1))
    done
    sorted_pace_dirs=("${sorted_pace_dirs[@]:0:${insert_index}}" "${pace_dir}" "${sorted_pace_dirs[@]:${insert_index}}")
  done

  for pace_dir in "${sorted_pace_dirs[@]}"; do
    pace_name="$(basename "${pace_dir}")"
    pace_value="${pace_name#pace}"
    pace_ps="$(awk -v pace="${pace_value}" 'BEGIN { printf "%.12g", pace * 5e-4 }')"
    output_json="${json_out}/${pace_name}.json"

    echo "Analyzing ${cv_name} ${bf_name} ${height_name} ${pace_name}"
    "${VENV_BIN}/eatr-analysis" \
      --input-glob "${pace_dir}/s*/*.colvar.dat" \
      --logfiles-glob "${pace_dir}/s*/*.plumed.log" \
      --temp "${TEMP_K}" \
      --energyunit "${ENERGY_UNIT}" \
      --timeunit "${TIMEUNIT_SECONDS}" \
      --plot-time-unit "${PLOT_TIME_UNIT}" \
      --tcol "${TCOL}" \
      --vcol "${VCOL}" \
      --acol "${ACOL}" \
      --threads "${THREADS}" \
      --bootstrap --numboots "${NUMBOOTS}" \
      -M -e -E \
      -q \
      -o "${output_json}"

    plot_inputs+=("${output_json}")
    plot_xvalues+=("${pace_ps}")
    plot_labels+=("${pace_value}")
    input_globs+=("--input-glob" "${pace_dir}/s*/*.colvar.dat")
    logfile_globs+=("--logfiles-glob" "${pace_dir}/s*/*.plumed.log")
    barrier_args+=("--barrier" "${pace_ps}")
  done

  if [[ ${#plot_inputs[@]} -eq 0 ]]; then
    echo "Skipping ${cv_name}/${bf_name}/${height_name}: no completed pace analyses" >&2
    return
  fi

  "${VENV_BIN}/eatr-analysis-plot" regular-series \
    -i "${plot_inputs[@]}" \
    --xvalues "${plot_xvalues[@]}" \
    --labels "${plot_labels[@]}" \
    --xlabel "MetaD hill deposition pace (ps)" \
    --method imetad-cdf eatr-cdf \
    --time-unit "${PLOT_TIME_UNIT}" \
    -o "${series_out}/${cv_name}_${bf_name}_${height_name}_cdf_rate_vs_pace.png"

  "${VENV_BIN}/eatr-analysis-plot" regular-series \
    -i "${plot_inputs[@]}" \
    --xvalues "${plot_xvalues[@]}" \
    --labels "${plot_labels[@]}" \
    --xlabel "MetaD hill deposition pace (ps)" \
    --method eatr-cdf \
    --time-unit "${PLOT_TIME_UNIT}" \
    --cdf-output "${series_out}/${cv_name}_${bf_name}_${height_name}_eatr_cdf_overlay.png" \
    -o "${series_out}/${cv_name}_${bf_name}_${height_name}_eatr_cdf_rate_vs_pace.png"

  "${VENV_BIN}/eatr-analysis-plot" regular-series \
    -i "${plot_inputs[@]}" \
    --xvalues "${plot_xvalues[@]}" \
    --labels "${plot_labels[@]}" \
    --xlabel "MetaD hill deposition pace (ps)" \
    --method eatr-comparison \
    --time-unit "${PLOT_TIME_UNIT}" \
    -o "${series_out}/${cv_name}_${bf_name}_${height_name}_eatr_comparison_vs_pace.png"

  "${VENV_BIN}/eatr-analysis-plot" regular-series \
    -i "${plot_inputs[@]}" \
    --xvalues "${plot_xvalues[@]}" \
    --labels "${plot_labels[@]}" \
    --xlabel "MetaD hill deposition pace (ps)" \
    --method imetad-cdf \
    --time-unit "${PLOT_TIME_UNIT}" \
    --cdf-output "${series_out}/${cv_name}_${bf_name}_${height_name}_imetad_cdf_overlay.png" \
    -o "${series_out}/${cv_name}_${bf_name}_${height_name}_imetad_cdf_rate_vs_pace.png"

  "${VENV_BIN}/eatr-flooding-analysis" \
    "${input_globs[@]}" \
    "${barrier_args[@]}" \
    "${logfile_globs[@]}" \
    --temp "${TEMP_K}" \
    --energyunit "${ENERGY_UNIT}" \
    --timeunit "${TIMEUNIT_SECONDS}" \
    --plot-time-unit "${PLOT_TIME_UNIT}" \
    --tcol "${TCOL}" \
    --vcol "${VCOL}" \
    --acol "${ACOL}" \
    --threads "${THREADS}" \
    --bootstrap --numboots "${NUMBOOTS}" \
    --nooffset \
    --condition-label "MetaD pace" \
    --condition-unit "ps" \
    --title-prefix "${cv_name} ${bf_name} ${height_name} flooding" \
    --plot-prefix "${flooding_prefix}" \
    -q \
    -o "${flooding_json}"
}

for cv_dir in $(collect_cv_dirs "${DATA_ROOT}"); do
  [[ -d "${cv_dir}" ]] || continue
  cv_name="$(basename "${cv_dir}")"

  bf_dirs=("${cv_dir}"/bf*)
  if [[ ${#bf_dirs[@]} -eq 0 || ! -d "${bf_dirs[0]}" ]]; then
    bf_dirs=("${cv_dir}")
  fi

  for bf_dir in "${bf_dirs[@]}"; do
    [[ -d "${bf_dir}" ]] || continue
    bf_name="$(basename "${bf_dir}")"

    for height_dir in "${bf_dir}"/height*; do
      [[ -d "${height_dir}" ]] || continue
      run_series "${cv_name}" "${bf_name}" "${height_dir}"
    done
  done
done

echo "Wrote analysis outputs under ${OUT_ROOT}"
