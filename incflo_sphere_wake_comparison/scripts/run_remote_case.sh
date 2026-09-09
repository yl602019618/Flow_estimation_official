#!/usr/bin/env bash
set -euo pipefail

case_name=${1:?case name required: smoke|sphere100_noio|sphere100_output|sphere100_double_strict_output|sphere100_double_strict_high144_output|sphere_mature40_double_strict_high144|baseline100_noio|baseline_t30p969_double_strict_high144}
gpu_index=${2:?physical GPU index required}
script_dir=$(cd "$(dirname "$0")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)
# shellcheck source=../versions.env
source "$project_dir/versions.env"
work_root=${INCFLO_WORK_ROOT:-$HOME/flow_estimation_incflo}
variant=${INCFLO_VARIANT:-float}
if [[ "$variant" != "float" && "$variant" != "double" ]]; then
    printf 'INCFLO_VARIANT must be float or double, got %s\n' "$variant" >&2
    exit 2
fi
executable="$work_root/bin/incflo3d_water_usct_${variant}"

case "$case_name" in
    smoke) input_name=inputs.sphere_8m_smoke ;;
    sphere100_noio) input_name=inputs.sphere_8m_production_100_noio ;;
    sphere100_output) input_name=inputs.sphere_8m_production_100_output ;;
    sphere100_double_strict_output) input_name=inputs.sphere_8m_double_strict_100_output ;;
    sphere100_double_strict_high144_output) input_name=inputs.sphere_8m_double_strict_high144_100_output ;;
    sphere_mature40_double_strict_high144) input_name=inputs.sphere_8m_double_strict_high144_mature_40s ;;
    baseline100_noio) input_name=inputs.baseline_8m_production_100_noio ;;
    baseline_t30p969_double_strict_high144) input_name=inputs.baseline_8m_double_strict_high144_t30p969 ;;
    *) printf 'Unknown case: %s\n' "$case_name" >&2; exit 2 ;;
esac
input_path="$project_dir/inputs/$input_name"
if [[ "$case_name" == *double_strict* && "$variant" != "double" ]]; then
    printf 'The double-strict case requires INCFLO_VARIANT=double\n' >&2
    exit 2
fi

python_bin=${INCFLO_PYTHON:-python}
env PYTHONPATH="$project_dir/.." "$python_bin" \
    -m incflo_sphere_wake_comparison.tools.validate_inputs "$input_path" >/dev/null
if [[ ! -x "$executable" ]]; then
    printf 'Missing executable; run build_remote.sh first: %s\n' "$executable" >&2
    exit 2
fi
if ! [[ "$gpu_index" =~ ^[0-9]+$ ]]; then
    printf 'GPU index must be a non-negative integer\n' >&2
    exit 2
fi

mapfile -t active_pids < <(
    nvidia-smi -i "$gpu_index" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | sed '/^[[:space:]]*$/d'
)
if [[ ${#active_pids[@]} -gt 0 ]]; then
    printf 'Refusing to start: GPU %s has active compute PIDs: %s\n' \
        "$gpu_index" "${active_pids[*]}" >&2
    exit 75
fi

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$work_root/runs/${case_name}_${variant}_${timestamp}"
mkdir -p "$run_dir"
cp "$input_path" "$run_dir/$input_name"
cp "$work_root/bin/build_manifest_${variant}.txt" "$run_dir/build_manifest.txt"
log_path="$run_dir/incflo.log"
time_path="$run_dir/time.txt"
gpu_csv="$run_dir/gpu.csv"

{
    printf 'timestamp_utc=%s\n' "$timestamp"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'gpu_index=%s\n' "$gpu_index"
    printf 'incflo_variant=%s\n' "$variant"
    nvidia-smi -i "$gpu_index" --query-gpu=name,uuid,driver_version,memory.total \
        --format=csv,noheader
    "$CUDA_HOME/bin/nvcc" --version | tail -n 1
} > "$run_dir/environment.txt"

nvidia-smi -i "$gpu_index" \
    --query-gpu=timestamp,index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits -lms 500 > "$gpu_csv" &
telemetry_pid=$!
cleanup() {
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
}
trap cleanup EXIT

set +e
(
    cd "$run_dir"
    env CUDA_VISIBLE_DEVICES="$gpu_index" AMREX_GPU_MAX_THREADS=256 \
        /usr/bin/time -v -o "$time_path" "$executable" "$run_dir/$input_name"
) > "$log_path" 2>&1
status=$?
set -e
cleanup
trap - EXIT

set +e
(
    cd "$project_dir/.."
    env PYTHONPATH="$project_dir/.." "$python_bin" \
        -m incflo_sphere_wake_comparison.tools.parse_incflo_log \
        --log "$log_path" --input "$run_dir/$input_name" \
        --time-file "$time_path" --gpu-csv "$gpu_csv" --gpu-index "$gpu_index" \
        --output "$run_dir/report.json"
) > "$run_dir/parser.log" 2>&1
parser_status=$?
set -e

printf '%s\n' "$status" > "$run_dir/exit_status.txt"
printf '%s\n' "$run_dir" > "$work_root/latest_${case_name}.txt"
if [[ $status -ne 0 || $parser_status -ne 0 ]]; then
    printf 'Case failed: solver=%s parser=%s; artifacts=%s\n' \
        "$status" "$parser_status" "$run_dir" >&2
    exit 2
fi
printf 'Case complete: %s\n' "$run_dir"
