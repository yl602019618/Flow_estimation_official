#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)
# shellcheck source=../versions.env
source "$project_dir/versions.env"
precision=${INCFLO_PRECISION:-$PRECISION}
precision=${precision^^}
if [[ "$precision" != "FLOAT" && "$precision" != "DOUBLE" ]]; then
    printf 'INCFLO_PRECISION must be FLOAT or DOUBLE, got %s\n' "$precision" >&2
    exit 2
fi
variant=${precision,,}

work_root=${INCFLO_WORK_ROOT:-$HOME/flow_estimation_incflo}
source_root="$work_root/src"
build_log_root="$work_root/build_logs"
mkdir -p "$source_root" "$build_log_root" "$work_root/bin"

clone_at_commit() {
    local repository=$1
    local commit=$2
    local destination=$3
    if [[ -e "$destination" && ! -d "$destination/.git" ]]; then
        printf 'Refusing non-git destination: %s\n' "$destination" >&2
        return 2
    fi
    if [[ ! -d "$destination/.git" ]]; then
        mkdir -p "$destination"
        git -C "$destination" init
        git -C "$destination" remote add origin "$repository"
    fi
    local actual_remote
    actual_remote=$(git -C "$destination" remote get-url origin)
    if [[ "$actual_remote" != "$repository" ]]; then
        printf 'Remote mismatch for %s: %s\n' "$destination" "$actual_remote" >&2
        return 2
    fi
    if ! git -C "$destination" cat-file -e "${commit}^{commit}" 2>/dev/null; then
        git -C "$destination" fetch --depth 1 origin "$commit"
    fi
    if [[ "$(git -C "$destination" rev-parse HEAD 2>/dev/null || true)" != "$commit" ]]; then
        git -C "$destination" checkout --detach "$commit"
    fi
    test "$(git -C "$destination" rev-parse HEAD)" = "$commit"
}

clone_at_commit "$INCFLO_REPOSITORY" "$INCFLO_COMMIT" "$source_root/incflo"
clone_at_commit "$AMREX_REPOSITORY" "$AMREX_COMMIT" "$source_root/amrex"
clone_at_commit "$AMREX_HYDRO_REPOSITORY" "$AMREX_HYDRO_COMMIT" "$source_root/AMReX-Hydro"

if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    printf 'CUDA compiler not found: %s/bin/nvcc\n' "$CUDA_HOME" >&2
    exit 2
fi

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
build_log="$build_log_root/build_${timestamp}.log"
make_dir="$source_root/incflo/test_3d"
jobs=${INCFLO_BUILD_JOBS:-8}

printf 'Building incflo at %s with %s jobs\n' "$INCFLO_COMMIT" "$jobs" | tee "$build_log"
(
    cd "$make_dir"
    env PATH="$CUDA_HOME/bin:$PATH" \
        AMREX_HOME="$source_root/amrex" \
        AMREX_HYDRO_HOME="$source_root/AMReX-Hydro" \
        CUDA_HOME="$CUDA_HOME" \
        CUDA_ARCH="$CUDA_ARCH" \
        nice -n 10 make -j"$jobs" \
        DIM=3 PRECISION="$precision" COMP=gnu \
        USE_MPI=FALSE USE_OMP=FALSE USE_CUDA=TRUE USE_EB=TRUE \
        USE_CUDA_FAST_MATH=TRUE CUDA_VERBOSE=FALSE \
        DEBUG=FALSE TINY_PROFILE=FALSE
) 2>&1 | tee -a "$build_log"

if [[ "$precision" == "FLOAT" ]]; then
    executable="$make_dir/incflo3d.gnu.FLOAT.CUDA.EB.ex"
else
    # AMReX GNUmake omits the precision token for its default DOUBLE build.
    executable="$make_dir/incflo3d.gnu.CUDA.EB.ex"
fi
if [[ ! -x "$executable" ]]; then
    printf 'Expected incflo executable was not produced: %s\n' "$executable" >&2
    exit 2
fi
target="$work_root/bin/incflo3d_water_usct_${variant}"
cp -f "$executable" "$target"
"$target" --describe > "$work_root/bin/build_description_${variant}.txt"

manifest="$work_root/bin/build_manifest_${variant}.txt"
cat > "$manifest" <<EOF
incflo=$INCFLO_COMMIT
amrex=$AMREX_COMMIT
amrex_hydro=$AMREX_HYDRO_COMMIT
cuda_home=$CUDA_HOME
cuda_arch=$CUDA_ARCH
precision=$precision
mpi=false
openmp=false
embedded_boundary=true
cuda_fast_math=true
build_log=$build_log
EOF

if [[ "$precision" == "FLOAT" ]]; then
    cp -f "$target" "$work_root/bin/incflo3d_water_usct"
    cp -f "$manifest" "$work_root/bin/build_manifest.txt"
    cp -f "$work_root/bin/build_description_${variant}.txt" \
        "$work_root/bin/build_description.txt"
fi
printf 'Executable: %s\n' "$target"
